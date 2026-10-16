"""Writes short replies to MELD lines with Liquid AI's LFM2.5-1.2B-Instruct on the local GPU, for serve.py to forward to.

Contents:
- Handler: answers /api/status and /api/reply on port 8768
- parse_history: reads and checks the optional list of the dialogue's earlier lines
- clean_reply: strips copied example layout from the model's output
- scene_transcript: writes those earlier lines as a plain transcript
- reply_length: picks the reply's word target, matched to the line or now and then a long turn
- write_reply: builds the prompt for one line, alone or after its dialogue so far, and generates the reply
- main: loads LFM2.5, warms it up and starts the server
"""

import json
import re
import sys
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text_fixes"))
from transcribe import CACHE  # noqa: E402

HOST     = "127.0.0.1"
PORT     = 8768
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
FEELS    = {
    "neutral": "calm",
    "joy": "happy",
    "surprise": "surprised",
    "anger": "angry",
    "sadness": "sad",
    "disgust": "disgusted",
    "fear": "scared",
}
"""Description: MELD's seven labels, the only emotions a request may name, each with the adjective the prompt uses."""
MAX_NEW_TOKENS = 30
"""Description: The longest reply in tokens, enough for one sentence of about 20 words."""
LONG_WORDS = 20
"""Description: The word limit of a long turn, the default reply length."""
MIN_WORDS = 2
"""Description: The shortest word target, so a one-word line still gets a real answer."""
LONG_EVERY = 4
"""Description: About one line in this many gets a long turn instead of a reply matched to its length."""
MAX_TEXT = 600
"""Description: The longest line in characters the server accepts."""
MAX_HISTORY = 40
"""Description: The most earlier lines one request may carry; MELD's longest dialogue has 24 lines."""
SYSTEM = (
    "You are a warm, funny friend in the room with these people, like a friend in a sitcom, not a counsellor. "
    "Reply only to the last speaker, in one short sentence, keeping to the length you are asked for. "
    "Answer what they actually said, "
    "in a tone that fits how they feel. Never tell them how they feel, never start with 'It sounds like' or "
    "'You're feeling', and never explain yourself. Don't open with 'Hey' and their name: in a one-on-one "
    "conversation, calling someone by name makes it sound slightly serious, even hostile.\n"
    "Examples:\n"
    'Monica (happy): "I got the job!" -> "No way! Drinks are on you tonight!"\n'
    'Joey (angry): "You ate my sandwich? Again?" -> "Okay, okay, I\'ll buy you two tomorrow, I promise."'
)
"""Description: The system prompt for every reply."""

model     = None
tokenizer = None


def ms(seconds):
    """
    Description: Converts seconds to whole milliseconds.
    Inputs:
    - seconds: a duration in seconds.
    Outputs: the duration in milliseconds, rounded.
    """
    return round(seconds * 1000)


def parse_history(raw):
    """
    Description: Reads the optional history query value and checks its shape.
    Inputs:
    - raw: a JSON list of {"speaker", "text"} objects, or an empty string.
    Outputs: the list (empty when raw is empty), or None when the value is malformed or too long.
    """
    if not raw:
        return []
    try:
        history = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        return None
    for line in history:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str) or len(line["text"]) > MAX_TEXT:
            return None
        line["speaker"] = str(line.get("speaker") or "")[:60]
    return history


def clean_reply(raw):
    """
    Description: Strips layout the model copies from the prompt's examples, so only the spoken reply is left.
    Inputs:
    - raw: the decoded model output.
    Outputs: the last non-empty line, without a leading "Name:" or "->" label and without wrapping quotes.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    reply = lines[-1] if lines else ""
    reply = re.sub(r"^(->\s*|[A-Z][A-Za-z .']{0,30}:\s+)", "", reply)
    # A quoted echo of the speaker's line can come before the real reply.
    reply = re.sub(r'^"[^"]*"\s+(?=\S)', "", reply)
    return reply.strip().strip('"').strip()


def scene_transcript(history):
    """
    Description: Writes the earlier lines of a dialogue as a plain transcript, so a reply can see the scene
    without taking part in it.
    Inputs:
    - history: a list of dicts with "speaker" and "text", oldest first.
    Outputs: one "Speaker: line" row per earlier line, joined by line breaks.
    """
    return "\n".join(f'{line["speaker"] or "Someone"}: {line["text"]}' for line in history)


def reply_length(text):
    """
    Description: Picks how long the reply to a line should be: about as long as the line, or now and then a
    long turn at the default length.
    Inputs:
    - text: the spoken line.
    Outputs: (word target, whether this is a long turn, the token cap for generation).
    Constraints: the long-turn choice comes from a checksum of the text, so the same line always gets the same
    length and greedy decoding stays repeatable.
    """
    line_words = len(text.split())
    long_turn  = zlib.crc32(text.encode("utf-8")) % LONG_EVERY == 0
    target     = LONG_WORDS if long_turn else max(MIN_WORDS, min(LONG_WORDS, line_words))
    # About 1.5 tokens per word plus room to finish the sentence, never above the overall cap.
    max_tokens = MAX_NEW_TOKENS if long_turn else min(MAX_NEW_TOKENS, round(target * 1.5) + 6)
    return target, long_turn, max_tokens


def write_reply(text, emotion, speaker, history=None):
    """
    Description: Generates a one-sentence reply to a line, given the emotion it was heard with.
    Inputs:
    - text: the spoken line.
    - emotion: one of the keys of FEELS.
    - speaker: the speaker's name, or an empty string.
    - history: the dialogue's earlier lines for scene_transcript, or None to reply to this line alone.
    Outputs: a dict with the reply, the tokens generated, the prompt length in tokens, the number of earlier
    lines seen, and the time spent in the model.
    Constraints: greedy decoding, so the same input always gives the same reply. serve.py makes calls take
    turns with the speech models, so this server runs one call at a time.
    """
    import torch

    target, long_turn, max_tokens = reply_length(text)

    history      = history or []
    who          = speaker or "Your friend"
    scene        = f"The conversation so far:\n{scene_transcript(history)}\n\n" if history else ""
    length       = (f"Use at most {target} words." if long_turn
                    else f"Match their length: use about {target} words, never more than {target + 3}.")
    user_message = (
        f"{scene}"
        f'{who} now says: "{text}"\n'
        f"{who} sounds {FEELS[emotion]}. Write what you say back to {who}. {length}"
    )
    messages     = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]

    started = time.perf_counter()
    inputs  = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    done = time.perf_counter()

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens    = out[0, prompt_length:]
    reply         = clean_reply(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return {
        "reply": reply,
        "tokens": int(new_tokens.shape[0]),
        "prompt_tokens": int(prompt_length),
        "history_lines": len(history),
        "target_words": target,
        "long_turn": long_turn,
        "model_ms": ms(done - started),
        "server_ms": ms(done - started),
    }


class Handler(BaseHTTPRequestHandler):
    """
    Description: Answers the two API paths.
    Methods:
    - do_GET: routes /api/status and /api/reply.
    - send_json: writes one JSON response.
    - log_message: keeps the console quiet.
    """

    def do_GET(self):
        """
        Description: Answers a GET request.
        Inputs: none; reads the request path.
        Outputs: none
        """
        url = urlparse(self.path)

        if url.path == "/api/status":
            compute = str(model.dtype).replace("torch.", "") if model else None
            self.send_json(200, {
                "model": "LFM2.5-1.2B-Instruct",
                "device": "cuda",
                "compute": compute,
                "ready": model is not None,
            })
            return

        if url.path == "/api/reply":
            query   = parse_qs(url.query)
            text    = query.get("text", [""])[0].strip()
            emotion = query.get("emotion", [""])[0]
            speaker = query.get("speaker", [""])[0].strip()[:60]
            history = parse_history(query.get("history", [""])[0])

            bad_request = emotion not in FEELS or not text or len(text) > MAX_TEXT or history is None
            if bad_request:
                self.send_json(400, {"error": "unknown emotion, bad history, or the line is empty or too long"})
                return

            self.send_json(200, write_reply(text, emotion, speaker, history))
            return

        self.send_json(404, {"error": "not found"})

    def send_json(self, status, body):
        """
        Description: Sends a JSON response.
        Inputs:
        - status: the HTTP status code.
        - body: a dict to send as JSON.
        Outputs: none
        """
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        """
        Description: Logs nothing, so serve.py's console stays readable.
        Inputs:
        - format, args: the standard log message parts.
        Outputs: none
        """


def main():
    """
    Description: Loads LFM2.5 in fp16 on the GPU, runs one warm-up reply, then serves until stopped.
    Inputs: none
    Outputs: none
    Constraints: exits with an error when CUDA is unavailable; it never falls back to the CPU. fp16 was as fast
    as fp32 on the GTX 1070 Ti for this model, at 2.2 GB of GPU memory against 4.5 GB.
    """
    global model, tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch; LFM2.5 needs the GPU")

    started   = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE),
        dtype=torch.float16,
    ).to("cuda").eval()
    loaded    = time.perf_counter()
    warm      = write_reply("Hi, how are you?", "neutral", "")

    gpu_name = torch.cuda.get_device_name(0)
    print(
        f"LFM2.5 loaded on {gpu_name} in {loaded - started:.1f} s, "
        f"warm-up reply {warm['model_ms']} ms",
        flush=True,
    )

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Reply server on http://{HOST}:{PORT} (Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
