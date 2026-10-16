"""Writes short replies to MELD lines with Liquid AI's LFM2.5-1.2B-Instruct on the local GPU, for serve.py to forward to.

Contents:
- Handler: answers /api/status and /api/reply on port 8768
- write_reply: builds the prompt for one line and generates the reply, with timings
- main: loads LFM2.5, warms it up and starts the server
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text_fixes"))
from transcribe import CACHE  # noqa: E402

HOST = "127.0.0.1"
PORT = 8768
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
FEELS = {"neutral": "calm", "joy": "happy", "surprise": "surprised", "anger": "angry", "sadness": "sad",
         "disgust": "disgusted", "fear": "scared"}
"""Description: MELD's seven labels, the only emotions a request may name, each with the adjective the prompt uses."""
MAX_NEW_TOKENS = 40
"""Description: The longest reply in tokens, the same cap the A40 reply benchmark used."""
MAX_TEXT = 600
"""Description: The longest line in characters the server accepts."""
SYSTEM = ("You are a warm, caring friend in a conversation. Speak directly to the person, addressing them as "
          "'you', in one short sentence of at most 20 words. Respond to how they feel, and never describe them "
          "or explain yourself.")
"""Description: The system prompt for every reply."""

model = None
tokenizer = None


def ms(seconds):
    """
    Description: Converts seconds to whole milliseconds.
    Inputs:
    - seconds: a duration in seconds.
    Outputs: the duration in milliseconds, rounded.
    """
    return round(seconds * 1000)


def write_reply(text, emotion, speaker):
    """
    Description: Generates a one-sentence reply to a line, given the emotion it was heard with.
    Inputs:
    - text: the spoken line.
    - emotion: one of the keys of FEELS.
    - speaker: the speaker's name, or an empty string.
    Outputs: a dict with the reply, the tokens generated, and the time spent in the model.
    Constraints: greedy decoding, so the same input always gives the same reply. serve.py makes calls take
    turns with the speech models, so this server runs one call at a time.
    """
    import torch
    who = speaker or "Your friend"
    user_message = (
        f'{who} says to you: "{text}"\n'
        f"{who} sounds {FEELS[emotion]}. Write what you say back."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]

    started = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    torch.cuda.synchronize()
    done = time.perf_counter()

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = out[0, prompt_length:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "reply": reply,
        "tokens": int(new_tokens.shape[0]),
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
            query = parse_qs(url.query)
            text = query.get("text", [""])[0].strip()
            emotion = query.get("emotion", [""])[0]
            speaker = query.get("speaker", [""])[0].strip()[:60]

            bad_request = emotion not in FEELS or not text or len(text) > MAX_TEXT
            if bad_request:
                self.send_json(400, {"error": "unknown emotion, or the line is empty or too long"})
                return

            self.send_json(200, write_reply(text, emotion, speaker))
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
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=str(CACHE), dtype=torch.float16).to("cuda").eval()
    loaded = time.perf_counter()
    warm = write_reply("Hi, how are you?", "neutral", "")
    print(f"LFM2.5 loaded on {torch.cuda.get_device_name(0)} in {loaded - started:.1f} s, "
          f"warm-up reply {warm['model_ms']} ms", flush=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Reply server on http://{HOST}:{PORT} (Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
