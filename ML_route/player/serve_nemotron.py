"""Transcribes MELD clips with NVIDIA Nemotron Speech Streaming on the local GPU, for serve.py to forward to.

Contents:
- Handler: answers /api/status and /api/transcribe on port 8766
- transcribe_clip: decodes one clip and runs Nemotron on it, with timings
- main: loads Nemotron, warms it up and starts the server
"""

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text_fixes"))
from transcribe import CACHE, CLIP_DIRS, RAW, SAMPLE_RATE, load_audio  # noqa: E402

HOST = "127.0.0.1"
PORT = 8766
MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
LOOKAHEAD = 13
"""Description: Right attention context in 80 ms frames; 13 is the card's 1.12 s setting."""
MAX_TOKENS = 448
"""Description: The most text tokens one clip may produce, enough for MELD's longest lines."""
CLIP_NAME = re.compile(r"^dia\d+_utt\d+$")
"""Description: The only clip names the server accepts, so no other file can be requested."""

model = None
processor = None
device = None


def ms(seconds):
    """
    Description: Converts seconds to whole milliseconds.
    Inputs:
    - seconds: a duration in seconds.
    Outputs: the duration in milliseconds, rounded.
    """
    return round(seconds * 1000)


def run_model(audio):
    """
    Description: Transcribes one clip's samples offline with the chosen lookahead.
    Inputs:
    - audio: 16 kHz mono float32 samples.
    Outputs: the text.
    Constraints: NVIDIA's docs say offline mode gives the same output as streaming with the same
    context, so this matches what the streaming model would have produced by the clip's end.
    """
    import torch
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = inputs.to(device, dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=MAX_TOKENS)
    if device == "cuda":
        torch.cuda.synchronize()
    return processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()


def transcribe_clip(split, clip):
    """
    Description: Decodes one clip's audio and transcribes it.
    Inputs:
    - split: "train", "dev" or "test".
    - clip: a clip name such as dia47_utt3.
    Outputs: a dict with the text, the audio length, and the time spent decoding and running the model.
    Constraints: raises RuntimeError when ffmpeg cannot decode the clip. serve.py makes calls take
    turns with Whisper, so this server runs one call at a time.
    """
    started = time.perf_counter()
    audio = load_audio(RAW / CLIP_DIRS[split] / f"{clip}.mp4")
    decoded = time.perf_counter()
    text = run_model(audio)
    done = time.perf_counter()
    return {"split": split, "clip": clip, "text": text, "audio_s": round(len(audio) / SAMPLE_RATE, 3),
            "decode_ms": ms(decoded - started), "model_ms": ms(done - decoded), "server_ms": ms(done - started)}


class Handler(BaseHTTPRequestHandler):
    """
    Description: Answers the two API paths.
    Methods:
    - do_GET: routes /api/status and /api/transcribe.
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
            self.send_json(200, {"model": "nemotron-speech-streaming-en-0.6b", "device": device,
                                 "compute": str(model.dtype).replace("torch.", "") if model else None,
                                 "lookahead_ms": 80 * (LOOKAHEAD + 1), "ready": model is not None})
            return
        if url.path == "/api/transcribe":
            query = parse_qs(url.query)
            split = query.get("split", [""])[0]
            clip = query.get("clip", [""])[0]
            if split not in CLIP_DIRS or not CLIP_NAME.match(clip):
                self.send_json(400, {"error": "unknown split or clip name"})
                return
            try:
                self.send_json(200, transcribe_clip(split, clip))
            except RuntimeError as err:
                self.send_json(422, {"error": str(err)})
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
    Description: Loads Nemotron in fp32 on the GPU, runs one warm-up call, then serves until stopped.
    Inputs: none
    Outputs: none
    Constraints: exits with an error when CUDA is unavailable; it never falls back to the CPU.
    fp32 is used because the GTX 1070 Ti has very slow fp16 and the weights are stored in fp32.
    """
    global model, processor, device
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch; Nemotron needs the GPU")
    device = "cuda"
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    processor.set_num_lookahead_tokens(LOOKAHEAD)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, cache_dir=str(CACHE), dtype=torch.float32).to(device).eval()
    loaded = time.perf_counter()
    run_model(np.zeros(SAMPLE_RATE, dtype=np.float32))
    warmed = time.perf_counter()
    print(f"Nemotron loaded on {torch.cuda.get_device_name(0)} in {loaded - started:.1f} s, "
          f"warm-up call {warmed - loaded:.2f} s, lookahead {80 * (LOOKAHEAD + 1)} ms", flush=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Nemotron server on http://{HOST}:{PORT} (Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
