"""Serves the MELD player and transcribes its clips live with Whisper on the local GPU.

Contents:
- Handler: serves ML_route/data plus the /api/status and /api/transcribe endpoints
- transcribe_clip: decodes one clip and runs Whisper on it, with timings
- main: loads Whisper, warms it up and starts the server
"""

import json
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text_fixes"))
from transcribe import CACHE, CLIP_DIRS, RAW, SAMPLE_RATE, add_cuda_dlls, load_audio  # noqa: E402

DATA = HERE.parent / "data"
HOST = "127.0.0.1"
PORT = 8765
MODEL_NAME = "large-v3-turbo"
DEVICE = "cuda"
COMPUTE = "int8"
CLIP_NAME = re.compile(r"^dia\d+_utt\d+$")
"""Description: The only clip names the server accepts, so no other file can be requested."""

model = None
model_lock = threading.Lock()


def ms(seconds):
    """
    Description: Converts seconds to whole milliseconds.
    Inputs:
    - seconds: a duration in seconds.
    Outputs: the duration in milliseconds, rounded.
    """
    return round(seconds * 1000)


def transcribe_clip(split, clip):
    """
    Description: Decodes one clip's audio and transcribes it with the loaded Whisper model.
    Inputs:
    - split: "train", "dev" or "test".
    - clip: a clip name such as dia47_utt3.
    Outputs: a dict with the text, the audio length, and the time spent decoding, waiting for
    the GPU and running Whisper.
    Constraints: raises RuntimeError when ffmpeg cannot decode the clip. Calls take turns on
    the GPU through model_lock.
    """
    started = time.perf_counter()
    audio = load_audio(RAW / CLIP_DIRS[split] / f"{clip}.mp4")
    decoded = time.perf_counter()
    with model_lock:
        locked = time.perf_counter()
        segments, _ = model.transcribe(
            audio, language="en", beam_size=5, word_timestamps=False,
            vad_filter=False, condition_on_previous_text=False)
        # faster-whisper decodes lazily, so the list must be built inside the timed section.
        segments = list(segments)
        done = time.perf_counter()
    return {
        "split": split,
        "clip": clip,
        "text": " ".join(s.text.strip() for s in segments).strip(),
        "audio_s": round(len(audio) / SAMPLE_RATE, 3),
        "decode_ms": ms(decoded - started),
        "wait_ms": ms(locked - decoded),
        "whisper_ms": ms(done - locked),
        "server_ms": ms(done - started),
    }


class Handler(SimpleHTTPRequestHandler):
    """
    Description: Serves the files in ML_route/data and answers the two API paths.
    Methods:
    - do_GET: routes /api/status and /api/transcribe, and serves files otherwise.
    - send_json: writes one JSON response.
    - log_message: logs API requests only, so video requests stay quiet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DATA), **kwargs)

    def do_GET(self):
        """
        Description: Answers a GET request.
        Inputs: none; reads the request path.
        Outputs: none
        """
        url = urlparse(self.path)
        if url.path == "/api/status":
            self.send_json(200, {"model": MODEL_NAME, "device": DEVICE,
                                 "compute": COMPUTE, "ready": model is not None})
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
        super().do_GET()

    def send_json(self, status, body):
        """
        Description: Sends a JSON response that the browser never caches.
        Inputs:
        - status: the HTTP status code.
        - body: a dict to send as JSON.
        Outputs: none
        """
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        """
        Description: Logs a request only when it is for the API.
        Inputs:
        - format, args: the standard log message parts.
        Outputs: none
        """
        if args and "/api/" in str(args[0]):
            super().log_message(format, *args)


def main():
    """
    Description: Loads Whisper on the GPU, runs one warm-up call, then serves until stopped.
    Inputs: none
    Outputs: none
    Constraints: exits with an error when the model cannot be loaded on CUDA; it never falls
    back to the CPU.
    """
    global model
    add_cuda_dlls()
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    try:
        model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE,
                             download_root=str(CACHE), local_files_only=True)
    except Exception as err:
        sys.exit(f"could not load Whisper on the GPU: {err}")
    loaded = time.perf_counter()
    segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en",
                                   beam_size=5, vad_filter=False)
    list(segments)
    warmed = time.perf_counter()
    print(f"Whisper {MODEL_NAME} ({COMPUTE}, {DEVICE}) loaded in {loaded - started:.1f} s, "
          f"warm-up call {warmed - loaded:.2f} s", flush=True)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}/meld-player.html (Ctrl+C stops the server)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
