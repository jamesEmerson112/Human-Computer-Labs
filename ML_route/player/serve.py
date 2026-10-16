"""Serves the MELD player and transcribes its clips live with Whisper, with Nemotron when serve_nemotron.py runs,
and writes replies with LFM2.5 when serve_reply.py runs.

Contents:
- Handler: serves ML_route/data plus the /api/status, /api/transcribe and /api/reply endpoints
- transcribe_clip: decodes one clip and runs Whisper on it, with timings
- nemotron_status: asks serve_nemotron.py whether Nemotron is loaded
- forward_nemotron: passes one clip to serve_nemotron.py, taking a turn on the GPU
- forward_reply: passes one line to serve_reply.py, taking a turn on the GPU
- main: loads Whisper, warms it up and starts the server
"""

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

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
NEMOTRON_URL = "http://127.0.0.1:8766"
"""Description: Where serve_nemotron.py listens."""
REPLY_URL = "http://127.0.0.1:8768"
"""Description: Where serve_reply.py listens."""

model = None
gpu_lock = threading.Lock()
"""Description: Held while any model runs, so Whisper, Nemotron and LFM2.5 take turns on the one GPU."""


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
    the GPU through gpu_lock.
    """
    started = time.perf_counter()
    audio = load_audio(RAW / CLIP_DIRS[split] / f"{clip}.mp4")
    decoded = time.perf_counter()
    with gpu_lock:
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
        "model_ms": ms(done - locked),
        "server_ms": ms(done - started),
    }


def helper_status(base, script):
    """
    Description: Asks a helper server for its status.
    Inputs:
    - base: the helper's base URL.
    - script: the helper's file name, for the error message.
    Outputs: its status dict, or {"ready": False, "error": ...} when it is not running.
    """
    try:
        with urllib.request.urlopen(f"{base}/api/status", timeout=1) as res:
            return json.loads(res.read())
    except (urllib.error.URLError, OSError, ValueError):
        return {"ready": False, "error": f"start {script}"}


def nemotron_status():
    """
    Description: Asks serve_nemotron.py for its status.
    Inputs: none
    Outputs: its status dict, or {"ready": False, "error": ...} when it is not running.
    """
    return helper_status(NEMOTRON_URL, "serve_nemotron.py")


def forward(base, script, path, params):
    """
    Description: Passes one request to a helper server while holding the GPU turn.
    Inputs:
    - base: the helper's base URL.
    - script: the helper's file name, for the error message.
    - path: the API path on the helper.
    - params: the query parameters to send.
    Outputs: (HTTP status, response dict), with wait_ms set to the time spent waiting for the turn.
    """
    started = time.perf_counter()
    with gpu_lock:
        locked = time.perf_counter()
        try:
            with urllib.request.urlopen(f"{base}{path}?{urlencode(params)}", timeout=60) as res:
                status, body = res.status, json.loads(res.read())
        except urllib.error.HTTPError as err:
            status, body = err.code, json.loads(err.read() or b"{}")
        except (urllib.error.URLError, OSError) as err:
            status, body = 502, {"error": f"{script} is not answering ({err})"}
    body["wait_ms"] = ms(locked - started)
    return status, body


def forward_nemotron(split, clip):
    """
    Description: Passes one clip to serve_nemotron.py while holding the GPU turn.
    Inputs:
    - split, clip: which clip to transcribe.
    Outputs: (HTTP status, response dict), with wait_ms set to the time spent waiting for the turn.
    """
    return forward(NEMOTRON_URL, "serve_nemotron.py", "/api/transcribe", {"split": split, "clip": clip})


def forward_reply(text, emotion, speaker):
    """
    Description: Passes one line to serve_reply.py while holding the GPU turn.
    Inputs:
    - text: the spoken line.
    - emotion: the emotion the reply should answer.
    - speaker: the speaker's name.
    Outputs: (HTTP status, response dict), with wait_ms set to the time spent waiting for the turn.
    """
    return forward(REPLY_URL, "serve_reply.py", "/api/reply", {"text": text, "emotion": emotion, "speaker": speaker})


class Handler(SimpleHTTPRequestHandler):
    """
    Description: Serves the files in ML_route/data and answers the two API paths.
    Methods:
    - do_GET: routes /api/status, /api/transcribe and /api/reply, and serves files otherwise.
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
            whisper = {"model": MODEL_NAME, "device": DEVICE, "compute": COMPUTE,
                       "ready": model is not None}
            self.send_json(200, {**whisper, "models": {"whisper": whisper, "nemotron": nemotron_status(),
                                                       "reply": helper_status(REPLY_URL, "serve_reply.py")}})
            return
        if url.path == "/api/reply":
            query = parse_qs(url.query)
            self.send_json(*forward_reply(query.get("text", [""])[0], query.get("emotion", [""])[0],
                                          query.get("speaker", [""])[0]))
            return
        if url.path == "/api/transcribe":
            query = parse_qs(url.query)
            split = query.get("split", [""])[0]
            clip = query.get("clip", [""])[0]
            which = query.get("model", ["whisper"])[0]
            if split not in CLIP_DIRS or not CLIP_NAME.match(clip) or which not in ("whisper", "nemotron"):
                self.send_json(400, {"error": "unknown model, split or clip name"})
                return
            if which == "nemotron":
                self.send_json(*forward_nemotron(split, clip))
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
