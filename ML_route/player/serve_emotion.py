"""Reads the emotion of a MELD line with the frozen encoders and the trained Rust head, for serve.py to forward to.

Contents:
- Handler: answers /api/status and /api/emotion on port 8769
- DecodeError: a clip whose audio ffmpeg cannot decode
- load_head: reads the head's JSON file into numpy arrays
- head_logits: runs the head's forward pass on one or more feature rows
- softmax: turns scores into probabilities
- weighted_f1: MELD's headline score, computed as the Rust head computes it
- wait_for_gemma: waits until EmbeddingGemma's weights have finished copying into the cache
- load_encoders: loads EmbeddingGemma, EmoBERTa-large and HuBERT-large
- text_vector: mean-pools one encoder's last layer over a line's real tokens
- voice_vector: HuBERT-large's last-layer and all-layer means over a clip's audio
- prepare_audio: cuts, pads and rounds a clip's samples the way the training audio was
- clip_features: the 3,840 joined features of one clip, with each encoder's time
- read_emotion: decodes a clip, reads its features and runs the head
- run_check: scores the head on the saved test features, with no encoders loaded
- run_parity: compares live features with the saved test features for the first clips
- main: parses the command line, then checks, compares or serves
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text_fixes"))
from transcribe import CACHE, CLIP_DIRS, RAW, SAMPLE_RATE, load_audio  # noqa: E402

DATA      = HERE.parent / "data"
HEAD_PATH = HERE.parent / "heads" / "meld-head.json"
"""Description: The trained head the Rust code in hello_world saves, committed so the player runs without training."""
FUSION    = DATA / "fusion"
"""Description: The joined training features the head was trained and scored on."""
WHISPER   = DATA / "MELD.Whisper"
"""Description: The label files whose Utterance column holds Whisper's text, which the text features read."""
HOST      = "127.0.0.1"
PORT      = 8769
EMOTIONS  = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
"""Description: MELD's seven emotions in the head's class order, as in hello_world's constants.rs."""
GEMMA_ID    = "google/embeddinggemma-300m"
EMOBERTA_ID = "tae898/emoberta-large"
HUBERT_ID   = "facebook/hubert-large-ll60k"
ENCODER_NAMES = ["embeddinggemma-300m", "emoberta-large", "hubert-large"]
"""Description: The encoders by short name, in the order their features are joined."""
BLOCKS = [("embeddinggemma-300m", 0, 768), ("emoberta-large", 768, 1792), ("hubert-large", 1792, 3840)]
"""Description: Each encoder's columns in the joined feature row: 768 + 1,024 + 2,048 = 3,840."""
MAX_TOKENS  = 128
"""Description: Longer lines are cut to this many tokens, as in training."""
MAX_AUDIO_S = 20.0
"""Description: Longer clips are cut to this many seconds before HuBERT, as in training."""
MAX_TEXT    = 600
"""Description: The longest line in characters the server accepts."""
CLIP_NAME   = re.compile(r"^dia\d+_utt\d+$")
"""Description: The only clip names the server accepts, so no other file can be read."""
GEMMA_POLL_S = 20
"""Description: Seconds between checks while EmbeddingGemma's weights are still being copied."""
GEMMA_WAIT_S = 20 * 60
"""Description: The longest wait for EmbeddingGemma's weights before giving up."""

head       = None
encoders   = None
fp32       = False
model_lock = threading.Lock()
"""Description: Held while the encoders run, so two requests never share the GPU at once."""


class DecodeError(Exception):
    """
    Description: Raised when ffmpeg cannot decode a clip, kept apart from RuntimeError because
    PyTorch reports its own failures, such as running out of GPU memory, as RuntimeError too.
    Methods: none
    """


def ms(seconds):
    """
    Description: Converts seconds to whole milliseconds.
    Inputs:
    - seconds: a duration in seconds.
    Outputs: the duration in milliseconds, rounded.
    """
    return round(seconds * 1000)


def load_head(path):
    """
    Description: Reads the head's JSON file, written by serde from hello_world's ClassifierHead.
    Inputs:
    - path: the JSON file.
    Outputs: a dict with the sizes and settings, w1 as (hidden_dim, feature_dim), w2 as
    (num_classes, hidden_dim), b1 and b2, and feature_mean and feature_std as arrays or None when
    standardisation is off. Every array is float32.
    Constraints: raises ValueError when an array's length does not match the sizes.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    feature_dim = raw["feature_dim"]
    hidden_dim  = raw["hidden_dim"]
    num_classes = raw["num_classes"]

    def array(name, size):
        values = np.asarray(raw[name], dtype=np.float32)
        if values.size != size:
            raise ValueError(f"{path.name}: {name} has {values.size} values, expected {size}")
        return values

    standardised = len(raw["feature_mean"]) > 0
    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "leaky_slope": np.float32(raw["leaky_slope"]),
        "square_activation": bool(raw["square_activation"]),
        "logit_softcap": np.float32(raw["logit_softcap"]),
        "feature_mean": array("feature_mean", feature_dim) if standardised else None,
        "feature_std": array("feature_std", feature_dim) if standardised else None,
        "w1": array("w1", hidden_dim * feature_dim).reshape(hidden_dim, feature_dim),
        "b1": array("b1", hidden_dim),
        "w2": array("w2", num_classes * hidden_dim).reshape(num_classes, hidden_dim),
        "b2": array("b2", num_classes),
    }


def head_logits(weights, x):
    """
    Description: Runs the head as step3_2_forward.rs does: standardise when the head says so, then
    W1 x + b1, Leaky ReLU, squared when square_activation is on, then W2 h + b2, then the softcap
    c * tanh(z / c) when logit_softcap is above 0.
    Inputs:
    - weights: the dict from load_head.
    - x: one feature row of feature_dim values, or a matrix with one row per line.
    Outputs: float32 scores of shape (lines, num_classes), one row per input row.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    if weights["feature_mean"] is not None:
        x = (x - weights["feature_mean"]) / weights["feature_std"]

    pre = x @ weights["w1"].T + weights["b1"]
    h   = np.where(pre > 0, pre, weights["leaky_slope"] * pre)
    if weights["square_activation"]:
        h = h * h

    z   = h @ weights["w2"].T + weights["b2"]
    cap = weights["logit_softcap"]
    if cap > 0:
        z = cap * np.tanh(z / cap)
    return z.astype(np.float32)


def softmax(z):
    """
    Description: Turns one row of scores into probabilities that sum to 1.
    Inputs:
    - z: a 1-D array of scores.
    Outputs: a float64 array of probabilities, the same length.
    """
    e = np.exp(np.asarray(z, dtype=np.float64) - np.max(z))
    return e / e.sum()


def weighted_f1(truth, predicted, num_classes):
    """
    Description: Weighted F1, as step3_6_eval.rs computes it: each emotion's F1 averaged with
    weights equal to how many lines carry that emotion.
    Inputs:
    - truth: the true class indices.
    - predicted: the predicted class indices, in the same order.
    - num_classes: the number of emotions.
    Outputs: weighted F1 times 100.
    """
    truth     = np.asarray(truth)
    predicted = np.asarray(predicted)
    weighted  = 0.0
    for c in range(num_classes):
        true_pos  = int(np.sum((truth == c) & (predicted == c)))
        pred_cnt  = int(np.sum(predicted == c))
        truth_cnt = int(np.sum(truth == c))
        precision = true_pos / pred_cnt if pred_cnt else 0.0
        recall    = true_pos / truth_cnt if truth_cnt else 0.0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        weighted += f1 * truth_cnt
    return 100.0 * weighted / max(len(truth), 1)


def wait_for_gemma():
    """
    Description: Waits until EmbeddingGemma's model.safetensors is in the cache and has stopped
    changing, since the gated weights are copied in from elsewhere rather than downloaded. A file
    untouched for GEMMA_POLL_S seconds, or whose size matches across two checks, counts as complete.
    Inputs: none
    Outputs: none
    Constraints: checks every GEMMA_POLL_S seconds and exits with an error after GEMMA_WAIT_S.
    """
    snapshots    = CACHE / "models--google--embeddinggemma-300m" / "snapshots"
    deadline     = time.monotonic() + GEMMA_WAIT_S
    last_size    = None
    told_missing = False
    while True:
        files   = sorted(snapshots.glob("*/model.safetensors"))
        stat    = files[0].stat() if files else None
        size    = stat.st_size if stat else None
        settled = stat is not None and time.time() - stat.st_mtime > GEMMA_POLL_S
        if settled or (size is not None and size == last_size):
            return
        if time.monotonic() > deadline:
            sys.exit(f"EmbeddingGemma's weights are not ready in {snapshots}")
        if size is None and not told_missing:
            print(f"EmbeddingGemma's weights are not in {snapshots} yet, waiting for them", flush=True)
            told_missing = True
        if size is not None and last_size is None:
            print(f"EmbeddingGemma weights at {size / 2**20:.0f} MiB, checking they are complete", flush=True)
        last_size = size
        time.sleep(GEMMA_POLL_S)


def load_encoders(use_fp32, hubert_cpu):
    """
    Description: Loads the three encoders and their tokenizer or feature extractor, in eval mode.
    Inputs:
    - use_fp32: loads EmoBERTa and HuBERT in fp32, which matches training exactly, instead of fp16.
    - hubert_cpu: runs HuBERT on the CPU in fp32, to leave GPU memory to Whisper and LFM2.5.
    Outputs: a dict from short name to {"model", "prep", "device", "dtype"}.
    Constraints: EmbeddingGemma always runs in fp32, because its activations overflow fp16 and turn
    to NaN, and the GTX 1070 Ti has no bf16. EmbeddingGemma is gated, so it loads from the cache only.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

    half = torch.float32 if use_fp32 else torch.float16
    wait_for_gemma()

    def text_encoder(repo, dtype, local):
        tok   = AutoTokenizer.from_pretrained(repo, cache_dir=str(CACHE), local_files_only=local)
        model = AutoModel.from_pretrained(repo, cache_dir=str(CACHE), local_files_only=local, dtype=dtype)
        return {"model": model.to("cuda").eval(), "prep": tok, "device": "cuda", "dtype": dtype}

    hubert_device = "cpu" if hubert_cpu else "cuda"
    hubert_dtype  = torch.float32 if hubert_cpu else half
    fe            = AutoFeatureExtractor.from_pretrained(HUBERT_ID, cache_dir=str(CACHE))
    hubert        = AutoModel.from_pretrained(HUBERT_ID, cache_dir=str(CACHE), dtype=hubert_dtype)
    return {
        "embeddinggemma-300m": text_encoder(GEMMA_ID, torch.float32, True),
        "emoberta-large": text_encoder(EMOBERTA_ID, half, False),
        "hubert-large": {"model": hubert.to(hubert_device).eval(), "prep": fe,
                         "device": hubert_device, "dtype": hubert_dtype},
    }


def text_vector(encoder, text):
    """
    Description: Encodes one line and mean-pools the last layer over its real tokens, as
    bench/features.py text_features does.
    Inputs:
    - encoder: one entry of the dict from load_encoders.
    - text: the line; an empty line is encoded as a single space, as in training.
    Outputs: a float32 vector of the encoder's hidden size.
    """
    import torch

    with torch.inference_mode():
        batch  = encoder["prep"]([text or " "], padding=True, truncation=True,
                                 max_length=MAX_TOKENS, return_tensors="pt").to(encoder["device"])
        hidden = encoder["model"](**batch).last_hidden_state.float()
        mask   = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return ((hidden * mask).sum(1) / mask.sum(1))[0].cpu().numpy()


def prepare_audio(audio):
    """
    Description: Cuts a clip at MAX_AUDIO_S, pads it to at least 0.1 s, and rounds its samples
    through float16, as the training audio was.
    Inputs:
    - audio: float32 16 kHz mono samples.
    Outputs: the prepared float32 samples.
    Constraints: the float16 rounding matches bench/audio_cache.py, which stored every clip as float16.
    """
    a = audio[: int(MAX_AUDIO_S * SAMPLE_RATE)]
    if len(a) < SAMPLE_RATE // 10:
        a = np.pad(a, (0, SAMPLE_RATE // 10 - len(a)))
    return a.astype(np.float16).astype(np.float32)


def voice_vector(encoder, audio):
    """
    Description: Encodes a clip's audio with HuBERT and averages over time, for the last layer and
    for the mean of every layer after the embedding output, as bench/features.py voice_features does.
    Inputs:
    - encoder: the HuBERT entry of the dict from load_encoders.
    - audio: samples from prepare_audio.
    Outputs: a float32 vector of 2,048 values, last layer first.
    """
    import torch

    with torch.inference_mode():
        inputs = encoder["prep"](audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        values = inputs["input_values"].to(encoder["device"], encoder["dtype"])
        hs     = encoder["model"](input_values=values, output_hidden_states=True).hidden_states
        last   = hs[-1][0].float().mean(0)
        every  = torch.stack(hs[1:]).float().mean(0)[0].mean(0)
        return torch.cat([last, every]).cpu().numpy()


def clip_features(audio, text):
    """
    Description: Computes one clip's joined features in training order: EmbeddingGemma, EmoBERTa,
    then HuBERT.
    Inputs:
    - audio: the clip's decoded 16 kHz mono samples.
    - text: the line's words.
    Outputs: (3,840 float32 features, dict of each encoder's time in ms keyed "<name>_ms").
    Constraints: each time is measured after torch.cuda.synchronize, so it covers the GPU work.
    """
    import torch

    parts, times = [], {}
    for name, short in (("embeddinggemma-300m", "embeddinggemma"), ("emoberta-large", "emoberta"),
                        ("hubert-large", "hubert")):
        started = time.perf_counter()
        if name == "hubert-large":
            vector = voice_vector(encoders[name], prepare_audio(audio))
        else:
            vector = text_vector(encoders[name], text)
        torch.cuda.synchronize()
        times[f"{short}_ms"] = ms(time.perf_counter() - started)
        parts.append(vector)
    return np.concatenate(parts).astype(np.float32), times


def read_emotion(split, clip, text):
    """
    Description: Decodes one clip, computes its features from its audio and the given words, and
    runs the head.
    Inputs:
    - split: "train", "dev" or "test".
    - clip: a clip name such as dia47_utt3.
    - text: the line's words, possibly empty.
    Outputs: a dict with the emotion, its probability, all seven probabilities, the words used, and
    the time spent decoding, in each encoder, in the head, in all models, and in the server.
    Constraints: raises DecodeError when ffmpeg cannot decode the clip, and FloatingPointError when
    the scores are not finite, which would otherwise reach the page as invalid JSON. Calls take
    turns through model_lock.
    """
    started = time.perf_counter()
    try:
        audio = load_audio(RAW / CLIP_DIRS[split] / f"{clip}.mp4")
    except RuntimeError as err:
        raise DecodeError(str(err)) from err
    decoded = time.perf_counter()
    with model_lock:
        locked          = time.perf_counter()
        features, times = clip_features(audio, text)
        head_started    = time.perf_counter()
        logits          = head_logits(head, features)[0]
        done            = time.perf_counter()

    if not np.isfinite(logits).all():
        raise FloatingPointError(f"the head's scores for {clip} are not finite")

    probs = softmax(logits)
    best  = int(np.argmax(logits))
    return {
        "emotion": EMOTIONS[best],
        "confidence": round(float(probs[best]), 4),
        "probs": {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs)},
        "text": text,
        "decode_ms": ms(decoded - started),
        **times,
        "head_ms": ms(done - head_started),
        "model_ms": ms(done - locked),
        "server_ms": ms(done - started),
    }


def run_check():
    """
    Description: Runs the head on the saved test features and prints its weighted F1, which must
    match the score the Rust run printed when it saved the head.
    Inputs: none
    Outputs: none
    """
    stat  = HEAD_PATH.stat()
    x     = np.load(FUSION / "x_test.npy")
    y     = np.load(FUSION / "y_test.npy")
    pred  = head_logits(head, x).argmax(1)
    score = weighted_f1(y, pred, head["num_classes"])
    saved = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    print(f"head {HEAD_PATH.name} ({stat.st_size} bytes, saved {saved}): "
          f"test weighted F1 {score:.2f} on {len(y)} lines, accuracy {100 * np.mean(pred == y):.2f}")


def run_parity(count):
    """
    Description: Recomputes the features of the first test clips from their audio and saved Whisper
    text, and prints each encoder block's cosine with the matching saved row.
    Inputs:
    - count: how many clips from the top of clips_test.txt to compare.
    Outputs: none
    """
    clips = (FUSION / "clips_test.txt").read_text(encoding="utf-8").split()[:count]
    x     = np.load(FUSION / "x_test.npy", mmap_mode="r")
    with (WHISPER / "test_sent_emo.csv").open(encoding="utf-8", newline="") as f:
        texts = {f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}": r["Utterance"] for r in csv.DictReader(f)}

    saved_rows = np.asarray(x[:count], dtype=np.float32)
    for i, clip in enumerate(clips):
        audio       = load_audio(RAW / CLIP_DIRS["test"] / f"{clip}.mp4")
        live, times = clip_features(audio, texts[clip])
        saved       = saved_rows[i]
        cosines     = []
        for name, lo, hi in BLOCKS:
            a, b = live[lo:hi].astype(np.float64), saved[lo:hi].astype(np.float64)
            cosines.append(f"{name} {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.6f}")
        same = EMOTIONS[int(head_logits(head, live).argmax())] == EMOTIONS[int(head_logits(head, saved).argmax())]
        print(f"{clip}: {', '.join(cosines)}; head agrees {same}; {times}; text {texts[clip]!r}", flush=True)


class Handler(BaseHTTPRequestHandler):
    """
    Description: Answers the two API paths.
    Methods:
    - do_GET: routes /api/status and /api/emotion.
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
            self.send_json(200, {
                "model": "meld-head",
                "encoders": ENCODER_NAMES,
                "dtype": "float32" if fp32 else "float16",
                "device": "cuda",
                "dtypes": {n: str(e["dtype"]).replace("torch.", "") for n, e in (encoders or {}).items()},
                "devices": {n: e["device"] for n, e in (encoders or {}).items()},
                "head": str(HEAD_PATH),
                "ready": encoders is not None and head is not None,
            })
            return

        if url.path == "/api/emotion":
            query = parse_qs(url.query)
            split = query.get("split", [""])[0]
            clip  = query.get("clip", [""])[0]
            text  = query.get("text", [""])[0].strip()

            bad_request = split not in CLIP_DIRS or not CLIP_NAME.fullmatch(clip) or len(text) > MAX_TEXT
            if bad_request:
                self.send_json(400, {"error": "unknown split or clip name, or the line is too long"})
                return
            try:
                self.send_json(200, read_emotion(split, clip, text))
            except DecodeError as err:
                self.send_json(422, {"error": str(err)})
            except Exception as err:
                # Without this, a failure such as running out of GPU memory drops the connection,
                # and serve.py reports the server as not answering.
                print(f"{split} {clip}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                self.send_json(500, {"error": f"{type(err).__name__}: {err}"})
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
    Description: Loads the head, then either scores it on the saved test features (--check),
    compares live and saved features (--parity N), or loads the encoders, warms them up and serves
    until stopped.
    Inputs: none; reads --check, --parity, --fp32 and --hubert-cpu from the command line.
    Outputs: none
    Constraints: exits with an error when CUDA is unavailable and encoders are needed; it never
    moves EmbeddingGemma or EmoBERTa to the CPU.
    """
    global head, encoders, fp32
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="score the head on x_test.npy, no encoders")
    ap.add_argument("--parity", type=int, default=0, help="compare live features for the first N test clips")
    ap.add_argument("--fp32", action="store_true", help="run EmoBERTa and HuBERT in fp32, as in training")
    ap.add_argument("--hubert-cpu", action="store_true", help="run HuBERT on the CPU")
    args = ap.parse_args()
    fp32 = args.fp32

    head = load_head(HEAD_PATH)
    if args.check:
        run_check()
        return

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch; the encoders need the GPU")

    started  = time.perf_counter()
    encoders = load_encoders(args.fp32, args.hubert_cpu)
    loaded   = time.perf_counter()
    _, warm  = clip_features(np.zeros(SAMPLE_RATE, dtype=np.float32), "Hello, how are you?")
    gpu_mib  = torch.cuda.memory_allocated() / 2**20
    print(f"Encoders loaded on {torch.cuda.get_device_name(0)} in {loaded - started:.1f} s "
          f"({'fp32' if args.fp32 else 'fp16'}, HuBERT on {encoders['hubert-large']['device']}, "
          f"{gpu_mib:.0f} MiB allocated), warm-up {warm}", flush=True)

    if args.parity:
        run_parity(args.parity)
        return

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Emotion server on http://{HOST}:{PORT} (Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
