"""Shared paths, data loading and scoring for the MELD benchmarks that run on the Runpod pod.

APIs:
- DATA, RESULTS, FEATS, AUDIO: the pod folders the benchmarks read and write
- LABELS: MELD's seven emotions in a fixed order
- CLIP_DIRS: the folder inside MELD.Raw that holds each split's clips
- read_split: reads one split's label file into a list of line dicts
- load_audio: decodes one clip to 16 kHz mono samples with ffmpeg
- load_audio_cache: reads a split's decoded audio written by audio_cache.py
- weighted_f1: MELD's headline score
- normalize: lower-cases and strips text for word error rate
- GpuMemory: samples the GPU's used memory in a background thread
- write_result: writes a benchmark's summary JSON and per-line JSONL
"""

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("MELD_DATA", "/workspace/data"))
RESULTS = Path(os.environ.get("MELD_RESULTS", "/workspace/results"))
FEATS = Path(os.environ.get("MELD_FEATS", "/workspace/feats"))
AUDIO = Path(os.environ.get("MELD_AUDIO", "/workspace/audio"))
SAMPLE_RATE = 16000
LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
"""Description: MELD's seven emotions. A label's class index is its position in this list."""
CLIP_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
"""Description: The folder inside MELD.Raw that holds each split's mp4 clips."""


def read_split(split, version="MELD.Completed"):
    """
    Description: Reads one split's label file.
    Inputs:
    - split: "train", "dev" or "test".
    - version: "MELD.Raw", "MELD.Completed" or "MELD.Whisper".
    Outputs: a list of dicts with clip, text, speaker, label (class index) and path (the mp4).
    Constraints: when the MELD_LIMIT environment variable is set, only that many first rows are
    returned, for quick local tests.
    """
    path = DATA / version / f"{split}_sent_emo.csv"
    if version == "MELD.Raw" and not path.exists():
        path = DATA / "csv_raw" / f"{split}_sent_emo.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        clip = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
        out.append({"clip": clip, "text": r["Utterance"], "speaker": r["Speaker"],
                    "label": LABELS.index(r["Emotion"]),
                    "path": DATA / "MELD.Raw" / CLIP_DIRS[split] / f"{clip}.mp4"})
    limit = os.environ.get("MELD_LIMIT")
    return out[:int(limit)] if limit else out


def load_audio(path):
    """
    Description: Decodes a clip's first audio stream to 16 kHz mono float32 samples.
    Inputs:
    - path: the mp4 clip.
    Outputs: a float32 numpy array.
    Constraints: raises RuntimeError when ffmpeg cannot decode the clip.
    """
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    run = subprocess.run(cmd, capture_output=True)
    if run.returncode != 0 or not run.stdout:
        raise RuntimeError(run.stderr.decode("utf-8", "replace").strip() or "no audio decoded")
    return np.frombuffer(run.stdout, dtype=np.float32)


def load_audio_cache(split):
    """
    Description: Reads the decoded audio of one split written by audio_cache.py.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: a dict from clip name to float32 samples. Clips that failed to decode are absent.
    """
    z = np.load(AUDIO / f"{split}.npz")
    samples, offsets, clips = z["samples"], z["offsets"], z["clips"]
    return {str(c): samples[offsets[i]:offsets[i + 1]].astype(np.float32) for i, c in enumerate(clips)}


def weighted_f1(gold, pred):
    """
    Description: Computes F1 for each emotion and averages them weighted by how often each occurs.
    Inputs:
    - gold, pred: equal-length sequences of class indices.
    Outputs: the weighted F1 as a percentage.
    """
    from sklearn.metrics import f1_score
    return 100.0 * f1_score(gold, pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0)


def normalize(text):
    """
    Description: Normalises English text for word error rate with Whisper's English normaliser.
    Inputs:
    - text: a transcript.
    Outputs: the normalised text.
    """
    global _NORMALIZER
    if _NORMALIZER is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _NORMALIZER = EnglishTextNormalizer()
    return _NORMALIZER(text)


_NORMALIZER = None


class GpuMemory:
    """
    Description: Samples the GPU's used memory with nvidia-smi every 0.2 s while running.
    Methods:
    - read: returns the GPU's used memory now, in MiB.
    - start: records the baseline and starts sampling.
    - loop: the sampling loop run by the background thread.
    - stop: stops sampling and returns (baseline MiB, peak MiB).
    """

    def __init__(self):
        self.peak = 0
        self.base = 0
        self.running = False

    def read(self):
        """Description: Returns the GPU's used memory now, in MiB."""
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])

    def start(self):
        self.base = self.read()
        self.peak = self.base
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while self.running:
            self.peak = max(self.peak, self.read())
            time.sleep(0.2)

    def stop(self):
        self.running = False
        return self.base, self.peak


def write_result(name, summary, lines=None):
    """
    Description: Writes a benchmark's summary as RESULTS/<name>.json and its per-line records as
    RESULTS/<name>.jsonl.
    Inputs:
    - name: the result's file stem.
    - summary: a dict.
    - lines: an optional list of dicts.
    Outputs: none
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if lines is not None:
        with (RESULTS / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
