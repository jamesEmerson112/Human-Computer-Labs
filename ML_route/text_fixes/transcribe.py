"""Transcribes every MELD clip with Whisper and writes the results next to the label files.

APIs:
- CLIP_DIRS: the folder that holds each split's clips
- add_cuda_dlls: makes the pip-installed cuBLAS and cuDNN DLLs loadable on Windows
- load_audio: decodes one clip to 16 kHz mono samples with ffmpeg
- clip_rows: lists the clips of one split from its label file
- transcribe_split: transcribes one split, appending one JSON line per clip
- run_batched: transcribes several short clips in one batched call
- shift_segment: makes a segment's times relative to its own clip
- make_record: builds the JSON record for one clip
- write_csv: writes a copy of a split's label file with Whisper's text as the utterance
- main: parses the command line and runs the chosen splits
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MELD.Raw"
OUT = ROOT / "data" / "MELD.Whisper"
CACHE = ROOT / "data" / "cache" / "hf"
SPLITS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
CLIP_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
"""Description: The folder inside MELD.Raw that holds each split's mp4 clips."""
SAMPLE_RATE = 16000
GAP_S = 1.0
"""Description: Seconds of silence placed between clips joined for one batched call."""
BATCH_SIZE = 8
"""Description: How many clips Whisper decodes at once in batched mode."""
MAX_BATCHED_S = 29.0
"""Description: Clips longer than this many seconds are transcribed one at a time."""
BatchedInferencePipeline = None


def add_cuda_dlls():
    """
    Description: Adds the bin folders of the nvidia-* pip packages to the DLL search path.
    Inputs: none
    Outputs: none
    Constraints: must run before faster_whisper is imported, and only matters on Windows.
    """
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for bin_dir in base.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def load_audio(path):
    """
    Description: Decodes a clip's audio, mixing every channel down to one, at 16 kHz.
    Inputs:
    - path: the mp4 clip.
    Outputs: a float32 numpy array of samples.
    Constraints: raises RuntimeError when ffmpeg cannot decode the clip.
    """
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    run = subprocess.run(cmd, capture_output=True)
    if run.returncode != 0 or not run.stdout:
        raise RuntimeError(run.stderr.decode("utf-8", "replace").strip() or "no audio decoded")
    return np.frombuffer(run.stdout, dtype=np.float32)


def clip_rows(split):
    """
    Description: Lists one split's clips in label-file order.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: a list of (clip name, mp4 path) pairs.
    """
    with (RAW / SPLITS[split]).open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    clips = [f"dia{r[5]}_utt{r[6]}" for r in rows]
    return [(c, RAW / CLIP_DIRS[split] / f"{c}.mp4") for c in clips]


def transcribe_split(model, split, only=None, limit=None, group=16):
    """
    Description: Transcribes the clips of one split that are not yet in its JSONL file and
    appends one JSON line per clip. A clip that fails to decode gets a line with an error field.
    Inputs:
    - model: a loaded faster_whisper.WhisperModel.
    - split: "train", "dev" or "test".
    - only: an optional set of clip names to restrict the run to.
    - limit: an optional maximum number of clips to transcribe in this call.
    - group: how many clips go into one batched call; 1 transcribes clips one at a time.
    Outputs: a list of (audio seconds, wall seconds) pairs, one per transcribed clip.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{split}_whisper.jsonl"
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            done = {json.loads(line)["clip"] for line in f if line.strip()}
    todo = [(c, p) for c, p in clip_rows(split) if c not in done and (only is None or c in only)]
    if limit is not None:
        todo = todo[:limit]
    timings = []
    batched = BatchedInferencePipeline(model=model) if group > 1 else None
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(c, pool.submit(load_audio, p)) for c, p in todo]
        for first in range(0, len(futures), group):
            loaded = []
            for clip, fut in futures[first:first + group]:
                try:
                    loaded.append((clip, fut.result()))
                except RuntimeError as err:
                    out.write(json.dumps({"split": split, "clip": clip, "error": str(err)}) + "\n")
            t0 = time.perf_counter()
            short = [(c, a) for c, a in loaded if batched and len(a) / SAMPLE_RATE <= MAX_BATCHED_S]
            short_names = {c for c, _ in short}
            single = [(c, a) for c, a in loaded if c not in short_names]
            results = run_batched(batched, short) if short else {}
            for clip, audio in single:
                segments, _ = model.transcribe(
                    audio, language="en", beam_size=5, word_timestamps=True,
                    vad_filter=False, condition_on_previous_text=False)
                results[clip] = list(segments)
            wall = time.perf_counter() - t0
            for clip, audio in loaded:
                segs = results[clip]
                duration = len(audio) / SAMPLE_RATE
                out.write(json.dumps(make_record(split, clip, duration, segs), ensure_ascii=False) + "\n")
                timings.append((duration, wall / len(loaded)))
            out.flush()
            if (first // group) % 25 == 0:
                print(f"{split}: {first + len(loaded)}/{len(futures)} clips", flush=True)
    return timings


def run_batched(batched, clips):
    """
    Description: Transcribes several short clips in one batched call by joining them with
    silence and giving Whisper each clip's span, then splits the segments back by clip.
    Inputs:
    - batched: a faster_whisper.BatchedInferencePipeline.
    - clips: a list of (clip name, samples) pairs, each at most MAX_BATCHED_S long.
    Outputs: a dict from clip name to its list of segments, with times relative to the clip.
    """
    gap = np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)
    parts, spans, pos = [], [], 0.0
    for clip, audio in clips:
        dur = len(audio) / SAMPLE_RATE
        spans.append((clip, pos, pos + dur))
        parts += [audio, gap]
        pos += dur + GAP_S
    joined = np.concatenate(parts)
    segments, _ = batched.transcribe(
        joined, language="en", beam_size=5, word_timestamps=True, batch_size=BATCH_SIZE,
        condition_on_previous_text=False,
        clip_timestamps=[{"start": s, "end": e} for _, s, e in spans])
    results = {clip: [] for clip, _, _ in spans}
    for seg in segments:
        for clip, s, e in spans:
            if s - 0.01 <= seg.start < e + GAP_S / 2:
                results[clip].append(shift_segment(seg, s))
                break
    return results


def shift_segment(seg, offset):
    """
    Description: Moves a segment and its words back by an offset, so times are relative to its clip.
    Inputs:
    - seg: a faster_whisper Segment.
    - offset: the clip's start time in the joined audio, in seconds.
    Outputs: the shifted segment.
    """
    words = [replace(w, start=w.start - offset, end=w.end - offset) for w in (seg.words or [])]
    return replace(seg, start=seg.start - offset, end=seg.end - offset, words=words)


def make_record(split, clip, duration, segs):
    """
    Description: Builds the JSON record for one transcribed clip.
    Inputs:
    - split, clip: which clip it is.
    - duration: the clip's audio length in seconds.
    - segs: the clip's segments, with times relative to the clip.
    Outputs: a dict ready for json.dumps.
    """
    return {
        "split": split, "clip": clip, "duration": round(duration, 3),
        "text": " ".join(s.text.strip() for s in segs).strip(),
        "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                      "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                      "compression_ratio": round(s.compression_ratio, 2)} for s in segs],
        "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word,
                   "p": round(w.probability, 3)} for s in segs for w in (s.words or [])],
    }


def write_csv(split):
    """
    Description: Writes MELD.Whisper/<split>_sent_emo.csv, a copy of the original label file
    whose Utterance column holds Whisper's text.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: the number of rows whose clip has no transcript.
    Constraints: rows without a transcript, such as clips that failed to decode, get an empty
    utterance.
    """
    texts = {}
    with (OUT / f"{split}_whisper.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts[rec["clip"]] = rec.get("text", "")
    with (RAW / SPLITS[split]).open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    missing = 0
    for row in rows[1:]:
        clip = f"dia{row[5]}_utt{row[6]}"
        missing += clip not in texts or not texts[clip]
        row[1] = texts.get(clip, "")
    with (OUT / SPLITS[split]).open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return missing


def main():
    """
    Description: Loads Whisper on the GPU and transcribes the chosen splits, or a pilot subset.
    Inputs: none; reads --splits, --clips, --limit, --model, --compute, --group and --out from
    the command line.
    Outputs: none
    Constraints: exits with an error when CUDA cannot be loaded; it never falls back to the CPU.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--clips", default="", help="comma-separated clip names, for a pilot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--out", default="", help="write to this folder instead of MELD.Whisper")
    args = ap.parse_args()

    global OUT, BatchedInferencePipeline
    if args.out:
        OUT = Path(args.out)
    add_cuda_dlls()
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    try:
        model = WhisperModel(args.model, device="cuda", compute_type=args.compute, download_root=str(CACHE))
    except Exception as err:
        sys.exit(f"CUDA model load failed: {err}")
    only = set(args.clips.split(",")) if args.clips else None
    all_timings = []
    for split in args.splits.split(","):
        t = transcribe_split(model, split, only, args.limit, args.group)
        all_timings += t
        if only is None and args.limit is None:
            print(f"{split}: {write_csv(split)} rows without a transcript")
    if all_timings:
        audio = np.array([a for a, _ in all_timings])
        wall = np.array([w for _, w in all_timings])
        slope, overhead = np.polyfit(audio, wall, 1) if len(audio) > 1 else (0.0, wall[0])
        print(f"clips {len(audio)}, audio {audio.sum():.0f}s, wall {wall.sum():.0f}s, "
              f"{audio.sum() / wall.sum():.1f}x real time, fixed cost per call about {overhead:.2f}s, "
              f"{slope:.3f}s per audio second")


if __name__ == "__main__":
    main()
