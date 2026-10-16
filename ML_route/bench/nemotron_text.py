"""Transcribes every MELD clip with Nemotron Speech Streaming 0.6B through Transformers and writes MELD.Nemotron.

Contents:
- MODEL_ID, LOOKAHEAD, PLAYER_MAX_TOKENS, LONG_S, BUDGET_S, NEMO_WER_PCT: the model, decoding and batching settings
- OUT, SPLITS: the output folder and the order the splits run in
- load_model: loads the processor and model the way the player's serve_nemotron.py does
- transcribe: transcribes a list of clips' samples in one model call
- batches: groups clips of similar length into batches
- read_done: reads a split's finished transcripts from its JSONL file
- attempt: transcribes one batch and builds its records, retrying clips alone when the batch fails
- run_clips: transcribes the clips a split still needs, appending each result to its JSONL file
- batch_check: compares batched output with one-clip output on the test clips
- write_csv: writes one split's MELD.Nemotron label file
- split_summary: counts and timings of one split
- test_scores: word error rate on test against MELD.Completed
- main: transcribes test at batch 1, checks batching, transcribes dev and train, writes the CSVs and the summary
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np

from common import DATA, RESULTS, SAMPLE_RATE, GpuMemory, load_audio_cache, normalize, read_split, write_result

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
"""Description: The Hugging Face repo of the model, the one the player's serve_nemotron.py loads."""
LOOKAHEAD = 13
"""Description: Right attention context in 80 ms frames; 13 is the card's 1.12 s setting."""
PLAYER_MAX_TOKENS = 448
"""Description: The decode-step limit the player uses. Each step is one 80 ms encoder frame or one emitted
token, so it cuts off clips longer than about 22 s; this script lets the length decide unless told otherwise."""
LONG_S = 20.0
"""Description: Clips longer than this many seconds always run alone, so their padding never slows a batch."""
BUDGET_S = 160.0
"""Description: The most padded audio seconds one batch may hold."""
NEMO_WER_PCT = 33.43
"""Description: Test word error rate of the same model at 1120 ms through the NeMo toolkit, from last night's
asr_race.py run (results/asr_nemotron-1120ms.json)."""
OUT = DATA / "MELD.Nemotron"
"""Description: The folder that receives the label files and the per-clip transcripts."""
SPLITS = ("test", "dev", "train")
"""Description: The splits in the order they run; test comes first because the batch check needs it."""


def load_model():
    """
    Description: Loads Nemotron's processor and model in fp32 on the GPU with the 1.12 s lookahead.
    Inputs: none
    Outputs: (model, processor).
    Constraints: exits with an error when CUDA is unavailable. fp32 matches the player and the stored weights.
    """
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch; Nemotron needs the GPU")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.set_num_lookahead_tokens(LOOKAHEAD)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda").eval()
    return model, processor


def transcribe(model, processor, arrays, max_tokens):
    """
    Description: Transcribes clips offline in one padded batch, as serve_nemotron.py's run_model does for one.
    Inputs:
    - model, processor: from load_model.
    - arrays: a list of 16 kHz mono float32 sample arrays.
    - max_tokens: the decode-step limit, or None to size it from the audio length.
    Outputs: (list of texts in input order, decode steps the batch used).
    """
    import torch
    inputs = processor(arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = inputs.to("cuda", dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=max_tokens)
    torch.cuda.synchronize()
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    return [t.strip() for t in texts], int(out.sequences.shape[1]) - 1


def batches(clips, audio, size):
    """
    Description: Sorts clips by length and groups neighbours into batches.
    Inputs:
    - clips: clip names.
    - audio: a dict from clip name to samples.
    - size: the most clips in one batch.
    Outputs: a list of lists of clip names.
    Constraints: clips longer than LONG_S run alone, and a batch's clip count times its longest clip stays
    within BUDGET_S.
    """
    out, cur = [], []
    for clip in sorted(clips, key=lambda c: len(audio[c])):
        seconds = len(audio[clip]) / SAMPLE_RATE
        if cur and (len(cur) >= size or seconds > LONG_S or (len(cur) + 1) * seconds > BUDGET_S):
            out.append(cur)
            cur = []
        cur.append(clip)
        if seconds > LONG_S:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def read_done(split):
    """
    Description: Reads the transcripts a split already has, so a killed run can continue.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: a dict from clip name to its record.
    Constraints: failed records and a line cut short by a killed run are left out, so those clips run again.
    """
    path = OUT / f"{split}_nemotron.jsonl"
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not rec.get("failed"):
                done[rec["clip"]] = rec
    return done


def attempt(model, processor, group, audio, max_tokens):
    """
    Description: Transcribes one batch and builds one record per clip. When the batch raises, each clip is
    retried alone, and a clip that still raises gets a failed record holding the error, so one bad clip never
    stops a split.
    Inputs:
    - model, processor: from load_model.
    - group: the batch's clip names.
    - audio: a dict from clip name to samples.
    - max_tokens: the decode-step limit, or None.
    Outputs: a list of record dicts in group order, each with clip, text, audio_s, ms, batch, steps_over_cap
    and failed, plus error on a failed one. A clip retried alone after its batch raised has batch 1.
    Constraints: ms is the batch's wall time divided by its clip count, and covers feature extraction and
    decoding with the audio already in memory. steps_over_cap marks clips that needed at least the player's
    PLAYER_MAX_TOKENS decode steps; for a batch it marks every clip in the batch. The retries run after the
    except block has ended, so the failed batch's tensors, which the traceback holds, are freed first.
    """
    import torch
    t = time.perf_counter()
    error = None
    try:
        texts, steps = transcribe(model, processor, [audio[c] for c in group], max_tokens)
    except Exception:
        error = traceback.format_exc()[-600:]
    if error is not None:
        torch.cuda.empty_cache()
        if len(group) > 1:
            return [rec for c in group for rec in attempt(model, processor, [c], audio, max_tokens)]
        return [{"clip": group[0], "text": "", "audio_s": round(len(audio[group[0]]) / SAMPLE_RATE, 3), "ms": None,
                 "batch": 1, "steps_over_cap": False, "failed": True, "error": error}]
    ms = 1000 * (time.perf_counter() - t) / len(group)
    return [{"clip": c, "text": text, "audio_s": round(len(audio[c]) / SAMPLE_RATE, 3), "ms": round(ms, 1),
             "batch": len(group), "steps_over_cap": steps >= PLAYER_MAX_TOKENS, "failed": False}
            for c, text in zip(group, texts)]


def run_clips(model, processor, split, clips, audio, size, max_tokens, done):
    """
    Description: Transcribes the clips not yet done and appends one JSON line per clip to the split's file.
    Inputs:
    - model, processor: from load_model.
    - split: the split the clips come from.
    - clips: the clip names to transcribe, all present in audio.
    - audio: a dict from clip name to samples.
    - size: the most clips in one batch.
    - max_tokens: the decode-step limit, or None.
    - done: the dict from read_done, updated in place.
    Outputs: none
    Constraints: when a killed run left the file without a final newline, one is written first, so the first
    new record does not join the cut-off line.
    """
    pending = [c for c in clips if c not in done]
    if not pending:
        print(f"{split}: all {len(clips)} clips already transcribed", flush=True)
        return
    t0 = time.perf_counter()
    count = 0
    path = OUT / f"{split}_nemotron.jsonl"
    torn = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as f:
        if torn:
            f.write("\n")
        for group in batches(pending, audio, size):
            for rec in attempt(model, processor, group, audio, max_tokens):
                done[rec["clip"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["failed"]:
                    print(f"{split}: {rec['clip']} failed: {rec['error'].strip().splitlines()[-1]}", flush=True)
            f.flush()
            if count // 500 != (count + len(group)) // 500:
                print(f"{split}: {count + len(group)}/{len(pending)} clips, {time.perf_counter() - t0:.0f} s",
                      flush=True)
            count += len(group)
    print(f"{split}: {count} clips transcribed in {time.perf_counter() - t0:.0f} s", flush=True)


def batch_check(model, processor, clips, audio, size, max_tokens, done):
    """
    Description: Transcribes the test clips again in batches and compares each text with its batch-1 text.
    Inputs:
    - model, processor: from load_model.
    - clips: the test clip names that have batch-1 transcripts in done.
    - audio: the test samples.
    - size: the batch size to check.
    - max_tokens: the decode-step limit, or None.
    - done: the test split's records from read_done.
    Outputs: a dict with the settings, the clip count, how many texts matched, how many clips fell back to
    one-clip calls, up to 20 differing examples and the batched milliseconds per clip.
    Constraints: a clip that fails in the batched pass, or whose batch raised so it was retried alone, counts
    as different, since its text says nothing about batched decoding.
    """
    t0 = time.perf_counter()
    batched, fell_back = {}, set()
    for group in batches(clips, audio, size):
        for rec in attempt(model, processor, group, audio, max_tokens):
            batched[rec["clip"]] = rec
            if len(group) > 1 and rec["batch"] == 1:
                fell_back.add(rec["clip"])
    seconds = time.perf_counter() - t0
    differ = [{"clip": c, "batch_1": done[c]["text"], "batched": batched[c]["text"],
               "batched_failed": batched[c]["failed"], "batch_raised": c in fell_back}
              for c in clips if batched[c]["failed"] or c in fell_back or batched[c]["text"] != done[c]["text"]]
    print(f"batch check at {size}: {len(clips) - len(differ)} of {len(clips)} texts match, "
          f"{len(fell_back)} clips fell back to one-clip calls", flush=True)
    return {"batch": size, "max_new_tokens": max_tokens, "clips": len(clips), "same": len(clips) - len(differ),
            "different": len(differ), "fell_back": len(fell_back),
            "examples": differ[:20], "batched_ms_per_clip": round(1000 * seconds / max(len(clips), 1), 1)}


def write_csv(split, done):
    """
    Description: Copies MELD.Completed's label file with each transcribed line's Utterance replaced by
    Nemotron's text, keeping every other column, the row order and the CSV dialect.
    Inputs:
    - split: "train", "dev" or "test".
    - done: a dict from clip name to its transcript record.
    Outputs: (rows written, rows that kept MELD.Completed's text because their clip has no audio or failed).
    Constraints: when MELD_LIMIT is set, only that many first rows are written, matching read_split.
    """
    with (DATA / "MELD.Completed" / f"{split}_sent_emo.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    limit = os.environ.get("MELD_LIMIT")
    if limit:
        body = body[:int(limit)]
    col, dia, utt = header.index("Utterance"), header.index("Dialogue_ID"), header.index("Utterance_ID")
    kept = 0
    for row in body:
        rec = done.get(f"dia{row[dia]}_utt{row[utt]}")
        if rec is None or rec["failed"]:
            kept += 1
        else:
            row[col] = rec["text"]
    with (OUT / f"{split}_sent_emo.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([header] + body)
    return len(body), kept


def split_summary(split, rows, audio, done, seconds):
    """
    Description: Summarises one split's transcripts.
    Inputs:
    - split: the split name.
    - rows: the split's lines from read_split.
    - audio: the split's samples.
    - done: the split's transcript records.
    - seconds: wall time spent transcribing it in this run.
    Outputs: a dict of counts and timings.
    """
    every = [done[r["clip"]] for r in rows if r["clip"] in done]
    recs = [r for r in every if not r["failed"]]
    ms = np.array([r["ms"] for r in recs]) if recs else np.zeros(1)
    return {"rows": len(rows), "transcribed": len(recs),
            "missing_audio": sum(r["clip"] not in audio for r in rows),
            "failed": {r["clip"]: r["error"].strip().splitlines()[-1] for r in every if r["failed"]},
            "empty_transcripts": sum(not r["text"] for r in recs),
            "batch_sizes": sorted({r["batch"] for r in recs}),
            "steps_over_player_cap": [r["clip"] for r in recs if r["steps_over_cap"]],
            "ms_median": round(float(np.median(ms)), 1), "ms_p95": round(float(np.percentile(ms, 95)), 1),
            "ms_mean": round(float(ms.mean()), 1), "audio_s_total": round(sum(r["audio_s"] for r in recs), 1),
            "seconds_this_run": round(seconds, 1)}


def test_scores(rows, done):
    """
    Description: Scores test transcripts against MELD.Completed's test text with asr_race.py's normaliser,
    once over every scored clip and once without clips over the player's step cap. Also counts how many
    texts equal last night's NeMo-toolkit transcripts when RESULTS/asr_nemotron-1120ms.jsonl exists.
    Inputs:
    - rows: the test lines from read_split.
    - done: the test transcript records.
    Outputs: a dict of word error rates and counts.
    """
    import jiwer
    refs, hyps, refs_short, hyps_short = [], [], [], []
    for r in rows:
        rec = done.get(r["clip"])
        if rec is None or rec["failed"]:
            continue
        ref, hyp = normalize(r["text"]), normalize(rec["text"])
        if ref.strip():
            refs.append(ref)
            hyps.append(hyp)
            if not rec["steps_over_cap"]:
                refs_short.append(ref)
                hyps_short.append(hyp)
    out = {"scored_clips": len(refs), "nemo_toolkit_wer_pct": NEMO_WER_PCT}
    if refs:
        w = jiwer.process_words(refs, hyps)
        out.update({"wer_pct": round(100 * w.wer, 2), "substitutions": w.substitutions,
                    "deletions": w.deletions, "insertions": w.insertions})
    if refs_short:
        out["wer_pct_without_clips_over_cap"] = round(100 * jiwer.wer(refs_short, hyps_short), 2)
    nemo = RESULTS / "asr_nemotron-1120ms.jsonl"
    if nemo.exists():
        old = {}
        with nemo.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    old[rec["clip"]] = rec["hyp"]
        both = [c for c in done if c in old and not done[c]["failed"]]
        out["same_text_as_nemo_toolkit"] = f"{sum(done[c]['text'] == old[c] for c in both)} of {len(both)}"
    return out


def main():
    """
    Description: Transcribes test one clip per call (timed like asr_race.py), checks that batched output
    equals it, transcribes dev and train in batches when it does and one at a time otherwise, then writes
    the three MELD.Nemotron label files and RESULTS/nemotron_text.json.
    Inputs: none; reads --batch and --max-new-tokens from the command line.
    Outputs: none
    Constraints: a rerun continues from the per-clip JSONL files in MELD.Nemotron.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="decode-step limit; 448 matches the player, the default sizes it from the clip")
    args = ap.parse_args()
    # Without a step limit, generate warns once per call with a different number, which would flood the log.
    warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    mem = GpuMemory().start()
    t0 = time.perf_counter()
    model, processor = load_model()
    load_s = time.perf_counter() - t0
    data = {s: (read_split(s), load_audio_cache(s)) for s in SPLITS}
    test_rows, test_audio = data["test"]
    for clip in [r["clip"] for r in test_rows if r["clip"] in test_audio][:3]:
        attempt(model, processor, [clip], test_audio, args.max_new_tokens)
    summary = {"model": MODEL_ID, "runtime": f"transformers {transformers.__version__}",
               "lookahead_ms": 80 * (LOOKAHEAD + 1), "dtype": "float32",
               "max_new_tokens": args.max_new_tokens or "sized from each clip's length",
               "device": torch.cuda.get_device_name(0), "load_s": round(load_s, 1), "splits": {}}
    size, lines = 1, []
    check_path = OUT / "batch_check.json"
    for split in SPLITS:
        rows, audio = data[split]
        done = read_done(split)
        clips = [r["clip"] for r in rows if r["clip"] in audio]
        t = time.perf_counter()
        run_clips(model, processor, split, clips, audio, size, args.max_new_tokens, done)
        seconds = time.perf_counter() - t
        if split == "test" and args.batch > 1:
            clips = [c for c in clips if not done[c]["failed"]]
            check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}
            if check.get("batch") != args.batch or check.get("max_new_tokens") != args.max_new_tokens:
                check = batch_check(model, processor, clips, audio, args.batch, args.max_new_tokens, done)
                check_path.write_text(json.dumps(check, indent=1, ensure_ascii=False), encoding="utf-8")
            summary["batch_check"] = check
            size = args.batch if check["different"] == 0 else 1
        written, kept = write_csv(split, done)
        summary["splits"][split] = split_summary(split, rows, audio, done, seconds)
        summary["splits"][split].update({"csv_rows": written, "csv_rows_kept_completed_text": kept})
        if split == "test":
            summary["test"] = test_scores(rows, done)
            lines = [{"clip": r["clip"], "ref": r["text"], "hyp": done[r["clip"]]["text"],
                      "ms": done[r["clip"]]["ms"], "audio_s": done[r["clip"]]["audio_s"],
                      "failed": done[r["clip"]]["failed"]}
                     for r in rows if r["clip"] in done]
    base, peak = mem.stop()
    summary["gpu_mib_peak_over_idle"] = peak - base
    summary["batch_used_for_dev_train"] = size
    summary["reference"] = "MELD.Completed test text, which keeps MELD's published words"
    summary["timing"] = ("test: one clip per call with the audio already decoded, as asr_race.py; dev and train: "
                         "batch wall time divided by its clips")
    write_result("nemotron_text", summary, lines)


if __name__ == "__main__":
    main()
