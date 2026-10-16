"""Extracts frozen vision-tower and reader features from Qwen3.5-family vision-language models for every MELD clip.

Contents:
- DEFAULT_MODELS, VIEWS, TEXT_VIEW: the models run by default and the feature views this script writes
- MAX_FAILED_IN_A_ROW: consecutive clip failures after which a split is abandoned
- TIMING_NOTES: how each view's milliseconds are measured, for vision_timing.json
- TowerHook: captures and mean-pools the visual tokens the vision module hands to the language model
- pick_dtype: bf16 where the GPU supports it natively, else fp16
- load_model: loads a model with vlm_zero_shot.load_vlm and moves its bf16 tensors to the chosen dtype
- find_visual: finds the model's vision module
- prepare: builds the model inputs for one clip exactly as vlm_zero_shot.ask does
- run_pass: one forward pass that returns the reader vector, the seven label scores and its timing
- partial_path, load_partial, save_partial: where a (model, split) checkpoint lives, and reading and writing it
- save_npz: writes a finished view atomically, in the layout of features.save
- update_timing, timing_entry: merge per-clip timing into RESULTS/vision_timing.json, and summarise it
- run_split: extracts the needed views of one split for one model, with checkpoints and a time limit
- run_model: loads one model and runs every chosen split
- main: parses the command line and runs each model in turn
"""

import argparse
import gc
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import DATA, FEATS, LABELS, RESULTS, read_split, weighted_f1
from frames import extract_frames
from vlm_zero_shot import build_prompt, label_token_ids, load_vlm

DEFAULT_MODELS = "qwen3.5-4b=Qwen/Qwen3.5-4B,oneemo=Jiaha0Hu4ng/OneEmo"
"""Description: The name=repo pairs run when --models is not given. OneEmo is a fine-tune of Qwen3.5-4B."""
VIEWS = ("tower", "reader")
"""Description: The two views taken from the frames-only pass. "tower" is the mean of the merged visual
tokens the vision module hands to the language model, over all eight frames; "reader" is the language
model's final normed hidden state at the last prompt position, the vector whose lm_head logits
vlm_zero_shot.ask reads."""
TEXT_VIEW = "reader_text"
"""Description: The reader view from a second pass whose prompt also gives the line's words."""
MAX_FAILED_IN_A_ROW = 20
"""Description: A split is abandoned, with its checkpoint kept, after this many clips fail one after another,
since that usually means the GPU is in a broken state rather than that the clips are bad."""
TIMING_NOTES = {
    "tower": "vision module only, timed by forward hooks with a CUDA sync, one clip of eight frames per call",
    "reader": "whole frames-only forward pass, vision plus language model, one clip per call",
    "reader_text": "whole forward pass with the line's words, vision plus language model, one clip per call",
}
"""Description: How each view's milliseconds are measured, written as the note of its vision_timing.json entry."""


class TowerHook:
    """
    Description: Hooks the vision module, mean-pools the visual tokens it returns and times its forward pass.
    Methods:
    - reset: clears the captured tokens and timing before a pass.
    - before: the forward pre-hook; starts the timer.
    - after: the forward hook; adds up the visual tokens and stops the timer.
    - pooled: returns the mean visual token over every call since the last reset.
    Constraints: get_image_features replaces pooler_output with a tuple of per-image splits right after
    the module returns, so the tokens are reduced inside the hook rather than kept by reference.
    """

    def __init__(self, module):
        self.calls = 0
        self.total = None
        self.count = 0
        self.ms = 0.0
        self.t = 0.0
        module.register_forward_pre_hook(self.before)
        module.register_forward_hook(self.after)

    def reset(self):
        self.calls = 0
        self.total = None
        self.count = 0
        self.ms = 0.0

    def before(self, module, args):
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def after(self, module, args, out):
        tokens = getattr(out, "pooler_output", None)
        if tokens is None:
            tokens = out[0] if isinstance(out, (tuple, list)) else out
        tokens = tokens.reshape(-1, tokens.shape[-1]).float()
        total = tokens.sum(0)
        self.total = total if self.total is None else self.total + total
        self.count += tokens.shape[0]
        self.calls += 1
        torch.cuda.synchronize()
        self.ms += 1000 * (time.perf_counter() - self.t)

    def pooled(self):
        """
        Description: Returns the mean visual token since the last reset as a float32 numpy vector.
        Constraints: raises RuntimeError when the hook has not fired since the last reset.
        """
        if not self.calls:
            raise RuntimeError("the vision module hook did not fire for this clip")
        return (self.total / self.count).cpu().numpy()


def pick_dtype():
    """
    Description: Chooses the compute dtype.
    Inputs: none
    Outputs: torch.bfloat16 on GPUs with native bf16 (the A40), else torch.float16 (the local GTX 1070 Ti).
    """
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def load_model(repo, dtype):
    """
    Description: Loads a model with vlm_zero_shot.load_vlm, then moves only its bf16 tensors to the chosen
    dtype, so tensors the model keeps in float32 on purpose stay in float32.
    Inputs:
    - repo: the Hugging Face repo id.
    - dtype: the compute dtype from pick_dtype.
    Outputs: (model, processor).
    """
    model, processor = load_vlm(repo)
    if dtype != torch.bfloat16:
        for t in list(model.parameters()) + list(model.buffers()):
            if t.dtype == torch.bfloat16:
                t.data = t.data.to(dtype)
    return model, processor


def find_visual(model):
    """
    Description: Finds the module that turns pixels into the visual tokens the language model reads.
    Inputs:
    - model: a loaded image-text model.
    Outputs: (module, its attribute path).
    Constraints: raises RuntimeError when no module named "visual" exists.
    """
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "visual", None) is not None:
        return inner.visual, "model.visual"
    for path, module in model.named_modules():
        if path.split(".")[-1] == "visual":
            return module, path
    raise RuntimeError("no vision module named 'visual' found")


def prepare(processor, frames, question):
    """
    Description: Builds the model inputs for one clip on the CPU, with the same message layout and chat
    template arguments as vlm_zero_shot.ask, so the last position is the one ask scores.
    Inputs:
    - processor: the model's processor.
    - frames: the clip's frame paths.
    - question: the prompt from build_prompt.
    Outputs: a BatchFeature of CPU tensors.
    """
    from PIL import Image
    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors="pt", enable_thinking=False)


@torch.inference_mode()
def run_pass(model, inputs, visual_dtype, label_ids):
    """
    Description: Runs one forward pass and reads the reader vector and the seven label scores at the
    last position.
    Inputs:
    - model: a loaded model with a TowerHook on its vision module.
    - inputs: the output of prepare.
    - visual_dtype: the vision module's dtype, for pixel_values.
    - label_ids: the output of vlm_zero_shot.label_token_ids.
    Outputs: (reader vector as float32 numpy, the seven label scores, milliseconds for the forward pass).
    Constraints: use_cache is off because no token is generated, which skips building a cache that the
    logits do not depend on.
    """
    inputs = inputs.to("cuda")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(visual_dtype)
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = model(**inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False)
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t)
    reader = out.hidden_states[-1][0, -1].float().cpu().numpy()
    logits = out.logits[0, -1].float()
    scores = [max(logits[i].item() for i in ids) for ids in label_ids]
    return reader, scores, ms


def partial_path(name, split):
    """
    Description: The checkpoint file of one model and split.
    Inputs:
    - name: the model's short name.
    - split: "train", "dev" or "test".
    Outputs: a Path.
    """
    return FEATS / name / f"partial_{split}.npz"


def load_partial(name, split):
    """
    Description: Reads a checkpoint written by save_partial.
    Inputs:
    - name: the model's short name.
    - split: "train", "dev" or "test".
    Outputs: (a dict from view to a dict from clip name to (vector, ms, scores), the label-file folder the
    checkpoint's reader_text rows were prompted with or None); an empty dict when there is no checkpoint.
    """
    path = partial_path(name, split)
    state = {}
    if not path.exists():
        return state, None
    z = np.load(path)
    for view in (*VIEWS, TEXT_VIEW):
        if f"{view}__clips" not in z:
            continue
        state[view] = {str(c): (z[f"{view}__x"][i], float(z[f"{view}__ms"][i]), z[f"{view}__scores"][i])
                       for i, c in enumerate(z[f"{view}__clips"])}
    source = str(z["text_source"]) if "text_source" in z else None
    return state, source


def save_partial(name, split, state, text_source):
    """
    Description: Writes a checkpoint atomically, so a run killed while writing keeps the previous one.
    Inputs:
    - name: the model's short name.
    - split: "train", "dev" or "test".
    - state: a dict from view to a dict from clip name to (vector, ms, scores).
    - text_source: the label-file folder the reader_text rows were prompted with, or None.
    Outputs: none
    """
    arrays = {}
    for view, done in state.items():
        if not done:
            continue
        clips = list(done)
        arrays[f"{view}__clips"] = np.array(clips)
        arrays[f"{view}__x"] = np.stack([done[c][0] for c in clips]).astype(np.float32)
        arrays[f"{view}__ms"] = np.array([done[c][1] for c in clips], dtype=np.float32)
        arrays[f"{view}__scores"] = np.stack([np.asarray(done[c][2], dtype=np.float32) for c in clips])
    if text_source is not None:
        arrays["text_source"] = np.array(text_source)
    path = partial_path(name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_npz(name, view, split, clips, labels, x):
    """
    Description: Writes FEATS/<name>/<view>_<split>.npz with arrays x, clips and labels, the layout of
    features.save, through a temporary file, so a run killed while writing never leaves a truncated file that
    a resumed run would take as finished.
    Inputs:
    - name: the model's short name.
    - view: the view name.
    - split: "train", "dev" or "test".
    - clips, labels: the clip names and class indices, in the order of x's rows.
    - x: the float32 feature array.
    Outputs: none
    """
    folder = FEATS / name
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{view}_{split}.part.npz"
    np.savez(tmp, x=x, clips=np.array(clips), labels=np.array(labels))
    os.replace(tmp, folder / f"{view}_{split}.npz")


def update_timing(entries):
    """
    Description: Merges timing entries into RESULTS/vision_timing.json, reading it just before writing so
    entries from other scripts are kept.
    Inputs:
    - entries: a dict from "<encoder>/<view>" to {"ms_median", "ms_p95", "clips", "device", "note"}.
    Outputs: none
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "vision_timing.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(entries)
    tmp = RESULTS / f"vision_timing.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def timing_entry(view, ms, dtype):
    """
    Description: Summarises per-clip milliseconds.
    Inputs:
    - view: the view name, which picks the note.
    - ms: a list of per-clip milliseconds.
    - dtype: the compute dtype, named after the GPU in "device".
    Outputs: {"ms_median", "ms_p95", "clips", "device", "note"}.
    """
    return {"ms_median": round(float(np.median(ms)), 1), "ms_p95": round(float(np.percentile(ms, 95)), 1),
            "clips": len(ms), "device": f"{torch.cuda.get_device_name(0)} {str(dtype).replace('torch.', '')}",
            "note": TIMING_NOTES[view] + "; the first clip of each run includes warm-up"}


def run_split(name, model, processor, hook, label_ids, split, with_text, minutes, every, timing_ms):
    """
    Description: Extracts the views of one split that have no final file yet, resuming from the
    checkpoint, saving it every few clips and stopping cleanly at the time limit. A clip whose inputs or
    forward pass fail, or whose vectors are not finite, is left out and named in the log, and is tried
    again by a later run. A view whose clips are all done or failed is written as
    FEATS/<name>/<view>_<split>.npz, and the checkpoint is removed once it holds nothing unfinished.
    Inputs:
    - name: the model's short name.
    - model, processor: the loaded model and processor.
    - hook: the TowerHook on the model's vision module.
    - label_ids: the output of vlm_zero_shot.label_token_ids.
    - split: "train", "dev" or "test".
    - with_text: also run the pass whose prompt gives the line's words.
    - minutes: the time limit for this split, or None for no limit.
    - every: clips between checkpoints.
    - timing_ms: a dict from view to per-clip milliseconds over this model's splits so far; this split's
      clips are added and the totals are merged into vision_timing.json.
    Outputs: a log record dict.
    Constraints: the checkpoint is saved before any exception leaves this function, and
    MAX_FAILED_IN_A_ROW failures in a row raise RuntimeError.
    """
    wanted = (*VIEWS, *([TEXT_VIEW] if with_text else []))
    views = [v for v in wanted if not (FEATS / name / f"{v}_{split}.npz").exists()]
    rec = {"model": name, "split": split, "views": views}
    if not views:
        print(f"{name} {split}: every view already extracted, skipped", flush=True)
        return {**rec, "skipped": True}
    rows = read_split(split)
    frames, failed_frames = extract_frames(rows, split)
    rows = [r for r in rows if r["clip"] in frames]
    labels = {r["clip"]: r["label"] for r in rows}
    whisper = (DATA / "MELD.Whisper" / f"{split}_sent_emo.csv").exists()
    text_version = "MELD.Whisper" if whisper else "MELD.Completed"
    words = {r["clip"]: r["text"] for r in read_split(split, text_version)} if with_text else {}
    state, text_source = load_partial(name, split)
    if with_text:
        if state.get(TEXT_VIEW) and text_source != text_version:
            print(f"{name} {split}: dropping {len(state[TEXT_VIEW])} checkpointed {TEXT_VIEW} rows prompted with "
                  f"{text_source}, since this run prompts with {text_version}", flush=True)
            state[TEXT_VIEW] = {}
        text_source = text_version
    for v in views:
        state.setdefault(v, {})
    resumed = {v: len(state[v]) for v in views}
    print(f"{name} {split}: {len(rows)} clips with frames, {len(failed_frames)} without; resuming with {resumed}",
          flush=True)

    def todo(r):
        """Description: Returns the passes a clip still needs: "frames", "text" or both."""
        need = []
        if any(v in views and r["clip"] not in state[v] for v in VIEWS):
            need.append("frames")
        if TEXT_VIEW in views and r["clip"] not in state[TEXT_VIEW]:
            need.append("text")
        return need

    def build(r, need):
        """Description: Prepares the inputs of the passes a clip needs on a worker thread, and times it."""
        t = time.perf_counter()
        out = {}
        if "frames" in need:
            out["frames"] = prepare(processor, frames[r["clip"]], build_prompt(None))
        if "text" in need:
            out["text"] = prepare(processor, frames[r["clip"]], build_prompt(words.get(r["clip"], r["text"])))
        return out, 1000 * (time.perf_counter() - t)

    def fail(clip, need, err):
        """Description: Records one failed pass of a clip against the views it would have filled."""
        reason = f"{type(err).__name__}: {str(err)[:200]}"
        for v in (VIEWS if need == "frames" else (TEXT_VIEW,)):
            if v in views:
                failed[v][clip] = reason
        print(f"{name} {split}: {clip} {need} pass failed, left out: {reason}", flush=True)

    work = [(r, todo(r)) for r in rows]
    work = [(r, n) for r, n in work if n]
    visual_dtype = next(find_visual(model)[0].parameters()).dtype
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    failed = {v: {} for v in views}
    stopped, nonfinite, hook_calls, prep_ms, wait_ms, since, done_n, in_a_row = False, [], [], [], [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(build, *work[0]) if work else None
        try:
            for i, (r, need) in enumerate(work):
                clip, ok = r["clip"], True
                t = time.perf_counter()
                current = pending
                pending = pool.submit(build, *work[i + 1]) if i + 1 < len(work) else None
                try:
                    batch, ms = current.result()
                except Exception as err:
                    for n in need:
                        fail(clip, n, err)
                    batch, ok = {}, False
                wait_ms.append(1000 * (time.perf_counter() - t))
                if ok:
                    prep_ms.append(ms)
                if "frames" in batch:
                    try:
                        hook.reset()
                        reader, scores, ms = run_pass(model, batch["frames"], visual_dtype, label_ids)
                        hook_calls.append(hook.calls)
                        tower = hook.pooled()
                        if not (np.isfinite(reader).all() and np.isfinite(tower).all()):
                            nonfinite.append(clip)
                            raise ValueError("non-finite tower or reader vector")
                        if "tower" in views:
                            state["tower"][clip] = (tower, hook.ms, np.zeros(len(LABELS), dtype=np.float32))
                        if "reader" in views:
                            state["reader"][clip] = (reader, ms, scores)
                    except Exception as err:
                        if isinstance(err, torch.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        fail(clip, "frames", err)
                        ok = False
                if "text" in batch:
                    try:
                        hook.reset()
                        reader, scores, ms = run_pass(model, batch["text"], visual_dtype, label_ids)
                        if not np.isfinite(reader).all():
                            nonfinite.append(clip)
                            raise ValueError("non-finite reader_text vector")
                        state[TEXT_VIEW][clip] = (reader, ms, scores)
                    except Exception as err:
                        if isinstance(err, torch.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        fail(clip, "text", err)
                        ok = False
                in_a_row = 0 if ok else in_a_row + 1
                if in_a_row >= MAX_FAILED_IN_A_ROW:
                    raise RuntimeError(f"{in_a_row} clips failed in a row, the last with: "
                                       f"{next(iter(f[clip] for f in failed.values() if clip in f), '')}")
                since += 1
                done_n = i + 1
                if since >= every:
                    save_partial(name, split, state, text_source)
                    since = 0
                    print(f"{name} {split}: {done_n} of {len(work)} clips this run, "
                          f"{len(set().union(*failed.values()))} clips with a failed pass, "
                          f"{time.perf_counter() - start:.0f} s, checkpoint saved", flush=True)
                if minutes is not None and time.perf_counter() - start > minutes * 60 and done_n < len(work):
                    stopped = True
                    break
        except BaseException:
            save_partial(name, split, state, text_source)
            raise
        finally:
            if pending is not None:
                pending.cancel()
    elapsed = time.perf_counter() - start
    finished = []
    for v in views:
        done = state[v]
        timing_ms.setdefault(v, []).extend(done[c][1] for c in done)
        if rows and all(r["clip"] in done or r["clip"] in failed[v] for r in rows) and done:
            clips = [r["clip"] for r in rows if r["clip"] in done]
            save_npz(name, v, split, clips, [labels[c] for c in clips],
                     np.stack([done[c][0] for c in clips]).astype(np.float32))
            finished.append(v)
    update_timing({f"{name}/{v}": timing_entry(v, ms, visual_dtype) for v, ms in timing_ms.items() if ms})
    if all(v in finished or (FEATS / name / f"{v}_{split}.npz").exists() for v in state):
        partial_path(name, split).unlink(missing_ok=True)
    else:
        save_partial(name, split, state, text_source)
    zero_shot = {}
    for v in ("reader", TEXT_VIEW):
        clips = [c for c in state.get(v, {}) if c in labels] if v in views else []
        if clips:
            zero_shot[v] = round(weighted_f1([labels[c] for c in clips],
                                             [int(np.argmax(state[v][c][2])) for c in clips]), 2)
    rec.update({
        "clips_with_frames": len(rows), "clips_without_frames": len(failed_frames), "resumed": resumed,
        "clips_this_run": done_n, "done": {v: len(state[v]) for v in views},
        "failed": {v: len(failed[v]) for v in views},
        "failed_clips": {v: dict(list(failed[v].items())[:50]) for v in views if failed[v]},
        "finished_views": finished, "stopped_by_time_limit": stopped, "seconds": round(elapsed, 1),
        "prep_ms_median": round(float(np.median(prep_ms)), 1) if prep_ms else None,
        "prep_wait_ms_median": round(float(np.median(wait_ms)), 1) if wait_ms else None,
        "hook_calls_per_clip": sorted(set(hook_calls)), "nonfinite_clips": nonfinite[:50],
        "gpu_mib_allocated_peak": round(torch.cuda.max_memory_allocated() / 2**20),
        "dims": {v: int(next(iter(state[v].values()))[0].shape[0]) for v in views if state[v]},
        "zero_shot_wf1_from_reader_logits": zero_shot,
    })
    if with_text:
        rec["reader_text_source"] = text_version
        rec["reader_text_clips_without_source_text"] = sum(r["clip"] not in words for r in rows)
    print(f"{name} {split}: finished {finished}, stopped by time limit {stopped}, "
          f"{len(set().union(*failed.values()))} clips with a failed pass, {elapsed:.0f} s", flush=True)
    return rec


def run_model(name, repo, splits, with_text, minutes, every):
    """
    Description: Loads one model, hooks its vision module and runs every chosen split, appending one log
    record per split to RESULTS/vlm_features_log.jsonl.
    Inputs:
    - name, repo: the model's short name and repo id.
    - splits: a list of splits.
    - with_text: also extract reader_text.
    - minutes: the time limit per split, or None.
    - every: clips between checkpoints.
    Outputs: none
    """
    views = (*VIEWS, *([TEXT_VIEW] if with_text else []))
    if all((FEATS / name / f"{v}_{s}.npz").exists() for v in views for s in splits):
        print(f"{name}: every view of every split already extracted, not loading the model", flush=True)
        return
    dtype = pick_dtype()
    t0 = time.perf_counter()
    model, processor = load_model(repo, dtype)
    load_s = time.perf_counter() - t0
    visual, visual_path = find_visual(model)
    hook = TowerHook(visual)
    label_ids = label_token_ids(getattr(processor, "tokenizer", processor))
    base = {"model": name, "repo": repo, "dtype": str(dtype).replace("torch.", ""), "load_s": round(load_s, 1),
            "params": sum(p.numel() for p in model.parameters()), "visual_module": visual_path,
            "visual_class": type(visual).__name__, "device": torch.cuda.get_device_name(0),
            "tower_pooling": "mean of the merged visual tokens (pooler_output) over all eight frames",
            "reader_position": "hidden_states[-1][0, -1], the final normed state whose lm_head logits ask reads"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    timing_ms = {}
    for split in splits:
        try:
            rec = run_split(name, model, processor, hook, label_ids, split, with_text, minutes, every, timing_ms)
        except Exception:
            rec = {"model": name, "split": split, "error": traceback.format_exc()[-1500:]}
            print(rec["error"], flush=True)
        with (RESULTS / "vlm_features_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({**base, **rec}) + "\n")
    del model, visual, hook, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """
    Description: Runs each chosen model in turn. A model that fails to load is logged and the next one still runs.
    Inputs: none; reads --models (name=repo pairs, comma-separated), --splits, --with-text, --limit-minutes
    (per model per split) and --checkpoint-every from the command line.
    Outputs: none
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--with-text", action="store_true")
    ap.add_argument("--limit-minutes", type=float, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()
    splits = [s for s in args.splits.split(",") if s]
    for pair in args.models.split(","):
        name, repo = pair.split("=", 1)
        try:
            run_model(name, repo, splits, args.with_text, args.limit_minutes, args.checkpoint_every)
        except Exception:
            err = {"model": name, "repo": repo, "error": traceback.format_exc()[-1500:]}
            print(err["error"], flush=True)
            RESULTS.mkdir(parents=True, exist_ok=True)
            with (RESULTS / "vlm_features_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(err) + "\n")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
