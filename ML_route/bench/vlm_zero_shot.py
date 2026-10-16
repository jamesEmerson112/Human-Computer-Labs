"""Asks vision-language models, with no MELD training, which emotion the speaker shows in each test clip.

Contents:
- load_vlm: loads an image-text model and its processor in bf16
- build_prompt: the question asked for one clip, with or without the line's words
- label_token_ids: the first token of each label word, in the spellings a model might start with
- ask: scores the seven label words as the model's first answer token for one clip
- run_model: answers both question forms for clips in a fixed shuffled order, up to a time limit
- main: extracts frames, runs the chosen models and writes vlm_<model>.json for each
"""

import argparse
import random
import time
import traceback

import numpy as np
import torch

from common import LABELS, read_split, weighted_f1, write_result
from frames import FRAME_WIDTH, N_FRAMES, extract_frames

def load_vlm(repo):
    """
    Description: Loads an image-text model and its processor in bf16 on the GPU.
    Inputs:
    - repo: the Hugging Face repo id.
    Outputs: (model, processor).
    """
    import transformers
    processor = transformers.AutoProcessor.from_pretrained(repo)
    errors = []
    classes = ["AutoModelForImageTextToText", "AutoModelForMultimodalLM"]
    if "omni" in repo.lower():
        # Only the thinker, the part that writes text, has a forward pass; the full Omni model also speaks.
        classes.insert(0, "Qwen2_5OmniThinkerForConditionalGeneration")
    for cls in classes:
        loader = getattr(transformers, cls, None)
        if loader is None:
            continue
        try:
            return loader.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval(), processor
        except Exception as err:
            errors.append(f"{cls}: {str(err)[:300]}")
    raise RuntimeError(" | ".join(errors))


def build_prompt(text):
    """
    Description: Builds the question for one clip.
    Inputs:
    - text: the line's words, or None to ask from the frames alone.
    Outputs: the question string.
    """
    said = f' The person speaking says: "{text}"' if text is not None else ""
    return (f"These are {N_FRAMES} frames, in order, from a short clip of the TV show Friends.{said} "
            "Which emotion does the person speaking show? Answer with exactly one word from this list: "
            + ", ".join(LABELS) + ".")


def label_token_ids(tokenizer):
    """
    Description: Finds the first token of each label word, lower-case and capitalised, with and
    without a leading space.
    Inputs:
    - tokenizer: the model's tokenizer.
    Outputs: a list of seven lists of token ids, in LABELS order.
    """
    out = []
    for lab in LABELS:
        ids = set()
        for v in (lab, lab.capitalize(), " " + lab, " " + lab.capitalize()):
            toks = tokenizer(v, add_special_tokens=False).input_ids
            if toks:
                ids.add(toks[0])
        out.append(sorted(ids))
    return out


@torch.inference_mode()
def ask(model, processor, frames, question, label_ids):
    """
    Description: Runs one forward pass over the clip's frames and question, and picks the label
    whose first token the model scores highest as the start of its answer. Every answer is
    therefore one of the seven labels; models that would first write an explanation are not
    penalised for it.
    Inputs:
    - model, processor: a loaded model and processor.
    - frames: the clip's frame paths.
    - question: the question text.
    - label_ids: the output of label_token_ids.
    Outputs: (predicted label, the seven label scores, milliseconds for the call).
    """
    from PIL import Image
    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False).to("cuda")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    logits = model(**inputs).logits[0, -1].float()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t)
    scores = [max(logits[i].item() for i in ids) for ids in label_ids]
    return LABELS[int(np.argmax(scores))], [round(s, 3) for s in scores], ms


def run_model(name, repo, rows, frames, minutes):
    """
    Description: Answers both question forms for each clip in a fixed shuffled order until the
    clips run out or the time limit passes, then scores both forms on the same clips.
    Inputs:
    - name, repo: the model's short name and repo id.
    - rows: test line dicts that have frames.
    - frames: the dict from extract_frames.
    - minutes: the time limit for this model.
    Outputs: none; writes vlm_<name>.json and .jsonl.
    """
    t0 = time.perf_counter()
    model, processor = load_vlm(repo)
    load_s = time.perf_counter() - t0
    tokenizer = getattr(processor, "tokenizer", processor)
    label_ids = label_token_ids(tokenizer)
    order = rows[:]
    random.Random(0).shuffle(order)
    lines = []
    start = time.perf_counter()
    for r in order:
        rec = {"clip": r["clip"], "gold": LABELS[r["label"]]}
        for form, text in (("frames", None), ("frames_text", r["text"])):
            pred, scores, ms = ask(model, processor, frames[r["clip"]], build_prompt(text), label_ids)
            rec[form] = {"pred": pred, "scores": scores, "ms": round(ms, 1)}
        lines.append(rec)
        if time.perf_counter() - start > minutes * 60:
            break
    summary = {"model": name, "repo": repo, "clips": len(lines), "test_clips_available": len(rows),
               "subset": len(lines) < len(rows), "load_s": round(load_s, 1),
               "params": sum(p.numel() for p in model.parameters()),
               "gpu_mib_allocated_peak": round(torch.cuda.max_memory_allocated() / 2**20)}
    gold = [LABELS.index(l["gold"]) for l in lines]
    summary["label_first_tokens"] = {lab: ids for lab, ids in zip(LABELS, label_ids)}
    for form in ("frames", "frames_text"):
        pred = [LABELS.index(l[form]["pred"]) for l in lines]
        summary[form] = {
            "wf1": round(weighted_f1(gold, pred), 2),
            "accuracy": round(100 * float(np.mean(np.array(pred) == np.array(gold))), 2),
            "unparsed": 0,
            "ms_median": round(float(np.median([l[form]["ms"] for l in lines])), 1),
            "predicted": {lab: sum(1 for p in pred if p == i) for i, lab in enumerate(LABELS)},
        }
    summary["setup"] = (f"{N_FRAMES} evenly spaced frames at {FRAME_WIDTH} px wide, bf16, one forward pass; the "
                        "label whose first token scores highest as the answer's first token is the prediction; "
                        "clips in a fixed shuffled order (seed 0)")
    write_result(f"vlm_{name}", summary, lines)
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    """
    Description: Extracts frames for the test split, then runs each chosen model in turn. A model
    that fails is written as an error result.
    Inputs: none; reads --models (name=repo pairs, comma-separated) and --minutes from the command line.
    Outputs: none
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    args = ap.parse_args()
    rows = read_split("test")
    frames, _ = extract_frames(rows, "test")
    rows = [r for r in rows if r["clip"] in frames]
    print(f"frames ready for {len(rows)} clips", flush=True)
    for pair in args.models.split(","):
        name, repo = pair.split("=", 1)
        try:
            run_model(name, repo, rows, frames, args.minutes)
        except Exception:
            write_result(f"vlm_{name}", {"model": name, "repo": repo, "error": traceback.format_exc()[-1500:]})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
