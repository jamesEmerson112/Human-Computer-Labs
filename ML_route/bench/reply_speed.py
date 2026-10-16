"""Times small language models writing one short reply per MELD line, as the demo's reply writer would.

Contents:
- REPLY_MODELS: the catalog models this script tries
- load_model: loads a model in bf16 on the GPU, as a text or an image-text model
- prompt_for: builds the reply prompt for one line
- time_model: measures time to first token and decode speed over the prompts
- main: times every model and writes reply_speed.json with one sample reply per model
"""

import argparse
import time
import traceback

import numpy as np
import torch

from common import LABELS, read_split, write_result

REPLY_MODELS = {
    "granite-4.1-3b": "ibm-granite/granite-4.1-3b",
    "nemotron-3-nano-4b": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "lfm2.5-1.2b-instruct": "LiquidAI/LFM2.5-1.2B-Instruct",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
}
"""Description: Reply-writer models by short name. Gated repos fail without a Hugging Face token and are logged."""
MAX_NEW_TOKENS = 40
PROMPTS = 50


def load_model(repo):
    """
    Description: Loads a model and its tokenizer in bf16 on the GPU, trying a text-only class first.
    Inputs:
    - repo: the Hugging Face repo id.
    Outputs: (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16)
    return model.cuda().eval(), tok


def prompt_for(row):
    """
    Description: Builds the reply prompt the explainer page uses, from a line and its gold emotion.
    Inputs:
    - row: a line dict from read_split.
    Outputs: the prompt string.
    """
    return (f'Someone said "{row["text"]}" and sounds {LABELS[row["label"]]}. '
            "Reply in one short, kind sentence.")


@torch.inference_mode()
def time_model(model, tok, prompts):
    """
    Description: Greedy-decodes each prompt twice, once for one token and once for up to
    MAX_NEW_TOKENS, to measure time to first token and decode speed at batch size one.
    Inputs:
    - model, tok: a loaded model and tokenizer.
    - prompts: a list of prompt strings.
    Outputs: a dict of timing statistics and the first three replies.
    """
    ttft, rate, totals, samples = [], [], [], []
    for i, p in enumerate(prompts):
        # enable_thinking=False keeps reasoning models from writing a hidden plan before the reply.
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, enable_thinking=False).to("cuda")
        n_in = ids["input_ids"].shape[1]
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(**ids, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        first = time.perf_counter() - t
        t = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        total = time.perf_counter() - t
        n_out = out.shape[1] - n_in
        if i >= 2:
            ttft.append(first)
            totals.append(total)
            if n_out > 1:
                rate.append((n_out - 1) / max(total - first, 1e-6))
        if len(samples) < 3:
            samples.append({"prompt": p, "reply": tok.decode(out[0, n_in:], skip_special_tokens=True).strip()})
    return {"ttft_ms_median": round(1000 * float(np.median(ttft)), 1),
            "ttft_ms_p95": round(1000 * float(np.percentile(ttft, 95)), 1),
            "decode_tokens_per_s_median": round(float(np.median(rate)), 1) if rate else None,
            "reply_ms_median": round(1000 * float(np.median(totals)), 1),
            "samples": samples}


def main():
    """
    Description: Times each reply model over PROMPTS test lines (the first two are warm-up and not
    counted) and writes the results. A model that fails to load is logged with its error.
    Inputs: none; reads --models and --out from the command line.
    Outputs: none
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--out", default="reply_speed", help="result file stem, so a rerun does not overwrite")
    args = ap.parse_args()
    names = list(REPLY_MODELS) if args.models == "all" else args.models.split(",")
    rows = read_split("test")[: PROMPTS + 2]
    prompts = [prompt_for(r) for r in rows]
    results = {}
    for name in names:
        try:
            t = time.perf_counter()
            model, tok = load_model(REPLY_MODELS[name])
            load_s = time.perf_counter() - t
            stats = time_model(model, tok, prompts)
            stats["load_s"] = round(load_s, 1)
            stats["params"] = sum(p.numel() for p in model.parameters())
            stats["gpu_mib_allocated_peak"] = round(torch.cuda.max_memory_allocated() / 2**20)
            results[name] = stats
            del model
        except Exception:
            results[name] = {"error": traceback.format_exc()[-1200:]}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(name, results[name].get("ttft_ms_median"), results[name].get("error", "")[-200:], flush=True)
    results["setup"] = (f"Hugging Face transformers generate, bf16, batch size 1, greedy, up to {MAX_NEW_TOKENS} "
                        f"new tokens, {PROMPTS} MELD test lines with the gold emotion in the prompt")
    write_result(args.out, results)


if __name__ == "__main__":
    main()
