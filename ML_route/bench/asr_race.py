"""Times speech-to-text models one clip at a time on MELD's test split and scores their word error rate.

Contents:
- load_whisper: loads faster-whisper and returns a transcribe function
- load_nemo: loads a NeMo model (Nemotron Streaming or Parakeet) and returns a transcribe function
- MODELS: the model variants this script can run
- main: transcribes every decodable test clip with one model variant and writes its results
"""

import argparse
import time

import numpy as np

from common import GpuMemory, SAMPLE_RATE, load_audio_cache, normalize, read_split, write_result


def load_whisper(compute):
    """
    Description: Loads Whisper large-v3-turbo with faster-whisper on the GPU.
    Inputs:
    - compute: the CTranslate2 compute type, such as "float16" or "int8".
    Outputs: a function from samples to text, using the same settings as the player's live Whisper.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type=compute,
                         download_root="/workspace/hf/whisper")

    def run(audio):
        segments, _ = model.transcribe(audio, language="en", beam_size=5, word_timestamps=False,
                                       vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
    return run


def load_nemo(name, context=None):
    """
    Description: Loads a NeMo speech model on the GPU.
    Inputs:
    - name: the Hugging Face repo id.
    - context: an optional [left, right] attention context in 80 ms frames, for cache-aware
      streaming models; [70, 13] is the 1.12 s setting and [70, 1] the 160 ms one.
    Outputs: a function from samples to text.
    Constraints: transcribes the whole clip in one offline call; NVIDIA's docs say a cache-aware
    model gives the same output this way as when streamed with the same context.
    """
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=name).cuda().eval()
    if context is not None:
        model.encoder.set_default_att_context_size(context)

    def run(audio):
        with torch.inference_mode():
            out = model.transcribe([audio], batch_size=1, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        first = out[0]
        return (first.text if hasattr(first, "text") else str(first)).strip()
    return run


MODELS = {
    "whisper-turbo-fp16": lambda: load_whisper("float16"),
    "whisper-turbo-int8": lambda: load_whisper("int8"),
    "nemotron-1120ms": lambda: load_nemo("nvidia/nemotron-speech-streaming-en-0.6b", [70, 13]),
    "nemotron-160ms": lambda: load_nemo("nvidia/nemotron-speech-streaming-en-0.6b", [70, 1]),
    "parakeet-tdt-v3": lambda: load_nemo("nvidia/parakeet-tdt-0.6b-v3"),
}
"""Description: The model variants, each a function that loads the model and returns its transcriber."""


def sync():
    """Description: Waits for queued GPU work when PyTorch is loaded, so timings are complete."""
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass


def main():
    """
    Description: Transcribes every decodable test clip one at a time with one model variant,
    timing each call, and writes word error rate, latency and GPU memory.
    Inputs: none; reads --model and --limit from the command line.
    Outputs: none
    """
    import jiwer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = read_split("test")
    audio = load_audio_cache("test")
    rows = [r for r in rows if r["clip"] in audio][:args.limit]
    mem = GpuMemory().start()
    t0 = time.perf_counter()
    run = MODELS[args.model]()
    load_s = time.perf_counter() - t0
    for r in rows[:3]:
        run(audio[r["clip"]])
    sync()

    lines, refs, hyps, times, lengths = [], [], [], [], []
    for i, r in enumerate(rows):
        a = audio[r["clip"]]
        t = time.perf_counter()
        text = run(a)
        sync()
        ms = 1000 * (time.perf_counter() - t)
        ref, hyp = normalize(r["text"]), normalize(text)
        lines.append({"clip": r["clip"], "ref": r["text"], "hyp": text, "ms": round(ms, 1),
                      "audio_s": round(len(a) / SAMPLE_RATE, 3)})
        if ref.strip():
            refs.append(ref)
            hyps.append(hyp)
        times.append(ms)
        lengths.append(len(a) / SAMPLE_RATE)
        if i % 500 == 0:
            print(f"{args.model}: {i}/{len(rows)}", flush=True)
    base, peak = mem.stop()
    t = np.array(times)
    out = jiwer.process_words(refs, hyps)
    write_result(f"asr_{args.model}", {
        "model": args.model, "clips": len(rows), "scored_clips": len(refs),
        "wer_pct": round(100 * out.wer, 2),
        "substitutions": out.substitutions, "deletions": out.deletions, "insertions": out.insertions,
        "ms_mean": round(float(t.mean()), 1), "ms_median": round(float(np.median(t)), 1),
        "ms_p95": round(float(np.percentile(t, 95)), 1), "ms_max": round(float(t.max()), 1),
        "audio_s_mean": round(float(np.mean(lengths)), 3),
        "real_time_factor": round(float(t.sum() / 1000 / np.sum(lengths)), 4),
        "load_s": round(load_s, 1), "gpu_mib_peak_over_idle": peak - base,
        "reference": "MELD.Completed test text, which keeps MELD's published words",
        "timing": "one clip per call, from the end of the clip's audio to the text, audio already decoded",
    }, lines)


if __name__ == "__main__":
    main()
