"""Builds the morning report, results/README.md, from the benchmark JSON files.

Contents:
- load: reads one result file, or None
- asr_table: the speech-to-text race table
- heads_tables: the frozen-encoder head tables
- reply_table: the reply-writer speed table
- vlm_table: the zero-shot VLM table
- failed_cell: a short table cell for a row that failed or had missing features
- seeded_table: a table of heads scored with several MLP seeds, optionally against text alone on the same clips
- vision_heads_tables: the text, vision, text-plus-vision and all-three head tables from heads_vision.json
- flatten: turns a nested result dict into table rows and prose notes
- generic_section: renders a result file whose layout is not fixed as a two-column table and notes
- cell: makes any value safe for one Markdown table cell
- timing_table: the per-encoder milliseconds per clip from vision_timing.json, with each writer's note
- faces_section: the face-finding notes from faces_stats.json and the frame counts from frames_failed.json and
  frames_repeated.json
- main: writes README.md next to the results

The first six Markdown tables are parsed by another tool, so every new section goes after the existing ones.
"""

import json
import os
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else
               os.environ.get("MELD_RESULTS") or Path(__file__).resolve().parent / "results")
"""Description: The results folder: the first command-line argument, else MELD_RESULTS when set, else the
bench folder's results, so a test run with MELD_RESULTS set never rewrites the saved README."""
ASR = ["whisper-turbo-fp16", "whisper-turbo-int8", "nemotron-1120ms", "nemotron-160ms", "parakeet-tdt-v3"]
VLM = ["qwen3.5-4b", "oneemo", "internvl3.5-4b", "gemma-4-e2b", "qwen2.5-omni-3b"]
NOTES_ASR = ("Word error rate is measured against MELD's own text after Whisper's English normaliser. Many "
             "clips also hold words from the line before or after, which no model can avoid, so insertions "
             "dominate every row and the absolute rates are high; compare the rows with each other. The saved "
             "MELD.Whisper test transcripts score 31.80% the same way, which matches the int8 row. Times are "
             "for one clip per call with the audio already decoded. GPU memory for the NeMo models includes "
             "PyTorch's cached pool, so it overstates what the model needs.")
NOTES_SPEED = ("Models built on Mamba, gated delta-rule or short-convolution layers (Nemotron 3 Nano, the "
               "Qwen3.5 family, LFM2.5) ran on transformers' slow reference kernels, because causal-conv1d, "
               "mamba-ssm and flash-linear-attention were not installed. Their speeds here understate what "
               "they reach with those kernels or with vLLM. "
               "Parameters loaded counts only the part of the model that writes text, so for Qwen3.5-4B (4.66B "
               "in the catalog), Qwen3.5-2B (2.27B) and Gemma 4 E2B it leaves out the vision and audio parts "
               "that the 6B limit counts.")
RUN_NOTES = [
    "The pod ran from 08:58 to 16:21 UTC on 2026-10-15, 7.4 hours. At $0.59 an hour that is about $4.35 of "
    "GPU time; Runpod's billing showed $3.93 in total, $3.76 GPU and $0.18 disk, when the pod was stopped, and "
    "its figures can lag. The pod is stopped with its 150 GB volume kept, which bills about $1 a day.",
    "The volume holds the MELD archive, the decoded audio, the extracted features, the uv package cache and "
    "the weights of the VLMs, the reply writers that were rerun, and EmbeddingGemma. The weights of every "
    "other finished model were deleted when the volume filled up during the night. The Python environments "
    "and the unpacked clips lived on the container disk, which Runpod erases on stop, so they need rebuilding "
    "from the cache next time.",
    "Things that went wrong and were fixed: the volume filled up and blocked two VLM downloads; Gemma 4 E2B's "
    "weights first failed to download; the free-text VLM answers were unusable (see that section); three "
    "reasoning models first wrote their thinking, so they were rerun with thinking turned off; Qwen2.5-Omni-3B "
    "first loaded as its speech-output model and was rerun with only its text part, the thinker.",
    "Frames failed to extract for 153 of the 2,610 test clips, so the VLM rows draw from 2,457 clips.",
]
"""Description: Facts about the run itself, written once the pod was stopped."""
NOTES_VLM = ("Each model saw 8 frames and was asked for one word from MELD's seven labels. The prediction is "
             "the label whose first token the model scores highest as the start of its answer, from one "
             "forward pass, so every answer is a label and the time is that one pass. A first try that let "
             "models write up to 8 tokens was thrown away: Qwen3.5-4B opened 1,579 of 1,585 answers with an "
             "explanation such as \"To determine the emotion\", so its 32.9 was the always-neutral score. "
             "Always answering neutral scores 31.27 on test. Some labels' first tokens are short fragments, "
             "such as a prefix of disgust or the letter f, which can also start other words, so those labels "
             "may be picked slightly too often; each result file lists the token ids used. Each model had a 40-minute limit, so a row may "
             "cover a random subset of the 2,457 clips whose frames extracted. The slow-kernel caveat above "
             "applies to the Qwen3.5-based models (Qwen3.5-4B and OneEmo). OneEmo's paper feeds 16 frames with "
             "its own prompt and reports a MELD hit rate, which is not weighted F1, so its lower score here "
             "with words is not a direct contradiction of that paper.")


def load(name):
    """
    Description: Reads one result file.
    Inputs:
    - name: the file stem in RESULTS.
    Outputs: the parsed dict, or None when the file is missing.
    """
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def asr_table():
    """
    Description: Builds the speech-to-text race table.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    rows = ["| Model | Word error rate | Median ms per line | 95th percentile ms | Real-time factor | GPU memory MiB |",
            "|---|---:|---:|---:|---:|---:|"]
    for m in ASR:
        r = load(f"asr_{m}")
        if r is None:
            rows.append(f"| {m} | not run | | | | |")
        else:
            rows.append(f"| {m} | {r['wer_pct']}% | {r['ms_median']} | {r['ms_p95']} | {r['real_time_factor']} | "
                        f"{r['gpu_mib_peak_over_idle']} |")
    return rows


def heads_tables():
    """
    Description: Builds the frozen-encoder head tables.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    r = load("heads")
    if r is None:
        return ["Not run."]
    out = []
    for part, title in (("text", "Text only"), ("voice", "Voice only"), ("text_plus_voice", "Text plus voice")):
        out += [f"**{title}.** Weighted F1 on MELD test.", "",
                "| Features | Logistic regression | MLP |", "|---|---:|---:|"]
        for name, v in r.get(part, {}).items():
            if "error" in v:
                out.append(f"| {name} | failed | failed |")
            else:
                out.append(f"| {name} | {v['logreg']['test_wf1']} | {v['mlp']['test_wf1']} |")
        out.append("")
    out.append(r.get("notes", ""))
    return out


def reply_table():
    """
    Description: Builds the reply-writer speed table.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    r = load("reply_speed")
    if r is None:
        return ["Not run."]
    extra = load("reply_speed_extra") or {}
    r.update({k: v for k, v in extra.items() if k != "setup"})
    out = ["| Model | Parameters loaded | Time to first token, median ms | Tokens per second | Whole reply, median ms |",
           "|---|---:|---:|---:|---:|"]
    for name, v in r.items():
        if name == "setup":
            continue
        if "error" in v:
            out.append(f"| {name} | failed: {v['error'].strip().splitlines()[-1][:120]} | | | |")
        else:
            out.append(f"| {name} | {v['params'] / 1e9:.2f}B | {v['ttft_ms_median']} | "
                       f"{v['decode_tokens_per_s_median']} | {v['reply_ms_median']} |")
    return out + ["", r.get("setup", "")]


def vlm_table():
    """
    Description: Builds the zero-shot VLM table.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    out = ["| Model | Clips | Frames only, weighted F1 | Frames and words, weighted F1 | Unparsed answers | Median ms per clip |",
           "|---|---:|---:|---:|---:|---:|"]
    for m in VLM:
        r = load(f"vlm_{m}")
        if r is None:
            out.append(f"| {m} | not run | | | | |")
        elif "error" in r:
            out.append(f"| {m} | failed: {r['error'].strip().splitlines()[-1][:120]} | | | | |")
        else:
            f, ft = r["frames"], r["frames_text"]
            out.append(f"| {m} | {r['clips']} | {f['wf1']} | {ft['wf1']} | {f['unparsed']} / {ft['unparsed']} | "
                       f"{f['ms_median']} / {ft['ms_median']} |")
    return out


def failed_cell(v):
    """
    Description: Shortens a failed row's error for a table cell.
    Inputs:
    - v: a result row holding "error".
    Outputs: "missing: <file>" for missing features, otherwise "failed: <last error line>".
    """
    last = (v["error"].strip().splitlines() or ["no message"])[-1][:120].replace("|", "/")
    if last.startswith("missing features:"):
        return last.replace("missing features:", "missing:")
    return f"failed: {last}"


def seeded_table(rows, same_clips):
    """
    Description: Builds a table of heads scored by heads.evaluate_seeds.
    Inputs:
    - rows: a dict from feature-set name to its scores.
    - same_clips: True to add the text-only scores on the same clips and the gains.
    Outputs: a list of Markdown lines.
    """
    head = "| Features | Test clips | Logistic regression | MLP, mean ± std |"
    rule = "|---|---:|---:|---:|"
    if same_clips:
        head += " Text only, same clips, logistic | Text only, same clips, MLP | Gain, logistic | Gain, MLP ± std |"
        rule += "---:|---:|---:|---:|"
    else:
        head += " Dev MLP, mean |"
        rule += "---:|"
    width = 8 if same_clips else 5
    out = [head, rule]
    if not rows:
        out.append("| none |" + " |" * (width - 1))
    for name, v in rows.items():
        if "error" in v:
            out.append(f"| {name} | {failed_cell(v)} |" + " |" * (width - 2))
            continue
        m = v["mlp"]
        line = (f"| {name} | {v['test_lines']} | {v['logreg']['test_wf1']} | "
                f"{m['test_wf1_mean']} ± {m['test_wf1_std']} |")
        if same_clips:
            t, g = v["text_only_same_clips"], v["gain"]
            line += (f" {t['logreg']['test_wf1']} | {t['mlp']['test_wf1_mean']} ± {t['mlp']['test_wf1_std']} | "
                     f"{g['logreg']:+.2f} | {g['mlp_mean']:+.2f} ± {g['mlp_std']} |")
        else:
            line += f" {m['dev_wf1_mean']} |"
        out.append(line)
    return out


def vision_heads_tables():
    """
    Description: Builds the text, vision, text-plus-vision and all-three head tables.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    r = load("heads_vision")
    if r is None:
        return ["Not run."]
    out = ["**Text only.** Weighted F1 on MELD test. The first row repeats last night's best fair text set to "
           "check that the run reproduces; the others use Whisper's transcripts, the words the demo reads.", ""]
    out += seeded_table({**r.get("reproducibility", {}), **r.get("text_live", {})}, False)
    base = next(iter(r.get("reproducibility", {}).values()), {})
    if "mlp" in base and r.get("last_night_mlp_test_wf1") is not None:
        out += ["", f"Last night's single MLP run on this set scored {r['last_night_mlp_test_wf1']}; seed 0 here "
                f"scored {base['mlp']['runs'][0]['test_wf1']}."]
    out += ["", "**Vision only.** Weighted F1 on MELD test.", ""] + seeded_table(r.get("vision", {}), False)
    out += ["", "**Text plus vision.** Weighted F1 on MELD test, with the text-only heads trained and scored on "
            "exactly the same clips and the gain over them.", ""] + seeded_table(r.get("text_plus_vision", {}), True)
    best = r.get("best_vision_by_dev")
    out += ["", "**Text, vision and voice.** Weighted F1 on MELD test" +
            (f", with {best}, the vision set with the best mean dev MLP F1." if best else "."), ""]
    out += seeded_table(r["all_three"], True) if r.get("all_three") else ["Not run."]
    return out + ["", r.get("notes", "")]


def flatten(d, prefix=""):
    """
    Description: Turns a nested result dict into table rows and prose notes.
    Inputs:
    - d: a dict from a result file.
    - prefix: the dotted key path so far.
    Outputs: (a list of (key, value) rows, a list of long strings to print as prose).
    Constraints: dicts with more than 12 entries and lists longer than 6 are shown as a count, since they
    usually hold one entry per clip.
    """
    rows, prose = [], []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and len(v) <= 12:
            r, p = flatten(v, f"{key}.")
            rows += r
            prose += p
        elif isinstance(v, list) and len(v) <= 6 and all(not isinstance(x, (dict, list)) for x in v):
            rows.append((key, ", ".join(str(x) for x in v) or "none"))
        elif isinstance(v, (dict, list)):
            rows.append((key, f"{len(v):,} listed"))
        elif isinstance(v, str) and len(v) > 100:
            prose.append(f"{key}: {v}")
        else:
            rows.append((key, v))
    return rows, prose


def generic_section(name):
    """
    Description: Renders a result file whose layout is not fixed by this report as a two-column table
    followed by its long strings as prose.
    Inputs:
    - name: the file stem in RESULTS.
    Outputs: a list of Markdown lines, or ["Not run."] when the file is missing.
    """
    r = load(name)
    if r is None:
        return ["Not run."]
    rows, prose = flatten(r)
    out = ["| Item | Value |", "|---|---|"] + [f"| {k} | {cell(v)} |" for k, v in rows]
    return out + [x for p in prose for x in ("", p)]


def cell(v):
    """
    Description: Makes a value safe for one Markdown table cell.
    Inputs:
    - v: any value; None gives an empty cell and lists are joined with commas.
    Outputs: a string with no pipes or line breaks.
    """
    if v is None:
        return ""
    if isinstance(v, list):
        v = ", ".join(str(x) for x in v)
    return " ".join(str(v).replace("|", "/").split())


def timing_table():
    """
    Description: Builds the per-encoder milliseconds-per-clip table from vision_timing.json.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    r = load("vision_timing")
    if r is None:
        return ["Not run."]
    out = ["| Encoder / view | Median ms per clip | 95th percentile ms | Clips | Device | Notes |",
           "|---|---:|---:|---:|---|---|"]
    known = ("ms_median", "ms_p95", "clips", "device")
    for name, v in sorted((k, v) for k, v in r.items() if isinstance(v, dict)):
        notes = "; ".join(cell(x) if k == "note" else f"{k} {cell(x)}" for k, x in v.items() if k not in known)
        out.append(f"| {name} | {cell(v.get('ms_median'))} | {cell(v.get('ms_p95'))} | {cell(v.get('clips'))} | "
                   f"{cell(v.get('device'))} | {notes} |")
    return out + ["", "Each row is milliseconds per clip as recorded by the script that extracted it, and its notes "
                  "say how it was measured: amortised batch time, one clip per call, or the time inside one of "
                  "several parallel CPU threads, so rows measured differently are not directly comparable. Face "
                  "finding has its own row, yunet/face-finding."]


def faces_section():
    """
    Description: Builds the face-finding notes from faces_stats.json, and the failed-clip and repeated-frame counts
    from frames_failed.json and frames_repeated.json.
    Inputs: none
    Outputs: a list of Markdown lines.
    """
    out = ["**Face finding.**", ""]
    r = load("faces_stats")
    per_split = {k: v for k, v in (r or {}).items() if isinstance(v, dict) and "clips" in v}
    if r is None:
        out.append("Not run.")
    elif not per_split:
        out += generic_section("faces_stats")
    else:
        out += ["| Split | Clips with frames | Clips without frames | Clips with no face | Frames | "
                "Frames with a face | Frames with several faces | Median face score |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for split, v in per_split.items():
            out.append(f"| {split} | {cell(v.get('clips'))} | {cell(v.get('clips_without_frames'))} | "
                       f"{cell(v.get('clips_with_no_face'))} | {cell(v.get('frames'))} | "
                       f"{cell(v.get('frames_with_face'))} | {cell(v.get('frames_with_several_faces'))} | "
                       f"{cell(v.get('median_face_score'))} |")
        det = r.get("detector") or {}
        if isinstance(det, dict) and det:
            out += ["", f"Detector: {det.get('file', '')} from {det.get('repo', '')}, score threshold "
                        f"{det.get('score_threshold', '')}."]
        if r.get("speaker_rule"):
            out += ["", f"Speaker rule: {r['speaker_rule']}"]
        out += ["", "A clip with no face in any frame has an all-zero face-view row whose has_face column is 0.0."]
    failed = load("frames_failed")
    out += ["", "**Clips whose frames failed to extract.**", ""]
    if failed is None:
        return out + ["Not run."]
    repeated = load("frames_repeated") or {}
    out += ["| Split | Failed clips | Clips with repeated frames |", "|---|---:|---:|"]
    out += [f"| {split} | {len(bad):,} | {cell(len(repeated[split]) if split in repeated else None)} |"
            for split, bad in failed.items()]
    return out + ["", "A short clip can hold fewer decodable frames than the eight slots; frames.py then fills an "
                      "empty slot with a copy of the nearest frame that worked rather than dropping the clip."]


def main():
    """
    Description: Writes README.md in RESULTS from whatever result files exist.
    Inputs: none; an optional results folder as the first command-line argument, else MELD_RESULTS.
    Outputs: none
    """
    lines = ["# Overnight benchmark results", "",
             "Hardware: one NVIDIA A40 48 GB Runpod pod (Secure Cloud, EU-SE-1), 9 vCPUs. "
             "Data: MELD test split, 2,610 lines.", "",
             "## Speech to text", ""] + asr_table() + ["", NOTES_ASR,
             "", "## Frozen encoders with a small head", ""] + heads_tables() + [
             "", "## Reply writers", ""] + reply_table() + ["", NOTES_SPEED,
             "", "## Zero-shot vision-language models", ""] + vlm_table() + ["", NOTES_VLM,
             "", "## About the run", ""] + [p + "\n" for p in RUN_NOTES] + [
             "## Vision features with a small head", ""] + vision_heads_tables() + [
             "", "## Vision encoder speed", ""] + timing_table() + [
             "", "## Faces and frames", ""] + faces_section()
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
