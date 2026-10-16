"""Extracts frozen text and voice features for every MELD line, for the head baselines in heads.py.

Contents:
- TEXT_ENCODERS, VOICE_ENCODERS: the catalog encoders this script runs
- TEXT_VERSIONS, MAX_AUDIO_S: the versions of the label text and the longest audio encoded
- chosen: turns an "all", "none" or comma-separated option into a list of names
- text_features: mean-pooled last-layer vectors for a list of lines
- voice_features: last-layer and all-layer mean vectors for each clip's audio
- emotion2vec_features: utterance vectors from emotion2vec+ through FunASR
- save: writes one feature file
- main: extracts the chosen encoders and writes FEATS/<encoder>/<version>_<split>.npz
"""

import argparse
import json
import time
import traceback

import numpy as np
import torch

from common import DATA, FEATS, RESULTS, SAMPLE_RATE, load_audio_cache, read_split

TEXT_ENCODERS = {
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "emoberta-large": "tae898/emoberta-large",
    "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
    "embeddinggemma-300m": "google/embeddinggemma-300m",
}
"""Description: Text encoders by short name. EmoBERTa's released weights were trained on MELD with no
splits named, and have likely seen MELD's test lines."""
VOICE_ENCODERS = {
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-large": "facebook/hubert-large-ll60k",
    "wav2vec2-superb-er": "superb/wav2vec2-base-superb-er",
    "emotion2vec-plus-large": "emotion2vec/emotion2vec_plus_large",
}
"""Description: Voice encoders by short name."""
TEXT_VERSIONS = {"raw": "MELD.Raw", "completed": "MELD.Completed", "whisper": "MELD.Whisper",
                 "nemotron": "MELD.Nemotron"}
"""Description: The versions of the label text, by short name. MELD.Nemotron is written by nemotron_text.py."""
MAX_AUDIO_S = 20.0
"""Description: Longer clips are cut to this many seconds before encoding."""


@torch.inference_mode()
def text_features(repo, texts):
    """
    Description: Encodes lines of text and mean-pools the last layer over the real tokens.
    Inputs:
    - repo: the Hugging Face repo id.
    - texts: a list of strings.
    Outputs: a float32 array of shape (lines, hidden size).
    """
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo, dtype=torch.float32).cuda().eval()
    out = []
    for i in range(0, len(texts), 64):
        batch = tok([t or " " for t in texts[i:i + 64]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to("cuda")
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        out.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


@torch.inference_mode()
def voice_features(repo, clips, audio):
    """
    Description: Encodes each clip's audio and averages over time, both for the last layer and for
    the mean of all layers.
    Inputs:
    - repo: the Hugging Face repo id.
    - clips: clip names in order.
    - audio: a dict from clip name to samples.
    Outputs: a float32 array of shape (clips, 2 * hidden size), last layer first.
    """
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).cuda().eval()
    out = []
    for clip in clips:
        a = audio[clip][: int(MAX_AUDIO_S * SAMPLE_RATE)]
        if len(a) < SAMPLE_RATE // 10:
            a = np.pad(a, (0, SAMPLE_RATE // 10 - len(a)))
        inputs = fe(a, sampling_rate=SAMPLE_RATE, return_tensors="pt").to("cuda")
        hs = model(**inputs, output_hidden_states=True).hidden_states
        last = hs[-1][0].mean(0)
        every = torch.stack(hs[1:]).mean(0)[0].mean(0)
        out.append(torch.cat([last, every]).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(out)


def emotion2vec_features(repo, clips, audio):
    """
    Description: Extracts emotion2vec+ utterance vectors through FunASR.
    Inputs:
    - repo: the Hugging Face repo id.
    - clips: clip names in order.
    - audio: a dict from clip name to samples.
    Outputs: a float32 array of shape (clips, 1024).
    """
    from funasr import AutoModel
    model = AutoModel(model=repo, hub="hf", disable_update=True, device="cuda")
    out = []
    for clip in clips:
        a = audio[clip][: int(MAX_AUDIO_S * SAMPLE_RATE)]
        res = model.generate(a, granularity="utterance", extract_embedding=True)
        out.append(np.asarray(res[0]["feats"], dtype=np.float32).reshape(-1))
    return np.stack(out)


def save(name, version, split, clips, labels, x):
    """
    Description: Writes one feature file.
    Inputs:
    - name: the encoder's short name.
    - version: the text version's short name, or "audio".
    - split: "train", "dev" or "test".
    - clips, labels: the lines' clip names and class indices.
    - x: the feature array.
    Outputs: none
    """
    folder = FEATS / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{version}_{split}.npz", x=x, clips=np.array(clips), labels=np.array(labels))


def chosen(arg, catalog, ap, flag):
    """
    Description: Turns a command-line choice into a list of short names.
    Inputs:
    - arg: "all", "none", an empty string, or comma-separated short names.
    - catalog: the dict of known short names.
    - ap: the argument parser, used to report unknown names.
    - flag: the option's name, for the error message.
    Outputs: a list of short names; "none" and an empty string give an empty list.
    Constraints: exits with a usage error when a name is not in the catalog.
    """
    if arg == "all":
        return list(catalog)
    if arg == "none":
        return []
    names = [n for n in arg.split(",") if n]
    unknown = [n for n in names if n not in catalog]
    if unknown:
        ap.error(f"{flag}: unknown {', '.join(unknown)}; choose from {', '.join(catalog)}")
    return names


def main():
    """
    Description: Extracts features for the chosen encoders over all three splits, and logs the
    time each took. An encoder that fails is logged and skipped.
    Inputs: none; reads --text and --voice (comma-separated short names, "all" or "none") and
    --versions (comma-separated text version short names, or "all") from the command line.
    Outputs: none
    Constraints: --versions all skips a text version whose label files are missing, such as MELD.Nemotron
    before nemotron_text.py has run; a version named explicitly must exist.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="all")
    ap.add_argument("--voice", default="all")
    ap.add_argument("--versions", default="all")
    args = ap.parse_args()
    text = chosen(args.text, TEXT_ENCODERS, ap, "--text")
    voice = chosen(args.voice, VOICE_ENCODERS, ap, "--voice")
    versions = chosen(args.versions, TEXT_VERSIONS, ap, "--versions")
    if args.versions == "all":
        versions = [v for v in versions if (DATA / TEXT_VERSIONS[v] / "test_sent_emo.csv").exists()
                    or (v == "raw" and (DATA / "csv_raw" / "test_sent_emo.csv").exists())]
    log = []
    for name in text:
        t0 = time.perf_counter()
        try:
            for vname in versions:
                for split in ("train", "dev", "test"):
                    rows = read_split(split, TEXT_VERSIONS[vname])
                    x = text_features(TEXT_ENCODERS[name], [r["text"] for r in rows])
                    save(name, vname, split, [r["clip"] for r in rows], [r["label"] for r in rows], x)
            log.append({"encoder": name, "kind": "text", "versions": versions,
                        "seconds": round(time.perf_counter() - t0)})
        except Exception:
            log.append({"encoder": name, "kind": "text", "error": traceback.format_exc()[-1500:]})
        print(log[-1], flush=True)
    for name in voice:
        t0 = time.perf_counter()
        try:
            for split in ("train", "dev", "test"):
                audio = load_audio_cache(split)
                rows = [r for r in read_split(split) if r["clip"] in audio]
                clips = [r["clip"] for r in rows]
                if name.startswith("emotion2vec"):
                    x = emotion2vec_features(VOICE_ENCODERS[name], clips, audio)
                else:
                    x = voice_features(VOICE_ENCODERS[name], clips, audio)
                save(name, "audio", split, clips, [r["label"] for r in rows], x)
            log.append({"encoder": name, "kind": "voice", "seconds": round(time.perf_counter() - t0)})
        except Exception:
            log.append({"encoder": name, "kind": "voice", "error": traceback.format_exc()[-1500:]})
        print(log[-1], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "features_log.jsonl").open("a", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
