"""Trains small heads on frozen MELD features and reports weighted F1 on the test split.

Contents:
- SUSPECT: encoders left out of the best-text pick
- VIEWS, SEEDS, TEXT_BASE, TEXT_NEMOTRON, VOICE_SETS: what the --vision mode scores
- load: reads one encoder's features for a split
- join: lines up text and voice features by clip
- subset: keeps only the named clips of one feature set
- checked_join: joins two feature sets and fails when they share no clips or disagree on a label
- fit_logreg: a class-weighted logistic regression, tuned on dev
- fit_mlp: a one-hidden-layer network, tuned and early-stopped on dev
- evaluate: runs both heads on one feature set and returns their scores
- evaluate_seeds: runs logistic regression once and the MLP once per seed, with mean and spread
- text_voice: scores text-only, voice-only and text-plus-voice heads and writes heads.json
- splits: reads one feature set for all three splits and fails on missing files or non-finite values
- gain: the joined set's test gain over text alone on the same clips, with its spread
- vision: scores text, vision, text-plus-vision and all-three heads and writes heads_vision.json
- main: runs text_voice, or vision with --vision
"""

import argparse
import json
import traceback

import numpy as np
import torch

from common import FEATS, LABELS, RESULTS, weighted_f1, write_result

SUSPECT = {"emoberta-large"}
"""Description: Encoders left out of the best-text pick. EmoBERTa-large's frozen features scored 77.9 on
MELD test against 63.46 on its own card, and its released weights were trained on MELD with no splits
named, so it has likely seen the test lines."""
VIEWS = ("frame", "face", "tower", "reader", "reader_text")
"""Description: The vision feature views, stored as FEATS/<encoder>/<view>_<split>.npz."""
SEEDS = (0, 1, 2)
"""Description: The MLP seeds used in --vision mode."""
TEXT_BASE = ("roberta-base", "completed")
"""Description: Last night's best fair text set, MELD's published words, whose MLP scored 61.36 on test."""
TEXT_NEMOTRON = [("roberta-base", "nemotron"), ("embeddinggemma-300m", "nemotron")]
"""Description: Text sets built from Nemotron's transcripts, the words a live demo would actually have."""
VOICE_SETS = [("emotion2vec-plus-large", "audio"), ("hubert-large", "audio")]
"""Description: The voice sets joined in the all-three row."""


def load(name, version, split):
    """
    Description: Reads one encoder's features for one split.
    Inputs:
    - name: the encoder's short name.
    - version: "raw", "completed", "whisper" or "audio".
    - split: "train", "dev" or "test".
    Outputs: (features, labels, clip names), or None when the file is missing.
    """
    path = FEATS / name / f"{version}_{split}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return z["x"], z["labels"], list(z["clips"])


def join(a, b):
    """
    Description: Concatenates two feature sets for the clips they share, in a's order.
    Inputs:
    - a, b: (features, labels, clips) triples.
    Outputs: one (features, labels, clips) triple.
    """
    where = {c: i for i, c in enumerate(b[2])}
    keep = [i for i, c in enumerate(a[2]) if c in where]
    other = [where[a[2][i]] for i in keep]
    return np.hstack([a[0][keep], b[0][other]]), a[1][keep], [a[2][i] for i in keep]


def subset(a, clips):
    """
    Description: Keeps only the named clips of one feature set, in the order of clips.
    Inputs:
    - a: a (features, labels, clips) triple.
    - clips: the clip names to keep; every one must be in a.
    Outputs: one (features, labels, clips) triple.
    """
    where = {c: i for i, c in enumerate(a[2])}
    keep = [where[c] for c in clips]
    return a[0][keep], a[1][keep], list(clips)


def checked_join(a, b, name):
    """
    Description: Joins two feature sets with join, after checking that they agree.
    Inputs:
    - a, b: (features, labels, clips) triples of the same split.
    - name: the joined set's name, for the error message.
    Outputs: one (features, labels, clips) triple.
    Constraints: raises ValueError when the sets share no clips or give a shared clip different labels, which
    means a clip-name or split mismatch rather than a real result.
    """
    labels = {c: int(y) for c, y in zip(b[2], b[1])}
    wrong = [c for c, y in zip(a[2], a[1]) if c in labels and labels[c] != int(y)]
    if wrong:
        raise ValueError(f"{name}: {len(wrong)} shared clips have different labels, such as {wrong[0]}")
    out = join(a, b)
    if not out[2]:
        raise ValueError(f"{name}: no shared clips")
    return out


def fit_logreg(train, dev):
    """
    Description: Fits a standardised softmax (multinomial logistic) regression on the GPU, full
    batch, for each weight decay and class weighting, and keeps the one with the best dev weighted F1.
    Inputs:
    - train, dev: (features, labels, clips) triples.
    Outputs: (predict function, dev F1, chosen settings).
    """
    mean, std = train[0].mean(0), train[0].std(0) + 1e-6
    xt = torch.tensor((train[0] - mean) / std, dtype=torch.float32, device="cuda")
    yt = torch.tensor(train[1], dtype=torch.long, device="cuda")
    xd = torch.tensor((dev[0] - mean) / std, dtype=torch.float32, device="cuda")
    counts = np.bincount(train[1], minlength=len(LABELS)) + 1
    w = torch.tensor(counts.sum() / counts / len(LABELS), dtype=torch.float32, device="cuda")
    best = None
    for weighted in (True, False):
        for decay in (1e-4, 1e-3, 1e-2, 1e-1):
            torch.manual_seed(0)
            lin = torch.nn.Linear(xt.shape[1], len(LABELS)).cuda()
            opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=decay)
            loss_fn = torch.nn.CrossEntropyLoss(weight=w if weighted else None)
            for _ in range(300):
                opt.zero_grad()
                loss_fn(lin(xt), yt).backward()
                opt.step()
            with torch.no_grad():
                f1 = weighted_f1(dev[1], lin(xd).argmax(1).cpu().numpy())
            if best is None or f1 > best[1]:
                best = (lin, f1, {"weight_decay": decay, "weighted_loss": weighted})
    lin = best[0]

    def predict(x):
        with torch.no_grad():
            return lin(torch.tensor((x - mean) / std, dtype=torch.float32, device="cuda")).argmax(1).cpu().numpy()
    return predict, best[1], best[2]


def fit_mlp(train, dev, seed=0):
    """
    Description: Trains a network with one hidden layer of 512 units for up to 60 epochs, with and
    without class weights, keeping the epoch and weighting with the best dev weighted F1.
    Inputs:
    - train, dev: (features, labels, clips) triples.
    - seed: the random seed.
    Outputs: (predict function, dev F1, chosen settings).
    """
    mean, std = train[0].mean(0), train[0].std(0) + 1e-6
    xt = torch.tensor((train[0] - mean) / std, dtype=torch.float32, device="cuda")
    yt = torch.tensor(train[1], dtype=torch.long, device="cuda")
    xd = torch.tensor((dev[0] - mean) / std, dtype=torch.float32, device="cuda")
    counts = np.bincount(train[1], minlength=len(LABELS)) + 1
    best = None
    for weighted in (True, False):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(xt.shape[1], 512), torch.nn.GELU(),
                                  torch.nn.Dropout(0.3), torch.nn.Linear(512, len(LABELS))).cuda()
        w = torch.tensor(counts.sum() / counts / len(LABELS), dtype=torch.float32, device="cuda")
        loss_fn = torch.nn.CrossEntropyLoss(weight=w if weighted else None)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        for epoch in range(60):
            net.train()
            order = torch.randperm(len(xt), device="cuda")
            for i in range(0, len(xt), 128):
                idx = order[i:i + 128]
                opt.zero_grad()
                loss_fn(net(xt[idx]), yt[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                f1 = weighted_f1(dev[1], net(xd).argmax(1).cpu().numpy())
            if best is None or f1 > best[1]:
                state = {k: v.clone() for k, v in net.state_dict().items()}
                best = (state, f1, {"weighted_loss": weighted, "epoch": epoch + 1})
    final = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(xt.shape[1], 512), torch.nn.GELU(),
                                torch.nn.Dropout(0.3), torch.nn.Linear(512, len(LABELS))).cuda()
    final.load_state_dict(best[0])
    final.eval()

    def predict(x):
        with torch.no_grad():
            return final(torch.tensor((x - mean) / std, dtype=torch.float32, device="cuda")).argmax(1).cpu().numpy()
    return predict, best[1], best[2]


def evaluate(train, dev, test):
    """
    Description: Runs both heads on one feature set.
    Inputs:
    - train, dev, test: (features, labels, clips) triples.
    Outputs: a dict with each head's dev and test weighted F1 and chosen settings.
    """
    out = {"dim": int(train[0].shape[1]), "test_lines": len(test[1])}
    for head, fit in (("logreg", fit_logreg), ("mlp", fit_mlp)):
        predict, dev_f1, settings = fit(train, dev)
        out[head] = {"dev_wf1": round(dev_f1, 2), "test_wf1": round(weighted_f1(test[1], predict(test[0])), 2),
                     "settings": settings}
    return out


def evaluate_seeds(train, dev, test, seeds=SEEDS):
    """
    Description: Runs logistic regression once and the MLP once per seed on one feature set.
    Inputs:
    - train, dev, test: (features, labels, clips) triples.
    - seeds: the MLP seeds.
    Outputs: a dict with the line counts, logistic regression's dev and test weighted F1 and settings,
    and the MLP's dev and test mean and sample standard deviation over the seeds plus each seed's run.
    """
    out = {"dim": int(train[0].shape[1]), "train_lines": len(train[1]), "dev_lines": len(dev[1]),
           "test_lines": len(test[1])}
    predict, dev_f1, settings = fit_logreg(train, dev)
    out["logreg"] = {"dev_wf1": round(dev_f1, 2), "test_wf1": round(weighted_f1(test[1], predict(test[0])), 2),
                     "settings": settings}
    runs = []
    for seed in seeds:
        predict, dev_f1, settings = fit_mlp(train, dev, seed)
        runs.append({"seed": seed, "dev_wf1": round(dev_f1, 2),
                     "test_wf1": round(weighted_f1(test[1], predict(test[0])), 2), "settings": settings})
    ddof = 1 if len(runs) > 1 else 0
    out["mlp"] = {"runs": runs}
    for part in ("dev", "test"):
        f1s = [r[f"{part}_wf1"] for r in runs]
        out["mlp"][f"{part}_wf1_mean"] = round(float(np.mean(f1s)), 2)
        out["mlp"][f"{part}_wf1_std"] = round(float(np.std(f1s, ddof=ddof)), 2)
    return out


def text_voice():
    """
    Description: Scores every text encoder on each text version, every voice encoder, and each
    voice encoder joined with the text encoder that did best on dev, then writes heads.json.
    Inputs: none
    Outputs: none
    """
    text_names = sorted(p.name for p in FEATS.iterdir() if (p / "completed_train.npz").exists())
    voice_names = sorted(p.name for p in FEATS.iterdir() if (p / "audio_train.npz").exists())
    results = {"text": {}, "voice": {}, "text_plus_voice": {}}
    for name in text_names:
        for version in ("raw", "completed", "whisper"):
            try:
                sets = [load(name, version, s) for s in ("train", "dev", "test")]
                results["text"][f"{name}/{version}"] = evaluate(*sets)
            except Exception:
                results["text"][f"{name}/{version}"] = {"error": traceback.format_exc()[-800:]}
            print(name, version, json.dumps(results["text"][f"{name}/{version}"]), flush=True)
    for name in voice_names:
        try:
            results["voice"][name] = evaluate(*[load(name, "audio", s) for s in ("train", "dev", "test")])
        except Exception:
            results["voice"][name] = {"error": traceback.format_exc()[-800:]}
        print(name, json.dumps(results["voice"][name]), flush=True)
    scored = [(k, v["mlp"]["dev_wf1"]) for k, v in results["text"].items()
              if k.endswith("/completed") and "mlp" in v and k.split("/")[0] not in SUSPECT]
    if scored:
        best_text = max(scored, key=lambda kv: kv[1])[0].split("/")[0]
        results["best_text_by_dev"] = best_text
        for name in voice_names:
            try:
                sets = [join(load(best_text, "completed", s), load(name, "audio", s)) for s in ("train", "dev", "test")]
                results["text_plus_voice"][f"{best_text}+{name}"] = evaluate(*sets)
            except Exception:
                results["text_plus_voice"][f"{best_text}+{name}"] = {"error": traceback.format_exc()[-800:]}
            print(best_text, name, json.dumps(results["text_plus_voice"][f"{best_text}+{name}"]), flush=True)
    results["notes"] = ("One line at a time with no dialogue context. Test text for raw and completed is "
                        "MELD's published test text; for whisper it is Whisper's test transcript. "
                        "EmoBERTa-large's released weights have likely seen MELD's test lines (77.9 here "
                        "against 63.46 on its own card), so its scores are not comparable and it is left "
                        "out of the best-text pick.")
    write_result("heads", results)


def splits(name, version):
    """
    Description: Reads one feature set for train, dev and test.
    Inputs:
    - name: the encoder's short name.
    - version: the text version, "audio" or a vision view.
    Outputs: a list of three (features, labels, clips) triples.
    Constraints: raises FileNotFoundError naming the first missing file, and ValueError when a file holds NaN or
    infinite values, which would otherwise train to an always-neutral head that looks like a weak encoder.
    """
    out = [load(name, version, s) for s in ("train", "dev", "test")]
    for s, v in zip(("train", "dev", "test"), out):
        if v is None:
            raise FileNotFoundError(f"missing features: {name}/{version}_{s}.npz")
        bad = int((~np.isfinite(v[0])).any(1).sum())
        if bad:
            raise ValueError(f"non-finite features: {name}/{version}_{s}.npz has {bad} rows with NaN or inf")
    return out


def gain(joined, alone):
    """
    Description: Compares a joined set's scores with the text-only scores on the same clips.
    Inputs:
    - joined, alone: evaluate_seeds outputs.
    Outputs: a dict with the logistic regression test gain, the MLP mean test gain, and the spread of
    that MLP gain, the square root of the sum of both sets' squared seed standard deviations.
    """
    return {"logreg": round(joined["logreg"]["test_wf1"] - alone["logreg"]["test_wf1"], 2),
            "mlp_mean": round(joined["mlp"]["test_wf1_mean"] - alone["mlp"]["test_wf1_mean"], 2),
            "mlp_std": round(float(np.hypot(joined["mlp"]["test_wf1_std"], alone["mlp"]["test_wf1_std"])), 2)}


def vision():
    """
    Description: Scores the reproducibility row, the Nemotron text rows, every vision set alone, every
    vision set joined with each base text set, and the best text with the best vision set and two voice
    sets, then writes heads_vision.json. Every joined row also carries the text-only scores on exactly
    the same train, dev and test clips, and the gain over them.
    Inputs: none
    Outputs: none
    Constraints: never writes heads.json; a missing or failing set is recorded and skipped. The file is
    rewritten after every row, so a run killed part way keeps the rows it finished.
    """
    results = {"reproducibility": {}, "text_nemotron": {}, "vision": {}, "text_plus_vision": {}, "all_three": {}}
    alone_cache = {}
    out_path = RESULTS / "heads_vision.json"

    def run(part, key, score):
        """
        Description: Stores one row's scores, or its error, prints it, and saves the results so far.
        Inputs:
        - part: the results section.
        - key: the row name.
        - score: a function of no arguments that returns the row's scores.
        Outputs: none
        """
        try:
            results[part][key] = score()
        except FileNotFoundError as e:
            results[part][key] = {"error": str(e)}
        except Exception:
            results[part][key] = {"error": traceback.format_exc()[-800:]}
        print(part, key, json.dumps(results[part][key]), flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_path)

    def with_text(text, others):
        """
        Description: Builds the scorer for one text set joined with other sets, which also scores the
        text alone on the joined clips, cached per clip list.
        Inputs:
        - text: a (name, version) pair.
        - others: (name, version) pairs joined in order.
        Outputs: a function of no arguments that returns the row's scores.
        """
        def score():
            """
            Description: Scores the joined set and the text alone on the same clips.
            Inputs: none
            Outputs: the joined set's evaluate_seeds output with text_only_same_clips and gain added.
            """
            base = splits(*text)
            joined = base
            name = "/".join(text)
            for other in others:
                name += "+" + "/".join(other)
                joined = [checked_join(a, b, name) for a, b in zip(joined, splits(*other))]
            cache_key = (text, tuple(tuple(j[2]) for j in joined))
            if cache_key not in alone_cache:
                alone_cache[cache_key] = evaluate_seeds(*[subset(b, j[2]) for b, j in zip(base, joined)])
            out = evaluate_seeds(*joined)
            out["text_only_same_clips"] = alone_cache[cache_key]
            out["gain"] = gain(out, alone_cache[cache_key])
            return out
        return score

    last = RESULTS / "heads.json"
    if last.exists():
        old = json.loads(last.read_text(encoding="utf-8")).get("text", {}).get("/".join(TEXT_BASE), {})
        results["last_night_mlp_test_wf1"] = old.get("mlp", {}).get("test_wf1")
    run("reproducibility", "/".join(TEXT_BASE), lambda: evaluate_seeds(*splits(*TEXT_BASE)))
    for text in TEXT_NEMOTRON:
        run("text_nemotron", "/".join(text), lambda text=text: evaluate_seeds(*splits(*text)))
    vision_sets = sorted((p.name, v) for p in FEATS.iterdir() if p.is_dir() for v in VIEWS
                         if (p / f"{v}_train.npz").exists())
    print("vision sets:", ["/".join(s) for s in vision_sets], flush=True)
    for vs in vision_sets:
        run("vision", "/".join(vs), lambda vs=vs: evaluate_seeds(*splits(*vs)))
    for vs in vision_sets:
        for text in (TEXT_BASE, TEXT_NEMOTRON[0]):
            run("text_plus_vision", f"{'/'.join(text)}+{'/'.join(vs)}", with_text(text, [vs]))
    scored = [(k, v["mlp"]["dev_wf1_mean"]) for k, v in results["vision"].items() if "mlp" in v]
    if scored:
        best = max(scored, key=lambda kv: kv[1])[0]
        results["best_vision_by_dev"] = best
        others = [tuple(best.split("/", 1))] + VOICE_SETS
        key = "+".join("/".join(s) for s in [TEXT_NEMOTRON[0]] + others)
        run("all_three", key, with_text(TEXT_NEMOTRON[0], others))
    results["notes"] = (
        "One line at a time with no dialogue context. Logistic regression runs once; the MLP runs with seeds "
        f"{', '.join(str(s) for s in SEEDS)} and is reported as the mean and sample standard deviation over "
        "them; seed 0 is the seed last night's single run used. The nemotron rows train, tune and test on "
        "Nemotron's transcripts of the clips, the words a live demo would have; completed rows use MELD's "
        "published words. Vision sets miss clips whose frames failed to extract, so a joined row drops those "
        "clips from train, dev and test alike; each joined row is therefore compared with the text-only heads "
        "trained and scored on exactly the same clips, and the gain is the difference. The gain's standard "
        "deviation is the square root of the sum of both rows' squared seed standard deviations, and a gain "
        "smaller than about two of those is not a real difference. Face-view rows end with a has_face column, "
        "1.0 when a face was found in any frame and 0.0 when the row is all zeros. The best vision set for the "
        "all-three row is picked by mean dev MLP F1.")
    write_result("heads_vision", results)


def main():
    """
    Description: Runs the text and voice heads, or the vision heads when --vision is given.
    Inputs: none; reads --vision from the command line.
    Outputs: none
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true", help="score vision sets and write heads_vision.json")
    if ap.parse_args().vision:
        vision()
    else:
        text_voice()


if __name__ == "__main__":
    main()
