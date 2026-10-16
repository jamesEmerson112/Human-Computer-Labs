"""Joins the frozen-encoder features of each MELD clip into one row per clip, for the Rust head to train on.

Contents:
- FEATS, OUT: where the encoder features are read from and the joined files are written to
- SETS: the feature sets joined, in column order, with their file version
- load_set: reads one encoder's features for one split
- join_split: joins every set for one split by clip and checks their labels agree
- main: writes x_<split>.npy, y_<split>.npy and clips_<split>.txt for train, dev and test
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEATS = HERE.parent / "data" / "feats"
"""Description: The encoder features copied from the pod, as FEATS/<encoder>/<version>_<split>.npz."""
OUT = HERE.parent / "data" / "fusion"
"""Description: The folder the joined files are written to, which the Rust head reads."""
SETS = [
    ("embeddinggemma-300m", "whisper"),
    ("emoberta-large", "whisper"),
    ("hubert-large", "audio"),
]
"""Description: The feature sets joined side by side, in this column order: 768 + 1,024 + 2,048 = 3,840."""


def load_set(encoder, version, split):
    """
    Description: Reads one encoder's features for one split.
    Inputs:
    - encoder: the encoder's folder name.
    - version: the text version or "audio".
    - split: "train", "dev" or "test".
    Outputs: a dict from clip name to (feature row, label).
    """
    z = np.load(FEATS / encoder / f"{version}_{split}.npz")
    return {str(c): (x, int(y)) for c, x, y in zip(z["clips"], z["x"], z["labels"])}


def join_split(split):
    """
    Description: Joins every feature set for one split by clip, keeping the first set's clip order.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: (feature matrix as float32, labels as int64, clip names, clips dropped for a missing set).
    Constraints: raises ValueError when two sets give one clip different labels.
    """
    sets = [load_set(e, v, split) for e, v in SETS]
    order = list(sets[0])
    shared = [c for c in order if all(c in s for s in sets)]
    dropped = [c for c in order if c not in shared]

    rows, labels = [], []
    for clip in shared:
        clip_labels = {s[clip][1] for s in sets}
        if len(clip_labels) != 1:
            raise ValueError(f"{split} clip {clip} has different labels across sets: {clip_labels}")
        rows.append(np.concatenate([s[clip][0] for s in sets]))
        labels.append(clip_labels.pop())

    x = np.stack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y, shared, dropped


def main():
    """
    Description: Writes the joined files for train, dev and test and prints what each holds.
    Inputs: none
    Outputs: none
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        x, y, clips, dropped = join_split(split)
        np.save(OUT / f"x_{split}.npy", x)
        np.save(OUT / f"y_{split}.npy", y)
        (OUT / f"clips_{split}.txt").write_text("\n".join(clips) + "\n", encoding="utf-8")
        print(f"{split}: {x.shape[0]} clips x {x.shape[1]} features, {len(dropped)} dropped "
              f"for a missing set{': ' + ', '.join(dropped[:5]) if dropped else ''}")


if __name__ == "__main__":
    main()
