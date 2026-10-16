"""Decodes every labelled MELD clip to 16 kHz mono once, so each benchmark reads the same audio.

Contents:
- decode: decodes one clip, returning None when it fails
- main: decodes the chosen splits in parallel and writes AUDIO/<split>.npz
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import AUDIO, load_audio, read_split


def decode(item):
    """
    Description: Decodes one clip.
    Inputs:
    - item: a (clip name, mp4 path) pair.
    Outputs: (clip, samples as float16) or (clip, error text).
    """
    clip, path = item
    try:
        return clip, load_audio(path).astype(np.float16)
    except RuntimeError as err:
        return clip, str(err)[:200]


def main():
    """
    Description: Decodes the chosen splits and writes one npz per split holding the joined samples,
    each clip's offset and the clip names, plus a JSON list of clips that failed.
    Inputs: none; reads --splits from the command line.
    Outputs: none
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test,dev,train")
    args = ap.parse_args()
    AUDIO.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        rows = read_split(split)
        with ProcessPoolExecutor(max_workers=48) as pool:
            done = list(pool.map(decode, [(r["clip"], r["path"]) for r in rows], chunksize=16))
        good = [(c, a) for c, a in done if not isinstance(a, str)]
        bad = {c: a for c, a in done if isinstance(a, str)}
        offsets = np.cumsum([0] + [len(a) for _, a in good])
        np.savez(AUDIO / f"{split}.npz", samples=np.concatenate([a for _, a in good]),
                 offsets=offsets, clips=np.array([c for c, _ in good]))
        (AUDIO / f"{split}_failed.json").write_text(json.dumps(bad, indent=1))
        print(f"{split}: {len(good)} clips decoded, {len(bad)} failed, "
              f"{offsets[-1] / 16000 / 3600:.2f} h of audio", flush=True)


if __name__ == "__main__":
    main()
