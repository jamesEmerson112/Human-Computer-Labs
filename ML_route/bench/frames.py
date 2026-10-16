"""Saves evenly spaced frames of every MELD clip as JPEG files, for the vision benchmarks.

APIs:
- FRAMES, N_FRAMES, FRAME_WIDTH: where frames go, how many per clip and how wide
- frame_paths: the expected frame files of one clip
- extract_frames: saves the frames of a list of clips and returns the ones that worked
- distinct_frames: counts a clip's byte-distinct frame files, to find slots filled with copies
- main: extracts every split and writes frames_failed.json and frames_repeated.json
"""

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import RESULTS, read_split

FRAMES = Path(os.environ.get("MELD_FRAMES", "/root/frames"))
"""Description: The folder that holds one subfolder of frames per split and clip."""
N_FRAMES = 8
"""Description: Frames taken from each clip, at the middle of eight equal slices of its length."""
FRAME_WIDTH = 640
"""Description: Frame width in pixels; the height keeps the clip's aspect ratio."""


def frame_paths(split, clip):
    """
    Description: Lists the frame files one clip should have.
    Inputs:
    - split: "train", "dev" or "test".
    - clip: the clip name, such as dia0_utt0.
    Outputs: a list of N_FRAMES paths.
    """
    return [FRAMES / split / clip / f"{k}.jpg" for k in range(N_FRAMES)]


def extract_frames(rows, split, workers=8):
    """
    Description: Saves N_FRAMES frames spaced evenly through each clip, FRAME_WIDTH pixels wide.
    Clips whose frames already exist are skipped. A seek that writes no frame is retried earlier
    in the clip, and a slot that still has none gets a copy of the nearest frame that worked.
    Inputs:
    - rows: line dicts from read_split.
    - split: the split the rows come from.
    - workers: parallel ffmpeg jobs.
    Outputs: (a dict from clip name to its frame paths for clips whose frames all exist,
    a dict from clip name to the reason it failed).
    Constraints: workers should stay at or below the pod's CPU count.
    """
    def one(r):
        paths = frame_paths(split, r["clip"])
        if all(p.exists() for p in paths):
            return r["clip"], paths, None
        if not r["path"].exists():
            return r["clip"], None, "clip file missing"
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(r["path"])], capture_output=True, text=True)
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            return r["clip"], None, f"no duration: {probe.stderr.strip()[:200]}"
        errors = []
        for k, p in enumerate(paths):
            if p.exists():
                continue
            t = (k + 0.5) / N_FRAMES * duration
            for seek in dict.fromkeys((t, t - 0.1, t - 0.3, 0.0)):
                run = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{max(seek, 0.0):.3f}",
                                      "-i", str(r["path"]), "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2",
                                      str(p)], capture_output=True, text=True)
                if p.exists():
                    break
            else:
                errors.append(f"frame {k}: {run.stderr.strip()[:160] or 'no frame written'}")
        good = [p for p in paths if p.exists()]
        if not good:
            return r["clip"], None, f"duration {duration:.2f} s; " + "; ".join(errors[:3])
        for p in paths:
            if not p.exists():
                # Short clips have fewer decodable frames than slots, so the nearest good frame fills the gap.
                shutil.copyfile(min(good, key=lambda g: abs(int(g.stem) - int(p.stem))), p)
        return r["clip"], paths, None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(one, rows))
    return {c: p for c, p, _ in done if p}, {c: e for c, _, e in done if e}


def distinct_frames(paths):
    """
    Description: Counts the different frame files of one clip, so slots filled with a copy of a neighbour show up.
    Inputs:
    - paths: the clip's frame files, all of which exist.
    Outputs: the number of byte-distinct files.
    Constraints: two real frames of a still shot can also encode to the same bytes, so a clip counted as having
    repeated frames may have no copied slot.
    """
    return len({p.read_bytes() for p in paths})


def main():
    """
    Description: Extracts frames for train, dev and test, writes the failed clips with their reasons to
    frames_failed.json, and writes the clips whose slots were partly filled with copies to frames_repeated.json
    as a dict from split to {clip: distinct frames}.
    Inputs: none
    Outputs: none
    """
    failed, repeated = {}, {}
    for split in ("train", "dev", "test"):
        rows = read_split(split)
        ok, bad = extract_frames(rows, split)
        failed[split] = bad
        counts = {c: distinct_frames(p) for c, p in ok.items()}
        repeated[split] = {c: n for c, n in counts.items() if n < N_FRAMES}
        print(f"{split}: {len(ok)} of {len(rows)} clips have frames, {len(bad)} failed, "
              f"{len(repeated[split])} with repeated frames", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "frames_failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
    (RESULTS / "frames_repeated.json").write_text(json.dumps(repeated, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
