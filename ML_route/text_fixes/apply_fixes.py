"""Builds MELD.Completed, the fixed copies of the MELD label files, from the originals in MELD.Raw.

Every split gets the character step. Train and dev also get the reviewed word fixes, then the
reviewed fixes taken from the audio.

APIs:
- CHAR_MAP: broken or typographic characters and their replacements
- normalize: applies CHAR_MAP and collapses repeated spaces in one utterance
- load_fixes: reads the fix log
- clean_split: cleans one split's rows in memory
- complete_split: applies the audio fixes to the cleaned rows and writes one label file
- main: builds all three splits and prints a summary
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MELD.Raw"
COMPLETED = ROOT / "data" / "MELD.Completed"
FIXES = Path(__file__).resolve().parent / "fixes.csv"
AUDIO_FIXES = Path(__file__).resolve().parent / "audio_fixes.csv"
SPLITS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
FIXED_SPLITS = {"train", "dev"}

CHAR_MAP = {
    "\x92": "'", "\x91": "'", "’": "'", "‘": "'",
    "\x93": '"', "\x94": '"', "“": '"', "”": '"',
    "\x85": "…", "\x97": "—", "\x96": "–", "\xa0": " ",
}
"""Description: MELD stores Windows-1252 punctuation as Unicode control characters. Each maps to
the character it was meant to be, with quotes and apostrophes made plain ASCII."""


def normalize(text):
    """
    Description: Replaces the characters in CHAR_MAP and collapses runs of spaces to one.
    Inputs:
    - text: one utterance from a MELD label file.
    Outputs: the normalized utterance, with leading and trailing spaces removed.
    Constraints: running it twice gives the same result as running it once.
    """
    for bad, good in CHAR_MAP.items():
        text = text.replace(bad, good)
    return re.sub(r" {2,}", " ", text).strip()


def load_fixes(path):
    """
    Description: Reads the fix log into a lookup by split and clip.
    Inputs:
    - path: the fix log, a CSV with split, clip, old, new and reason columns.
    Outputs: a dict from (split, clip) to (old, new).
    Constraints: exits with an error when a split and clip pair appears twice.
    """
    fixes = {}
    if not path.exists():
        return fixes
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["split"], row["clip"])
            if key in fixes:
                sys.exit(f"duplicate fix for {key}")
            fixes[key] = (row["old"], row["new"])
    return fixes


def clean_split(split, fixes):
    """
    Description: Normalizes every utterance in one split and applies that split's logged word
    fixes, without writing anything. MELD.Raw is only read.
    Inputs:
    - split: "train", "dev" or "test".
    - fixes: the lookup from load_fixes.
    Outputs: a (header, rows, characters changed, fixes applied) tuple, where rows are the
    cleaned data rows.
    Constraints: exits with an error when a fix's old text does not match the normalized
    utterance. Test rows only get character normalization.
    """
    path = RAW / SPLITS[split]
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    changed = applied = 0
    for row in body:
        text = normalize(row[1])
        changed += text != row[1]
        key = (split, f"dia{row[5]}_utt{row[6]}")
        if split in FIXED_SPLITS and key in fixes:
            old, new = fixes[key]
            if text != old:
                sys.exit(f"{key}: log says {old!r} but the text is {text!r}")
            text = new
            applied += 1
        row[1] = text
    return header, body, changed, applied


def complete_split(split, header, body, audio_fixes):
    """
    Description: Applies the audio fixes to a split's cleaned rows and writes the result to
    MELD.Completed.
    Inputs:
    - split: "train", "dev" or "test".
    - header, body: the cleaned header and rows from clean_split.
    - audio_fixes: the lookup from load_fixes for audio_fixes.csv.
    Outputs: the number of audio fixes applied.
    Constraints: exits with an error when an audio fix's old text does not match the cleaned
    utterance. Test rows are copied unchanged.
    """
    applied = 0
    rows = [list(r) for r in body]
    for row in rows:
        key = (split, f"dia{row[5]}_utt{row[6]}")
        if split in FIXED_SPLITS and key in audio_fixes:
            old, new = audio_fixes[key]
            if row[1] != old:
                sys.exit(f"{key}: audio log says {old!r} but the cleaned text is {row[1]!r}")
            row[1] = new
            applied += 1
    COMPLETED.mkdir(parents=True, exist_ok=True)
    with (COMPLETED / SPLITS[split]).open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    return applied


def main():
    """
    Description: Builds MELD.Completed for all three splits and prints how many rows each step
    changed.
    Inputs: none
    Outputs: none
    Constraints: exits with an error when the log names a clip that does not exist.
    """
    fixes = load_fixes(FIXES)
    audio_fixes = load_fixes(AUDIO_FIXES)
    found = audio_found = 0
    for split in SPLITS:
        header, body, changed, applied = clean_split(split, fixes)
        audio_applied = complete_split(split, header, body, audio_fixes)
        found += applied
        audio_found += audio_applied
        print(f"{split}: {len(body)} rows, {changed} with characters normalized, {applied} word fixes, "
              f"{audio_applied} audio fixes")
    if found != len(fixes):
        sys.exit(f"{len(fixes) - found} logged fixes matched no row")
    if audio_found != len(audio_fixes):
        sys.exit(f"{len(audio_fixes) - audio_found} audio fixes matched no row")


if __name__ == "__main__":
    main()
