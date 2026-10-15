"""Builds a local HTML page that plays MELD dialogues clip by clip.

Contents:
- load_split: reads one split's lines from the three label versions
- main: writes ML_route/data/meld-player.html next to MELD.Raw
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TEMPLATE = HERE / "player.html"
OUTPUT = DATA / "meld-player.html"
SPLITS = ("train", "dev", "test")


def read_rows(path):
    """
    Description: Reads a MELD label file into a dict keyed by clip.
    Inputs:
    - path: the label CSV.
    Outputs: {(dialogue, utterance): row} with each row as a dict.
    """
    with path.open(encoding="utf-8", newline="") as f:
        return {
            (int(r["Dialogue_ID"]), int(r["Utterance_ID"])): r
            for r in csv.DictReader(f)
        }


def read_word_times(path):
    """
    Description: Reads Whisper's word timings for every clip of one split.
    Inputs:
    - path: the split's Whisper JSONL file.
    Outputs: {clip: [[start in hundredths of a second, word], ...]}.
    """
    times = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            record = json.loads(raw)
            words = [[round(w["start"] * 100), w["word"].strip()]
                     for w in record.get("words") or []]
            if words:
                times[record["clip"]] = words
    return times


def load_split(split):
    """
    Description: Reads one split's lines from MELD.Clean, MELD.Completed and
    MELD.Whisper.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: {dialogue: [[utterance, speaker, emotion, clean, completed,
    whisper, words], ...]}, where completed is 0 when it equals clean,
    whisper is None when the clip has no transcript, and words holds
    Whisper's word timings or 0.
    """
    name = f"{split}_sent_emo.csv"
    clean = read_rows(DATA / "MELD.Clean" / name)
    completed = read_rows(DATA / "MELD.Completed" / name)
    whisper = read_rows(DATA / "MELD.Whisper" / name)
    word_times = read_word_times(DATA / "MELD.Whisper" / f"{split}_whisper.jsonl")
    dialogues = {}
    for (dia, utt), row in sorted(clean.items()):
        text = row["Utterance"]
        done = completed[(dia, utt)]["Utterance"]
        heard = whisper.get((dia, utt), {}).get("Utterance") or None
        dialogues.setdefault(dia, []).append([
            utt, row["Speaker"], row["Emotion"], text,
            0 if done == text else done, heard,
            word_times.get(f"dia{dia}_utt{utt}", 0),
        ])
    return dialogues


def main():
    """
    Description: Writes the player page with every split's lines embedded.
    Inputs: none
    Outputs: none
    """
    data = {split: load_split(split) for split in SPLITS}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__DATA__", payload.replace("</", "<\\/"))
    OUTPUT.write_text(page, encoding="utf-8")
    lines = sum(len(v) for d in data.values() for v in d.values())
    print(f"wrote {OUTPUT} with {lines} lines")


if __name__ == "__main__":
    main()
