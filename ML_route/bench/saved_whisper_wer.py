"""Scores the saved MELD.Whisper test transcripts the same way asr_race.py scores live runs, as a cross-check.

Contents:
- main: prints the word error rate of MELD.Whisper's test text against MELD.Completed's test text
"""

import csv
import json
import os
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

DATA = Path(os.environ.get("MELD_DATA", Path(__file__).resolve().parent.parent / "data"))


def main():
    """
    Description: Prints the word error rate of the saved Whisper test transcripts.
    Inputs: none
    Outputs: none
    """
    norm = EnglishTextNormalizer()
    with (DATA / "MELD.Completed" / "test_sent_emo.csv").open(encoding="utf-8", newline="") as f:
        refs = {f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}": r["Utterance"] for r in csv.DictReader(f)}
    hyps = {}
    with (DATA / "MELD.Whisper" / "test_whisper.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            hyps[rec["clip"]] = rec.get("text", "")
    pairs = [(norm(refs[c]), norm(hyps[c])) for c in refs if c in hyps and norm(refs[c]).strip()]
    out = jiwer.process_words([p[0] for p in pairs], [p[1] for p in pairs])
    print(f"saved MELD.Whisper test (large-v3-turbo int8, batched, word timestamps): {len(pairs)} clips, "
          f"WER {100 * out.wer:.2f}% (S {out.substitutions}, D {out.deletions}, I {out.insertions})")


if __name__ == "__main__":
    main()
