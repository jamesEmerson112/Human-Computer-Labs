# MELD data quality

This note records the problems found in MELD's training and dev data on 2026-10-14, what was cleaned, and how to find out whether the cleaning helps. The counts come from scripts run on the CSVs and clips in `ML_route/data/MELD.Raw/`.

## The label files

`ML_route/data/MELD.Raw/` holds MELD exactly as published, including the three label CSVs. The original `train_sent_emo.csv` sits inside the inner archive `MELD.Raw/train.tar.gz`, while the dev and test CSVs are in the outer `MELD.Raw.tar.gz`.

`ML_route/data/MELD.Completed/` holds fixed copies of the three CSVs only. The video clips stay in `MELD.Raw`. Running `python apply_fixes.py` in `ML_route/text_fixes/` rebuilds `MELD.Completed` from `MELD.Raw`, and it never writes to `MELD.Raw`.

The fixing has three steps. The first two are described here, and the third, the audio fixes, under "Transcripts from the audio" below. Until 2026-10-15 a separate `MELD.Clean` folder held the text after the first two steps only. The user dropped it that day, so the word fixes and the audio fixes are no longer measured separately.

1. A character step runs on all three splits. MELD stores Windows-1252 punctuation as Unicode control characters, such as U+0092 for an apostrophe and U+0085 for an ellipsis. The step maps each one back to real punctuation and collapses repeated spaces. It changed 3,142 train lines, 346 dev lines and 856 test lines.
2. A word step runs on train and dev only. It applies the 195 reviewed fixes in `ML_route/text_fixes/fixes.csv`, of which 177 are in train and 18 in dev. Eight Opus agents proposed 196 fixes. Two were dropped as unconfirmed guesses, one was corrected, and one missed fix (dia1036_utt0) was added by hand. The fixes cover clear written mistakes such as you're for your, misspellings, and `?` in place of an apostrophe. Spoken style, stutters and cut-off lines were left alone.

The test split keeps its published words, so test scores stay comparable with other MELD results.

## Transcripts from the audio

On 2026-10-14 every clip was transcribed from its own audio, and agents used those transcripts to complete MELD's cut-off lines. This added the audio fixes to `MELD.Completed` and gave one more version of the label files.

`ML_route/data/MELD.Whisper/` holds Whisper's transcript of every clip. `{split}_whisper.jsonl` has one JSON line per clip with the text, the segments with their times and confidence, and the words with their times. `{split}_sent_emo.csv` is MELD's label file with the Utterance column replaced by Whisper's text. `ML_route/text_fixes/transcribe.py` builds both and skips clips already done, so a rerun resumes.

The third step of `MELD.Completed` applies the reviewed audio fixes in `ML_route/text_fixes/audio_fixes.csv` on top of the first two steps. The audio fixes cover train and dev only, so the test split gets the character step alone. Every audio fix changes only the Utterance column.

### How the transcripts were made

The model is Whisper large-v3-turbo, run with faster-whisper 1.2.1 on the local GTX 1070 Ti with int8 weights. The settings are English, beam size 5, word timestamps, no voice-activity filter, and no conditioning on the previous clip. ffmpeg decodes each clip and mixes all channels down to mono at 16 kHz, which keeps the centre channel of the 6-channel clips.

The training split took 105 minutes for 8.6 hours of audio, about 4.9 times faster than real time. Dev and test took 44 minutes for 3.3 hours, about 4.4 times faster than real time. Each call costs about 0.57 seconds whatever the clip length, because Whisper always encodes a 30-second window, so the per-call cost dominates on MELD's short clips. Batched mode was about 20% faster on a 50-clip pilot but gave worse text, including an empty result for dia1011_utt4 and more invented "Thank you." lines. Two processes sharing the GPU ran slower than one. Turning off word timestamps saved 3%, and beam size 1 saved 10%.

Two clips have no audio. The file for train dia125_utt3 is broken ("moov atom not found"), and dev dia110_utt7 has no clip file. Whisper returned no text for 49 train clips, 6 dev clips and 15 test clips, which are mostly very short or quiet. In `MELD.Whisper/{split}_sent_emo.csv` these clips, and the two without audio, have an empty utterance, so a model trained on that version must handle blank lines.

### How Whisper's text compares with MELD's

After lowercasing and removing punctuation, Whisper's text matches MELD's cleaned text from the first two steps exactly for 34.8% of train lines, 33.4% of dev lines and 33.1% of test lines. Most other differences are small. Whisper writes "going to" for "gonna", "okay" for "OK", and numbers as words, and it fixes stutters. Whisper's text is at least three words longer than MELD's for 13.6% of train lines, 11.5% of dev lines and 13.2% of test lines. Some of these are lines MELD cut off. Others are clips whose audio also holds a neighbour's line, such as dia2_utt6, where Whisper adds "Not me." from Monica's next line "Got me."

### How the audio fixes were chosen

Eight Opus agents compared MELD's cleaned text from the first two steps and Whisper line by line, seven for train and one for dev. Each saw the speaker, the clip timing and overlaps, Whisper's segments with their confidence, and the neighbouring lines. They could propose three kinds of fix.

1. `completed` finishes a cut-off line with the rest of the same speaker's sentence from Whisper.
2. `missing_words` puts back a phrase that MELD skipped inside the speaker's own turn.
3. `misheard_in_meld` replaces a wrong word when Whisper and the conversation agree.

The rules kept MELD's text whenever Whisper was doubtful, never added words from a neighbouring line, and kept MELD's spoken style such as "gonna" and digits. When an agent was unsure, it made no change.

The agents proposed 344 fixes. Three Opus verifier agents reviewed every one against the clip's segments and neighbouring lines and dropped three, and the parent session dropped one more.

- dia440_utt1 wrote "Fresca", but Whisper heard "fresco".
- dia671_utt10 took "that mean?" from the next clip's transcript.
- dia833_utt1 added "Ross", which more likely starts Monica's next line.
- dia308_utt1 turned Whisper's "I'd love my office" into "I love my office", which changes the meaning beyond Whisper's words.

The 340 accepted fixes are 279 `completed`, 52 `missing_words` and 9 `misheard_in_meld`. 318 are in train and 22 in dev, which is 3.2% and 2.0% of lines. A few accepted fixes bend the style rules slightly. dia475_utt5 and dia540_utt3 lowercase a word that started a sentence in MELD, dia0_utt13 adds a period inside MELD's text, and dia89_utt15 writes "gonna" where Whisper wrote "going to", matching MELD's style earlier in the line. Examples are dia700_utt2, which now ends "We register, and you get to keep all the presents.", and dia6_utt12, which becomes "Thanks a lot for coming buddy." The grammar pass had left dia6_utt12 alone because the fix was a guess, and the audio settles it.

## Problems that cleaning does not fix

- Overlapping clips. In 650 of 8,951 back-to-back pairs of train lines (7.3%), the next line starts before the previous one ends, and 309 pairs end at the same moment. For example, dia2_utt6 also contains Monica's next line, "Got me." The text is unaffected, but the audio of the earlier clip holds two speakers.
- Missing lines. In 478 of 8,951 back-to-back pairs (5.3%), more than 5 seconds pass between the lines. Dialogue 700 has gaps of 4.1, 8.8 and 3.5 seconds where lines appear to be missing.
- Cut-off transcripts. 508 of 9,989 train lines (5.1%) end without punctuation, and many of them stop mid-sentence. dia700_utt2 has 25 words in a 12.6-second clip that holds more speech than that. After the first two steps, 520 train lines end without a period, question mark, exclamation mark, ellipsis, quote or bracket, and `MELD.Completed` finishes 255 of them from the audio, as described above. The agents left the rest unchanged.
- Label errors. Some speaker names are wrong, such as "A week?" in dia1036_utt10, which is labeled as Stanley but is almost certainly Joey.
- Surround audio. Some clips have 6-channel audio with the dialogue in the center channel. dia700_utt6 has the center channel about 15 dB louder than front left and right. Audio extraction must mix all channels down to mono, as ffmpeg's `-ac 1` does, or the speech is mostly lost.

## Does cleaning help?

It is unmeasured. Fixing the broken characters likely helps, because without it a text model sees two spellings of words like "don't". The 195 word fixes touch under 2% of lines, and pretrained text encoders cope well with typos, so their effect is probably small. Matching the training text to the text the live system sees is likely the bigger lever, because the demo reads words from Whisper, not from MELD's CSV.

To measure it, train the same model on three versions of the text and compare dev-set weighted F1: `MELD.Raw`, `MELD.Completed` and `MELD.Whisper`. Completed against Raw measures the character, word and audio fixes together. Whisper against the others tests whether matching the text the live system sees matters more than tidy text. Score every run on the same dev text, or on each version's own dev text, and say which. Report the result either way, because the brief asks for evidence behind decisions.
