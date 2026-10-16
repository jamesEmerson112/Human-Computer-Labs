# MELD emotion prototype: words and voice in, emotion and a reply out

This repository is a take-home submission for the ML route described in `ML_route/README.md`, on the text plus audio track. Each MELD clip goes through Whisper, which turns its sound into words. Three frozen encoders then turn the words and the sound into features. A small head trained in Rust picks one of MELD's seven emotions, and a small language model writes a short reply for that emotion.

The whole inference path holds about 2.8 billion learned parameters, under the brief's limit of 6 billion.

| Part | Model | Parameters |
|---|---|---:|
| Speech to text | Whisper large-v3-turbo (int8) | 808.9M |
| Text encoder | EmoBERTa-large | 355.4M |
| Text encoder | EmbeddingGemma-300m | 302.9M |
| Voice encoder | HuBERT-large | 315.4M |
| Emotion head | trained here, in Rust | about 1M |
| Reply writer | LFM2.5-1.2B-Instruct | 1.17B |

The trained head scores 61.96 weighted F1 on MELD's test split (62.5 ± 0.8 across three seeds). EmoBERTa-large was released already trained on MELD, and its weights have likely seen MELD's test lines, so this score may be higher than the head would reach on new speech.

The SWE route in `SWE_route/` is a separate challenge and is not part of this submission.

## What you need

**Hardware:**
- A computer with an NVIDIA GPU.
  - The live player was tested on a GTX 1070 Ti with 8 GB.
  - Feature extraction ran on a Runpod A40 with 48 GB. An 8 GB card also works, more slowly.
- About 15 GB of free disk space for MELD and the features.

**Tools:**
- Python 3.12, with [uv](https://docs.astral.sh/uv/) to create the environments.
- Rust 1.92 or later, with `cargo`.
- `ffmpeg` on the `PATH`, used to decode the clips' audio.

## Steps to reproduce

Every command runs from the repository root unless a step says otherwise. On Windows, use `.venv/Scripts/python.exe`; on Linux and macOS, use `.venv/bin/python`.

### 1. Download MELD

Download `MELD.Raw.tar.gz` from https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz.

1. Unpack it into `ML_route/data/`, which gives `ML_route/data/MELD.Raw/`.
2. Inside that folder, unpack `train.tar.gz`, `dev.tar.gz` and `test.tar.gz`. That gives:
   - `train_splits/` (9,989 clips);
   - `dev_splits_complete/` (1,112 clips);
   - `output_repeated_splits_test/` (2,747 clips).

The test folder also holds macOS `._*` stub files; the scripts skip them.

### 2. Create the two Python environments

The speech-to-text environment runs Whisper through faster-whisper:

```
uv venv --python 3.12 ML_route/text_fixes/.venv
uv pip install --python ML_route/text_fixes/.venv faster-whisper==1.2.1 numpy av nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The model environment runs the encoders, the emotion server and the reply writer:

```
uv venv --python 3.12 ML_route/player/.venv
uv pip install --python ML_route/player/.venv torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python ML_route/player/.venv transformers==5.19.0 numpy safetensors sentencepiece av
```

PyTorch's cu126 build is the newest that still runs on older Pascal cards such as the GTX 1070 Ti. On a newer GPU, any recent CUDA build works.

### 3. Build the cleaned labels (MELD.Completed)

```
ML_route/text_fixes/.venv/Scripts/python.exe ML_route/text_fixes/apply_fixes.py
```

This writes `ML_route/data/MELD.Completed/` from `MELD.Raw`, and never changes `MELD.Raw`.

- **Every split:** it repairs MELD's broken punctuation characters.
- **Train and dev only:** it also applies the reviewed word fixes in `ML_route/text_fixes/fixes.csv` and the fixes taken from the audio in `ML_route/text_fixes/audio_fixes.csv`.
- **Test** keeps MELD's published words, so the test scores stay comparable with other MELD results.

`docs/research/meld-data-quality.md` explains how the fixes were made.

### 4. Transcribe every clip with Whisper (MELD.Whisper)

```
ML_route/text_fixes/.venv/Scripts/python.exe ML_route/text_fixes/transcribe.py
```

This writes `ML_route/data/MELD.Whisper/`: Whisper's transcript of every clip, as JSONL plus a copy of each label file with Whisper's words. The head reads Whisper's words rather than MELD's, because the live demo only ever has Whisper's words. Expect this to take a few hours on an 8 GB GPU. A rerun skips clips already done.

**Shortcut:** download `MELD.Whisper.zip` from the repository's [GitHub Release](https://github.com/jamesEmerson112/Human-Computer-Labs/releases/tag/data-v1) and unpack it into `ML_route/data/`.

### 5. Extract the encoder features

```
export MELD_DATA=ML_route/data MELD_FEATS=ML_route/data/feats MELD_AUDIO=ML_route/data/audio MELD_RESULTS=ML_route/data/results
cd ML_route/bench
../player/.venv/Scripts/python.exe audio_cache.py
../player/.venv/Scripts/python.exe features.py --text embeddinggemma-300m,emoberta-large --voice hubert-large --versions whisper
cd ../..
```

- `audio_cache.py` decodes every clip once to 16 kHz mono.
- `features.py` writes one feature file per encoder and split into `ML_route/data/feats/`:
  - EmbeddingGemma: the mean of its last layer over the line's tokens, 768 numbers.
  - EmoBERTa-large: the same pooling, 1,024 numbers.
  - HuBERT-large: its last layer and the mean of all its layers, each averaged over time, 2,048 numbers.
- **EmbeddingGemma is a gated model.** Accept its terms on Hugging Face and log in with `hf auth login` first.
- **Run time:** on an A40 this step takes under an hour. On an 8 GB card it takes a few hours.

**Shortcut:** download `feats.zip` from the same [GitHub Release](https://github.com/jamesEmerson112/Human-Computer-Labs/releases/tag/data-v1) and unpack it into `ML_route/data/`.

### 6. Join the features

```
ML_route/player/.venv/Scripts/python.exe ML_route/hello_world/prepare_features.py
```

This joins the three encoders' features for each clip into one row of 3,840 numbers, and writes them to `ML_route/data/fusion/`. One train clip and one dev clip have no audio and are left out.

### 7. Train the emotion head in Rust

```
cd ML_route/hello_world
cargo run --release
```

- **Settings:** training reads `hyperparameters.toml`. Each setting is commented, including the five extra training techniques that were measured and the scores behind keeping or dropping them.
- **What it does:** it trains for 2,000 steps, scores the dev split every 100 steps, keeps the head with the best dev score, then scores that head on test.
- **Output:** it saves the head to `ML_route/heads/meld-head.json`.
- **Time:** it takes about 2.5 minutes on a CPU. The `--release` flag matters, because a debug build is many times slower.
- **Tests:** `cargo test` checks the hand-written gradients against numerical ones.

**Shortcut:** the trained head is committed at `ML_route/heads/meld-head.json`, so steps 4 to 7 can be skipped to just run the player. Training writes to the same path.

### 8. Run the live player

Run each server in its own terminal, from the `ML_route` folder:

```
text_fixes/.venv/Scripts/python.exe player/build_player.py
text_fixes/.venv/Scripts/python.exe player/serve.py          # page, Whisper speech to text (port 8765)
player/.venv/Scripts/python.exe player/serve_reply.py         # LFM2.5 reply writer (port 8768)
player/.venv/Scripts/python.exe player/serve_emotion.py       # encoders and emotion head (port 8769)
```

Then:

1. Open http://127.0.0.1:8765/meld-player.html.
2. Tick **Live speech-to-text**, **Live emotion** and **Live reply**.
3. Play a test dialogue, for example by opening `#test-149`.

For each clip, Whisper writes the words, the head picks the emotion, and LFM2.5 writes a reply to it. Each step shows its time. The Score tab compares the head's emotions with MELD's labels.

- **Use test dialogues only.** The head trained on the train dialogues, so it would look too good there. The page opens on test by default.
- **Check the head:** `player/.venv/Scripts/python.exe player/serve_emotion.py --check` scores the saved head on the test features. It should match the Rust run's test score.

## Results and evidence

- `ML_route/bench/results/README.md` holds the A40 benchmarks: speech-to-text models, frozen encoders with trained heads, reply-writer speed, and vision-language models.
- `ML_route/hello_world/hyperparameters.toml` records the head's training settings and the measured effect of each technique.
- `docs/explainers/meld-head-training.html` explains the approach, with an interactive version at https://claude.ai/artifact/D7XqirEfGAJDvMPBq1xemQ.
