#!/usr/bin/env bash
# Prepares a freshly started pod for run_vision.sh: builds the hf environment in /root/envs, and in parallel installs
# ffmpeg and unpacks the MELD clips to /root/meld, since a stopped pod loses its container disk.
# Progress and the ENV, FFMPEG and CLIPS markers (each ending in _OK or _FAILED) go to
# /workspace/logs/setup_vision_status.txt, which run_vision.sh waits on. Rerunning it skips whatever is in place.
set -u
LOG=/workspace/logs
STATUS=$LOG/setup_vision_status.txt
ENV=/root/envs/hf
ARCHIVE=/workspace/data/MELD.Raw.tar.gz
MELD=/root/meld
LINK=/workspace/data/MELD.Raw
export UV_CACHE_DIR=/workspace/uvcache UV_LINK_MODE=copy
mkdir -p "$LOG" /root/envs "$MELD"
: > "$STATUS"

# Appends one timestamped line to the status file.
note() {
  echo "$(date -u +%FT%TZ) $*" >> "$STATUS"
}

# Checks that the hf environment imports everything the vision run needs and sees the GPU.
env_works() {
  "$ENV/bin/python" - <<'EOF' >> "$LOG/setup_env.log" 2>&1
import accelerate, av, cv2, huggingface_hub, jiwer, librosa, numpy, pandas, PIL, sentencepiece, sklearn, soundfile
import timm, torch, torchaudio, torchvision, transformers, whisper_normalizer
assert torch.cuda.is_available(), "CUDA is not available"
assert hasattr(cv2, "FaceDetectorYN"), "OpenCV has no FaceDetectorYN"
print("torch", torch.__version__, "transformers", transformers.__version__, "cv2", cv2.__version__,
      "gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
EOF
}

# Builds /root/envs/hf on Python 3.12 from the uv cache: torch 2.8 from the cu128 index first, then the rest with
# torch pinned so no later package can replace it.
build_env() {
  if [ -x "$ENV/bin/python" ] && env_works; then
    note "ENV_OK (already built)"
    return
  fi
  if ! command -v uv > /dev/null; then
    python3 -m pip install -q uv || python3 -m pip install -q --break-system-packages uv || \
      curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  rm -rf "$ENV"
  note "env: building $ENV"
  if uv venv --python 3.12 "$ENV" \
    && uv pip install --python "$ENV/bin/python" "torch==2.8.*" torchaudio torchvision \
         --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip freeze --python "$ENV/bin/python" | grep -E '^torch(audio|vision)?==' | sed 's/+.*//' \
         > /root/envs/torch_pins.txt \
    && uv pip install --python "$ENV/bin/python" -c /root/envs/torch_pins.txt "transformers==5.19.*" accelerate \
         scikit-learn soundfile pandas av pillow timm sentencepiece opencv-python-headless librosa huggingface_hub \
         jiwer whisper-normalizer \
    && env_works; then
    note "ENV_OK $(tail -n 1 "$LOG/setup_env.log")"
  else
    note "ENV_FAILED (see $LOG/setup_env.log)"
  fi
}

# Installs ffmpeg and ffprobe with apt when they are missing, since frames.py needs them and apt packages live on
# the container disk.
install_ffmpeg() {
  if command -v ffmpeg > /dev/null && command -v ffprobe > /dev/null; then
    note "FFMPEG_OK (already installed)"
  elif { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg; } \
    >> "$LOG/setup_ffmpeg.log" 2>&1; then
    note "FFMPEG_OK $(ffmpeg -version | head -n 1)"
  else
    note "FFMPEG_FAILED (see $LOG/setup_ffmpeg.log)"
  fi
}

# Counts the mp4 clips in one folder, leaving out the macOS ._ stub files.
count_clips() {
  find "$1" -maxdepth 1 -name '*.mp4' ! -name '._*' 2> /dev/null | wc -l
}

# Finds the folder that holds all three split folders under /root/meld, or prints nothing.
clip_root() {
  local dir
  while IFS= read -r dir; do
    dir=$(dirname "$dir")
    if [ -d "$dir/dev_splits_complete" ] && [ -d "$dir/output_repeated_splits_test" ]; then
      echo "$dir"
      return
    fi
  done < <(find "$MELD" -maxdepth 4 -type d -name train_splits)
}

# Unpacks the clips. The archive's top level is listed first: when it holds nested train, dev and test tar.gz
# files, each is streamed straight into the inner tar so no intermediate tarball touches the disk. Success is
# judged by the clip counts, not by tar's exit codes, so CLIPS_OK means every split is complete.
unpack_clips() {
  local root member
  root=$(clip_root)
  if [ -n "$root" ] && [ "$(count_clips "$root/train_splits")" -ge 9989 ] \
    && [ "$(count_clips "$root/dev_splits_complete")" -ge 1112 ] \
    && [ "$(count_clips "$root/output_repeated_splits_test")" -ge 2747 ]; then
    note "clips: already unpacked in $root"
  elif [ ! -f "$ARCHIVE" ]; then
    note "CLIPS_FAILED: $ARCHIVE is missing"
    return
  else
    note "clips: listing $ARCHIVE"
    tar -tzf "$ARCHIVE" > "$MELD/archive_list.txt" 2>> "$LOG/setup_clips.log"
    head -n 20 "$MELD/archive_list.txt" >> "$LOG/setup_clips.log"
    if grep -Eq '(^|/)(train|dev|test)\.tar\.gz$' "$MELD/archive_list.txt"; then
      mkdir -p "$MELD/MELD.Raw"
      while IFS= read -r member; do
        note "clips: streaming $member"
        tar -xzOf "$ARCHIVE" --occurrence=1 "$member" \
          | tar -xzf - -C "$MELD/MELD.Raw" --exclude='._*' >> "$LOG/setup_clips.log" 2>&1
        note "clips: $member finished with exit codes ${PIPESTATUS[*]}"
      done < <(grep -E '(^|/)(train|dev|test)\.tar\.gz$' "$MELD/archive_list.txt")
    else
      note "clips: no nested archives, extracting directly"
      tar -xzf "$ARCHIVE" -C "$MELD" --exclude='._*' >> "$LOG/setup_clips.log" 2>&1
    fi
    root=$(clip_root)
  fi
  if [ -z "$root" ]; then
    note "CLIPS_FAILED: no folder under $MELD holds train_splits, dev_splits_complete and output_repeated_splits_test"
    return
  fi
  if [ -L "$LINK" ]; then
    rm "$LINK"
  elif [ -e "$LINK" ]; then
    mv "$LINK" "$LINK.old.$(date +%s)"
    note "clips: moved an existing non-link $LINK aside"
  fi
  ln -sfn "$root" "$LINK"
  local train dev test
  train=$(count_clips "$root/train_splits")
  dev=$(count_clips "$root/dev_splits_complete")
  test=$(count_clips "$root/output_repeated_splits_test")
  if [ "$train" -ge 9989 ] && [ "$dev" -ge 1112 ] && [ "$test" -ge 2747 ]; then
    note "CLIPS_OK $LINK -> $root: train $train, dev $dev, test $test clips"
  else
    note "CLIPS_FAILED: incomplete unpack in $root: train $train, dev $dev, test $test (expected 9989, 1112, 2747)"
  fi
}

note "setup start"
{ install_ffmpeg; unpack_clips; } &
build_env
wait
note "setup end"
