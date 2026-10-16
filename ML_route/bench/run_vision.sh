#!/usr/bin/env bash
# Runs the text-plus-vision benchmark in order on the pod, one GPU job at a time, logging each step.
# It waits for setup_vision.sh's markers, and a failed step is logged while the next one still runs. When the
# clips are missing or incomplete, the frame and vision steps are skipped rather than run on a partial corpus.
# Each step's start, end, exit code and duration go to /workspace/logs/run_vision_status.txt for cost tracking.
set -u
cd /workspace/bench || exit 1
export HF_HOME=/workspace/hf MELD_FRAMES=/root/frames PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG=/workspace/logs
STATUS=$LOG/run_vision_status.txt
SETUP=$LOG/setup_vision_status.txt
HF=/root/envs/hf/bin/python
mkdir -p "$LOG"

# Runs one step and records its start, end, exit code and duration.
step() {
  local name=$1 t0 rc
  shift
  t0=$(date +%s)
  echo "$(date -u +%FT%TZ) start $name" >> "$STATUS"
  "$@" > "$LOG/$name.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc $(( $(date +%s) - t0 ))s" >> "$STATUS"
}

# Waits up to 90 minutes for a setup marker to end in _OK or _FAILED, and succeeds only on _OK when the given
# path also exists, so a marker left from an earlier boot is not trusted on its own.
wait_for() {
  local i
  for i in $(seq 1 180); do
    grep -q "${1}_OK" "$SETUP" 2> /dev/null && [ -e "$2" ] && return 0
    grep -q "${1}_FAILED" "$SETUP" 2> /dev/null && return 1
    [ "$i" -eq 1 ] && echo "$(date -u +%FT%TZ) waiting for setup marker ${1}" >> "$STATUS"
    sleep 30
  done
  return 1
}


# Records that a step was skipped and why, in the same status file.
skip() {
  echo "$(date -u +%FT%TZ) skip $1: $2" >> "$STATUS"
}

echo "$(date -u +%FT%TZ) RUN_START" >> "$STATUS"
if ! wait_for ENV "$HF"; then
  echo "$(date -u +%FT%TZ) STOP: the hf environment is not ready; see $SETUP" >> "$STATUS"
  exit 1
fi
CLIPS=1
if ! wait_for CLIPS /workspace/data/MELD.Raw/train_splits; then
  CLIPS=0
  echo "$(date -u +%FT%TZ) warning: the clips are missing or incomplete; the vision steps are skipped" >> "$STATUS"
fi
if ! command -v ffprobe > /dev/null; then
  CLIPS=0
  echo "$(date -u +%FT%TZ) warning: ffmpeg is not installed; the vision steps are skipped" >> "$STATUS"
fi

nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 30 \
  > "$LOG/gpu_trace_vision.csv" 2>&1 &
TRACE=$!
trap 'kill "$TRACE" 2> /dev/null' EXIT

if [ "$CLIPS" -eq 1 ]; then
  step frames "$HF" frames.py
else
  skip frames "no complete clips"
fi
# nemotron_text.py reads the decoded audio that last night's audio_cache.py left on the volume; rebuild it if gone.
if [ ! -f /workspace/audio/train.npz ] || [ ! -f /workspace/audio/dev.npz ] || [ ! -f /workspace/audio/test.npz ]; then
  if [ "$CLIPS" -eq 1 ]; then
    step audio_cache "$HF" audio_cache.py
  else
    skip audio_cache "no complete clips, so nemotron_text.py will fail"
  fi
fi
step nemotron_text "$HF" nemotron_text.py
step features_nemotron "$HF" features.py --text roberta-base --voice none --versions nemotron
# EmbeddingGemma is gated; without a token, the cached weights from the earlier run are used offline.
if [ -s "$HF_HOME/token" ] || [ -n "${HF_TOKEN:-}" ]; then
  step features_nemotron_gemma "$HF" features.py --text embeddinggemma-300m --voice none --versions nemotron
else
  echo "$(date -u +%FT%TZ) note: no Hugging Face token, EmbeddingGemma runs offline from the cache" >> "$STATUS"
  step features_nemotron_gemma env HF_HUB_OFFLINE=1 "$HF" features.py --text embeddinggemma-300m --voice none \
    --versions nemotron
fi
if [ "$CLIPS" -eq 1 ]; then
  step vision_features "$HF" vision_features.py --encoders all
  step vlm_qwen3.5-4b "$HF" vlm_features.py --models qwen3.5-4b=Qwen/Qwen3.5-4B
  step vlm_oneemo "$HF" vlm_features.py --models oneemo=Jiaha0Hu4ng/OneEmo
else
  skip vision_features "no complete clips"
  skip vlm "no complete clips"
fi
step heads_vision "$HF" heads.py --vision
step report "$HF" report.py /workspace/results
echo "$(date -u +%FT%TZ) ALL_DONE" >> "$STATUS"
