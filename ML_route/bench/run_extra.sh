#!/usr/bin/env bash
# Runs the steps that needed the Hugging Face token or a fix, after run_all.sh has finished.
# EmbeddingGemma features, the heads again (EmoBERTa left out of the best-text pick), and Llama 3.2 3B
# reply speed when Meta has approved access.
set -u
cd /workspace/bench
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG=/workspace/logs
STATUS=$LOG/run_all_status.txt
HF=/root/envs/hf/bin/python

# Runs one step and records its exit code and duration.
step() {
  local name=$1; shift
  local t0=$(date +%s)
  echo "$(date -u +%FT%TZ) start $name" >> $STATUS
  "$@" > $LOG/$name.log 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc $(( $(date +%s) - t0 ))s" >> $STATUS
}

until grep -q ALL_DONE $STATUS; do sleep 60; done
step features_extra $HF features.py --text embeddinggemma-300m --voice ""
step heads_rerun $HF heads.py
# Reasoning models rerun with thinking turned off; Gemma 4 E2B reruns after its weights finished downloading.
MODELS=nemotron-3-nano-4b,smollm3-3b,qwen3.5-4b,gemma-4-e2b-it
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat $HF_HOME/token)" \
  https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json)
if [ "$code" = "200" ]; then
  MODELS=$MODELS,llama-3.2-3b-instruct
else
  echo "$(date -u +%FT%TZ) skip llama-3.2-3b-instruct, access http $code" >> $STATUS
fi
step reply_speed_extra $HF reply_speed.py --models $MODELS --out reply_speed_extra
echo "$(date -u +%FT%TZ) EXTRA_DONE" >> $STATUS
