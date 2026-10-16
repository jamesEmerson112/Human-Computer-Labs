#!/usr/bin/env bash
# Reruns the zero-shot VLM step with label scoring in place of free-text answers, then the extra pass.
# The first VLM run asked for free text with an 8-token limit, and Qwen3.5-4B explained itself first,
# so almost no answer held a label.
set -u
cd /workspace/bench
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG=/workspace/logs
STATUS=$LOG/run_all_status.txt
HF=/root/envs/hf/bin/python
mkdir -p /workspace/results/vlm_first_try && mv /workspace/results/vlm_*.json* /workspace/results/vlm_first_try/ 2>/dev/null
t0=$(date +%s)
echo "$(date -u +%FT%TZ) start vlm_scored" >> $STATUS
$HF vlm_zero_shot.py --models "$(cat vlm_models.txt)" --minutes 40 > $LOG/vlm_scored.log 2>&1
echo "$(date -u +%FT%TZ) end vlm_scored rc=$? $(( $(date +%s) - t0 ))s" >> $STATUS
echo "$(date -u +%FT%TZ) ALL_DONE" >> $STATUS
bash run_extra.sh > $LOG/run_extra.log 2>&1
