#!/usr/bin/env bash
# Reruns the zero-shot VLM test for Qwen2.5-Omni-3B with its thinker loaded, after the extra pass.
set -u
cd /workspace/bench
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG=/workspace/logs
STATUS=$LOG/run_all_status.txt
until grep -q EXTRA_DONE $STATUS; do sleep 30; done
t0=$(date +%s)
echo "$(date -u +%FT%TZ) start vlm_omni" >> $STATUS
/root/envs/hf/bin/python vlm_zero_shot.py --models qwen2.5-omni-3b=Qwen/Qwen2.5-Omni-3B --minutes 40 > $LOG/vlm_omni.log 2>&1
echo "$(date -u +%FT%TZ) end vlm_omni rc=$? $(( $(date +%s) - t0 ))s" >> $STATUS
echo "$(date -u +%FT%TZ) OMNI_DONE" >> $STATUS
