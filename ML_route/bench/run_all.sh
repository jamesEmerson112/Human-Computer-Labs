#!/usr/bin/env bash
# Runs every overnight benchmark in order on the pod, one GPU job at a time, logging each step.
# A failed step is logged and the next one still runs. Results land in /workspace/results.
set -u
cd /workspace/bench
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG=/workspace/logs
STATUS=$LOG/run_all_status.txt

# Picks the copy of an environment on the fast local disk when its build finished, else the network copy.
pick() {
  if grep -q "${2}_LOCAL_OK" $LOG/local_envs.log 2>/dev/null; then echo /root/envs/$1/bin/python; else echo /workspace/envs/$1/bin/python; fi
}

# Runs one step and records its exit code and duration.
step() {
  local name=$1; shift
  local t0=$(date +%s)
  echo "$(date -u +%FT%TZ) start $name" >> $STATUS
  "$@" > $LOG/$name.log 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc $(( $(date +%s) - t0 ))s" >> $STATUS
}

nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 30 > $LOG/gpu_trace.csv 2>&1 &

for i in $(seq 1 40); do grep -q ASR_LOCAL_OK $LOG/local_envs.log 2>/dev/null && break; sleep 30; done
ASR=$(pick asr ASR)
for m in whisper-turbo-int8; do
  step asr_$m $ASR asr_race.py --model $m
done

HF=$(pick hf HF)
for i in $(seq 1 60); do [ -f /workspace/audio/train.npz ] && break; sleep 30; done
step features $HF features.py
step heads $HF heads.py
step reply_speed $HF reply_speed.py
step vlm $HF vlm_zero_shot.py --models "$(cat /workspace/bench/vlm_models.txt)" --minutes 50
echo "$(date -u +%FT%TZ) ALL_DONE" >> $STATUS
