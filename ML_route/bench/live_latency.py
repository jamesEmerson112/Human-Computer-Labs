"""Measures the live player chain's latency and resource use on this PC through serve.py.

Contents:
- pick_dialogues: picks seeded test dialogues from MELD.Completed
- call: one GET to serve.py, returning the JSON body and the round trip
- GpuSampler: reads nvidia-smi's used memory and utilisation in the background
- ProcessSampler: reads the three servers' CPU and RAM, and the machine's load, in the background
- summarize: median, 95th percentile and maximum of a list of numbers
- main: plays every line of the picked dialogues and writes bench/results/live_1070ti.json

Run it from ML_route with the three player servers up:
text_fixes/.venv/Scripts/python.exe bench/live_latency.py
"""

import csv
import json
import platform
import random
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_CSV = HERE.parent / "data" / "MELD.Completed" / "test_sent_emo.csv"
OUTPUT = HERE / "results" / "live_1070ti.json"
SERVER = "http://127.0.0.1:8765"
DIALOGUES = 20
SEED = 7
MAX_HISTORY = 40
MAX_TEXT = 600
SERVER_SCRIPTS = ("serve.py", "serve_emotion.py", "serve_reply.py")


def pick_dialogues(count, seed):
    """
    Description: Picks seeded test dialogues and their lines in order.
    Inputs:
    - count: how many dialogues to pick.
    - seed: the random seed.
    Outputs: a list of (dialogue, [(utterance, speaker, text), ...]) sorted by dialogue.
    """
    dialogues = {}
    with TEST_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            line = (int(row["Utterance_ID"]), row["Speaker"], row["Utterance"])
            dialogues.setdefault(int(row["Dialogue_ID"]), []).append(line)
    chosen = random.Random(seed).sample(sorted(dialogues), count)
    return [(dia, sorted(dialogues[dia])) for dia in sorted(chosen)]


def call(path, params):
    """
    Description: Sends one GET to serve.py.
    Inputs:
    - path: the endpoint, such as /api/emotion.
    - params: the query parameters.
    Outputs: (body, round trip in milliseconds), with body None on any failure.
    """
    url = f"{SERVER}{path}?{urllib.parse.urlencode(params)}"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            body = json.load(response)
    except Exception as err:
        body = {"error": str(err)}
    elapsed = (time.perf_counter() - started) * 1000
    return (None if "error" in body else body), round(elapsed, 1)


class GpuSampler(threading.Thread):
    """
    Description: Reads the GPU's used memory and utilisation twice a second from one nvidia-smi process.
    Fields:
    - samples: a list of (seconds since start, used MiB, utilisation percent).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, text=True,
        )
        self.started = time.perf_counter()

    def run(self):
        """
        Description: Collects samples until the nvidia-smi process ends.
        Inputs: none
        Outputs: none
        """
        for raw in self.process.stdout:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                self.samples.append((time.perf_counter() - self.started, int(parts[0]), int(parts[1])))

    def stop(self):
        """
        Description: Ends the nvidia-smi process.
        Inputs: none
        Outputs: none
        """
        self.process.terminate()


PROCESS_LOOP = r"""
$ids = @{}
# Each venv python.exe is a small launcher with the real interpreter as its child, so keep the largest.
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($s in @('serve_emotion.py','serve_reply.py','serve.py')) {
  $best = $all | Where-Object { $_.CommandLine -match ([regex]::Escape($s) + '\s*$') } |
    Sort-Object { (Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue).WorkingSet64 } -Descending |
    Select-Object -First 1
  if ($best) { $ids[$best.ProcessId] = $s }
}
while ($true) {
  $load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
  $rows = foreach ($id in $ids.Keys) {
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p) { @{ script = $ids[$id]; pid = $id; cpu_s = $p.TotalProcessorTime.TotalSeconds; ram_mb = [math]::Round($p.WorkingSet64 / 1MB, 1) } }
  }
  @{ load = $load; procs = @($rows) } | ConvertTo-Json -Compress -Depth 4
  Start-Sleep -Milliseconds 1000
}
"""


class ProcessSampler(threading.Thread):
    """
    Description: Reads each server's CPU time and RAM, and the machine's CPU load, once a second from
    one PowerShell process.
    Fields:
    - samples: a list of (seconds since start, machine load percent, {script: (cpu seconds, RAM MB)}).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.process = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", PROCESS_LOOP],
            stdout=subprocess.PIPE, text=True,
        )
        self.started = time.perf_counter()

    def run(self):
        """
        Description: Collects samples until the PowerShell process ends.
        Inputs: none
        Outputs: none
        """
        for raw in self.process.stdout:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            procs = {p["script"]: (p["cpu_s"], p["ram_mb"]) for p in row.get("procs") or []}
            self.samples.append((time.perf_counter() - self.started, row.get("load"), procs))

    def stop(self):
        """
        Description: Ends the PowerShell process.
        Inputs: none
        Outputs: none
        """
        self.process.terminate()

    def per_server(self):
        """
        Description: Turns the samples into each server's mean CPU use and peak RAM over the run.
        Inputs: none
        Outputs: {script: {"cpu_percent_of_machine": ..., "ram_mb_peak": ...}}.
        Constraints: CPU use is the server's CPU time over the run's wall time, divided by the logical CPUs.
        """
        if len(self.samples) < 2:
            return {}
        first, last = self.samples[0], self.samples[-1]
        wall = last[0] - first[0]
        threads = _logical_cpus()
        result = {}
        for script in SERVER_SCRIPTS:
            if script not in first[2] or script not in last[2]:
                continue
            cpu = (last[2][script][0] - first[2][script][0]) / wall / threads * 100
            ram = max(s[2][script][1] for s in self.samples if script in s[2])
            result[script] = {"cpu_percent_of_machine": round(cpu, 1), "ram_mb_peak": ram}
        return result


def _logical_cpus():
    """
    Description: The number of logical CPUs.
    Inputs: none
    Outputs: an int, at least 1.
    """
    import os
    return os.cpu_count() or 1


def summarize(values):
    """
    Description: Median, 95th percentile and maximum of a list of numbers.
    Inputs:
    - values: the numbers; None entries are skipped.
    Outputs: {"n", "median", "p95", "max"}, or {"n": 0} when nothing is left.
    """
    kept = sorted(v for v in values if v is not None)
    if not kept:
        return {"n": 0}
    p95 = kept[min(len(kept) - 1, round(0.95 * (len(kept) - 1)))]
    return {"n": len(kept), "median": round(statistics.median(kept), 1), "p95": round(p95, 1), "max": round(kept[-1], 1)}


def main():
    """
    Description: Plays every line of the picked dialogues through Whisper, the emotion head and LFM2.5,
    samples the GPU and the servers, then writes and prints the summary.
    Inputs: none
    Outputs: none
    """
    status, _ = call("/api/status", {})
    if not status or not all(status["models"][m].get("ready") for m in ("whisper", "emotion", "reply")):
        raise SystemExit("Start serve.py, serve_emotion.py and serve_reply.py first, and wait until all are ready.")

    gpu = GpuSampler()
    procs = ProcessSampler()
    gpu.start()
    procs.start()
    time.sleep(2)

    lines = []
    picked = pick_dialogues(DIALOGUES, SEED)
    for dia, utterances in picked:
        history = []
        for utt, speaker, text in utterances:
            clip = f"dia{dia}_utt{utt}"
            heard, transcribe_rt = call("/api/transcribe", {"model": "whisper", "split": "test", "clip": clip})
            words = (heard or {}).get("text", "")[:MAX_TEXT]
            emotion, emotion_rt = call("/api/emotion", {"split": "test", "clip": clip, "text": words})
            label = (emotion or {}).get("emotion", "neutral")
            reply, reply_rt = call("/api/reply", {
                "text": (words or text)[:MAX_TEXT],
                "emotion": label,
                "speaker": speaker,
                "history": json.dumps(history[-MAX_HISTORY:]),
            })
            history.append({"speaker": speaker, "text": text[:MAX_TEXT]})
            lines.append({
                "clip": clip,
                "audio_s": (heard or {}).get("audio_s"),
                "whisper_model_ms": (heard or {}).get("model_ms"),
                "whisper_decode_ms": (heard or {}).get("decode_ms"),
                "whisper_wait_ms": (heard or {}).get("wait_ms"),
                "whisper_round_trip_ms": transcribe_rt if heard else None,
                "embeddinggemma_ms": (emotion or {}).get("embeddinggemma_ms"),
                "emoberta_ms": (emotion or {}).get("emoberta_ms"),
                "hubert_ms": (emotion or {}).get("hubert_ms"),
                "head_ms": (emotion or {}).get("head_ms"),
                "emotion_model_ms": (emotion or {}).get("model_ms"),
                "emotion_wait_ms": (emotion or {}).get("wait_ms"),
                "emotion_round_trip_ms": emotion_rt if emotion else None,
                "reply_model_ms": (reply or {}).get("model_ms"),
                "reply_tokens": (reply or {}).get("tokens"),
                "reply_wait_ms": (reply or {}).get("wait_ms"),
                "reply_round_trip_ms": reply_rt if reply else None,
                "clip_end_to_emotion_ms": round(transcribe_rt + emotion_rt, 1) if heard and emotion else None,
                "clip_end_to_reply_ms": round(transcribe_rt + emotion_rt + reply_rt, 1) if heard and emotion and reply else None,
                "emotion": label,
                "reply": (reply or {}).get("reply"),
            })
            print(f"{clip}: {label}, reply after {lines[-1]['clip_end_to_reply_ms']} ms", flush=True)

    time.sleep(1)
    gpu.stop()
    procs.stop()
    gpu.join(timeout=5)
    procs.join(timeout=5)

    stages = [k for k in lines[0] if k.endswith("_ms") or k == "reply_tokens"]
    summary = {k: summarize([line[k] for line in lines]) for k in stages}
    gpu_used = [s[1] for s in gpu.samples]
    gpu_util = [s[2] for s in gpu.samples]
    loads = [s[1] for s in procs.samples if s[1] is not None]
    result = {
        "hardware": {
            "machine": platform.platform(),
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
            "logical_cpus": _logical_cpus(),
        },
        "dialogues": [dia for dia, _ in picked],
        "lines": len(lines),
        "failures": sum(1 for line in lines if line["clip_end_to_reply_ms"] is None),
        "summary_ms": summary,
        "gpu": {
            "used_mib_peak": max(gpu_used, default=None),
            "used_mib_median": round(statistics.median(gpu_used), 1) if gpu_used else None,
            "utilisation_percent_median": round(statistics.median(gpu_util), 1) if gpu_util else None,
        },
        "cpu": {
            "machine_load_percent_median": round(statistics.median(loads), 1) if loads else None,
            "machine_load_percent_max": max(loads, default=None),
            "servers": procs.per_server(),
        },
        "per_line": lines,
    }
    OUTPUT.write_bytes((json.dumps(result, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))

    print(f"\n{len(lines)} lines from {len(picked)} dialogues, {result['failures']} failures, written to {OUTPUT}")
    print("| Stage | Median ms | 95th percentile ms | Max ms |")
    print("|---|---:|---:|---:|")
    for k, s in summary.items():
        if s["n"]:
            print(f"| {k} | {s['median']} | {s['p95']} | {s['max']} |")
    print(json.dumps({"gpu": result["gpu"], "cpu": result["cpu"]}, indent=1))


if __name__ == "__main__":
    main()
