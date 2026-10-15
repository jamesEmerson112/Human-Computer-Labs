"""Run native headless trials and verify every frame against the saved baseline.

The Python default-physics reference is remeasured immediately before Rust.
Historical traces establish behavioral equivalence; current resource/timing
observations do not imply performance on different hardware or an idle host.
"""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.measure_viewer import save_json
from lamp_sim.benchmark import cpu_name, fingerprints
from lamp_sim.gestures import load_gestures
from lamp_sim.motion import load_limits


def verify_reference(reference: Path) -> dict:
    """Reject stale workloads before creating output or launching any measurement."""
    fresh = "benchmarks/python-reference-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    instruction = ("Generate a fresh reference with: uv run --locked python -m lamp_sim.benchmark "
                   f"--output {fresh} --repeats 1 --speed 0.5; then pass "
                   f"--reference {fresh} to this command. Keep historical baselines unchanged.")
    try:
        manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
        recorded = manifest["source_sha256"]
        expected = fingerprints()
        prefix = "simulation/gestures/"
        old_catalog = {key: value for key, value in recorded.items() if key.startswith(prefix)}
        current_catalog = {key: value for key, value in expected.items() if key.startswith(prefix)}
        if old_catalog != current_catalog:
            added = sorted(current_catalog.keys() - old_catalog.keys())
            removed = sorted(old_catalog.keys() - current_catalog.keys())
            changed = sorted(key for key in current_catalog.keys() & old_catalog.keys() if current_catalog[key] != old_catalog[key])
            raise ValueError(f"Gesture workload differs: {len(added)} added, {len(removed)} removed, {len(changed)} changed files")
        for key, value in {"speed": 0.5, "control_hz": 30, "physics_hz": 240, "settling_tail_s": 1.0}.items():
            if manifest.get(key) != value:
                raise ValueError(f"Reference {key} is {manifest.get(key)!r}; expected {value}")
        # Exact model and controller inputs matter even when gesture names match.
        physical_inputs = [key for key in expected if key.startswith("starter_pack/robot/")]
        physical_inputs += ["simulation/lamp_sim/motion.py", "simulation/lamp_sim/backends.py"]
        for key in physical_inputs:
            if recorded.get(key) != expected[key]:
                raise ValueError(f"Reference model/controller input changed: {key}")
        names = set(load_gestures(load_limits()))
        for backend in ("mujoco-kinematic", "mujoco-source", "mujoco"):
            if not (reference / f"{backend}.csv.gz").is_file():
                raise ValueError(f"Reference trace is missing: {backend}.csv.gz")
            result = json.loads((reference / f"{backend}.json").read_text(encoding="utf-8"))
            if result.get("backend") != backend or {trial["gesture"] for trial in result["trials"]} != names:
                raise ValueError(f"Reference {backend} trial names do not match the current catalog")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot compare unmatched benchmark workloads: {error}. {instruction}") from error
    return {"path": str(reference.resolve()), "manifest_sha256": hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest(),
            "gesture_count": len(names), "catalog_match": True}


def run_sampled(command: list[str], log: Path) -> dict:
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    samples = []
    psutil.cpu_percent()
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, **hidden)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                info = process.memory_info()
                samples.append({"rss_mib": info.rss / 2**20,
                                "peak_rss_mib": getattr(info, "peak_wset", info.rss) / 2**20,
                                "host_cpu_percent": psutil.cpu_percent()})
            except psutil.NoSuchProcess:
                break
            time.sleep(.1)
        code = child.wait()
    if code:
        raise RuntimeError(log.read_text())
    return {"command": command, "peak_rss_mib": max(s["peak_rss_mib"] for s in samples),
            "mean_host_cpu_percent": statistics.mean(s["host_cpu_percent"] for s in samples),
            "sample_period_s": .1, "samples": samples}


def compare_trace(reference: Path, measured: Path, expected_repeats: int = 3) -> dict:
    with gzip.open(reference, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [column for column in reader.fieldnames
                   if column.startswith(("desired_rad.", "actual_rad.", "velocity_rad_s."))]
        old = {(row["gesture"], int(row["frame"])): np.array([float(row[c]) for c in columns])
               for row in reader if row["repeat"] == "1"}
    maxima = np.zeros(len(columns))
    frames = 0
    seen = set()
    unique_rows = set()
    by_repeat = {}
    opener = gzip.open if measured.suffix == ".gz" else open
    with opener(measured, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["gesture"], int(row["frame"])
            repeat = int(row["repeat"])
            full_key = (key[0], repeat, key[1])
            if full_key in unique_rows:
                raise AssertionError(f"Duplicate native frame: {full_key}")
            unique_rows.add(full_key)
            by_repeat.setdefault(repeat, set()).add(key)
            values = np.array([float(row[c]) for c in columns])
            if key not in old:
                raise AssertionError(f"Unexpected native frame: {key}")
            maxima = np.maximum(maxima, np.abs(values - old[key]))
            frames += 1
            seen.add(key)
    if seen != old.keys():
        raise AssertionError(f"Native trace misses {len(old.keys() - seen)} reference frames")
    if set(by_repeat) != set(range(1, expected_repeats + 1)):
        raise AssertionError(f"Unexpected repetitions: {sorted(by_repeat)}")
    if any(keys != old.keys() for keys in by_repeat.values()):
        raise AssertionError("A repetition does not contain every reference gesture/frame")
    np.testing.assert_array_less(maxima, np.full(len(columns), 1e-9), err_msg="Python/native trajectory mismatch")
    return {"frames_compared": frames, "unique_gesture_frames": len(seen), "tolerance": 1e-9,
            "complete_repetitions": sorted(by_repeat), "duplicate_rows": 0,
            "max_absolute_differences": dict(zip(columns, map(float, maxima)))}


def summarize(path: Path, resources: dict) -> dict:
    result = json.loads(path.read_text())
    trials = result["trials"]
    total_frames = sum(t["frames"] for t in trials)
    return {"trials": len(trials), "frames": total_frames,
            "worst_trial_p95_ms": max(t["compute_p95_ms"] for t in trials),
            "median_realtime_factor": statistics.median(t["realtime_factor"] for t in trials),
            "median_cpu_ms_per_simulated_s": statistics.median(t["cpu_ms_per_simulated_s"] for t in trials),
            "pooled_cpu_ms_per_simulated_s": 1000 * sum(t["cpu_s"] for t in trials) / sum(t["simulated_s"] for t in trials),
            # Windows venv python.exe may be a small redirector process. Prefer
            # the worker's own measured peak over sampling that launcher.
            "peak_process_rss_mib": result.get("process_peak_rss_mib", resources["peak_rss_mib"]),
            "tracking_rmse_deg": (sum(t["tracking_rmse_deg"]**2 * t["frames"] for t in trials) / total_frames)**.5,
            "tracking_max_deg": max(t["tracking_max_deg"] for t in trials),
            "soft_limit_violation_max_deg": max(t["soft_limit_violation_max_deg"] for t in trials),
            "max_penetration_mm": max(t["max_penetration_mm"] for t in trials),
            "physics_warning_count": max(t["physics_warning_count"] for t in trials),
            "over_budget_frames": sum(t["over_budget_frames"] for t in trials),
            "mean_host_cpu_percent": resources["mean_host_cpu_percent"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", type=Path, default=PROJECT / "benchmarks/2026-10-13-current-setup/headless")
    parser.add_argument("--executable", type=Path, default=PROJECT / "controller/target/release/benchmark.exe")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    try:
        reference_validation = verify_reference(args.reference)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "cpu": cpu_name(), "python": platform.python_version(),
                "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                "ram_gib": psutil.virtual_memory().total / 2**30, "repeats": args.repeats, "speed": .5,
                "control_hz": 30, "physics_hz": 240, "settling_tail_frames": 31,
                "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                "model_manifest": json.loads((PROJECT / "models/derived/manifest.json").read_text()),
                "reference_validation": reference_validation,
                "measurement_scope": "Sequential fresh processes, warmup per gesture. Compute covers controller+physics; loop includes observation/log buffers. CSV serialization excluded. No rendering."}
    native_sources = sorted((PROJECT / "controller/src").rglob("*.rs"))
    native_sources += sorted((PROJECT / "controller/native").glob("*"))
    native_sources += [PROJECT / "controller/Cargo.toml", PROJECT / "controller/Cargo.lock",
                       PROJECT / "controller/build.rs", Path(__file__)]
    metadata["source_sha256"] = fingerprints() | {
        str(path.relative_to(PROJECT.parent)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in native_sources if path.is_file()}
    save_json(output / "manifest.json", metadata)
    results = {}
    python_output = output / "python-current"
    python_output.mkdir()
    command = [sys.executable, "-m", "lamp_sim.benchmark", "--worker", "mujoco", "--output", str(python_output),
               "--repeats", str(args.repeats), "--speed", "0.5"]
    print("Measuring current Python default physics...", flush=True)
    resources = run_sampled(command, output / "python-current.log")
    save_json(python_output / "resources.json", resources)
    results["python-mujoco"] = summarize(python_output / "mujoco.json", resources)
    for backend in ("mujoco-kinematic", "mujoco-source", "mujoco"):
        destination = output / ("rust-" + backend)
        command = [str(args.executable), "--root", str(PROJECT), "--output", str(destination),
                   "--backend", backend, "--repeats", str(args.repeats), "--speed", "0.5"]
        print(f"Measuring Rust {backend}...", flush=True)
        resources = run_sampled(command, output / ("rust-" + backend + ".log"))
        save_json(destination / "resources.json", resources)
        parity = compare_trace(args.reference / f"{backend}.csv.gz", destination / f"{backend}.csv", args.repeats)
        save_json(destination / "parity.json", parity)
        results["rust-" + backend] = summarize(destination / f"{backend}.json", resources)
        results["rust-" + backend]["parity"] = parity
        # Compress bulky traces after all measurements; preserve exact CSV bytes.
        csv_path = destination / f"{backend}.csv"
        with csv_path.open("rb") as source, gzip.open(str(csv_path) + ".gz", "wb") as target:
            import shutil
            shutil.copyfileobj(source, target)
        csv_path.unlink()
    save_json(output / "summary.json", results)
    print(json.dumps({key: {k:v for k,v in value.items() if k != "parity"} for key,value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
