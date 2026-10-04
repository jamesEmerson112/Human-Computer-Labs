"""Repeatable headless comparisons; run with python -m lamp_sim.benchmark."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

from .gestures import GESTURE_PATH, load_gestures
from .motion import MotionController, ROBOT_PATH


BACKENDS = ("kinematic", "mujoco-kinematic", "mujoco-source", "mujoco")
PROJECT = Path(__file__).resolve().parents[1]
DT = 1 / 30


def worker(backend_name: str, output: Path, repeats: int, speed: float) -> None:
    # Each backend runs in its own fresh process, so RSS includes its own imports.
    started = time.perf_counter()
    from .backends import make_backend
    backend = make_backend(backend_name)
    library = load_gestures(backend.names)
    initialization_s = time.perf_counter() - started
    process = psutil.Process()
    results = []
    names = backend.names
    header = ["gesture", "repeat", "frame", "simulation_s", "compute_ms", "contacts", "penetration_m", "contact_force_n", "max_effort_nm"]
    header += [f"{prefix}.{n}" for prefix in ("desired_rad", "actual_rad", "velocity_rad_s") for n in names]
    with gzip.open(output / f"{backend_name}.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for gesture in library.values():
            # Warm up the actual workload once, then reset for every measured trial.
            run_trial(backend, gesture, speed, record=False)
            for repeat in range(repeats):
                trial, rows = run_trial(backend, gesture, speed, record=True)
                trial.update(backend=backend_name, gesture=gesture.name, repeat=repeat + 1)
                results.append(trial)
                # Disk serialization is outside the measured playback loop.
                for row in rows:
                    writer.writerow([gesture.name, repeat + 1, *row])
    result = {
        "backend": backend_name, "initialization_s": initialization_s,
        "rss_before_first_measured_trial_mib": results[0]["rss_start_mib"],
        "process_peak_rss_mib": getattr(process.memory_info(), "peak_wset", process.memory_info().rss) / 2**20,
        "trials": results,
    }
    (output / f"{backend_name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_trial(backend, gesture, speed: float, record: bool):
    backend.reset()
    motion = MotionController(backend.limits)
    motion.set_speed(speed)
    motion.play(gesture)
    names = backend.names
    rows, timings, errors = [], [], []
    contact_frames = 0
    max_contacts = 0 if backend.diagnostics.contacts is not None else None
    max_penetration = 0.0 if max_contacts is not None else None
    max_force = 0.0 if backend.name in ("mujoco", "mujoco-source") else None
    max_effort = 0.0 if max_force is not None else None
    max_velocity_ratio = max_soft_violation = 0.0
    pairs = set()
    process = psutil.Process()
    rss_start = process.memory_info().rss / 2**20
    rss_peak = rss_start
    cpu_start = time.process_time()
    start = time.perf_counter()
    tail_frames = 0
    for frame in range(20000):
        tick_start = time.perf_counter_ns()
        desired = motion.step(DT)
        actual = backend.advance(desired, DT)
        elapsed_ms = (time.perf_counter_ns() - tick_start) / 1e6
        diag = backend.diagnostics
        if record:
            timings.append(elapsed_ms)
            delta = np.array([actual[n] - desired[n] for n in names])
            errors.append(delta)
            if diag.contacts is not None:
                max_contacts = max(max_contacts, diag.contacts)
                contact_frames += int(diag.contacts > 0)
                max_penetration = max(max_penetration, diag.penetration_m)
                pairs.update(diag.pairs)
            if max_force is not None:
                max_force = max(max_force, diag.contact_force_n)
                max_effort = max(max_effort, diag.max_effort_nm)
            for name, limit in backend.limits.items():
                max_velocity_ratio = max(max_velocity_ratio, abs(backend.velocity[name]) / limit.velocity)
                max_soft_violation = max(max_soft_violation, limit.lower - actual[name], actual[name] - limit.upper)
            if frame % 30 == 0:
                rss_peak = max(rss_peak, process.memory_info().rss / 2**20)
            rows.append([
                frame, (frame + 1) * DT, elapsed_ms, diag.contacts, diag.penetration_m,
                diag.contact_force_n, diag.max_effort_nm,
                *[desired[n] for n in names], *[actual[n] for n in names],
                *[backend.velocity[n] for n in names],
            ])
        if not motion.moving:
            if tail_frames >= 30:
                break
            tail_frames += 1
    else:
        raise RuntimeError(f"Gesture {gesture.name} exceeded the benchmark frame limit")
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    if not record:
        return None, None
    error = np.array(errors)
    simulated = len(rows) * DT
    trial = {
        "frames": len(rows), "simulated_s": simulated, "loop_wall_s": wall,
        "cpu_s": cpu, "cpu_ms_per_simulated_s": 1000 * cpu / simulated,
        "realtime_factor": simulated / wall,
        "compute_p50_ms": float(np.percentile(timings, 50)),
        "compute_p95_ms": float(np.percentile(timings, 95)),
        "compute_max_ms": max(timings),
        "over_budget_frames": sum(t > 1000 * DT for t in timings),
        "rss_start_mib": rss_start, "rss_sampled_peak_mib": rss_peak,
        "tracking_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error**2)))),
        "tracking_max_deg": float(np.rad2deg(np.max(np.abs(error)))),
        "final_error_max_deg": float(np.rad2deg(np.max(np.abs(error[-1])))),
        "max_velocity_ratio": max_velocity_ratio,
        "soft_limit_violation_max_deg": float(np.rad2deg(max_soft_violation)),
        "contact_frames": contact_frames if max_contacts is not None else None,
        "max_contacts": max_contacts, "max_penetration_mm": None if max_penetration is None else max_penetration * 1000,
        "max_contact_force_n": max_force, "max_effort_nm": max_effort,
        "contact_pairs": sorted(pairs), "physics_warning_count": backend.diagnostics.warnings,
    }
    return trial, rows


def fingerprints() -> dict[str, str]:
    files = [ROBOT_PATH, ROBOT_PATH.parent / "assets/lamp_shade.stl", PROJECT / "uv.lock", PROJECT / "gesture.schema.json"]
    files += sorted(GESTURE_PATH.glob("*.json")) + sorted((PROJECT / "lamp_sim").glob("*.py"))
    return {str(path.relative_to(PROJECT.parent)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def report(output: Path, metadata: dict) -> None:
    all_results = [json.loads((output / f"{name}.json").read_text()) for name in BACKENDS]
    summaries = []
    for result in all_results:
        trials = result["trials"]
        summary = {
            "backend": result["backend"],
            "median_realtime_factor": statistics.median(t["realtime_factor"] for t in trials),
            "worst_trial_p95_ms": max(t["compute_p95_ms"] for t in trials),
            "peak_rss_mib": result["process_peak_rss_mib"],
            "tracking_rmse_deg": float(np.sqrt(sum(t["tracking_rmse_deg"]**2*t["frames"] for t in trials)/sum(t["frames"] for t in trials))),
            "max_error_deg": max(t["tracking_max_deg"] for t in trials),
            "max_penetration_mm": None if trials[0]["max_penetration_mm"] is None else max(t["max_penetration_mm"] for t in trials),
            "over_budget_frames": sum(t["over_budget_frames"] for t in trials),
            "physics_warning_count": max(t["physics_warning_count"] for t in trials),
            "max_soft_limit_violation_deg": max(t["soft_limit_violation_max_deg"] for t in trials),
            "max_velocity_ratio": max(t["max_velocity_ratio"] for t in trials),
            "median_cpu_ms_per_simulated_s": statistics.median(t["cpu_ms_per_simulated_s"] for t in trials),
        }
        summaries.append(summary)
    (output / "summary.json").write_text(json.dumps({"metadata": metadata, "summary": summaries}, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# Lamp backend benchmark", "", f"Measured at {metadata['timestamp_utc']}. {metadata['repeats']} measured trials per gesture and backend; one warmup each. Pace: {metadata['speed']}.", "",
        "Headless backend + trajectory generation; browser rendering, network transport, and disk writes excluded from per-frame compute times. Loop throughput includes in-memory metric collection. No real-time sleeping.", "",
        "| Backend | Median real-time factor | Worst trial p95 compute (ms) | Peak process RSS (MiB) | Tracking RMSE (deg) | Max error (deg) | Max penetration (mm) | Over 33.3 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        penetration = "not measured" if item["max_penetration_mm"] is None else f"{item['max_penetration_mm']:.3f}"
        lines.append(f"| {item['backend']} | {item['median_realtime_factor']:.1f}x | {item['worst_trial_p95_ms']:.3f} | {item['peak_rss_mib']:.1f} | {item['tracking_rmse_deg']:.3f} | {item['max_error_deg']:.3f} | {penetration} | {item['over_budget_frames']} |")
    lines += ["", "Kinematic zero tracking error is by construction: poses are assigned directly. It is not evidence of motor accuracy. Physics errors and contacts are additional diagnostic capabilities, not a regression against a physical baseline.", "",
        "`mujoco-source` retains the source self-contact geometry. `mujoco` excludes only upper_arm_link / lamp_head_link because those proxies overlap at zero pose. Other self/table contacts remain enabled. The exclusion can also hide genuine contact between that pair; keep the source mode for audits.", "",
        "Each raw .csv.gz row contains simulation time, desired/actual joint angles, velocities, contact diagnostics, effort, and compute time. JSON files include per-gesture trials, initialization cost, settling error, warnings, and soft-limit violations. None of these results establish real-hardware accuracy or Ubuntu performance."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.speed <= 1:
        parser.error("repeats must be positive and speed must be in (0, 1]")
    if args.worker:
        worker(args.worker, args.output, args.repeats, args.speed)
        return
    timestamp = datetime.now(timezone.utc)
    output = (args.output or PROJECT / "benchmarks" / timestamp.strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from .backends import GAINS, CONTACT_EXCLUSIONS, PHYSICS_HZ
    metadata = {
        "timestamp_utc": timestamp.isoformat(), "platform": platform.platform(),
        "cpu": cpu_name(), "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(), "ram_gib": psutil.virtual_memory().total / 2**30,
        "available_ram_gib_at_start": psutil.virtual_memory().available / 2**30,
        "python": platform.python_version(), "packages": {p: version(p) for p in ("mujoco", "numpy", "yourdfpy", "viser", "psutil")},
        "repeats": args.repeats, "speed": args.speed, "control_hz": 30, "physics_hz": PHYSICS_HZ,
        "settling_tail_s": 1.0, "servo_gains_kp_kv": GAINS, "armature_kg_m2": 0.02,
        "gravity_m_s2": [0, 0, -9.81], "filtered_contact_exclusions": CONTACT_EXCLUSIONS,
        "source_sha256": fingerprints(), "scope": "headless single-lamp comparison; no browser FPS measurement",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for backend in BACKENDS:
        print(f"Benchmarking {backend}...", flush=True)
        subprocess.run([sys.executable, "-m", "lamp_sim.benchmark", "--worker", backend,
                        "--output", str(output), "--repeats", str(args.repeats), "--speed", str(args.speed)], cwd=PROJECT, check=True)
    report(output, metadata)
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
