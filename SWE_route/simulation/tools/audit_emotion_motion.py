"""Record controller checks and similarity candidates for the selected wheel.

This report never approves emotional meaning or edits a review. Optional physics
results must come from a fresh, hash-matched Python benchmark of this library.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lamp_sim.benchmark import fingerprints
from lamp_sim.gestures import Gesture, load_gestures
from lamp_sim.motion import MotionController, load_limits


DT = 1 / 30
NEAR_RMSE_DEG = 2.0
NEAR_MAX_DEG = 6.0
NEAR_DURATION_RATIO = 1.25


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motion_signature(gesture: Gesture) -> str:
    data = [{**asdict(step), "angles": sorted(step.angles)} for step in gesture.steps]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def command_trial(gesture: Gesture, start: str, speed: float) -> tuple[dict, np.ndarray]:
    limits = load_limits()
    motion = MotionController(limits)
    if start != "neutral":
        motion.synchronize({name: limit.lower + 0.01 if start == "lower" else limit.upper - 0.01
                            for name, limit in limits.items()})
    origin = motion.positions.copy()
    motion.set_speed(speed)
    motion.play(gesture)
    frames = [list(origin.values())]
    violation = speed_ratio = 0.0
    for _ in range(1800):
        before = motion.positions.copy()
        pose = motion.step(DT)
        for name, value in pose.items():
            limit = limits[name]
            violation = max(violation, limit.lower - value, value - limit.upper)
            speed_ratio = max(speed_ratio, abs(value - before[name]) / (limit.velocity * speed * DT))
        frames.append(list(pose.values()))
        if not motion.moving:
            break
    trace = np.array(frames)
    moved = bool(np.max(np.abs(trace - trace[0])) > 1e-12)
    returned = motion.positions == origin
    failures = []
    if motion.moving:
        failures.append("did not complete within 60 simulation seconds")
    if not np.all(np.isfinite(trace)):
        failures.append("nonfinite requested angles")
    if violation > 1e-12 or speed_ratio > 1 + 1e-10:
        failures.append("requested position or speed exceeded its envelope")
    if gesture.group and (not moved or not returned):
        failures.append("emotion must move and return to its starting pose")

    # Interrupt an independently restarted clip, including near-limit poses.
    motion.synchronize(origin)
    motion.play(gesture)
    motion.step(0.3)
    motion.stop()
    held = motion.positions.copy()
    if motion.step(60) != held or motion.moving:
        failures.append("Stop did not clear requested motion")
    return {"start": start, "pace": speed, "frames": len(trace) - 1,
            "duration_s": (len(trace) - 1) * DT, "moved": moved,
            "returned_to_start": returned, "max_requested_speed_ratio": speed_ratio,
            "max_requested_limit_violation_rad": violation, "failures": failures}, trace


def reset_checks() -> list[dict]:
    limits = load_limits()
    result = []
    for start in ("lower", "upper"):
        motion = MotionController(limits)
        motion.synchronize({name: limit.lower + 0.01 if start == "lower" else limit.upper - 0.01
                            for name, limit in limits.items()})
        motion.reset()
        for _ in range(1800):
            motion.step(DT)
            if not motion.moving:
                break
        result.append({"start": start, "passed": not motion.moving and all(
            value == limits[name].clamp(0.0) for name, value in motion.positions.items())})
    return result


def similarity(gestures: dict[str, Gesture], traces: dict[str, np.ndarray]) -> tuple[list, list, dict]:
    emotions = [gesture for gesture in gestures.values() if gesture.group]
    signatures: dict[str, list[str]] = {}
    resampled = {}
    for gesture in emotions:
        signatures.setdefault(motion_signature(gesture), []).append(gesture.name)
        trace = traces[gesture.name]
        resampled[gesture.name] = np.array([
            np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(trace)), trace[:, joint])
            for joint in range(trace.shape[1])
        ]).T
    exact = [names for names in signatures.values() if len(names) > 1]
    candidates = []
    nearest: dict[str, list[dict]] = {gesture.name: [] for gesture in emotions}
    for first, second in itertools.combinations(emotions, 2):
        delta = np.rad2deg(resampled[first.name] - resampled[second.name])
        rmse = float(np.sqrt(np.mean(delta**2)))
        maximum = float(np.max(np.abs(delta)))
        durations = [(len(traces[name]) - 1) * DT for name in (first.name, second.name)]
        ratio = max(durations) / min(durations)
        metrics = {"rmse_deg": rmse, "max_difference_deg": maximum, "duration_ratio": ratio}
        if rmse <= NEAR_RMSE_DEG and maximum <= NEAR_MAX_DEG and ratio <= NEAR_DURATION_RATIO:
            candidates.append({"names": [first.name, second.name], **metrics})
        if first.group == second.group:
            nearest[first.name].append({"name": second.name, **metrics})
            nearest[second.name].append({"name": first.name, **metrics})
    nearest = {name: sorted(values, key=lambda item: item["rmse_deg"])[:3] for name, values in nearest.items()}
    return exact, sorted(candidates, key=lambda item: item["rmse_deg"]), nearest


def physics_results(reference: Path, names: set[str]) -> dict:
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    current = fingerprints()
    mismatched = [path for path, revision in current.items() if manifest.get("source_sha256", {}).get(path) != revision]
    if mismatched:
        raise ValueError(f"Physics reference does not match the current library/model: {mismatched[:5]}")
    if manifest.get("speed") != 0.5 or manifest.get("control_hz") != 30 or manifest.get("physics_hz") != 240:
        raise ValueError("Physics reference must use pace 0.5, 30 Hz control and 240 Hz physics")
    data = json.loads((reference / "mujoco.json").read_text(encoding="utf-8"))
    trials = data["trials"]
    if {trial["gesture"] for trial in trials} != names:
        raise ValueError("Physics reference is missing current gestures or contains retired gestures")
    keys = [(trial["gesture"], trial["repeat"]) for trial in trials]
    if len(set(keys)) != len(keys) or set(keys) != set(itertools.product(names, range(1, manifest["repeats"] + 1))):
        raise ValueError("Physics reference has incomplete or duplicate repetitions")
    fields = ["gesture", "repeat", "frames", "tracking_rmse_deg", "tracking_max_deg",
              "final_error_max_deg", "soft_limit_violation_max_deg", "max_velocity_ratio",
              "contact_frames", "max_penetration_mm", "contact_pairs", "physics_warning_count"]
    return {"reference": str(reference.relative_to(PROJECT)) if reference.is_relative_to(PROJECT) else str(reference),
            "reference_manifest_sha256": digest(reference / "manifest.json"),
            "reference_results_sha256": digest(reference / "mujoco.json"),
            "scope": "Default MuJoCo, neutral starts, unchanged gains/contact policy; actual angles may overshoot.",
            "trials": [{key: trial[key] for key in fields} for trial in trials]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-reference", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory to preserve earlier evidence")
    limits = load_limits()
    gestures = load_gestures(limits)
    trials, traces = {}, {}
    for gesture in gestures.values():
        trials[gesture.name] = []
        for start, speed in itertools.product(("neutral", "lower", "upper"), (0.5, 1.0)):
            trial, trace = command_trial(gesture, start, speed)
            trials[gesture.name].append(trial)
            if start == "neutral" and speed == 0.5:
                traces[gesture.name] = trace
    exact, near, nearest = similarity(gestures, traces)
    resets = reset_checks()
    failed = [name for name, cases in trials.items() if any(case["failures"] for case in cases)]
    physics = physics_results(args.physics_reference.resolve(), set(gestures)) if args.physics_reference else None
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Technical audit and human-review candidates; never human animation approval",
              "catalog_count": len(gestures), "emotion_count": sum(bool(g.group) for g in gestures.values()),
              "source_sha256": fingerprints(), "harness_sha256": digest(Path(__file__)),
              "control_hz": 30, "cases": sum(map(len, trials.values())), "trials": trials,
              "failed_gestures": failed, "reset_checks": resets,
              "similarity_method": {"source": "requested neutral trajectories at pace 0.5",
                                    "phase_samples": 128, "rmse_deg_at_most": NEAR_RMSE_DEG,
                                    "max_difference_deg_at_most": NEAR_MAX_DEG,
                                    "duration_ratio_at_most": NEAR_DURATION_RATIO,
                                    "meaning": "Review prompts only; phase resampling can hide timing differences"},
              "identical_motion_groups": exact, "similarity_candidates": near,
              "nearest_within_family": nearest, "physics": physics}
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Animation motion audit", "",
             f"{len(gestures)} animations; {report['emotion_count']} emotions; {report['cases']} controller cases.",
             f"Command failures: {len(failed)}. Reset failures: {sum(not r['passed'] for r in resets)}.", "",
             "Controller checks cover neutral and both near-limit starting poses at Normal and Full pace, completion, requested limits/speed, and interruption. Emotion clips must move and return to their starting pose. Utility clips have their original hold/reach behavior.", "",
             "## Similarity review candidates", "",
             f"{len(exact)} identical step groups and {len(near)} close trajectory pairs. These flags do not assign human pass/fail.", "",
             "The comparison resamples neutral requested trajectories to 128 phase samples. A pair is flagged at RMS difference <=2 degrees, maximum difference <=6 degrees, and duration ratio <=1.25. It can miss perceptual similarity and timing distinctions. Full nearest-neighbor results are in audit.json.", "",
             "| First animation | Second animation | RMS difference (degrees) | Duration ratio |",
             "|---|---|---:|---:|"]
    for pair in near:
        first, second = pair["names"]
        lines.append(f"| {gestures[first].label} | {gestures[second].label} | {pair['rmse_deg']:.3f} | {pair['duration_ratio']:.3f} |")
    if physics:
        observations = physics["trials"]
        lines += ["", "## Actual physics observations", "",
                  f"Hash-matched reference: `{physics['reference']}`. {len(observations)} neutral-start default-MuJoCo trials.", "",
                  f"Maximum actual soft-limit overshoot: {max(t['soft_limit_violation_max_deg'] for t in observations):.6f} degrees. Maximum tracking error: {max(t['tracking_max_deg'] for t in observations):.6f} degrees.",
                  f"Trials with contacts: {sum(t['contact_frames'] > 0 for t in observations)}. Trials with solver warnings: {sum(t['physics_warning_count'] > 0 for t in observations)}.", "",
                  "These observations do not establish physical tracking from near-limit starting poses, hardware accuracy, or emotional recognition. Requested-angle checks and actual physics observations are separate."]
    else:
        lines += ["", "Actual physics was not included in this report; supply a current --physics-reference."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited {len(gestures)} clips / {report['cases']} cases: {len(failed)} command failures, {len(near)} similarity candidates")
    if failed or any(not result["passed"] for result in resets):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
