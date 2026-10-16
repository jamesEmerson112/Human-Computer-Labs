"""Emit fresh Python-controller reference traces for the Rust parity test.

Run through the simulation's locked Python environment; no saved golden results
can become stale when an authored gesture changes.
"""

from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lamp_sim.gestures import load_gestures, parse_gesture
from lamp_sim.motion import MotionController, load_limits


def snapshot(motion):
    return {
        "positions": list(motion.positions[name] for name in names),
        "targets": list(motion.targets[name] for name in names),
        "moving": motion.moving,
        "activity": motion.activity,
    }


limits = load_limits()
names = sorted(limits)
gestures = load_gestures(limits)
print(json.dumps({"limits": {name: asdict(limit) for name, limit in limits.items()},
                  "gesture_names": sorted(gestures)}))
for gesture in gestures.values():
    for start in ("neutral", "lower", "upper"):
        for speed in (0.5, 2.0, 3.0):
            motion = MotionController(limits)
            if start != "neutral":
                motion.synchronize({
                    name: limit.lower + 0.01 if start == "lower" else limit.upper - 0.01
                    for name, limit in limits.items()
                })
            origin = motion.positions.copy()
            motion.set_speed(speed)
            motion.play(gesture)
            frames = []
            while motion.moving:
                motion.step(1 / 30)
                frames.append(snapshot(motion))
                if len(frames) > 6000:
                    raise RuntimeError(f"Gesture did not finish: {gesture.name}")
            motion.synchronize(origin)
            motion.set_speed(max(speed, 1.0))
            motion.play(gesture)
            motion.step(0.3)
            interrupted = snapshot(motion)
            motion.stop()
            motion.step(60)
            stopped = snapshot(motion)
            print(json.dumps({
                "name": gesture.name, "start": start, "speed": speed, "origin": origin,
                "frames": frames, "interrupted": interrupted, "stopped": stopped,
            }, separators=(",", ":")))

# Uneven dt values cross segment boundaries and preserve remaining time; changing
# pace mid-segment must not jump. Synchronization represents an observed physics
# pose, including a small excursion beyond a soft limit.
motion = MotionController(limits)
actions = [
    {"targets": {"head_pitch_joint": 0.1, "base_yaw_joint": 0.3}},
    {"dt": 0.1}, {"speed": 0.25}, {"dt": 0.47}, {"speed": 1.0},
    {"play": "nod"}, {"dt": 0.0}, {"dt": 0.36},
    {"targets": {"base_yaw_joint": -0.4}}, {"dt": 0.23},
    {"stop": True}, {"dt": 100.0}, {"reset": True}, {"dt": 3.0},
    {"synchronize": {name: limit.upper + 0.02 for name, limit in limits.items()}},
    {"play": "nod"}, {"dt": 0.613}, {"dt": 10.0},
    {"reset": True}, {"dt": 10.0},
    {"authored": {
        "schema_version": 1, "name": "pace_test", "label": "Pace test", "units": "radians",
        "steps": [
            {"targets": {"base_yaw_joint": 0.1}, "duration_seconds": 1.0},
            {"hold_seconds": 0.4},
            {"targets": {"base_yaw_joint": 1.3}},
            {"return_to_start": True, "duration_seconds": 2.0},
        ],
    }},
    {"dt": 0.2}, {"speed": 2.0}, {"dt": 0.0}, {"dt": 0.713},
    {"speed": 0.25}, {"dt": 0.351}, {"speed": 1.3}, {"dt": 0.97},
    {"speed": 3.0}, {"dt": 0.43}, {"dt": 10.0},
]
for action in actions:
    if "targets" in action:
        motion.set_targets(action["targets"])
    elif "speed" in action:
        motion.set_speed(action["speed"])
    elif "play" in action:
        motion.play(gestures[action["play"]])
    elif "authored" in action:
        motion.play(parse_gesture(action["authored"], limits))
    elif "dt" in action:
        motion.step(action["dt"])
    elif "stop" in action:
        motion.stop()
    elif "reset" in action:
        motion.reset()
    elif "synchronize" in action:
        motion.synchronize(action["synchronize"])
    print(json.dumps({"action": action, "expected": snapshot(motion)}, separators=(",", ":")))
