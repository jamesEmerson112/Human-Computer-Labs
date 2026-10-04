"""Exercise the authored emotion clips through the production motion controller."""

import pytest

from lamp_sim.gestures import load_gestures
from lamp_sim.motion import MotionController, load_limits


LIMITS = load_limits()
EMOTIONS = [gesture for gesture in load_gestures(LIMITS).values() if gesture.group]


@pytest.mark.parametrize("gesture", EMOTIONS, ids=lambda gesture: gesture.name)
@pytest.mark.parametrize("start", ["neutral", "lower", "upper"])
def test_emotion_moves_within_limits_and_returns_to_start(gesture, start):
    motion = MotionController(LIMITS)
    if start != "neutral":
        motion.synchronize({
            name: limit.lower + 0.01 if start == "lower" else limit.upper - 0.01
            for name, limit in LIMITS.items()
        })
    origin = motion.positions.copy()
    motion.set_speed(0.5)
    motion.play(gesture)
    moved = False
    for _ in range(1800):
        before = motion.positions.copy()
        current = motion.step(1 / 30)
        moved |= current != origin
        for name, value in current.items():
            limit = LIMITS[name]
            assert limit.lower - 1e-12 <= value <= limit.upper + 1e-12
            assert abs(value - before[name]) <= limit.velocity * motion.speed / 30 + 1e-12
        if not motion.moving:
            break
    assert moved
    assert not motion.moving, "Emotion clip did not finish within 60 seconds at Normal pace"
    assert motion.positions == origin


@pytest.mark.parametrize("gesture", EMOTIONS, ids=lambda gesture: gesture.name)
def test_emotion_can_be_interrupted(gesture):
    motion = MotionController(LIMITS)
    motion.play(gesture)
    motion.step(0.3)
    motion.stop()
    held = motion.positions.copy()
    assert motion.step(60) == held
    assert not motion.moving
