import math

import numpy as np
import pytest
import yourdfpy

from lamp_sim.motion import MotionController, ROBOT_PATH, load_limits
from lamp_sim.gestures import load_gestures


def play_gesture(motion, name):
    motion.play(load_gestures(motion.limits)[name])


@pytest.fixture
def motion():
    return MotionController(load_limits())


def test_model_limits_and_joint_count():
    limits = load_limits()
    assert len(limits) == 5
    assert limits["shoulder_pitch_joint"].lower == -0.65
    assert limits["shoulder_pitch_joint"].upper == 0.95
    assert limits["shoulder_pitch_joint"].velocity == 0.95


@pytest.mark.parametrize("target", [-100.0, 100.0])
def test_all_joints_reach_clamped_targets_without_exceeding_speed(motion, target):
    motion.set_targets(dict.fromkeys(motion.limits, target))
    for _ in range(300):
        previous = motion.positions.copy()
        current = motion.step(1 / 120)
        for name, value in current.items():
            limit = motion.limits[name]
            assert limit.lower <= value <= limit.upper
            assert abs(value - previous[name]) <= limit.velocity / 120 + 1e-12
    assert not motion.moving
    for name, value in motion.positions.items():
        limit = motion.limits[name]
        assert value == (limit.upper if target > 0 else limit.lower)


def test_stop_cancels_nod_return_and_holds_pose(motion):
    play_gesture(motion, "nod")
    motion.step(0.1)
    assert motion.positions["head_pitch_joint"] > 0
    motion.stop()
    frozen = motion.positions.copy()
    assert motion.step(10) == frozen
    assert not motion.moving
    assert motion.targets == frozen


def test_nod_returns_to_nonzero_pose_even_near_limit(motion):
    motion.set_targets({"head_pitch_joint": 0.59, "base_yaw_joint": 0.3})
    motion.step(10)
    origin = motion.positions.copy()
    play_gesture(motion, "nod")
    motion.step(0.2)
    assert motion.positions["head_pitch_joint"] < origin["head_pitch_joint"]
    motion.step(10)
    assert motion.positions == origin


def test_manual_target_interrupts_nod_and_reset_returns_zero(motion):
    play_gesture(motion, "nod")
    motion.step(0.1)
    motion.set_targets({"head_pitch_joint": -0.2})
    motion.step(10)
    assert motion.positions["head_pitch_joint"] == -0.2
    motion.reset()
    motion.step(10)
    assert all(v == 0 for v in motion.positions.values())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_target_rejected_without_changing_motion(motion, value):
    before = motion.targets.copy()
    with pytest.raises(ValueError):
        motion.set_targets({"base_yaw_joint": value})
    assert motion.targets == before


def test_unknown_joint_and_invalid_time_rejected(motion):
    with pytest.raises(ValueError):
        motion.set_targets({"not_a_joint": 0.1})
    for dt in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError):
            motion.step(dt)


@pytest.mark.parametrize("gesture", ["nod", "shake", "curious"])
@pytest.mark.parametrize("near_limit", [False, True])
def test_gestures_stay_bounded_and_return_to_start(motion, gesture, near_limit):
    if near_limit:
        motion.set_targets({n: lim.upper - 0.01 for n, lim in motion.limits.items()})
        motion.step(10)
    origin = motion.positions.copy()
    motion.set_speed(0.5)
    play_gesture(motion, gesture)
    changed = False
    for _ in range(900):
        before = motion.positions.copy()
        current = motion.step(1 / 120)
        changed |= current != origin
        for name, value in current.items():
            limit = motion.limits[name]
            assert limit.lower <= value <= limit.upper
            assert abs(value - before[name]) <= limit.velocity * motion.speed / 120 + 1e-12
        if not motion.moving:
            break
    assert changed
    assert not motion.moving
    assert motion.positions == origin


@pytest.mark.parametrize("gesture", ["shake", "curious"])
def test_stop_cancels_compound_gesture(motion, gesture):
    play_gesture(motion, gesture)
    motion.step(0.1)
    motion.stop()
    held = motion.positions.copy()
    assert motion.step(30) == held
    assert not motion.moving


def test_slow_playback_scales_time_and_changing_pace_does_not_jump(motion):
    normal = MotionController(load_limits())
    targets = {"base_yaw_joint": 1.0}
    normal.set_targets(targets)
    motion.set_targets(targets)
    motion.set_speed(0.25)
    assert motion.step(0.4) == normal.step(0.1)
    held = motion.positions.copy()
    motion.set_speed(1.0)
    assert motion.positions == held
    motion.step(10)
    assert motion.positions["base_yaw_joint"] == 1.0


@pytest.mark.parametrize("speed", [0.0, -1.0, 1.1, math.inf, math.nan])
def test_invalid_playback_speed_is_rejected(motion, speed):
    with pytest.raises(ValueError):
        motion.set_speed(speed)
    assert motion.speed == 1.0


def test_urdf_mesh_loads_and_light_direction_follows_head():
    model = yourdfpy.URDF.load(str(ROBOT_PATH), mesh_dir=str(ROBOT_PATH.parent))
    shade = next(link for link in model.robot.links if link.name == "lamp_head_link")
    assert shade.visuals[0].geometry.mesh.filename == "assets/lamp_shade.stl"
    assert any(len(mesh.faces) == 360 for mesh in model.scene.geometry.values())
    light = model.get_transform("light_emitter_link", "base_link")
    np.testing.assert_allclose(light[:3, 0], [-1, 0, 0], atol=1e-10)
    model.update_cfg({"head_pitch_joint": 0.3})
    tilted = model.get_transform("light_emitter_link", "base_link")
    assert tilted[2, 0] < 0  # Positive pitch sends the beam down toward the table.
    np.testing.assert_allclose(model.get_transform("base_link"), np.eye(4))
