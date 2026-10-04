import json
import math

import pytest

from lamp_sim.gestures import load_gesture, load_gestures, parse_gesture
from lamp_sim.motion import MotionController, load_limits


def definition(steps, **changes):
    return {
        "schema_version": 1, "name": "test", "label": "Test", "units": "radians",
        "steps": steps, **changes,
    }


def play(motion, steps):
    motion.play(parse_gesture(definition(steps), motion.limits))


def test_offsets_reference_start_but_omitted_joints_keep_previous_waypoint():
    motion = MotionController(load_limits())
    motion.set_targets({"head_pitch_joint": 0.1})
    motion.step(10)
    play(motion, [
        {"offsets": {"head_pitch_joint": 0.2}},
        {"targets": {"base_yaw_joint": 0.4}},
        {"offsets": {"head_pitch_joint": -0.2}},
    ])
    motion.step(0.25)
    assert motion.positions["head_pitch_joint"] == pytest.approx(0.3)
    motion.step(0.4)
    assert motion.positions["base_yaw_joint"] == pytest.approx(0.4)
    assert motion.positions["head_pitch_joint"] == pytest.approx(0.3)
    motion.step(10)
    assert motion.positions["head_pitch_joint"] == pytest.approx(-0.1)
    assert motion.positions["base_yaw_joint"] == 0.4


def test_hold_is_exact_and_does_not_get_minimum_move_duration():
    motion = MotionController(load_limits())
    play(motion, [
        {"targets": {"head_pitch_joint": 0.1}, "duration_seconds": 0.5},
        {"hold_seconds": 0.05},
        {"return_to_start": True},
    ])
    motion.step(0.5)
    assert motion.positions["head_pitch_joint"] == 0.1
    motion.step(0.04)
    assert motion.positions["head_pitch_joint"] == 0.1
    motion.step(0.02)
    assert motion.positions["head_pitch_joint"] < 0.1
    motion.step(10)
    assert motion.positions["head_pitch_joint"] == 0.0


def test_large_dt_crosses_move_hold_and_return_without_losing_time():
    motion = MotionController(load_limits())
    play(motion, [
        {"targets": {"head_pitch_joint": 0.1}},
        {"hold_seconds": 0.1},
        {"return_to_start": True},
    ])
    motion.step(0.61)
    assert not motion.moving
    assert motion.positions["head_pitch_joint"] == 0.0


def test_requested_duration_is_stretched_to_obey_velocity():
    motion = MotionController(load_limits())
    play(motion, [{"targets": {"base_yaw_joint": 2.0}, "duration_seconds": 0.01}])
    for _ in range(200):
        before = motion.positions["base_yaw_joint"]
        motion.step(0.01)
        assert abs(motion.positions["base_yaw_joint"] - before) <= 0.015 + 1e-12
    assert motion.positions["base_yaw_joint"] == pytest.approx(2.0)


def test_absolute_targets_are_clamped_and_return_is_optional():
    motion = MotionController(load_limits())
    play(motion, [{"targets": {"head_pitch_joint": 99}}])
    motion.step(10)
    assert motion.positions["head_pitch_joint"] == 0.6


@pytest.mark.parametrize("start,expected", [(0.59, 0.29), (-0.79, -0.49)])
def test_reverse_policy_handles_both_bounds(start, expected):
    motion = MotionController(load_limits())
    motion.set_targets({"head_pitch_joint": start})
    motion.step(10)
    offset = 0.3 if start > 0 else -0.3
    play(motion, [{"offsets": {"head_pitch_joint": offset}, "on_limit": "reverse"}])
    motion.step(10)
    assert motion.positions["head_pitch_joint"] == pytest.approx(expected)


@pytest.mark.parametrize("steps", [
    [], [{"targets": {}}], [{"targets": {"typo_joint": 0.1}}],
    [{"targets": {"head_pitch_joint": True}}],
    [{"targets": {"head_pitch_joint": "0.3"}}],
    [{"targets": {"head_pitch_joint": math.nan}}],
    [{"offsets": {"head_pitch_joint": math.inf}}],
    [{"targets": {"head_pitch_joint": 0.1}, "hold_seconds": 1}],
    [{"hold_seconds": 0}], [{"hold_seconds": -1}], [{"hold_seconds": 61}],
    [{"hold_seconds": True}], [{"hold_seconds": math.nan}],
    [{"return_to_start": False}], [{"return_to_start": 1}],
    [{"offsets": {"head_pitch_joint": 0.3}, "on_limit": "ignore"}],
    [{"targets": {"head_pitch_joint": 0.3}, "on_limit": "reverse"}],
    [{"target": {"head_pitch_joint": 0.3}}],
    [{"hold_seconds": 0.1}] * 129,
])
def test_invalid_steps_are_rejected(steps):
    with pytest.raises(ValueError):
        parse_gesture(definition(steps), load_limits())


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"schema_version": True}, {"units": "degrees"},
    {"name": "../nod"}, {"label": " "}, {"unexpected": 1},
])
def test_invalid_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        parse_gesture(definition([{"hold_seconds": 1}], **changes), load_limits())


@pytest.mark.parametrize("text", [
    '{"name": "one", "name": "two"}', '{"steps": [}',
    '{"steps": [{"hold_seconds": NaN}]}',
])
def test_file_errors_include_filename(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        load_gesture(path, load_limits())


def test_library_rejects_duplicate_ids_and_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No gesture"):
        load_gestures(load_limits(), tmp_path)
    data = definition([{"hold_seconds": 0.1}])
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate gesture name"):
        load_gestures(load_limits(), tmp_path)


def test_library_reloads_edited_files_and_discovers_new_ones(tmp_path):
    path = tmp_path / "custom.json"
    data = definition([{"targets": {"head_pitch_joint": 0.1}}])
    path.write_text(json.dumps(data), encoding="utf-8")
    original = load_gestures(load_limits(), tmp_path)
    data["steps"][0]["targets"]["head_pitch_joint"] = 0.2
    path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "another.json").write_text(json.dumps(definition(
        [{"hold_seconds": 1}], name="another", label="Another",
    )), encoding="utf-8")
    updated = load_gestures(load_limits(), tmp_path)
    assert set(updated) == {"test", "another"}
    assert dict(original["test"].steps[0].angles)["head_pitch_joint"] == 0.1
    assert dict(updated["test"].steps[0].angles)["head_pitch_joint"] == 0.2


def test_failed_reload_keeps_last_valid_library(tmp_path):
    # Exercise the real reload boundary without starting a browser/server.
    from types import SimpleNamespace
    from lamp_sim.viewer import LampViewer

    viewer = LampViewer.__new__(LampViewer)
    viewer.motion = MotionController(load_limits())
    viewer.gestures = load_gestures(viewer.motion.limits)
    viewer.gesture_notice = SimpleNamespace(content="")
    original = viewer.gestures
    with pytest.MonkeyPatch.context() as patch:
        def fail(*args):
            raise ValueError("broken.json: unknown joint")
        patch.setattr("lamp_sim.viewer.load_gestures", fail)
        viewer.reload_gestures()
    assert viewer.gestures is original
    assert "Previous gestures kept" in viewer.gesture_notice.content
    assert "broken.json" in viewer.gesture_notice.content
