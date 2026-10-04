import math

import numpy as np
import pytest

from lamp_sim.backends import KinematicBackend, MujocoBackend
from lamp_sim.gestures import load_gestures
from lamp_sim.motion import MotionController


@pytest.fixture(scope="module")
def physics():
    return MujocoBackend()


def test_model_preserves_five_joints_visuals_mesh_and_effort_limits(physics):
    model = physics.model
    assert model.nq == model.nv == model.nu == 5
    assert model.nmesh == 1
    assert model.ngeom == 23  # The original 22 plus a table plane.
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], [6, 12, 9, 3.5, 3.2])
    assert model.opt.timestep == pytest.approx(1 / 240)


@pytest.mark.parametrize("pose", [
    {},
    {"base_yaw_joint": 0.4, "shoulder_pitch_joint": -0.3, "elbow_pitch_joint": -0.2, "neck_yaw_joint": 0.6, "head_pitch_joint": 0.25},
])
def test_mujoco_forward_kinematics_matches_original(pose):
    original = KinematicBackend()
    mujoco = MujocoBackend("mujoco-kinematic")
    targets = dict.fromkeys(original.names, 0.0) | pose
    original.advance(targets, 1 / 30)
    mujoco.advance(targets, 1 / 30)
    np.testing.assert_allclose(mujoco.emitter_transform(), original.emitter_transform(), atol=2e-6)
    assert mujoco.positions == original.positions


def test_source_contact_overlap_is_reproduced_and_exclusion_is_specific(physics):
    source = MujocoBackend("mujoco-source")
    assert source.diagnostics.penetration_m == pytest.approx(0.001, abs=1e-8)
    assert "lamp_head_link / upper_arm_link" in source.diagnostics.pairs
    physics.reset()
    assert physics.diagnostics.contacts == 0
    assert physics.model.nexclude == 1
    assert source.model.nexclude == 0
    # Other robot collision proxies and the ground still participate in contacts.
    assert np.count_nonzero(physics.model.geom_contype) == 7


def test_physics_tracks_targets_through_dynamics_and_respects_effort(physics):
    physics.reset()
    targets = dict.fromkeys(physics.names, 0.0) | {"base_yaw_joint": 0.5}
    actual = physics.advance(targets, 1 / 30)
    assert 0 < actual["base_yaw_joint"] < 0.5
    for _ in range(90):
        physics.advance(targets, 1 / 30)
        assert np.all(np.abs(physics.data.actuator_force) <= physics.model.actuator_forcerange[:, 1] + 1e-10)
    assert physics.positions["base_yaw_joint"] == pytest.approx(0.5, abs=0.02)
    assert physics.diagnostics.warnings == 0
    np.testing.assert_allclose(physics.data.body("base_link").xpos, [0, 0, 0])


def test_fractional_control_ticks_do_not_lose_physics_time(physics):
    physics.reset()
    targets = dict.fromkeys(physics.names, 0.0)
    for _ in range(100):
        physics.advance(targets, 0.001)
    assert physics.time == pytest.approx(0.1)
    assert physics._pending_time < 1e-10


def test_pose_transfer_and_target_clamping(physics):
    physics.reset({"base_yaw_joint": 0.4})
    assert physics.positions["base_yaw_joint"] == 0.4
    desired = dict.fromkeys(physics.names, 100.0)
    physics.advance(desired, 0.0)
    for name, index in zip(physics.names, physics.a_indices):
        assert physics.data.ctrl[index] == physics.limits[name].upper


def test_collision_ablation_exposes_head_tracking_problem():
    errors = {}
    for name in ("mujoco-source", "mujoco"):
        backend = MujocoBackend(name)
        motion = MotionController(backend.limits)
        motion.set_speed(0.5)
        motion.play(load_gestures(backend.names)["nod"])
        errors[name] = 0.0
        for _ in range(120):
            desired = motion.step(1 / 30)
            actual = backend.advance(desired, 1 / 30)
            errors[name] = max(errors[name], abs(actual["head_pitch_joint"] - desired["head_pitch_joint"]))
    assert math.degrees(errors["mujoco-source"]) > 10
    assert math.degrees(errors["mujoco"]) < 5


@pytest.mark.parametrize("gesture", ["scan", "bow", "excited", "reach_hold", "limit_sweep", "idle_hold"])
def test_new_gestures_complete_without_physics_warnings(physics, gesture):
    physics.reset()
    motion = MotionController(physics.limits)
    motion.set_speed(0.5)
    motion.play(load_gestures(physics.names)[gesture])
    for _ in range(5000):
        desired = motion.step(1 / 30)
        physics.advance(desired, 1 / 30)
        assert physics.diagnostics.warnings == 0
        if not motion.moving:
            break
    assert not motion.moving
    assert np.all(np.isfinite(physics.data.qpos))


def test_synchronize_cancels_old_waypoints_and_uses_observed_pose():
    backend = KinematicBackend()
    motion = MotionController(backend.limits)
    motion.play(load_gestures(backend.names)["shake"])
    motion.synchronize(dict.fromkeys(backend.names, 0.1))
    assert not motion.moving
    assert all(v == 0.1 for v in motion.positions.values())
