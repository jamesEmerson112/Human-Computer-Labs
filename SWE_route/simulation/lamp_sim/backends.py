"""Interchangeable pose playback and CPU physics for the challenge's own URDF."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .motion import ROBOT_PATH, load_limits


PHYSICS_HZ = 240
CONTROL_HZ = 30
# Assumed simulation gains, not specifications for a real LeLamp servo.
GAINS = {
    "base_yaw_joint": (80.0, 4.0),
    "shoulder_pitch_joint": (120.0, 8.0),
    "elbow_pitch_joint": (100.0, 6.0),
    "neck_yaw_joint": (40.0, 2.0),
    "head_pitch_joint": (25.0, 1.5),
}
CONTACT_EXCLUSIONS = [("upper_arm_link", "lamp_head_link")]
BACKEND_NAMES = ("kinematic", "mujoco-kinematic", "mujoco", "mujoco-source")


@dataclass
class Diagnostics:
    contacts: int | None = None
    penetration_m: float | None = None
    contact_force_n: float | None = None
    pairs: set[str] = field(default_factory=set)
    max_effort_nm: float | None = None
    warnings: int = 0


class KinematicBackend:
    name = "kinematic"

    def __init__(self):
        import yourdfpy
        self.limits = load_limits()
        self.names = tuple(self.limits)
        self.model = yourdfpy.URDF.load(str(ROBOT_PATH), load_meshes=False)
        self.reset()

    def reset(self, positions: dict[str, float] | None = None) -> None:
        self.positions = {n: self.limits[n].clamp((positions or {}).get(n, 0.0)) for n in self.names}
        self.model.update_cfg(self.positions)
        self.velocity = dict.fromkeys(self.names, 0.0)
        self.diagnostics = Diagnostics()
        self.time = 0.0

    def advance(self, targets: dict[str, float], dt: float) -> dict[str, float]:
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        bounded = {n: self.limits[n].clamp(targets[n]) for n in self.names}
        self.velocity = {n: (bounded[n] - self.positions[n]) / dt if dt else 0.0 for n in self.names}
        self.positions = bounded
        self.model.update_cfg(bounded)
        self.time += dt
        return bounded.copy()

    def emitter_transform(self) -> np.ndarray:
        return self.model.get_transform("light_emitter_link", "base_link").copy()


def build_mujoco_model(filtered: bool = True):
    """Convert in memory; temporary MJCF serialization never changes the fixture.

    Binary STL and retained visual/fixed frames are import adaptations. The
    sole optional self-contact exclusion is explicit and benchmarked separately.
    """
    import mujoco
    import trimesh

    source = ET.parse(ROBOT_PATH).getroot()
    extension = ET.SubElement(source, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false")
    mesh = trimesh.load(ROBOT_PATH.parent / "assets/lamp_shade.stl", force="mesh")
    assets = {"lamp_shade.stl": mesh.export(file_type="stl")}
    imported = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"), assets)
    cache = Path(__file__).resolve().parents[1] / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".xml", delete=False) as handle:
        path = Path(handle.name)
    try:
        mujoco.mj_saveLastXML(str(path), imported)
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)
    ET.SubElement(root, "option", timestep=str(1 / PHYSICS_HZ), integrator="implicitfast", gravity="0 0 -9.81")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="plane", group="1", size="2 2 0.1", rgba="0.2 0.25 0.3 1")
    ET.SubElement(world, "light", name="room_light", pos="0 -1 2", dir="0 0 -1")
    emitter = world.find(".//body[@name='light_emitter_link']")
    ET.SubElement(emitter, "light", name="lamp_light", type="spot", pos="0 0 0", dir="1 0 0", cutoff="24", diffuse="1 0.88 0.64")
    limits = load_limits()
    actuators = ET.SubElement(root, "actuator")
    efforts = {j.attrib["name"]: j.find("limit").attrib["effort"] for j in source.findall("joint") if j.get("type") == "revolute"}
    for joint in root.findall(".//joint"):
        if joint.get("name") in limits:
            joint.set("armature", "0.02")
    for name, limit in limits.items():
        kp, kv = GAINS[name]
        ET.SubElement(
            actuators, "position", name=name + "_servo", joint=name,
            kp=str(kp), kv=str(kv), ctrlrange=f"{limit.lower} {limit.upper}",
            forcerange=f"-{efforts[name]} {efforts[name]}",
        )
    if filtered:
        contact = ET.SubElement(root, "contact")
        for first, second in CONTACT_EXCLUSIONS:
            ET.SubElement(contact, "exclude", body1=first, body2=second)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return model


class MujocoBackend:
    def __init__(self, name: str = "mujoco"):
        import mujoco
        if name not in BACKEND_NAMES[1:]:
            raise ValueError(f"Unknown MuJoCo backend: {name}")
        self.name = name
        self.limits = load_limits()
        self.names = tuple(self.limits)
        self.model = build_mujoco_model(filtered=name != "mujoco-source")
        self.data = mujoco.MjData(self.model)
        self.q_indices = np.array([self.model.joint(n).qposadr[0] for n in self.names])
        self.v_indices = np.array([self.model.joint(n).dofadr[0] for n in self.names])
        self.a_indices = np.array([self.model.actuator(n + "_servo").id for n in self.names])
        self.emitter_id = self.model.body("light_emitter_link").id
        self.reset()

    def reset(self, positions: dict[str, float] | None = None) -> None:
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        pose = np.array([self.limits[n].clamp((positions or {}).get(n, 0.0)) for n in self.names])
        self.data.qpos[self.q_indices] = pose
        self.data.ctrl[self.a_indices] = pose
        mujoco.mj_forward(self.model, self.data)
        self._pending_time = 0.0
        self.time = 0.0
        self.positions = dict(zip(self.names, map(float, pose)))
        self.velocity = dict.fromkeys(self.names, 0.0)
        self.diagnostics = Diagnostics(0, 0.0, 0.0, max_effort_nm=0.0)
        self._observe_contacts()

    def _observe_contacts(self) -> None:
        import mujoco
        diag = self.diagnostics
        diag.contacts = max(diag.contacts, self.data.ncon)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            diag.penetration_m = max(diag.penetration_m, max(0.0, -float(contact.dist)))
            bodies = [self.model.body(self.model.geom_bodyid[g]).name or "world" for g in (contact.geom1, contact.geom2)]
            diag.pairs.add(" / ".join(sorted(bodies)))
            if self.name != "mujoco-kinematic":
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                diag.contact_force_n = max(diag.contact_force_n, float(np.linalg.norm(force[:3])))
        diag.max_effort_nm = max(diag.max_effort_nm, float(np.max(np.abs(self.data.actuator_force))))
        diag.warnings = sum(int(w.number) for w in self.data.warning)

    def advance(self, targets: dict[str, float], dt: float) -> dict[str, float]:
        import mujoco
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        desired = np.array([self.limits[n].clamp(targets[n]) for n in self.names])
        self.data.ctrl[self.a_indices] = desired
        self.diagnostics = Diagnostics(0, 0.0, 0.0, max_effort_nm=0.0)
        if self.name == "mujoco-kinematic":
            before = self.data.qpos[self.q_indices].copy()
            self.data.qpos[self.q_indices] = desired
            self.data.qvel[self.v_indices] = (desired - before) / dt if dt else 0.0
            self.data.time += dt
            mujoco.mj_forward(self.model, self.data)
            self._observe_contacts()
            # No physical force interpretation when we directly assign poses.
            self.diagnostics.contact_force_n = None
            self.diagnostics.max_effort_nm = None
        else:
            self._pending_time += dt
            while self._pending_time + 1e-12 >= self.model.opt.timestep:
                mujoco.mj_step(self.model, self.data)
                self._observe_contacts()
                self._pending_time = max(0.0, self._pending_time - self.model.opt.timestep)
            # mj_step leaves derived poses at the pre-integration state.
            mujoco.mj_forward(self.model, self.data)
        self.positions = dict(zip(self.names, map(float, self.data.qpos[self.q_indices])))
        self.velocity = dict(zip(self.names, map(float, self.data.qvel[self.v_indices])))
        self.time = float(self.data.time)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise RuntimeError("MuJoCo produced a nonfinite state")
        return self.positions.copy()

    def emitter_transform(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.data.xmat[self.emitter_id].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[self.emitter_id]
        return transform


def make_backend(name: str):
    if name not in BACKEND_NAMES:
        raise ValueError(f"Unknown backend: {name}")
    return KinematicBackend() if name == "kinematic" else MujocoBackend(name)
