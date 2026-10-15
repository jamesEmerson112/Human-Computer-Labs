"""Export the existing Python model for the native runtime, without altering fixtures.

Run with the locked Python environment. Python is a build/validation tool only;
the Rust executable loads the resulting MJCF and STL directly.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lamp_sim.backends import CONTACT_EXCLUSIONS, GAINS, build_mujoco_model
from lamp_sim.motion import ROBOT_PATH, load_limits

VERSION = "3.14.0"


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def quaternion(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = (angle / 2 for angle in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


def visual_scene() -> dict:
    root = ET.parse(ROBOT_PATH).getroot()
    materials = {item.attrib["name"]: list(map(float, item.find("color").attrib["rgba"].split()))
                 for item in root.findall("material")}
    visuals = []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            attrs = {} if origin is None else origin.attrib
            geometry = next(iter(visual.find("geometry")))
            material = visual.find("material")
            name = material.get("name", "") if material is not None else ""
            item = {"link": link.attrib["name"], "type": geometry.tag,
                    "position": list(map(float, attrs.get("xyz", "0 0 0").split())),
                    "quaternion": quaternion(list(map(float, attrs.get("rpy", "0 0 0").split()))),
                    "rgba": materials.get(name, [0.8, 0.8, 0.8, 1.0]), "material": name}
            if geometry.tag == "cylinder":
                item["size"] = [float(geometry.get("radius")), float(geometry.get("length"))]
            elif geometry.tag == "sphere":
                item["size"] = [float(geometry.get("radius"))]
            elif geometry.tag == "box":
                item["size"] = list(map(float, geometry.attrib["size"].split()))
            elif geometry.tag == "mesh":
                if geometry.attrib["filename"] != "assets/lamp_shade.stl":
                    raise ValueError("Unexpected fixture mesh; update the exporter explicitly")
                item["mesh"] = "lamp_shade.stl"
                item["scale"] = list(map(float, geometry.get("scale", "1 1 1").split()))
            else:
                raise ValueError(f"Unsupported visual: {geometry.tag}")
            visuals.append(item)
    return {"schema_version": 1, "coordinate_system": "z-up", "quaternion_order": "xyzw",
            "limits": {name: asdict(limit) for name, limit in load_limits().items()},
            "links": [link.attrib["name"] for link in root.findall("link")], "visuals": visuals}


def export_files() -> dict[str, bytes]:
    if mujoco.__version__ != VERSION:
        raise RuntimeError(f"Expected MuJoCo {VERSION}, got {mujoco.__version__}; use uv sync --locked")
    mesh = trimesh.load(ROBOT_PATH.parent / "assets/lamp_shade.stl", force="mesh")
    files = {"lamp_shade.stl": mesh.export(file_type="stl")}
    with tempfile.TemporaryDirectory(prefix="lamp-export-") as directory:
        destination = Path(directory) / "model.xml"
        for filename, filtered in (("lamp.xml", True), ("lamp-source.xml", False)):
            model = build_mujoco_model(filtered=filtered)
            mujoco.mj_saveLastXML(str(destination), model)
            xml = ET.fromstring(destination.read_bytes())
            # Preserve the exact 240 Hz step if MuJoCo's serializer rounds it.
            xml.find("option").set("timestep", repr(float(model.opt.timestep)))
            for asset in xml.findall("asset/mesh"):
                asset.set("file", Path(asset.attrib["file"]).name)
            files[filename] = ET.tostring(xml, encoding="utf-8", xml_declaration=True) + b"\n"
            reloaded = mujoco.MjModel.from_xml_string(files[filename].decode(), {"lamp_shade.stl": files["lamp_shade.stl"]})
            before, after = mujoco.MjData(model), mujoco.MjData(reloaded)
            for frame in range(240):
                target = 0.15 * math.sin(frame / 30)
                before.ctrl[:] = target
                after.ctrl[:] = target
                mujoco.mj_step(model, before)
                mujoco.mj_step(reloaded, after)
                np.testing.assert_allclose(before.qpos, after.qpos, rtol=0, atol=1e-12,
                                           err_msg=f"Serialized model differs: {filename}, frame {frame}")
    files["scene.json"] = json_bytes(visual_scene())
    sources = [ROBOT_PATH, ROBOT_PATH.parent / "assets/lamp_shade.stl",
               PROJECT / "lamp_sim/backends.py", PROJECT / "lamp_sim/motion.py",
               PROJECT / "uv.lock", Path(__file__)]
    files["manifest.json"] = json_bytes({
        "schema_version": 1, "mujoco_version": VERSION,
        "generator": "uv run --locked python tools/export_native_model.py",
        "source_sha256": {str(path.relative_to(PROJECT.parent)).replace("\\", "/"):
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "generated_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "physics_hz": 240, "control_hz": 30, "servo_gains_kp_kv": GAINS,
        "armature_kg_m2": 0.02, "filtered_contact_exclusions": CONTACT_EXCLUSIONS,
        "notes": "Derived build assets. The supplied URDF/STL are unchanged. Python is not required at runtime.",
    })
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed exports differ")
    parser.add_argument("--output", type=Path, default=PROJECT / "models/derived")
    args = parser.parse_args()
    files = export_files()
    if args.check:
        stale = [name for name, content in files.items()
                 if not (args.output / name).is_file() or (args.output / name).read_bytes() != content]
        if stale:
            raise SystemExit(f"Stale model exports: {', '.join(stale)}. Regenerate before building.")
        print("Model export verified: identical source hashes, assets, and physics replay.")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (args.output / name).write_bytes(content)
        print(f"Exported {len(files)} native model files to {args.output}")


if __name__ == "__main__":
    main()
