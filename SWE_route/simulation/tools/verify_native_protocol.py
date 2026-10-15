"""Verify the built Rust server through its real HTTP/WebSocket interface.

Run from simulation/: uv run --locked python tools/verify_native_protocol.py
The script starts and cleans up its own hidden process on an unused loopback
port. Every attempt writes a fresh JSON result and server log, including failed
attempts; existing evidence is never overwritten. It does not edit gestures.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import mujoco
from jsonschema import Draft202012Validator
import yourdfpy
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmarks.measure_viewer import available_port, terminate_tree
from lamp_sim.motion import ROBOT_PATH
from tools.verify_native_viewer import core_gestures, local_catalog, verify_catalog


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite(value, path="state") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{path}.{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f"Nonfinite value at {path}")


def quaternion_matrix(quaternion) -> np.ndarray:
    x, y, z, w = quaternion
    require(abs(np.linalg.norm(quaternion) - 1) < 1e-9, "Link quaternion is not unit length")
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


class Client:
    def __init__(self, url, schema, report):
        self.socket = connect(url.replace("http://", "ws://") + "/ws", origin=url, proxy=None, open_timeout=5)
        self.validator = Draft202012Validator(schema)
        self.report = report
        self.counter = 0
        self.last_state = None
        self.hello = self.receive_until(lambda message: message["type"] == "hello")
        self.state()

    def close(self):
        self.socket.close()

    def receive_until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = json.loads(self.socket.recv(timeout=max(0.001, deadline - time.monotonic())))
            finite(message)
            require(message.get("v") == 1, "Unexpected protocol version")
            if message["type"] == "state":
                if self.last_state:
                    require(message["sequence"] >= self.last_state["sequence"], "State sequence moved backwards")
                    require(message["server_time_s"] >= self.last_state["server_time_s"], "Server time moved backwards")
                self.last_state = message
                self.report["state_messages"] += 1
            if predicate(message):
                return message
        raise TimeoutError("Expected protocol message did not arrive")

    def state(self, predicate=lambda state: True, after=0, timeout=10):
        return self.receive_until(lambda message: message["type"] == "state" and message["server_time_s"] >= after and predicate(message), timeout)

    def command(self, kind, *, valid=True, raw=None, **fields):
        self.counter += 1
        identifier = f"verify-{self.counter}"
        data = {"v": 1, "id": identifier, "type": kind, **fields}
        if valid:
            self.validator.validate(data)
        elif raw is None:
            require(not self.validator.is_valid(data), f"Invalid test command unexpectedly matches schema: {data}")
        payload = raw.replace("$ID", identifier) if raw is not None else json.dumps(data, allow_nan=False)
        started = time.perf_counter()
        self.socket.send(payload)
        # Malformed/nonfinite JSON may prevent the server from decoding even its
        # correlation ID. With one request in flight, its empty-ID error is clear.
        reply = self.receive_until(lambda message: message["type"] == "ack" and (message["id"] == identifier or (not valid and message["id"] == "")))
        require(reply["ok"] is valid, f"Unexpected acknowledgement for {payload}: {reply}")
        if not valid:
            require(bool(reply.get("error")), "Rejected command has no explanation")
        self.report["commands"].append({"command": data if raw is None else payload, "ack": reply, "round_trip_ms": (time.perf_counter() - started) * 1000})
        return reply


def http(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5) as response:
        return response.status, response.headers.get_content_type(), response.read()


def denied(url, headers):
    try:
        http(url, headers)
    except urllib.error.HTTPError as error:
        require(error.code == 403, f"Expected HTTP 403, got {error.code}")
        return
    raise AssertionError("Foreign host/origin was accepted")


def verify(args, report):
    expected = local_catalog()
    cores = core_gestures(expected)
    report["expected_catalog"] = expected
    report["families"] = [gesture["group"] for gesture in cores]
    schema = json.loads((PROJECT / "protocol/commands.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    command = [str(args.binary), "--root", str(PROJECT), "serve", "--port", str(port)]
    report["server_command"] = command
    report["url"] = url
    process = None
    client = None
    with args.log.open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            report["server_pid"] = process.pid
            deadline = time.monotonic() + 30
            while True:
                require(process.poll() is None, "Owned Rust server exited during startup")
                try:
                    _, _, body = http(url + "/api/scene")
                    scene = json.loads(body)
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Rust server did not become ready")
                    time.sleep(0.1)

            require(scene["coordinate_system"] == "z-up" and scene["quaternion_order"] == "xyzw", "Unexpected scene axes/quaternion convention")
            status, content_type, body = http(url)
            require(status == 200 and content_type == "text/html", "Viewer HTML is unavailable")
            html = body.decode("utf-8")
            assets = set(re.findall(r'''(?:src|href)=["']([^"']+)["']''', html))
            assets |= {"/assets/" + visual["mesh"] for visual in scene["visuals"] if visual["type"] == "mesh"}
            report["assets"] = []
            for asset in sorted(assets):
                if asset.startswith("data:"):
                    continue
                asset_url = urllib.parse.urljoin(url, asset)
                require(urllib.parse.urlparse(asset_url).netloc == urllib.parse.urlparse(url).netloc, "Viewer has a remote asset dependency")
                status, content_type, data = http(asset_url)
                require(status == 200 and len(data) > 0, f"Asset failed to load: {asset}")
                require(content_type != "text/html", f"Asset returned fallback HTML: {asset}")
                report["assets"].append({"path": asset, "bytes": len(data), "content_type": content_type, "sha256": hashlib.sha256(data).hexdigest()})
            require(any(asset["path"].endswith(".js") for asset in report["assets"]), "No bundled viewer JavaScript found")
            for path in ("/api/scene", "/ws"):
                denied(url + path, {"Host": "attacker.invalid"})
                denied(url + path, {"Origin": "https://attacker.invalid"})
            try:
                with connect(url.replace("http://", "ws://") + "/ws", origin="https://attacker.invalid", proxy=None, open_timeout=5):
                    raise AssertionError("Foreign WebSocket origin was accepted")
            except InvalidStatus as error:
                require(error.response.status_code == 403, "Foreign WebSocket origin did not get 403")
            report["checks"].append("local HTML, bundled scripts/styles and mesh assets; HTTP Host/Origin and WebSocket Origin guards")

            client = Client(url, schema, report)
            verify_catalog(client.hello["gestures"], expected)
            require(client.hello["limits"] == scene["limits"], "Hello limits differ from scene")
            require(client.hello["mujoco_version"] == "3.14.0", "Unexpected MuJoCo version")
            require(client.hello["physics_hz"] == 240 and client.hello["control_hz"] == 30, "Unexpected step rates")
            catalog = client.hello["gestures"]
            model = yourdfpy.URDF.load(str(ROBOT_PATH), mesh_dir=str(ROBOT_PATH.parent))
            reference_models = {
                mode: mujoco.MjModel.from_xml_path(str(PROJECT / "models/derived" / filename))
                for mode, filename in (("mujoco", "lamp.xml"), ("mujoco-source", "lamp-source.xml"), ("mujoco-kinematic", "lamp.xml"))
            }
            report["max_link_position_error_m"] = 0.0
            report["max_link_rotation_error"] = 0.0
            report["max_mujoco_link_position_error_m"] = 0.0
            report["max_mujoco_link_rotation_error"] = 0.0
            report["fk_tolerances"] = {
                "derived_mujoco": 1e-10, "urdf_position_m": 1e-9, "urdf_rotation_matrix": 1e-6,
                "reason": "MuJoCo XML serialization rounds the fixed speaker quaternion to six decimals; compare the same derived model at higher precision as well.",
            }
            report["fk_states_checked"] = 0

            def check_fk(state):
                require(set(state["actual"]) == set(scene["limits"]), "State is missing actual joints")
                model.update_cfg(state["actual"])
                reference_model = reference_models[state["backend"]]
                reference_data = mujoco.MjData(reference_model)
                for joint, value in state["actual"].items():
                    joint_id = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    reference_data.qpos[reference_model.jnt_qposadr[joint_id]] = value
                mujoco.mj_forward(reference_model, reference_data)
                for name in scene["links"]:
                    pose = state["links"][name]
                    transform = model.get_transform(name, "base_link")
                    rotation = quaternion_matrix(pose["quaternion"])
                    position_error = float(np.max(np.abs(np.array(pose["position"]) - transform[:3, 3])))
                    rotation_error = float(np.max(np.abs(rotation - transform[:3, :3])))
                    report["max_link_position_error_m"] = max(report["max_link_position_error_m"], position_error)
                    report["max_link_rotation_error"] = max(report["max_link_rotation_error"], rotation_error)
                    require(position_error < 1e-9 and rotation_error < 1e-6, f"Observed link pose disagrees with URDF FK: {name}")
                    body_id = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_BODY, name)
                    position_error = float(np.max(np.abs(np.array(pose["position"]) - reference_data.xpos[body_id])))
                    rotation_error = float(np.max(np.abs(rotation - reference_data.xmat[body_id].reshape(3, 3))))
                    report["max_mujoco_link_position_error_m"] = max(report["max_mujoco_link_position_error_m"], position_error)
                    report["max_mujoco_link_rotation_error"] = max(report["max_mujoco_link_rotation_error"], rotation_error)
                    require(position_error < 1e-10 and rotation_error < 1e-10, f"Observed link pose disagrees with Python MuJoCo FK: {name}")
                report["fk_states_checked"] += 1
                return quaternion_matrix(state["links"]["light_emitter_link"]["quaternion"])[:, 0]

            for _ in range(10):
                check_fk(client.state())
            report["checks"].append("hello catalog/limits/rates and ten finite live state snapshots with all link poses checked against URDF FK")

            ack = client.command("pace", speed=1.0)
            client.state(lambda state: state["speed"] == 1.0, after=ack["server_time_s"])
            ack = client.command("play", name="nod")
            playing = client.state(lambda state: state["activity"] == "Playing Nod", after=ack["server_time_s"])
            start_head = playing["requested"]["head_pitch_joint"]
            changed = client.state(lambda state: abs(state["requested"]["head_pitch_joint"] - start_head) > 1e-3)
            check_fk(changed)
            ack = client.command("stop")
            stopped = client.state(lambda state: state["activity"] == "Stopped", after=ack["server_time_s"])
            for _ in range(10):
                state = client.state()
                require(state["requested"] == stopped["requested"] and state["activity"] == "Stopped", "Stop did not cancel requested playback")
            report["checks"].append("Nod moves and Stop freezes requested pose while physics continues")

            for gesture in cores:
                ack = client.command("play", name=gesture["name"])
                state = client.state(lambda state: state["activity"] == f"Playing {gesture['label']}", after=ack["server_time_s"])
                check_fk(state)
                ack = client.command("stop")
                client.state(lambda state: state["activity"] == "Stopped", after=ack["server_time_s"])
            report["checks"].append(f"All {len(cores)} current emotion families play and stop through the live protocol")

            ack = client.command("targets", targets={"base_yaw_joint": 99, "head_pitch_joint": -99})
            clamped = client.state(lambda state: state["activity"] == "Ready", after=ack["server_time_s"], timeout=10)
            require(clamped["requested"]["base_yaw_joint"] == scene["limits"]["base_yaw_joint"]["upper"], "Upper target was not clamped")
            require(clamped["requested"]["head_pitch_joint"] == scene["limits"]["head_pitch_joint"]["lower"], "Lower target was not clamped")
            check_fk(clamped)
            ack = client.command("reset")
            client.state(lambda state: state["activity"] == "Ready" and all(abs(value) < 1e-12 for value in state["requested"].values()), after=ack["server_time_s"], timeout=10)
            for enabled in (False, True):
                ack = client.command("light", enabled=enabled, brightness=3.5)
                client.state(lambda state: state["light"] == {"enabled": enabled, "brightness": 3.5}, after=ack["server_time_s"])
            report["checks"].append("manual target soft clamping, reset, pace, light on/off and brightness")

            for mode in ("mujoco-source", "mujoco", "mujoco-kinematic"):
                ack = client.command("mode", mode=mode)
                state = client.state(lambda state: state["backend"] == mode, after=ack["server_time_s"])
                require((state["diagnostics"]["contact_force_n"] is None) == (mode == "mujoco-kinematic"), "Mode contact-force semantics differ")
                check_fk(state)
            ack = client.command("reset")
            neutral = client.state(lambda state: state["activity"] == "Ready", after=ack["server_time_s"])
            require(np.max(np.abs(check_fk(neutral) - [-1, 0, 0])) < 1e-9, "Neutral emitter does not point along base -x")
            for pitch in (0.3, -0.3):
                ack = client.command("targets", targets={"head_pitch_joint": pitch})
                tilted = client.state(lambda state: state["activity"] == "Ready", after=ack["server_time_s"])
                direction = check_fk(tilted)
                require(direction[2] * pitch < 0, "Head pitch does not tilt the emitter in the expected direction")
            report["checks"].append("all three MuJoCo modes and emitter neutral/down/up orientation including the URDF 180-degree head yaw")

            client.command("pace", speed=0.25)
            ack = client.command("play", name="nod")
            client.state(lambda state: state["activity"] == "Playing Nod", after=ack["server_time_s"])
            invalids = [
                ("targets", {"targets": {"unknown_joint": 0.1}}),
                ("pace", {"speed": 0}), ("pace", {"speed": 1.1}),
                ("light", {"enabled": False, "brightness": 11}),
                ("light", {"enabled": "yes", "brightness": 3}),
                ("mode", {"mode": "unknown"}), ("stop", {"extra": 1}),
            ]
            for kind, fields in invalids:
                client.command(kind, valid=False, **fields)
            # Unknown names are schema-valid but must fail catalog lookup atomically.
            invalid_raw = [
                '{"v":1,"id":"$ID","type":"play","name":"missing_gesture"}',
                '{"v":1,"id":"$ID","type":"targets","targets":{"head_pitch_joint":0.1,"head_pitch_joint":0.2}}',
                '{"v":1,"id":"$ID","type":"stop","type":"reset"}',
                '{"v":2,"id":"$ID","type":"stop"}',
                '{"v":1,"id":"$ID","type":"targets","targets":{"head_pitch_joint":1e999}}',
            ]
            for raw in invalid_raw:
                client.command("raw-invalid", valid=False, raw=raw)
            preserved = client.state()
            require(preserved["activity"] == "Playing Nod", "Rejected commands interrupted active motion")
            require(preserved["speed"] == 0.25 and preserved["light"] == {"enabled": True, "brightness": 3.5}, "Rejected commands changed pace or light")
            require(preserved["backend"] == "mujoco-kinematic", "Rejected command changed backend")
            client.command("stop")
            report["checks"].append("twelve rejected commands preserve active playback, pace, light and backend; duplicate keys are rejected recursively")

            client.command("reload")
            refreshed = client.receive_until(lambda message: message["type"] == "hello")
            require(refreshed["gestures"] == catalog, "Reload changed an unmodified catalog")
            client.close()
            client = Client(url, schema, report)
            require(client.hello["gestures"] == catalog, "Reconnect did not return the complete catalog")
            require(client.last_state["light"] == {"enabled": True, "brightness": 3.5}, "Reconnect lost authoritative light state")
            require(client.last_state["speed"] == 0.25 and client.last_state["backend"] == "mujoco-kinematic", "Reconnect lost controller state")
            report["checks"].append("reload publishes a complete catalog; reconnect sends full catalog and current authoritative state")
            report["catalog_count"] = len(catalog)
            report["status"] = "passed"
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if process is not None:
                    terminate_tree(process)
                    process.wait(timeout=10)
                    report["owned_server_exited"] = process.poll() is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=PROJECT / "controller/target/release" / ("lamp-controller.exe" if os.name == "nt" else "lamp-controller"))
    parser.add_argument("--output", type=Path, default=PROJECT / ".verification/native-protocol.json")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    args.log = args.output.with_suffix(".server.log")
    require(args.binary.is_file(), "Build the release Rust server first with cargo build --release --locked")
    require(not args.output.exists() and not args.log.exists(), "Output evidence already exists; select a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "checks": [], "commands": [], "state_messages": 0, "platform": sys.platform}
    sources = [Path(__file__), PROJECT / "protocol/commands.schema.json", PROJECT / "models/derived/scene.json", ROBOT_PATH, args.binary]
    sources += list((PROJECT / "controller/src").glob("*.rs")) + [PROJECT / "controller/native/bridge.c"]
    sources += [path for path in (PROJECT / "models/derived").iterdir() if path.is_file()]
    sources += sorted((PROJECT / "gestures").glob("*.json"))
    report["sha256"] = {str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    with args.output.open("x", encoding="utf-8") as output:
        try:
            verify(args, report)
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            report["traceback"] = traceback.format_exc()
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    print(f"Protocol verification {report['status']}: {args.output}")
    if report["status"] != "passed":
        print(report.get("traceback", "Unknown verification error"), file=sys.stderr)
        return 1
    print(f"{len(report['checks'])} check groups, {len(report['commands'])} commands, {report['state_messages']} states, {report['fk_states_checked']} FK samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
