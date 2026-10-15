"""Verify Vite's local HTTP/WebSocket proxy and origin guard against native MuJoCo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.measure_viewer import save_json, terminate_tree, wait_json
from tools.verify_native_viewer import core_gestures, local_catalog, verify_catalog


def http_request(path: str, host: str = "127.0.0.1:5173", origin: str | None = None):
    headers = {"Host": host}
    if origin is not None:
        headers["Origin"] = origin
    connection = http.client.HTTPConnection("127.0.0.1", 5173, timeout=10)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def receive_until(websocket, predicate):
    for _ in range(100):
        message = json.loads(websocket.recv(timeout=5))
        if predicate(message):
            return message
    raise AssertionError("Expected proxy response did not arrive")


def verify(output: Path) -> None:
    expected = local_catalog()
    migration = json.loads((PROJECT / "catalogs/emotion-wheel-migration.json").read_text(encoding="utf-8"))
    cores = core_gestures(expected)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "catalog_count": len(expected), "families": [gesture["group"] for gesture in cores],
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    backend = dev = None
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        for port in (8082, 5173):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        source = PROJECT / "controller/target/release" / ("lamp-controller.exe" if os.name == "nt" else "lamp-controller")
        cache = PROJECT / ".cache"
        cache.mkdir(exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="dev-proxy-check-", dir=cache)) / source.name
        shutil.copy2(source, private)
        result["executable_sha256"] = hashlib.sha256(private.read_bytes()).hexdigest()
        with (output / "native.log").open("w", encoding="utf-8") as log:
            backend = subprocess.Popen([str(private), "serve", "--port", "8082"], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, **hidden)
        wait_json("http://127.0.0.1:8082/api/scene")
        with (output / "vite.log").open("w", encoding="utf-8") as log:
            dev = subprocess.Popen(["node", "node_modules/vite/bin/vite.js"], cwd=PROJECT / "viewer", stdout=log, stderr=subprocess.STDOUT, **hidden)
        wait_json("http://127.0.0.1:5173/api/scene")
        for host in ("127.0.0.1:5173", "localhost:5173"):
            for origin in (None, f"http://{host}"):
                for path in ("/api/scene", "/api/review-migration"):
                    status, body = http_request(path, host, origin)
                    assert status == 200, (host, origin, path, status)
                    assert json.loads(body)["schema_version"] == 1
                    if path == "/api/review-migration":
                        assert json.loads(body) == migration, "Proxied review migration differs from the source file"
        status, body = http_request("/assets/lamp_shade.stl")
        assert status == 200 and len(body) > 100
        checks.append("Both local hostnames proxy scene, exact review migration JSON and mesh requests, with or without matching HTTP Origin")
        for host, origin in (("127.0.0.1:5173", "https://example.com"), ("127.0.0.1:5173", "http://localhost:5173"), ("attacker.example:5173", None)):
            for path in ("/api/scene", "/api/review-migration"):
                status, _ = http_request(path, host, origin)
                assert status in (403, 404), (host, origin, path, status)
        checks.append("HTTP proxy rejects foreign/mismatched Origin and foreign Host")
        for host in ("127.0.0.1", "localhost"):
            with connect(f"ws://{host}:5173/ws", origin=f"http://{host}:5173", proxy=None, max_size=4_000_000) as websocket:
                hello = receive_until(websocket, lambda message: message.get("type") == "hello")
                verify_catalog(hello["gestures"], expected)
                receive_until(websocket, lambda message: message.get("type") == "state")
                for identifier, command in (("dev-play", {"type": "play", "name": "nod"}), ("dev-stop", {"type": "stop"})):
                    websocket.send(json.dumps({"v": 1, "id": identifier, **command}))
                    ack = receive_until(websocket, lambda message: message.get("type") == "ack" and message.get("id") == identifier)
                    assert ack["ok"], ack
        checks.append(f"Both local WebSocket origins receive all {len(expected)} local gestures/current state and acknowledge Nod/Stop")
        for origin in (None, "https://example.com", "http://localhost:5173", "null"):
            try:
                with connect("ws://127.0.0.1:5173/ws", origin=origin, proxy=None, open_timeout=5):
                    raise AssertionError(f"Rejected origin was accepted: {origin}")
            except InvalidStatus as error:
                assert error.response.status_code == 404, (origin, error.response.status_code)
        checks.append("WebSocket proxy rejects absent, foreign, mismatched, and opaque origins before forwarding")
        try:
            with connect("ws://127.0.0.1:8082/ws", origin="http://127.0.0.1:5173", proxy=None, open_timeout=5):
                raise AssertionError("Native origin guard was bypassed")
        except InvalidStatus as error:
            assert error.response.status_code == 403
        checks.append("Native server still rejects direct dev-origin connections")
        result["passed"] = True
        print(f"Passed {len(checks)} real development-proxy checks", flush=True)
    except BaseException as error:
        result["passed"] = False
        result["failure"] = repr(error)
        raise
    finally:
        for process in (dev, backend):
            if process:
                terminate_tree(process)
        save_json(output / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
