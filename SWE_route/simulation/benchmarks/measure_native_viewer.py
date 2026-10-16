"""Measure the native viewer using the baseline's Chrome/resource probes.

Use a release build. Run sequentially with measure_viewer.py for a contemporary
comparison. The benchmark query fixes camera and canvas size; shading/frontend
implementation and background desktop load can still differ.
"""

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import urllib.request

import psutil

from measure_viewer import (DevTools, PROBE, PROJECT, ProcessSampler,
                            available_port, distribution, save_json, terminate_tree, wait_json)
from tools.verify_native_viewer import core_gestures, local_catalog, verify_catalog

STATE_PROBE = r"""(() => {
  const NativeSocket = window.WebSocket;
  const probe = window.nativeProbe = {states: [], acks: [], active: false, last: null, hello: null};
  window.WebSocket = class extends NativeSocket {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (message.type === 'hello') probe.hello = message;
        if (message.type === 'state') {
          probe.last = message;
          if (probe.active) probe.states.push({received_ms: performance.now(), sequence: message.sequence,
            server_time_s: message.server_time_s, simulation_time_s: message.simulation_time_s,
            timing: message.timing, activity: message.activity});
        }
        if (message.type === 'ack' && probe.active) probe.acks.push(message);
      });
    }
  };
})();
"""


def wait_for(page: DevTools, expression: str, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if page.evaluate(expression):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Browser condition timed out: {expression}")


def click(page: DevTools, identifier: str) -> None:
    page.evaluate(f"document.getElementById({json.dumps(identifier)}).click()")


def play_family(page: DevTools, gesture: dict) -> float:
    return page.evaluate("""new Promise((resolve, reject) => {
      const group = document.getElementById('group-filter');
      group.value = 'All'; group.dispatchEvent(new Event('change', {bubbles:true}));
      const review = document.getElementById('review-filter');
      if (review) {review.value = 'all'; review.dispatchEvent(new Event('change', {bubbles:true}));}
      const select = document.getElementById('gesture-select');
      const name = %s;
      if (select.tagName === 'SELECT') {
        select.value = name; select.dispatchEvent(new Event('change', {bubbles:true}));
      } else {
        const button = [...select.querySelectorAll('[data-gesture]')].find(button => button.dataset.gesture === name);
        if (!button) {reject('Gesture is not in the visible list: ' + name); return;}
        button.click();
      }
      const started = performance.now();
      const observer = new MutationObserver(() => {
        if (document.getElementById('activity').textContent.trim() === %s) {
          observer.disconnect(); clearTimeout(timeout); resolve(performance.now() - started);
        }
      });
      const timeout = setTimeout(() => {observer.disconnect(); reject('Status timeout');}, 5000);
      observer.observe(document.getElementById('activity'), {subtree:true,childList:true,characterData:true});
      document.getElementById('play').click();
    })""" % (json.dumps(gesture["name"]), json.dumps("Playing " + gesture["label"])))


def measure(args) -> None:
    expected = local_catalog()
    cores = core_gestures(expected)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    server = browser = page = None
    port, debug_port = available_port(), available_port()
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    profile = Path(tempfile.mkdtemp(prefix="lamp-native-chrome-"))
    phases, commands, resources = [], [], []
    sources = sorted((PROJECT / "controller/src").rglob("*.rs"))
    sources += sorted((PROJECT / "controller/native").glob("*"))
    sources += sorted((PROJECT / "viewer/src").glob("*"))
    sources += [PROJECT / "controller/Cargo.lock", PROJECT / "viewer/package-lock.json",
                PROJECT / "models/derived/manifest.json", Path(__file__)]
    sources += sorted((PROJECT / "gestures").glob("*.json"))
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "logical_cpu_count": psutil.cpu_count(),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_ram_mib": psutil.virtual_memory().total / 2**20,
                "backend": "mujoco", "speed": .5, "repeats": args.repeats,
                "idle_seconds": args.idle_seconds, "active_seconds": args.active_seconds,
                "families": [gesture["group"] for gesture in cores], "catalog_count": len(expected),
                "gesture_schedule": [{"name": gesture["name"], "label": gesture["label"], "at_seconds": index * args.active_seconds / len(cores)}
                                     for index, gesture in enumerate(cores)],
                "warmup_s": 5, "browser_mode": "headless=new",
                "source_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources if p.is_file()},
                "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                "built_viewer_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted((PROJECT / "viewer/dist").rglob("*")) if p.is_file()},
                "cpu_percent_definition": "100% = one logical core; process-tree CPU seconds / wall seconds",
                "rss_definition": "sum of process-tree working sets; shared pages can be counted more than once",
                "latency_definition": "browser click to DOM activity state; not first motion or GPU presentation",
                "preset": args.preset,
                "camera": ({"position": [-.72, -1.15, .85], "target": [-.08, 0, .38], "fov": 75}
                           if args.preset == "matched" else {"position": [-1.1, -1.55, 1.07], "target": [-.03, 0, .36], "fov": 42}),
                "comparison_limits": ("Matched gesture schedule, pace, camera, canvas, probes. Frontend implementation differs; host load recorded, not controlled."
                                      if args.preset == "matched" else "Delivered visual settings including shadows; camera/canvas/shading differ from legacy. This is an observation of the delivered viewer, not a controlled renderer comparison.")}
    try:
        started = time.perf_counter()
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([str(args.executable), "--root", str(PROJECT), "--speed", "0.5", "serve", "--port", str(port),
                                       "--timing-log", str(output / "server-ticks.csv")],
                                      cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, **hidden)
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError((output / "server.log").read_text())
            try:
                with urllib.request.urlopen(url + "/api/scene", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.25)
        else:
            raise RuntimeError("Native viewer HTTP startup timed out")
        manifest["server_http_ready_s"] = time.perf_counter() - started
        flags = ["--headless=new", "--no-first-run", "--no-default-browser-check",
                 f"--remote-debugging-port={debug_port}", "--window-size=1280,900",
                 "--force-device-scale-factor=1", f"--user-data-dir={profile}", "about:blank"]
        browser = subprocess.Popen([str(args.chrome), *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden)
        endpoint = f"http://127.0.0.1:{debug_port}"
        version = wait_json(endpoint + "/json/version")
        manifest["browser_version"] = version["Browser"]
        manifest["browser_flags"] = [flag for flag in flags if not flag.startswith("--user-data-dir")]
        browser_tools = DevTools(version["webSocketDebuggerUrl"])
        try:
            manifest["graphics"] = browser_tools.call("SystemInfo.getInfo")
        finally:
            browser_tools.connection.close()
        targets = wait_json(endpoint + "/json/list")
        page = DevTools(next(target["webSocketDebuggerUrl"] for target in targets if target["type"] == "page"))
        page.call("Page.enable")
        page.call("Runtime.enable")
        page.call("Page.addScriptToEvaluateOnNewDocument", {"source": PROBE + STATE_PROBE})
        started = time.perf_counter()
        page.call("Page.navigate", {"url": url + ("/?benchmark=1" if args.preset == "matched" else "/")})
        wait_for(page, "window.nativeProbe?.last && document.getElementById('play') && !document.getElementById('play').disabled")
        verify_catalog(page.evaluate("nativeProbe.hello.gestures"), expected)
        manifest["navigation_to_controls_ready_s"] = time.perf_counter() - started
        time.sleep(5)
        manifest["viewport"] = page.evaluate("({width:innerWidth,height:innerHeight,dpr:devicePixelRatio})")
        manifest["webgl"] = page.evaluate("""(() => {
          const canvas = document.querySelector('canvas'); const gl = canvas.getContext('webgl2');
          const info = gl.getExtension('WEBGL_debug_renderer_info');
          return {width:canvas.width,height:canvas.height,
            renderer:info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)};
        })()""")
        if args.preset == "matched" and (manifest["webgl"]["width"], manifest["webgl"]["height"]) != (940, 805):
            raise RuntimeError(f"Unexpected benchmark canvas size: {manifest['webgl']}")
        save_json(output / "manifest.json", manifest)
        screenshot = page.call("Page.captureScreenshot", {"format": "png"})
        (output / "viewer-ready.png").write_bytes(base64.b64decode(screenshot["data"]))
        psutil.cpu_percent()
        for repeat in range(1, args.repeats + 1):
            for mode, seconds in (("idle", args.idle_seconds), ("active", args.active_seconds)):
                click(page, "reset")
                time.sleep(1.5)
                server_sample, browser_sample = ProcessSampler(server.pid), ProcessSampler(browser.pid)
                phase_start = time.perf_counter()
                page.evaluate("lampProbe.start(); nativeProbe.states=[]; nativeProbe.acks=[]; nativeProbe.active=true")
                index, next_sample, phase_samples = 0, phase_start, []
                while time.perf_counter() - phase_start < seconds:
                    now = time.perf_counter()
                    if mode == "active" and index < len(cores) and now - phase_start >= index * seconds / len(cores):
                        latency = play_family(page, cores[index])
                        commands.append({"repeat": repeat, "family": cores[index]["group"], "gesture": cores[index]["name"], "click_to_status_ms": latency})
                        index += 1
                    if now >= next_sample:
                        row = {"monotonic_s": now, "repeat": repeat, "phase": mode,
                               "server_rss_mib": server_sample.sample(), "browser_tree_rss_mib": browser_sample.sample(),
                               "host_cpu_percent": psutil.cpu_percent(), "available_ram_mib": psutil.virtual_memory().available / 2**20}
                        phase_samples.append(row)
                        resources.append(row)
                        next_sample = now + .5
                    time.sleep(.025)
                browser_result = page.evaluate("lampProbe.stop()")
                native_result = page.evaluate("nativeProbe.active=false; ({states:nativeProbe.states,acks:nativeProbe.acks})")
                server_sample.sample()
                browser_sample.sample()
                elapsed = time.perf_counter() - phase_start
                phases.append({"repeat": repeat, "phase": mode, "wall_s": elapsed,
                               "browser": browser_result, "native": native_result,
                               "server_cpu_percent_one_core": 100 * server_sample.cpu_seconds() / elapsed,
                               "browser_cpu_percent_one_core": 100 * browser_sample.cpu_seconds() / elapsed,
                               "server_peak_sampled_rss_mib": max(s["server_rss_mib"] for s in phase_samples),
                               "browser_peak_sampled_tree_rss_mib": max(s["browser_tree_rss_mib"] for s in phase_samples),
                               "host_cpu_percent": distribution([s["host_cpu_percent"] for s in phase_samples])})
                click(page, "stop")
                print(f"Completed repeat {repeat}/{args.repeats}: {mode}, {elapsed:.1f}s", flush=True)
        screenshot = page.call("Page.captureScreenshot", {"format": "png"})
        (output / "viewer.png").write_bytes(base64.b64decode(screenshot["data"]))
        time.sleep(1.1)  # Allow the bounded timing-log buffer to flush the final phase.
    finally:
        save_json(output / "measurements.json", {"phases": phases, "commands": commands, "resources": resources})
        if page:
            page.connection.close()
        if server:
            terminate_tree(server)
        if browser:
            terminate_tree(browser)
    summaries = []
    ticks = list(csv.DictReader((output / "server-ticks.csv").open(encoding="utf-8")))
    for phase in phases:
        draw = phase["browser"]
        states = phase["native"]["states"]
        first, last = states[0], states[-1]
        server_elapsed = last["server_time_s"] - first["server_time_s"]
        draw_times = draw["draw_frame_times_ms"]
        phase_ticks = [row for row in ticks if first["server_time_s"] <= float(row["started_s"]) <= last["server_time_s"]]
        summaries.append({key: value for key, value in phase.items() if key not in ("browser", "native")} | {
            "received_state_hz": (len(states) - 1) / server_elapsed,
            "server_tick_hz": (last["sequence"] - first["sequence"]) / server_elapsed,
            "simulation_seconds_per_wall_second": (last["simulation_time_s"] - first["simulation_time_s"]) / server_elapsed,
            "server_tick_compute_ms": distribution([float(row["compute_ms"]) / int(row["steps"]) for row in phase_ticks]),
            "server_control_physics_compute_ms": distribution([state["timing"]["tick_compute_ms"] for state in states]),
            "dropped_simulation_time_s": last["timing"]["dropped_time_s"] - first["timing"]["dropped_time_s"],
            "browser_raf_hz": 1000 * len(draw["raf_intervals_ms"]) / sum(draw["raf_intervals_ms"]),
            "browser_draw_frames_per_s": len(draw_times) * 1000 / draw["elapsed_ms"],
            "browser_draw_interval_ms": distribution([b-a for a,b in zip(draw_times, draw_times[1:])]),
            "browser_long_tasks": distribution(draw["long_tasks_ms"]),
        })
    save_json(output / "summary.json", {"phases": summaries,
              "click_to_status_ms": distribution([row["click_to_status_ms"] for row in commands])})
    print(f"Saved native browser measurement: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--executable", type=Path, default=PROJECT / "controller/target/release/lamp-controller.exe")
    parser.add_argument("--chrome", type=Path, default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preset", choices=("matched", "default"), default="matched")
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--active-seconds", type=float, default=30)
    args = parser.parse_args()
    family_count = len(core_gestures(local_catalog()))
    if args.repeats < 1 or args.idle_seconds < 1 or args.active_seconds < family_count:
        parser.error(f"Use at least one repeat, one idle second, and {family_count} active seconds (one per current emotion family)")
    measure(args)


if __name__ == "__main__":
    main()
