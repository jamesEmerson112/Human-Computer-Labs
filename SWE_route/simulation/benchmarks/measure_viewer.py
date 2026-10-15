"""Record the current viewer via an isolated Chrome instance and its DevTools API.

Run from simulation/: uv run --locked python benchmarks/measure_viewer.py --help
This instruments the existing loop without editing application code.
"""

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import psutil
from websockets.sync.client import connect


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

PROBE = r"""
(() => {
  const state = {active: false, draws: 0};
  window.lampProbe = state;
  for (const type of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!type) continue;
    for (const name of ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced']) {
      const original = type.prototype[name];
      if (!original) continue;
      type.prototype[name] = function(...args) {
        if (state.active) state.draws++;
        return original.apply(this, args);
      };
    }
  }
  state.start = () => {
    state.started = performance.now(); state.last = null; state.draws = 0;
    state.rafIntervals = []; state.drawTimes = []; state.longTasks = [];
    state.active = true;
  };
  state.stop = () => {
    state.active = false;
    return {elapsed_ms: performance.now() - state.started,
      raf_intervals_ms: state.rafIntervals, draw_frame_times_ms: state.drawTimes,
      long_tasks_ms: state.longTasks, visibility: document.visibilityState};
  };
  const frame = (now) => {
    if (state.active) {
      if (state.last !== null) state.rafIntervals.push(now - state.last);
      if (state.draws > 0) state.drawTimes.push(now - state.started);
      state.draws = 0; state.last = now;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  new PerformanceObserver(list => {
    if (state.active) state.longTasks.push(...list.getEntries().map(e => e.duration));
  }).observe({entryTypes: ['longtask']});
})();
"""


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "max": float(max(values))}


def worker(output: Path, port: int) -> None:
    from lamp_sim.viewer import LampViewer, run

    original = LampViewer.tick
    rows = []
    next_stop_check = 0.0

    def measured_tick(viewer, dt):
        nonlocal next_stop_check
        started = time.perf_counter()
        if started >= next_stop_check:
            next_stop_check = started + 1
            if (output / "STOP").exists():
                raise KeyboardInterrupt
        original(viewer, dt)
        finished = time.perf_counter()
        rows.append((started, dt, (finished - started) * 1000,
                     viewer.backend.time, viewer.motion.activity))

    LampViewer.tick = measured_tick
    try:
        run(port=port, backend="mujoco")
    finally:
        with (output / "server-ticks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("monotonic_s", "requested_dt_s", "tick_compute_ms", "physics_time_s", "activity"))
            writer.writerows(rows)


class DevTools:
    def __init__(self, url: str):
        self.connection = connect(url, max_size=32_000_000)
        self.counter = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        self.connection.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.connection.recv(timeout=20))
            if message.get("id") == self.counter:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def click(self, label: str) -> None:
        found = self.evaluate("""(() => {
          const e = [...document.querySelectorAll('button')].find(e => e.textContent.trim() === %s);
          if (!e) return false; e.click(); return true;
        })()""" % json.dumps(label))
        if not found:
            raise RuntimeError(f"Button not found: {label}")

    def toggle_family(self, family: str) -> None:
        found = self.evaluate("""(() => {
          const e = [...document.querySelectorAll('div')].find(e =>
            e.textContent.trim() === %s && e.style.cursor === 'pointer');
          if (!e) return false; e.click(); return true;
        })()""" % json.dumps(family))
        if not found:
            raise RuntimeError(f"Family not found: {family}")

    def play_family(self, family: str) -> float:
        # The promise measures browser click -> server status -> DOM acknowledgment.
        # It does NOT measure time to first motor movement or completed GPU presentation.
        return self.evaluate("""new Promise((resolve, reject) => {
          const buttons = [...document.querySelectorAll('button')].filter(e =>
            e.textContent.trim() === 'Play emotion' && e.getBoundingClientRect().height > 0 &&
            !e.closest('[aria-hidden=true]'));
          if (buttons.length !== 1) { reject('Expected one visible Play button'); return; }
          const started = performance.now();
          const observer = new MutationObserver(() => {
            if ([...document.querySelectorAll('strong')].some(e => e.textContent === %s)) {
              observer.disconnect(); clearTimeout(timeout); resolve(performance.now() - started);
            }
          });
          const timeout = setTimeout(() => { observer.disconnect(); reject('Status acknowledgment timeout'); }, 5000);
          observer.observe(document.body, {subtree: true, childList: true, characterData: true});
          buttons[0].click();
        })""" % json.dumps("Playing " + family))


class ProcessSampler:
    def __init__(self, root_pid: int):
        self.root = psutil.Process(root_pid)
        self.cpu = {}
        self.initial = {}
        self.sample()
        self.initial = self.cpu.copy()

    def sample(self) -> float:
        rss = 0
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                with process.oneshot():
                    key = (process.pid, process.create_time())
                    cpu = process.cpu_times()
                    self.cpu[key] = cpu.user + cpu.system
                    rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss / 2**20

    def cpu_seconds(self) -> float:
        return sum(value - self.initial.get(key, 0) for key, value in self.cpu.items())


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_json(url: str):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.load(response)
        except (OSError, ValueError):
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}")


def terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([parent, *children], timeout=5)
        for remaining in alive:
            remaining.kill()
    except psutil.NoSuchProcess:
        pass


def measure(args) -> None:
    from lamp_sim.benchmark import fingerprints
    from lamp_sim.gestures import load_gestures
    from lamp_sim.motion import load_limits

    library = load_gestures(load_limits())
    families = tuple(sorted({gesture.group for gesture in library.values() if gesture.group}))
    if not families or any(sum(gesture.group == family and gesture.label == family
                               for gesture in library.values()) != 1 for family in families):
        raise ValueError("Each emotion family must contain exactly one same-label core clip")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    chrome = args.chrome
    if not chrome.is_file():
        raise FileNotFoundError(f"Set --chrome to a Chrome executable: {chrome}")
    port, debug_port = available_port(), available_port()
    profile = Path(tempfile.mkdtemp(prefix="lamp-baseline-chrome-"))
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    server = browser = page = None
    started = time.perf_counter()
    phases, latencies, samples = [], [], []
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "backend": "mujoco",
        "speed": 0.5, "repeats": args.repeats, "idle_seconds": args.idle_seconds,
        "active_seconds": args.active_seconds, "families": families, "catalog_count": len(library),
        "source_sha256": fingerprints(), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "browser_mode": "headless=new", "requested_window": [1280, 900],
        "resource_sample_period_s": 0.5, "warmup_s": 5,
        "cpu_percent_definition": "100 * process-tree CPU seconds / wall seconds; 100% is one logical core",
        "rss_definition": "sum of process-tree working sets; shared pages can be counted more than once",
    }
    try:
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                                       "--port", str(port)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, **hidden)
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError((output / "server.log").read_text())
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.25)
        else:
            raise RuntimeError("Viewer HTTP startup timed out")
        manifest["server_http_ready_s"] = time.perf_counter() - started
        flags = ["--headless=new", "--no-first-run", "--no-default-browser-check",
                 f"--remote-debugging-port={debug_port}", "--window-size=1280,900",
                 "--force-device-scale-factor=1", f"--user-data-dir={profile}", "about:blank"]
        browser = subprocess.Popen([str(chrome), *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden)
        endpoint = f"http://127.0.0.1:{debug_port}"
        version_info = wait_json(endpoint + "/json/version")
        manifest["browser_version"] = version_info["Browser"]
        manifest["browser_flags"] = [flag for flag in flags if not flag.startswith("--user-data-dir")]
        browser_tools = DevTools(version_info["webSocketDebuggerUrl"])
        try:
            manifest["graphics"] = browser_tools.call("SystemInfo.getInfo")
        finally:
            browser_tools.connection.close()
        targets = wait_json(endpoint + "/json/list")
        page = DevTools(next(target["webSocketDebuggerUrl"] for target in targets if target["type"] == "page"))
        page.call("Page.enable")
        page.call("Runtime.enable")
        page.call("Page.addScriptToEvaluateOnNewDocument", {"source": PROBE})
        navigation_started = time.perf_counter()
        page.call("Page.navigate", {"url": url})
        for _ in range(120):
            if page.evaluate("document.body && document.body.innerText.includes(%s)" % json.dumps(f"Loaded {len(library)} gestures.")):
                break
            time.sleep(.25)
        else:
            raise RuntimeError("Browser did not load the complete gesture library")
        manifest["navigation_to_controls_ready_s"] = time.perf_counter() - navigation_started
        time.sleep(5)
        if not page.evaluate("typeof window.lampProbe === 'object'"):
            raise RuntimeError("Browser timing probe was not installed before navigation")
        manifest["viewport"] = page.evaluate("({width:innerWidth,height:innerHeight,dpr:devicePixelRatio})")
        manifest["webgl"] = page.evaluate("""(() => {
          const canvas = [...document.querySelectorAll('canvas')].sort((a,b) => b.width*b.height - a.width*a.height)[0];
          if (!canvas || !canvas.width || !canvas.height) throw new Error('No rendered canvas');
          const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
          const info = gl.getExtension('WEBGL_debug_renderer_info');
          return {width:canvas.width,height:canvas.height,
            renderer:info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)};
        })()""")
        ready_image = page.call("Page.captureScreenshot", {"format": "png"})
        (output / "viewer-ready.png").write_bytes(base64.b64decode(ready_image["data"]))
        print(f"Browser ready: {manifest['webgl']}", flush=True)
        save_json(output / "manifest.json", manifest)
        psutil.cpu_percent()
        opened = None
        for repeat in range(1, args.repeats + 1):
            for mode, seconds in (("idle", args.idle_seconds), ("active", args.active_seconds)):
                if opened:
                    page.toggle_family(opened)
                    opened = None
                page.click("Reset pose")
                time.sleep(1.5)
                server_sample = ProcessSampler(server.pid)
                browser_sample = ProcessSampler(browser.pid)
                phase_start = time.perf_counter()
                page.evaluate("lampProbe.start()")
                index = 0
                next_sample = phase_start
                phase_samples = []
                while time.perf_counter() - phase_start < seconds:
                    now = time.perf_counter()
                    if mode == "active" and index < len(families) and now - phase_start >= index * seconds / len(families):
                        if opened:
                            page.toggle_family(opened)
                        opened = families[index]
                        page.toggle_family(opened)
                        time.sleep(.15)
                        latency = page.play_family(opened)
                        latencies.append({"repeat": repeat, "family": opened, "click_to_status_ms": latency})
                        index += 1
                    if now >= next_sample:
                        row = {"monotonic_s": time.perf_counter(), "repeat": repeat, "phase": mode,
                               "server_rss_mib": server_sample.sample(), "browser_tree_rss_mib": browser_sample.sample(),
                               "host_cpu_percent": psutil.cpu_percent(), "available_ram_mib": psutil.virtual_memory().available / 2**20}
                        phase_samples.append(row)
                        samples.append(row)
                        next_sample = now + .5
                    time.sleep(.025)
                browser_result = page.evaluate("lampProbe.stop()")
                server_sample.sample()
                browser_sample.sample()
                ended = time.perf_counter()
                elapsed = ended - phase_start
                phases.append({"repeat": repeat, "phase": mode, "started_monotonic_s": phase_start,
                               "ended_monotonic_s": ended, "wall_s": elapsed, "browser": browser_result,
                               "server_cpu_percent_one_core": 100 * server_sample.cpu_seconds() / elapsed,
                               "browser_cpu_percent_one_core": 100 * browser_sample.cpu_seconds() / elapsed,
                               "server_peak_sampled_rss_mib": max(s["server_rss_mib"] for s in phase_samples),
                               "browser_peak_sampled_tree_rss_mib": max(s["browser_tree_rss_mib"] for s in phase_samples),
                               "host_cpu_percent": distribution([s["host_cpu_percent"] for s in phase_samples])})
                page.click("Stop motion")
                print(f"Completed repeat {repeat}/{args.repeats}: {mode}, {elapsed:.1f}s", flush=True)
        screenshot = page.call("Page.captureScreenshot", {"format": "png"})
        (output / "viewer.png").write_bytes(base64.b64decode(screenshot["data"]))
    finally:
        save_json(output / "measurements.json", {"phases": phases, "commands": latencies, "resources": samples})
        if page:
            page.connection.close()
        if server:
            (output / "STOP").write_text("Stop benchmark worker\n", encoding="utf-8")
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_tree(server)
        if browser:
            terminate_tree(browser)
    ticks = list(csv.DictReader((output / "server-ticks.csv").open(encoding="utf-8")))
    summary = []
    for phase in phases:
        frames = [row for row in ticks if phase["started_monotonic_s"] <= float(row["monotonic_s"]) < phase["ended_monotonic_s"]]
        times = [float(row["monotonic_s"]) for row in frames]
        intervals = [1000 * (b - a) for a, b in zip(times, times[1:])]
        simulation_s = sum(float(row["requested_dt_s"]) for row in frames)
        browser_data = phase["browser"]
        summary.append({key: value for key, value in phase.items() if key != "browser"} | {
            "server_tick_hz": (len(times) - 1) / (times[-1] - times[0]),
            "server_tick_compute_ms": distribution([float(row["tick_compute_ms"]) for row in frames]),
            "server_tick_interval_ms": distribution(intervals),
            "simulation_seconds_per_wall_second": simulation_s / phase["wall_s"],
            "browser_raf_interval_ms": distribution(browser_data["raf_intervals_ms"]),
            "browser_raf_hz": 1000 * len(browser_data["raf_intervals_ms"]) / sum(browser_data["raf_intervals_ms"]),
            "browser_draw_frames_per_s": len(browser_data["draw_frame_times_ms"]) * 1000 / browser_data["elapsed_ms"],
            "browser_draw_interval_ms": distribution(list(np.diff(browser_data["draw_frame_times_ms"]))),
            "browser_long_tasks": distribution(browser_data["long_tasks_ms"]),
        })
    save_json(output / "summary.json", {"phases": summary, "click_to_status_ms": distribution([row["click_to_status_ms"] for row in latencies])})
    print(f"Saved viewer baseline: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New result directory; existing directories are rejected")
    parser.add_argument("--chrome", type=Path, default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--active-seconds", type=float, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8093, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.port)
    else:
        if args.repeats < 1 or args.idle_seconds < 1 or args.active_seconds < 6:
            parser.error("Use at least one repeat, one idle second, and six active seconds")
        measure(args)


if __name__ == "__main__":
    main()
