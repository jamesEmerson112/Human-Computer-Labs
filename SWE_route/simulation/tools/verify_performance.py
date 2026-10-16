"""Exercise the hardcoded speaking lamp with real browser audio and native physics.

Uses an isolated browser profile and an owned server. The result records functional
checks and a small observed latency/resource sample, not a matched benchmark or
human approval of the acting. Existing output directories are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.measure_viewer import DevTools, available_port, save_json, terminate_tree, wait_json
from tools.verify_native_viewer import PROBE, capture, change, click, wait_for


AUDIO_PROBE = r"""
(() => {
  const probe = window.performanceVerification = {sources: [], samples: [], commands: [], sockets: [], contexts: []};
  const Socket = window.WebSocket;
  window.WebSocket = class extends Socket {
    constructor(...args) {
      super(...args); probe.sockets.push(this);
      this.addEventListener('message', event => {
        const m = JSON.parse(event.data);
        if (m.type === 'state' && probe.samples.length < 15000) probe.samples.push({
          receivedMs: performance.now(), time: m.server_time_s, simulation: m.simulation_time_s,
          performance: m.performance, requested: m.requested, actual: m.actual,
          diagnostics: m.diagnostics, timing: m.timing
        });
      });
    }
    send(value) { probe.commands.push({atMs: performance.now(), ...JSON.parse(value)}); super.send(value); }
  };
  const start = AudioBufferSourceNode.prototype.start;
  const stop = AudioBufferSourceNode.prototype.stop;
  AudioBufferSourceNode.prototype.start = function(when = 0, ...args) {
    if (!probe.contexts.includes(this.context)) probe.contexts.push(this.context);
    const samples = this.buffer?.getChannelData(0) ?? [];
    let peak = 0; for (const value of samples) peak = Math.max(peak, Math.abs(value));
    const record = {atMs:performance.now(), when, contextTime:this.context.currentTime,
      state:this.context.state, duration:this.buffer?.duration, peak, endedMs:null, stoppedMs:null,
      connections: this.__performanceConnections ?? []};
    this.__performanceRecord = record; probe.sources.push(record);
    this.addEventListener('ended', () => { record.endedMs = performance.now(); });
    return start.call(this, when, ...args);
  };
  AudioBufferSourceNode.prototype.stop = function(...args) {
    if (this.__performanceRecord) this.__performanceRecord.stoppedMs = performance.now();
    return stop.apply(this, args);
  };
  const connect = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function(destination, ...args) {
    if (this instanceof AudioBufferSourceNode) {
      this.__performanceConnections ??= [];
      this.__performanceConnections.push(destination.constructor.name);
    }
    return connect.call(this, destination, ...args);
  };
})();
"""


def trusted_click(page: DevTools, identifier: str) -> None:
    """Use a real browser user gesture so the ordinary audio policy is tested."""
    point = page.evaluate("""(() => {
      const e = document.getElementById(%s); e.scrollIntoView({block:'center'});
      const r = e.getBoundingClientRect(); return {x:r.x+r.width/2, y:r.y+r.height/2};
    })()""" % json.dumps(identifier))
    page.call("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **point})
    page.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **point})


def verify(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = args.executable.resolve()
    if not executable.is_file() or not args.chrome.is_file():
        raise FileNotFoundError("Build the native release binary and install Chrome before verification")
    port, debug_port = available_port(), available_port()
    profile = Path(tempfile.mkdtemp(prefix="lamp-performance-chrome-"))
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    server = browser = page = observer = None
    checks: list[str] = []
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Functional browser/audio/physics checks; synthetic inputs, no human performance approval",
        "measurement_limits": "Small functional-run sample on this Windows host with background work; no external audio capture, no Ubuntu or matched performance claim",
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256((PROJECT / "performances/demo.json").read_bytes()).hexdigest(),
        "viewer_dist_sha256": {p.relative_to(PROJECT / "viewer/dist").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((PROJECT / "viewer/dist").rglob("*")) if p.is_file()},
        "checks": checks,
        "scenes": [],
    }
    resources: list[dict] = []

    def sample_resources() -> None:
        if server:
            process = psutil.Process(server.pid)
            resources.append({"at_s": time.monotonic(), "rss_bytes": process.memory_info().rss,
                              "cpu_s": sum(process.cpu_times()[:2])})

    def perform(preset: str) -> str:
        change(page, "performance-preset", preset)
        previous = page.evaluate("lampVerification.latest.performance?.id ?? null")
        trusted_click(page, "performance-play")
        return wait_for(page, "lampVerification.latest.performance?.id && lampVerification.latest.performance.id !== %s && lampVerification.latest.performance.id" % json.dumps(previous))

    def stopped() -> None:
        wait_for(page, "!lampVerification.latest.performance && lampVerification.latest.activity === 'Stopped'")
        wait_for(page, "performanceVerification.sources.every(source => source.endedMs !== null || source.stoppedMs !== null)")

    try:
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([str(executable), "serve", "--port", str(port)], cwd=PROJECT,
                                      stdout=log, stderr=subprocess.STDOUT, **hidden)
        url = f"http://127.0.0.1:{port}"
        wait_json(url + "/api/scene")
        browser = subprocess.Popen([str(args.chrome), "--headless=new", "--no-first-run",
                                    "--no-default-browser-check",
                                    f"--remote-debugging-port={debug_port}", "--window-size=1440,1000",
                                    "--force-device-scale-factor=1", f"--user-data-dir={profile}", "about:blank"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden)
        endpoint = f"http://127.0.0.1:{debug_port}"
        report["browser"] = wait_json(endpoint + "/json/version")["Browser"]
        targets = wait_json(endpoint + "/json/list")
        page = DevTools(next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"))

        def prepare(target: DevTools) -> None:
            target.call("Page.enable")
            target.call("Runtime.enable")
            target.call("Page.addScriptToEvaluateOnNewDocument", {"source": PROBE + AUDIO_PROBE})
            target.call("Page.navigate", {"url": url})
            wait_for(target, "document.querySelector('#performance-play') && !document.querySelector('#performance-play').disabled")

        prepare(page)
        assert page.evaluate("document.querySelector('#performance-preset').options.length") == 9
        assert len(page.evaluate("lampVerification.hellos.at(-1).gestures")) == 126
        checks.append("Nine hardcoded performances and existing 126-clip library load together")
        capture(page, output / "ready.png")

        # Force slow preparation so Stop exercises async decode/fetch cancellation.
        page.evaluate("""window.originalPerformanceFetch = window.fetch;
          window.fetch = async (...args) => {
            if (String(args[0]).endsWith('.wav')) await new Promise(resolve => setTimeout(resolve, 700));
            return originalPerformanceFetch(...args);
          };""")
        trusted_click(page, "performance-play")
        wait_for(page, "document.querySelector('#performance-panel').dataset.preparation === 'true'")
        click(page, "stop")
        stopped()
        time.sleep(1)
        assert page.evaluate("performanceVerification.sources.length") == 0
        assert page.evaluate("performanceVerification.commands.filter(c => c.type === 'perform').length") == 0
        page.evaluate("window.fetch = window.originalPerformanceFetch")
        checks.append("Stop while audio is preparing prevents a delayed performance or speech start")

        for preset in ("greeting-happy", "greeting-sad", "greeting-angry", "question-happy", "reassurance-happy"):
            click(page, "reset")
            wait_for(page, "Object.values(lampVerification.latest.requested).every(v => Math.abs(v) < .002)")
            source_count = page.evaluate("performanceVerification.sources.length")
            performance_id = perform(preset)
            wait_for(page, f"performanceVerification.sources.length === {source_count + 1}")
            wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
            assert page.evaluate("document.querySelector('#pace').disabled")
            source = page.evaluate("performanceVerification.sources.at(-1)")
            assert source["state"] == "running" and source["peak"] > .01
            assert source["when"] >= source["contextTime"]
            assert source["connections"], "Audio source was never connected"
            if preset == "greeting-happy":
                capture(page, output / "speaking.png")
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                sample_resources()
                if page.evaluate("lampVerification.latest.performance?.phase === 'idle'"):
                    break
                time.sleep(.15)
            else:
                raise AssertionError(f"Performance did not settle: {preset}")
            wait_for(page, "performanceVerification.sources.at(-1).endedMs !== null")
            samples = page.evaluate("performanceVerification.samples.filter(s => s.performance?.id === %s)" % json.dumps(performance_id))
            phases = {sample["performance"]["phase"] for sample in samples}
            assert {"anticipating", "speaking", "settling", "idle"}.issubset(phases), phases
            assert any(abs(v) > .02 for sample in samples for v in sample["requested"].values()) or preset.endswith("neutral")
            state = page.evaluate("lampVerification.latest")
            limits = page.evaluate("lampVerification.hellos.at(-1).limits")
            assert samples[-1]["simulation"] - samples[0]["simulation"] > 1
            actual = np.array([list(sample["actual"].values()) for sample in samples])
            assert np.isfinite(actual).all() and np.ptp(actual, axis=0).max() > .01
            assert abs(source["duration"] - state["performance"]["audio_duration_s"]) < .001
            for sample in samples:
                assert sample["diagnostics"]["warnings"] == 0
                for joint, value in sample["requested"].items():
                    assert limits[joint]["lower"] <= value <= limits[joint]["upper"]
            report["scenes"].append({"preset": preset, "id": performance_id, "phases": sorted(phases),
                                     "audio": page.evaluate("performanceVerification.sources.at(-1)"),
                                     "duration_s": state["performance"]["audio_duration_s"],
                                     "max_tracking_error_deg": max(s["diagnostics"]["tracking_error_rad"] for s in samples) * 180 / np.pi,
                                     "max_actual_soft_limit_overshoot_deg": max(max(0, limits[j]["lower"] - v, v - limits[j]["upper"])
                                                                                for s in samples for j, v in s["actual"].items()) * 180 / np.pi,
                                     "max_contacts": max(s["diagnostics"]["contacts"] for s in samples), "samples": samples})
            click(page, "performance-stop")
            stopped()
        checks.append("Five complete audible-buffer performances traverse anticipation, speaking, settling and idle with bounded requested angles")

        def trajectory(scene: dict) -> np.ndarray:
            samples = scene["samples"]
            times = [s["time"] - s["performance"]["start_at_s"] for s in samples]
            grid = np.linspace(0, scene["duration_s"], 120)
            return np.array([np.interp(grid, times, [s["requested"][j] for s in samples]) for j in sorted(samples[0]["requested"])])

        traces = [trajectory(scene) for scene in report["scenes"]]
        differences = {f"{a}-{b}": float(np.sqrt(np.mean((traces[a] - traces[b]) ** 2)))
                       for a, b in ((0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4))}
        assert all(value > .005 for value in differences.values()), differences
        report["trajectory_rms_differences_rad"] = differences
        checks.append("Measured requested trajectories differ with both emotion and phrase content; recognizability remains a human judgment")

        # New input replaces current playback. A scoped late cancellation must
        # never cancel the replacement (covered on the server too).
        old = perform("greeting-happy")
        wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
        old_source = page.evaluate("performanceVerification.sources.length - 1")
        new = perform("greeting-sad")
        assert new != old
        wait_for(page, f"performanceVerification.sources[{old_source}]?.stoppedMs != null")
        click(page, "stop")
        stopped()
        checks.append("Replacement cancels old audio and Stop cancels the replacement")

        perform("greeting-angry")
        assert page.evaluate("lampVerification.latest.performance?.phase") == "anticipating"
        click(page, "stop")
        stopped()
        time.sleep(1)
        assert page.evaluate("!lampVerification.latest.performance")
        checks.append("Stop during anticipation prevents speech and idle from restarting")

        for override in ("reset", "manual"):
            perform("greeting-happy")
            wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
            if override == "reset":
                click(page, "reset")
            else:
                change(page, "base_yaw_joint", .12)
            wait_for(page, "!lampVerification.latest.performance")
            wait_for(page, "performanceVerification.sources.every(source => source.endedMs !== null)")
            click(page, "stop")
            stopped()
        checks.append("Reset and manual joint edits cancel speech and pending performance movement")

        perform("greeting-happy")
        wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
        page.evaluate("performanceVerification.contexts.at(-1).suspend()")
        stopped()
        assert page.evaluate("performanceVerification.sources.at(-1).stoppedMs !== null")
        page.evaluate("performanceVerification.contexts.at(-1).resume()")
        wait_for(page, "performanceVerification.sources.every(source => source.endedMs !== null)")
        assert page.evaluate("!lampVerification.latest.performance")
        checks.append("Audio suspension cancels the performance instead of silently drifting from motion")

        # A second viewer observes without creating an audio source; its Stop
        # still overrides the shared physical character.
        target_id = page.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        targets = wait_json(endpoint + "/json/list")
        observer = DevTools(next(t["webSocketDebuggerUrl"] for t in targets if t["id"] == target_id))
        prepare(observer)
        performance_id = perform("question-happy")
        wait_for(observer, "lampVerification.latest.performance?.id === %s" % json.dumps(performance_id))
        wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
        assert observer.evaluate("performanceVerification.sources.length") == 0
        click(observer, "stop")
        stopped()
        checks.append("Second tab observes silently; its Stop cancels owner audio and shared motion")

        perform("reassurance-happy")
        wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
        page.evaluate("performanceVerification.sockets.at(-1).close()")
        wait_for(observer, "!lampVerification.latest.performance")
        wait_for(page, "performanceVerification.sources.every(source => source.endedMs !== null)")
        wait_for(page, "document.querySelector('#performance-play') && !document.querySelector('#performance-play').disabled", seconds=15)
        time.sleep(1)
        assert page.evaluate("!lampVerification.latest.performance")
        checks.append("Owner disconnect cancels audio and motion; reconnect does not replay")

        perform("greeting-happy")
        wait_for(page, "lampVerification.latest.performance?.phase === 'speaking'")
        click(page, "play")
        wait_for(page, "!lampVerification.latest.performance && lampVerification.latest.activity.startsWith('Playing ')")
        wait_for(page, "performanceVerification.sources.every(source => source.endedMs !== null)")
        click(page, "stop")
        stopped()
        checks.append("Existing gesture playback overrides the performance and cancels its speech")

        page.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        time.sleep(.3)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.evaluate("document.querySelector('#performance-panel').scrollIntoView()")
        capture(page, output / "mobile.png")
        checks.append("Performance controls fit the 390px mobile layout")
        assert not page.evaluate("lampVerification.errors"), page.evaluate("lampVerification.errors")
        assert not observer.evaluate("lampVerification.errors"), observer.evaluate("lampVerification.errors")
        report["resources"] = resources
        report["native_peak_sampled_rss_mib"] = max(r["rss_bytes"] for r in resources) / 1048576
        if len(resources) > 1:
            report["native_mean_cpu_cores"] = (resources[-1]["cpu_s"] - resources[0]["cpu_s"]) / (resources[-1]["at_s"] - resources[0]["at_s"])
        report["commands"] = page.evaluate("performanceVerification.commands")
        report["audio_sources"] = page.evaluate("performanceVerification.sources")
        report["passed"] = True
        print(f"Passed {len(checks)} performance check groups; evidence: {output}", flush=True)
    except BaseException as error:
        report["passed"] = False
        report["failure"] = repr(error)
        if page:
            try:
                report["page_text"] = page.evaluate("document.body.innerText")
                report["browser_probe"] = page.evaluate("lampVerification")
                report["audio_sources"] = page.evaluate("performanceVerification.sources")
                capture(page, output / "failure.png")
            except Exception:
                pass
        raise
    finally:
        save_json(output / "result.json", report)
        for target in (observer, page):
            if target:
                target.connection.close()
        if browser:
            terminate_tree(browser)
        if server:
            terminate_tree(server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    binary = "lamp-controller.exe" if os.name == "nt" else "lamp-controller"
    parser.add_argument("--executable", type=Path, default=PROJECT / "controller/target/release" / binary)
    parser.add_argument("--chrome", type=Path, default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    verify(parser.parse_args())
