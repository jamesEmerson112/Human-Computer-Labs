"""Exercise the built Three.js viewer against an owned native server and Chrome.

This is a functional check, not a performance benchmark. Existing output folders
are never overwritten. Uses the locked Python environment and existing CDP helper.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.measure_viewer import DevTools, available_port, save_json, terminate_tree, wait_json
from lamp_sim.gestures import load_gestures
from lamp_sim.motion import load_limits


def local_catalog() -> list[dict]:
    """Use validated source files as the expectation, independently of the server."""
    revisions = {json.loads(path.read_bytes().decode("utf-8-sig"))["name"]: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (PROJECT / "gestures").glob("*.json")}
    return [{"name": gesture.name, "label": gesture.label, "description": gesture.description,
             "group": gesture.group, "revision": revisions[gesture.name]}
            for gesture in load_gestures(load_limits()).values()]


def verify_catalog(actual: list[dict], expected: list[dict]) -> None:
    if len({gesture["name"] for gesture in actual}) != len(actual):
        raise AssertionError("Server catalog contains duplicate gesture names")
    fields = ("name", "label", "description", "group", "revision")
    normalized = lambda catalog: {gesture["name"]: {key: gesture.get(key, "") for key in fields} for gesture in catalog}
    if normalized(actual) != normalized(expected):
        raise AssertionError("Server catalog differs from the validated local gesture files")


def core_gestures(catalog: list[dict]) -> list[dict]:
    """Each nonempty emotion group has exactly one same-label core clip."""
    cores = []
    for family in dict.fromkeys(gesture["group"] for gesture in catalog if gesture["group"]):
        matches = [gesture for gesture in catalog if gesture["group"] == family and gesture["label"] == family]
        if len(matches) != 1:
            raise AssertionError(f"Expected one core gesture for emotion family {family!r}")
        cores.append(matches[0])
    if not cores:
        raise AssertionError("No emotion families found in the local gesture catalog")
    return cores


PROBE = r"""
(() => {
  const probe = window.lampVerification = {errors: [], commands: [], acks: [], hellos: [], states: 0, latest: null};
  addEventListener('error', event => probe.errors.push({kind:'error', message:event.message, file:event.filename, line:event.lineno}));
  addEventListener('unhandledrejection', event => probe.errors.push({kind:'rejection', message:String(event.reason)}));
  const originalError = console.error;
  console.error = (...args) => {probe.errors.push({kind:'console',message:args.map(String).join(' ')}); originalError(...args);};
  const OriginalSocket = WebSocket;
  window.WebSocket = class extends OriginalSocket {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (message.type === 'state') {probe.latest = message; probe.states++;}
        if (message.type === 'hello') probe.hellos.push(message);
        if (message.type === 'ack') probe.acks.push(message);
      });
    }
    send(value) {probe.commands.push(JSON.parse(value)); super.send(value);}
  };
})();
"""


def wait_for(page: DevTools, expression: str, seconds: float = 10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = page.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"Browser condition did not become true: {expression}")


def click(page: DevTools, identifier: str) -> None:
    page.evaluate(f"document.getElementById({json.dumps(identifier)}).click()")


def change(page: DevTools, identifier: str, value: object) -> None:
    if identifier == "gesture-select":
        page.evaluate("""(() => {
          const list = document.getElementById('gesture-select');
          if (list.tagName === 'SELECT') {
            list.value = %s;
            list.dispatchEvent(new Event('change', {bubbles:true}));
          } else {
            const button = [...list.querySelectorAll('[data-gesture]')].find(button => button.dataset.gesture === %s);
            if (!button) throw new Error('Gesture is not in the visible list');
            button.click();
          }
        })()""" % (json.dumps(value), json.dumps(value)))
        return
    page.evaluate("""(() => {
      const input = document.getElementById(%s);
      input.value = %s;
      input.dispatchEvent(new Event('input', {bubbles:true}));
      input.dispatchEvent(new Event('change', {bubbles:true}));
    })()""" % (json.dumps(identifier), json.dumps(value)))


def gesture_names(page: DevTools) -> list[str]:
    return page.evaluate("""(() => {
      const list = document.getElementById('gesture-select');
      return list.tagName === 'SELECT'
        ? [...list.options].map(option => option.value)
        : [...list.querySelectorAll('[data-gesture]')].map(button => button.dataset.gesture);
    })()""")


def capture(page: DevTools, path: Path) -> None:
    result = page.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.write_bytes(base64.b64decode(result["data"]))


def verify(args) -> None:
    expected = local_catalog()
    expected_names = {gesture["name"] for gesture in expected}
    cores = core_gestures(expected)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = PROJECT / "controller/target/release/lamp-controller.exe"
    if os.name != "nt":
        executable = executable.with_suffix("")
    if not executable.is_file():
        raise FileNotFoundError(f"Build the native release executable first: {executable}")
    chrome = args.chrome
    if not chrome.is_file():
        raise FileNotFoundError(chrome)
    port, debug_port = available_port(), available_port()
    profile = Path(tempfile.mkdtemp(prefix="lamp-native-verify-chrome-"))
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    server = browser = page = None
    checks: list[str] = []
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Functional verification only; not a performance measurement",
        "catalog_count": len(expected),
        "families": [gesture["group"] for gesture in cores],
        "expected_catalog": expected,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "viewer_dist_sha256": {str(path.relative_to(PROJECT / "viewer/dist")): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((PROJECT / "viewer/dist").rglob("*")) if path.is_file()},
        "checks": checks,
    }

    def launch_server(suffix: str = ""):
        with (output / f"server{suffix}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(executable), "serve", "--port", str(port)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, **hidden)
        wait_json(f"http://127.0.0.1:{port}/api/scene")
        return process

    try:
        server = launch_server()
        browser = subprocess.Popen([str(chrome), "--headless=new", "--no-first-run", "--no-default-browser-check", f"--remote-debugging-port={debug_port}", "--window-size=1440,1000", "--force-device-scale-factor=1", f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden)
        endpoint = f"http://127.0.0.1:{debug_port}"
        version = wait_json(endpoint + "/json/version")
        manifest["browser_version"] = version["Browser"]
        targets = wait_json(endpoint + "/json/list")
        page = DevTools(next(target["webSocketDebuggerUrl"] for target in targets if target["type"] == "page"))
        page.call("Page.enable")
        page.call("Runtime.enable")
        page.call("Page.addScriptToEvaluateOnNewDocument", {"source": PROBE})
        url = f"http://127.0.0.1:{port}"
        page.call("Page.navigate", {"url": url})
        wait_for(page, "document.querySelector('#play') && !document.querySelector('#play').disabled")
        time.sleep(1)
        assert set(gesture_names(page)) == expected_names and len(gesture_names(page)) == len(expected)
        verify_catalog(page.evaluate("lampVerification.hellos.at(-1).gestures"), expected)
        assert page.evaluate("document.querySelectorAll('#joints input').length") == 5
        assert page.evaluate("Number(document.querySelector('#pace').max)") == 3
        assert page.evaluate("Number(document.querySelector('#pace').value)") == 1
        assert page.evaluate("lampVerification.latest.speed") == 1
        manifest["viewport"] = page.evaluate("({width:innerWidth,height:innerHeight,dpr:devicePixelRatio})")
        manifest["canvas"] = page.evaluate("({width:document.querySelector('canvas').width,height:document.querySelector('canvas').height})")
        checks.append(f"Loads all {len(expected)} local gestures, five joint controls, rendered scene, and fresh controller state")
        capture(page, output / "viewer-ready.png")

        for gesture in cores:
            family = gesture["group"]
            change(page, "group-filter", family)
            change(page, "gesture-select", gesture["name"])
            click(page, "play")
            wait_for(page, "lampVerification.latest.activity === %s" % json.dumps("Playing " + family))
            wait_for(page, "document.querySelector('#activity').textContent === %s" % json.dumps("Playing " + family))
            time.sleep(0.15)
            click(page, "stop")
            wait_for(page, "lampVerification.latest.activity === 'Stopped'")
        checks.append(f"All {len(cores)} core emotion gestures play through the UI and Stop interrupts each")

        change(page, "group-filter", "All")
        change(page, "gesture-search", "Nod")
        assert "nod" in gesture_names(page)
        change(page, "gesture-select", "nod")
        click(page, "play")
        wait_for(page, "lampVerification.latest.activity === 'Playing Nod'")
        page.call("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
        wait_for(page, "lampVerification.latest.activity === 'Stopped'")
        change(page, "gesture-search", "")
        assert set(gesture_names(page)) == expected_names and len(gesture_names(page)) == len(expected)
        checks.append("Search selects Nod; Escape interrupts it; clearing search restores the catalog")

        change(page, "pace", "0.35")
        wait_for(page, "Math.abs(lampVerification.latest.speed - 0.35) < 1e-9")
        change(page, "pace", "2")
        wait_for(page, "lampVerification.latest.speed === 2 && document.querySelector('#pace-value').textContent === '2.00×'")
        change(page, "pace", "3")
        wait_for(page, "lampVerification.latest.speed === 3 && document.querySelector('#pace-value').textContent === '3.00×'")
        page.evaluate("document.querySelector('#pace').scrollIntoView({block: 'center'})")
        capture(page, output / "viewer-pace-3x.png")
        for joint, target in (("base_yaw_joint", 0.15), ("shoulder_pitch_joint", 0.1), ("elbow_pitch_joint", -0.1), ("neck_yaw_joint", 0.1), ("head_pitch_joint", 0.435)):
            change(page, joint, target)
            wait_for(page, "Math.abs(lampVerification.latest.requested[%s] - %s) < 0.001" % (json.dumps(joint), target))
        checks.append("Pace defaults to 1x, accepts 2x and 3x, and all five manual targets update the native controller")
        change(page, "pace", "1")
        wait_for(page, "lampVerification.latest.speed === 1")
        change(page, "brightness", "7.5")
        wait_for(page, "lampVerification.latest.light.brightness === 7.5")
        click(page, "light-enabled")
        wait_for(page, "lampVerification.latest.light.enabled === false")
        time.sleep(0.3)
        capture(page, output / "viewer-light-off.png")
        click(page, "light-enabled")
        wait_for(page, "lampVerification.latest.light.enabled === true")
        click(page, "axes")
        click(page, "light-view")
        time.sleep(0.5)
        capture(page, output / "viewer-light-view.png")
        click(page, "axes")
        click(page, "reset-view")
        change(page, "brightness", "5")
        checks.append("Brightness/on/off and local axes/light-view/reset-view controls respond")

        page.evaluate("document.querySelector('.diagnostics').open = true")
        for mode in ("mujoco-source", "mujoco-kinematic", "mujoco"):
            change(page, "backend-mode", mode)
            wait_for(page, "lampVerification.latest.backend === %s" % json.dumps(mode))
        hello_count = page.evaluate("lampVerification.hellos.length")
        click(page, "reload")
        wait_for(page, f"lampVerification.hellos.length > {hello_count}")
        assert set(gesture_names(page)) == expected_names and len(gesture_names(page)) == len(expected)
        click(page, "reset")
        wait_for(page, "Object.values(lampVerification.latest.requested).every(value => Math.abs(value) < 0.001)")
        checks.append("All three simulation modes, gesture reload, and Reset pose work")

        pre_reload_errors = page.evaluate("lampVerification.errors")
        pre_reload_commands = page.evaluate("lampVerification.commands")
        assert not pre_reload_errors, pre_reload_errors
        assert page.evaluate("lampVerification.acks.every(ack => ack.ok)")
        page.evaluate("document.body.dataset.verifyReload = 'pending'")
        page.call("Page.reload")
        wait_for(page, "document.body && !document.body.dataset.verifyReload && document.querySelector('#play') && !document.querySelector('#play').disabled")
        assert page.evaluate("lampVerification.latest.backend") == "mujoco"
        checks.append("Browser reload obtains a complete scene/catalog/current state")

        terminate_tree(server)
        server = None
        wait_for(page, "[...document.querySelectorAll('[data-command]')].every(element => element.disabled)")
        before_offline = page.evaluate("lampVerification.commands.length")
        click(page, "play")
        change(page, "pace", "0.7")
        assert page.evaluate("lampVerification.commands.length") == before_offline
        assert page.evaluate("document.querySelector('#connection-status').textContent").lower().find("connect") >= 0
        server = launch_server("-restart")
        wait_for(page, "document.querySelector('#play') && !document.querySelector('#play').disabled", seconds=15)
        assert page.evaluate("lampVerification.commands.length") == before_offline
        assert page.evaluate("lampVerification.latest.speed") == 1
        assert page.evaluate("lampVerification.latest.activity") == "Ready"
        checks.append("Server loss disables commands; offline actions are not sent or replayed; restart receives fresh state")

        page.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        time.sleep(0.5)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        capture(page, output / "viewer-mobile.png")
        checks.append("390px responsive layout has no horizontal page overflow")
        page.call("Emulation.clearDeviceMetricsOverride")
        time.sleep(0.5)
        capture(page, output / "viewer-final.png")
        manifest["commands_before_reload"] = pre_reload_commands
        manifest["browser_probe"] = page.evaluate("lampVerification")
        assert not manifest["browser_probe"]["errors"], manifest["browser_probe"]["errors"]
        manifest["passed"] = True
        print(f"Passed {len(checks)} browser checks; screenshots: {output}", flush=True)
    except BaseException as error:
        manifest["passed"] = False
        manifest["failure"] = repr(error)
        if page:
            try:
                manifest["browser_probe"] = page.evaluate("window.lampVerification")
                manifest["page_text"] = page.evaluate("document.body.innerText")
                capture(page, output / "viewer-failure.png")
            except Exception:
                pass
        raise
    finally:
        save_json(output / "result.json", manifest)
        if page:
            page.connection.close()
        if browser:
            terminate_tree(browser)
        if server:
            terminate_tree(server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome", type=Path, default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    verify(parser.parse_args())
