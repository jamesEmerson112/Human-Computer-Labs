"""Exercise animation pass/fail review in an isolated browser profile.

Run after building the native executable and viewer. Existing evidence folders
are preserved, and this harness never touches the user's browser or review data.
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


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.measure_viewer import DevTools, available_port, save_json, terminate_tree, wait_json
from tools.verify_native_viewer import PROBE, capture, change, click, core_gestures, gesture_names, local_catalog, verify_catalog, wait_for


STORAGE_KEY = "lamp-animation-review-v1"


def checkbox(page: DevTools, name: str) -> None:
    page.evaluate("""(() => {
      const input = [...document.querySelectorAll('[data-review-name]')].find(input => input.dataset.reviewName === %s);
      if (!input) throw new Error('Review checkbox missing');
      input.click();
    })()""" % json.dumps(name))


def import_file(page: DevTools, path: Path) -> None:
    document = page.call("DOM.getDocument")
    node = page.call("DOM.querySelector", {"nodeId": document["root"]["nodeId"], "selector": "#review-file"})
    page.call("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": [str(path.resolve())]})


def review_state(page: DevTools):
    return page.evaluate(f"JSON.parse(localStorage.getItem({json.dumps(STORAGE_KEY)}))")


def user_click(page: DevTools, identifier: str) -> None:
    point = page.evaluate("""(() => {
      const button = document.getElementById(%s);
      button.scrollIntoView({block:'center'});
      const box = button.getBoundingClientRect();
      return {x:box.x+box.width/2,y:box.y+box.height/2};
    })()""" % json.dumps(identifier))
    for event in ("mousePressed", "mouseReleased"):
        page.call("Input.dispatchMouseEvent", {"type": event, **point, "button": "left", "clickCount": 1})


def verify(args) -> None:
    expected = local_catalog()
    expected_names = {gesture["name"] for gesture in expected}
    count = len(expected)
    cores = core_gestures(expected)
    empty_summary = f"0 passed · {count} failed"
    passed_summary = f"1 passed · {count - 1} failed"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    downloads = output / "downloads"
    downloads.mkdir()
    executable = PROJECT / "controller/target/release" / ("lamp-controller.exe" if os.name == "nt" else "lamp-controller")
    port, debug_port = available_port(), available_port()
    profile = Path(tempfile.mkdtemp(prefix="lamp-review-verify-chrome-"))
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    server = browser = page = None
    checks = []
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Functional review verification with an isolated profile; not human animation approval",
        "catalog_count": count,
        "families": [gesture["group"] for gesture in cores],
        "expected_catalog": expected,
        "checks": checks,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "viewer_dist_sha256": {str(path.relative_to(PROJECT / "viewer/dist")): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((PROJECT / "viewer/dist").rglob("*")) if path.is_file()},
    }
    try:
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([str(executable), "serve", "--port", str(port)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, **hidden)
        url = f"http://127.0.0.1:{port}"
        wait_json(url + "/api/scene")
        browser = subprocess.Popen([str(args.chrome), "--headless=new", "--no-first-run", "--no-default-browser-check", f"--remote-debugging-port={debug_port}", "--window-size=1440,1000", "--force-device-scale-factor=1", f"--user-data-dir={profile}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden)
        endpoint = f"http://127.0.0.1:{debug_port}"
        version = wait_json(endpoint + "/json/version")
        result["browser_version"] = version["Browser"]
        targets = wait_json(endpoint + "/json/list")
        page = DevTools(next(target["webSocketDebuggerUrl"] for target in targets if target["type"] == "page"))
        page.call("Page.enable")
        page.call("Runtime.enable")
        page.call("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(downloads)})
        page.call("Page.addScriptToEvaluateOnNewDocument", {"source": PROBE})
        page.call("Page.navigate", {"url": url})
        wait_for(page, "document.querySelector('#play') && !document.querySelector('#play').disabled")
        verify_catalog(page.evaluate("lampVerification.hellos.at(-1).gestures"), expected)
        assert set(gesture_names(page)) == expected_names and len(gesture_names(page)) == count
        assert page.evaluate("document.querySelectorAll('[data-review-name]').length") == count
        assert page.evaluate("[...document.querySelectorAll('[data-review-name]')].every(input => !input.checked)")
        assert page.evaluate("document.querySelector('#review-summary').textContent") == empty_summary
        checks.append(f"All {count} gesture rows begin unchecked and count as failed in an isolated profile")
        capture(page, output / "review-initial.png")

        initial_commands = page.evaluate("lampVerification.commands.length")
        change(page, "gesture-search", "Nod")
        change(page, "gesture-select", "nod")
        checkbox(page, "nod")
        wait_for(page, f"document.querySelector('#review-summary').textContent === {json.dumps(passed_summary)}")
        assert page.evaluate("lampVerification.commands.length") == initial_commands
        notes = 'Smoke-test note only: <img src=x onerror="window.reviewInjected=true"> & "timing"\nSecond line.'
        change(page, "review-notes", notes)
        wait_for(page, f"JSON.parse(localStorage.getItem('{STORAGE_KEY}'))?.animations.some(animation => animation.name === 'nod' && animation.notes === {json.dumps(notes)})")
        assert not page.evaluate("window.reviewInjected === true")
        assert page.evaluate("lampVerification.commands.length") == initial_commands
        checks.append("Checking Nod records pass and safe text notes without sending any motion command")

        change(page, "gesture-search", "")
        change(page, "review-filter", "passed")
        assert gesture_names(page) == ["nod"]
        change(page, "review-filter", "failed")
        assert set(gesture_names(page)) == expected_names - {"nod"} and len(gesture_names(page)) == count - 1
        change(page, "review-filter", "all")
        for gesture in cores:
            family = gesture["group"]
            change(page, "group-filter", family)
            assert set(gesture_names(page)) == {item["name"] for item in expected if item["group"] == family}
            assert "nod" not in gesture_names(page)
        change(page, "group-filter", "All")
        change(page, "gesture-select", "nod")
        assert page.evaluate("document.querySelector('[data-review-name=nod]').checked")
        assert page.evaluate("document.querySelector('#review-notes').value") == notes
        checks.append("Family/search/pass-fail filtering preserves each animation's review and notes")

        for expected in (False, True):
            page.evaluate("document.querySelector('[data-review-name=nod]').focus()")
            page.call("Input.dispatchKeyEvent", {"type": "keyDown", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32})
            page.call("Input.dispatchKeyEvent", {"type": "keyUp", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32})
            wait_for(page, f"document.querySelector('[data-review-name=nod]').checked === {json.dumps(expected)}")
        assert page.evaluate("lampVerification.commands.length") == initial_commands
        checks.append("Space toggles the focused native review checkbox without playing a gesture")

        before_next = page.evaluate("document.querySelector('.gesture-option[aria-pressed=true]').dataset.gesture")
        click(page, "next-gesture")
        after_next = page.evaluate("document.querySelector('.gesture-option[aria-pressed=true]').dataset.gesture")
        assert before_next != after_next
        assert page.evaluate("lampVerification.commands.length") == initial_commands
        checks.append("Next advances the selected animation without autoplay")

        page.evaluate("document.body.dataset.verifyReload = 'pending'")
        page.call("Page.reload")
        wait_for(page, "document.body && !document.body.dataset.verifyReload && document.querySelector('#play') && !document.querySelector('#play').disabled")
        assert page.evaluate("document.querySelector('#review-summary').textContent") == passed_summary
        change(page, "gesture-select", "nod")
        assert page.evaluate("document.querySelector('#review-notes').value") == notes
        checks.append("Browser reload restores saved review status and notes")

        change(page, "review-filter", "passed")
        user_click(page, "export-review")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            reports = list(downloads.glob("*.json"))
            if reports:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Review export did not download a JSON file")
        exported = json.loads(reports[0].read_text(encoding="utf-8"))
        assert exported["type"] == "lamp-animation-review"
        assert exported["schema_version"] == 1
        assert exported["rule"] == "checked_pass_unchecked_fail"
        assert len(exported["animations"]) == count
        assert {animation["name"] for animation in exported["animations"]} == expected_names
        assert exported["summary"] == {"total": count, "passed": 1, "failed": count - 1}
        nod = next(animation for animation in exported["animations"] if animation["name"] == "nod")
        assert nod["status"] == "pass" and nod["notes"] == notes
        result["downloaded_report"] = reports[0].name
        checks.append(f"Actual browser JSON download contains all {count} animations even when only Passed is visible")

        change(page, "review-filter", "all")
        change(page, "gesture-select", "nod")
        checkbox(page, "nod")
        change(page, "review-notes", "temporary change")
        wait_for(page, f"document.querySelector('#review-summary').textContent === {json.dumps(empty_summary)}")
        import_file(page, reports[0])
        wait_for(page, f"document.querySelector('#review-summary').textContent === {json.dumps(passed_summary)}")
        change(page, "gesture-select", "nod")
        assert page.evaluate("document.querySelector('#review-notes').value") == notes
        before_invalid = review_state(page)
        previous_message = page.evaluate("document.querySelector('#review-message').textContent")
        invalid = json.loads(json.dumps(exported))
        invalid["animations"][0]["status"] = "unrecognized"
        invalid_path = output / "invalid-import.json"
        save_json(invalid_path, invalid)
        import_file(page, invalid_path)
        wait_for(page, f"document.querySelector('#review-message').textContent !== {json.dumps(previous_message)}")
        assert review_state(page) == before_invalid
        assert page.evaluate("lampVerification.commands.length") == 0
        checks.append("Import restores saved reviews; malformed import rejects atomically without changing saved data")
        capture(page, output / "review-invalid-import.png")
        import_file(page, reports[0])
        wait_for(page, "document.querySelector('#review-message').textContent.startsWith('Imported ')")
        change(page, "review-filter", "passed")
        page.evaluate("document.querySelector('.control-panel').scrollTop = 0")
        capture(page, output / "review-desktop.png")

        page.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        page.evaluate("document.querySelector('#review-summary').scrollIntoView({block:'start'})")
        time.sleep(0.3)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        capture(page, output / "review-mobile.png")
        checks.append("Mobile review controls remain within a390px viewport")
        result["browser_probe"] = page.evaluate("lampVerification")
        assert not result["browser_probe"]["errors"], result["browser_probe"]["errors"]
        result["passed"] = True
        print(f"Passed {len(checks)} animation-review browser checks", flush=True)
    except BaseException as error:
        result["passed"] = False
        result["failure"] = repr(error)
        if page:
            try:
                result["browser_probe"] = page.evaluate("window.lampVerification")
                result["page_text"] = page.evaluate("document.body.innerText")
                capture(page, output / "review-failure.png")
            except Exception:
                pass
        raise
    finally:
        save_json(output / "result.json", result)
        if page:
            page.connection.close()
        for process in (browser, server):
            if process:
                terminate_tree(process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome", type=Path, default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    verify(parser.parse_args())
