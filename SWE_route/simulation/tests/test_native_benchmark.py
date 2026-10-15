"""A stale benchmark must fail before starting expensive, unmatched trials."""

import json

import pytest

from benchmarks import compare_native


@pytest.fixture
def reference(tmp_path, monkeypatch):
    hashes = {
        "simulation/gestures/nod.json": "a" * 64,
        "starter_pack/robot/lamp.urdf": "b" * 64,
        "simulation/lamp_sim/motion.py": "c" * 64,
        "simulation/lamp_sim/backends.py": "d" * 64,
    }
    monkeypatch.setattr(compare_native, "fingerprints", lambda: hashes.copy())
    monkeypatch.setattr(compare_native, "load_limits", lambda: {})
    monkeypatch.setattr(compare_native, "load_gestures", lambda limits: {"nod": object()})
    manifest = {"source_sha256": hashes.copy(), "speed": 0.5, "control_hz": 30,
                "physics_hz": 240, "settling_tail_s": 1.0}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for backend in ("mujoco-kinematic", "mujoco-source", "mujoco"):
        (tmp_path / f"{backend}.csv.gz").write_bytes(b"trace")
        (tmp_path / f"{backend}.json").write_text(
            json.dumps({"backend": backend, "trials": [{"gesture": "nod", "repeat": 1}]}), encoding="utf-8")
    return tmp_path, manifest, hashes


def test_matching_reference_is_accepted(reference):
    path, _, _ = reference
    result = compare_native.verify_reference(path)
    assert result["catalog_match"] is True
    assert result["gesture_count"] == 1
    assert len(result["manifest_sha256"]) == 64


@pytest.mark.parametrize("change", ["add", "remove", "edit", "motion", "physics", "speed", "names", "trace"])
def test_unmatched_reference_explains_how_to_regenerate(reference, change):
    path, manifest, hashes = reference
    if change == "add":
        hashes["simulation/gestures/new.json"] = "e" * 64
    elif change == "remove":
        del hashes["simulation/gestures/nod.json"]
    elif change == "edit":
        hashes["simulation/gestures/nod.json"] = "e" * 64
    elif change in ("motion", "physics"):
        hashes[f"simulation/lamp_sim/{'motion' if change == 'motion' else 'backends'}.py"] = "e" * 64
    elif change == "speed":
        manifest["speed"] = 1.0
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif change == "names":
        (path / "mujoco.json").write_text(json.dumps({"backend": "mujoco", "trials": [{"gesture": "other"}]}), encoding="utf-8")
    elif change == "trace":
        (path / "mujoco.csv.gz").unlink()
    with pytest.raises(ValueError, match="Cannot compare unmatched benchmark workloads.*Generate a fresh reference"):
        compare_native.verify_reference(path)


def test_preflight_failure_creates_no_output_or_process(reference, monkeypatch):
    path, _, hashes = reference
    hashes["simulation/gestures/nod.json"] = "e" * 64
    destination = path / "must-not-exist"
    monkeypatch.setattr(compare_native.sys, "argv", ["compare_native.py", "--reference", str(path), "--output", str(destination)])
    monkeypatch.setattr(compare_native, "run_sampled", lambda *args: pytest.fail("A measurement started before preflight"))
    with pytest.raises(SystemExit) as error:
        compare_native.main()
    assert error.value.code == 2
    assert not destination.exists()
