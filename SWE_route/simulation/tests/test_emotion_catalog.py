"""Protect wheel coverage, archived evidence, and conservative review migration."""

from collections import Counter
import json
from pathlib import Path
import shutil

import pytest

from lamp_sim.gestures import load_gestures
from lamp_sim.motion import load_limits
from tools.build_emotion_library import (
    ROOT, SOURCE_ID, TARGET_ID, catalog_paths, metadata, read_baseline, revision, run,
)


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_bytes())


def copied_project(tmp_path: Path) -> Path:
    for folder in ("catalogs", "gestures"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    return tmp_path


def test_selected_wheel_coverage_uses_full_paths_and_excludes_only_bad_branch():
    catalog = read_json("catalogs/emotion-wheel.json")
    paths = catalog_paths(catalog)
    assert catalog["catalog_id"] == TARGET_ID
    assert len(paths) == len(set(paths)) == 117
    assert Counter(len(path.split(" / ")) for path in paths) == {1: 6, 2: 37, 3: 74}
    assert Counter(path.split(" / ")[0] for path in paths) == {
        "Fearful": 19, "Angry": 25, "Disgusted": 13,
        "Sad": 19, "Happy": 28, "Surprised": 13,
    }
    words = Counter(path.split(" / ")[-1] for path in paths)
    assert len(words) == 114
    assert {word: count for word, count in words.items() if count > 1} == {
        "Inferior": 2, "Embarrassed": 2, "Disappointed": 2,
    }
    excluded = catalog["excluded"]["paths"]
    assert len(excluded) == len(set(excluded)) == 13
    assert all(path.split(" / ")[0] == "Bad" for path in excluded)
    assert not set(paths) & set(excluded)
    assert "Fearful / Anxious / Overwhelmed" in paths
    assert "Bad / Stressed / Overwhelmed" in excluded
    assert catalog["neutral"]["kind"] == "control_action"
    assert not catalog["neutral"]["included_in_emotion_count"]


def test_active_library_matches_selected_paths_and_preserves_utility_content():
    catalog = read_json("catalogs/emotion-wheel.json")
    library = load_gestures(load_limits())
    assert len(library) == 126
    assert {clip.label for clip in library.values() if clip.group} == set(catalog_paths(catalog))
    utility_names = {"nod", "shake", "bow", "curious", "scan", "excited", "idle_hold", "reach_hold", "limit_sweep"}
    assert {clip.name for clip in library.values() if not clip.group} == utility_names
    for name in utility_names:
        raw = (ROOT / "gestures" / f"{name}.json").read_bytes()
        assert b"\r\n" not in raw
        assert json.loads(raw) == json.loads((
            ROOT / "gestures" / "archive" / "six-core-wheel" / "clips" / f"{name}.json"
        ).read_bytes())
    assert all("neutral" not in name and "reset" not in name for name in library)
    assert library["idle_hold"].steps[0].kind == "hold_seconds"


def test_archive_manifest_accounts_for_every_original_file_and_document():
    archive = ROOT / "gestures" / "archive" / "six-core-wheel"
    manifest = read_json("gestures/archive/six-core-wheel/manifest.json")
    assert manifest["catalog_id"] == SOURCE_ID
    assert manifest["counts"] == {"emotion": 114, "utility": 9, "total": 123}
    files = {item["path"]: item for item in manifest["files"]}
    assert len(files) == len(manifest["files"]) == 125
    actual = {path.relative_to(archive).as_posix() for path in archive.rglob("*") if path.is_file()}
    assert actual == set(files) | {"manifest.json"}
    for relative, item in files.items():
        raw = (archive / relative).read_bytes()
        assert len(raw) == item["bytes"]
        assert revision(raw) == item["revision"]
    source, _ = read_baseline(ROOT, archive=False)
    assert len(source) == 123


def test_migration_has_complete_unambiguous_identity_and_hash_coverage():
    migration = read_json("catalogs/emotion-wheel-migration.json")
    source, _ = read_baseline(ROOT, archive=False)
    target = {path.stem: path.read_bytes() for path in (ROOT / "gestures").glob("*.json")}
    entries = migration["entries"]
    sources = [row["source"]["name"] for row in entries]
    targets = [row["target"]["name"] for row in entries]
    retired = {row["name"] for row in migration["retired"]}
    added = {row["name"] for row in migration["added"]}
    assert migration["source_catalog_id"] == SOURCE_ID
    assert migration["target_catalog_id"] == TARGET_ID
    assert len(sources) == len(set(sources)) == 103
    assert len(targets) == len(set(targets)) == 103
    assert set(sources).isdisjoint(retired)
    assert set(targets).isdisjoint(added)
    assert set(sources) | retired == source.keys()
    assert set(targets) | added == target.keys()
    assert len(retired) == 20
    assert len(added) == 23
    for row in entries:
        assert row["source"] == metadata(source[row["source"]["name"]])
        assert row["target"] == metadata(target[row["target"]["name"]])
        old = json.loads(source[row["source"]["name"]])
        new = json.loads(target[row["target"]["name"]])
        assert old["steps"] == new["steps"]
        assert old["units"] == new["units"] == "radians"
    for row in migration["added"]:
        assert {key: row[key] for key in ("name", "label", "group", "revision")} == metadata(target[row["name"]])
    for row in migration["retired"]:
        assert {key: row[key] for key in ("name", "label", "group", "revision")} == metadata(source[row["name"]])


def test_only_equivalent_meanings_and_steps_are_eligible_to_carry_reviews():
    audit = read_json("catalogs/emotion-wheel-audit.json")
    migration = read_json("catalogs/emotion-wheel-migration.json")
    assert audit["coverage"] == {
        "exact": 25, "family_rename": 40, "contextual_reuse": 29, "authored": 23,
    }
    assert len(audit["rows"]) == 117
    assert audit["exact_step_duplicates"] == []
    for row in audit["rows"]:
        assert row["preserve_review"] == (row["classification"] in {"exact", "family_rename"})
        assert row["motion_changed"] == (row["classification"] == "authored")
        assert row["reason"]
        if row["classification"] == "exact":
            assert {key: row["source"][key] for key in ("name", "label", "group")} == {
                key: row["target"][key] for key in ("name", "label", "group")
            }
        elif row["classification"] == "family_rename":
            assert row["source"]["label"].split(" / ")[1:] == row["path"].split(" / ")[1:]
    assert sum(row["preserve_review"] for row in migration["entries"]) == 74


def test_generator_reproduces_frozen_active_catalog_and_audit_without_writes():
    assert run(ROOT, apply=False) == {
        "source": 123, "target": 126, "mapped": 103,
        "added": 23, "retired": 20, "preserve_review": 74,
    }


def test_rebuild_refuses_edited_active_clip_without_changing_other_files(tmp_path):
    project = copied_project(tmp_path)
    edited = project / "gestures" / "emotion_happy.json"
    edited.write_bytes(edited.read_bytes() + b"\n")
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="unknown or edited active clip: emotion_happy"):
        run(project, apply=True)
    assert {path: path.read_bytes() for path in project.rglob("*") if path.is_file()} == before


def test_rebuild_refuses_corrupted_frozen_evidence(tmp_path):
    project = copied_project(tmp_path)
    edited = project / "gestures" / "archive" / "six-core-wheel" / "clips" / "nod.json"
    edited.write_bytes(edited.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen archive changed: clips/nod.json"):
        run(project, apply=True)


def test_initial_archive_refuses_an_unrecognized_catalog_without_writing(tmp_path):
    (tmp_path / "gestures").mkdir()
    (tmp_path / "gestures" / "unexpected.json").write_text(
        '{"name": "unexpected", "group": "New"}', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="123-clip original six-core library"):
        read_baseline(tmp_path, archive=True)
    assert not (tmp_path / "gestures" / "archive").exists()
