"""Build the selected wheel from a frozen, byte-preserved six-core baseline.

Run with --apply once to archive, write the audit, and activate the target.
The default --check verifies reproducibility without changing any files.
An edited active clip is never overwritten by --apply: changes must be reviewed
and incorporated into this authoring source before a later rebuild.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "six-core-wheel-2026-10-13"
TARGET_ID = "emotion-wheel-without-bad-2026-10-14"
SOURCE_COMMIT = "4ec9606"
ROOT_ALIASES = {
    "Angry": "Anger", "Fearful": "Fear", "Disgusted": "Disgust",
    "Surprised": "Surprise", "Happy": "Happy", "Sad": "Sad",
}
JOINTS = (
    "base_yaw_joint", "shoulder_pitch_joint", "elbow_pitch_joint",
    "neck_yaw_joint", "head_pitch_joint",
)

# Each contextual reuse has a single source. None transfers a human approval:
# the branch, depth, or intended meaning changed even when the motion did not.
REUSES = {
    "Fearful / Weak": ("emotion_fear_submissive", "A small sustained bow can suggest reduced agency; the new Weak meaning needs review."),
    "Fearful / Weak / Worthless": ("emotion_fear_submissive_worthless", "Retain the slow contraction for the same leaf word, now under Weak."),
    "Fearful / Weak / Insignificant": ("emotion_fear_submissive_insignificant", "Retain the two-stage contraction for the same leaf word, now under Weak."),
    "Fearful / Rejected / Excluded": ("emotion_fear_rejected_alienated", "A withdrawn side-facing stance is a candidate for exclusion; Alienated is not an exact synonym."),
    "Fearful / Threatened": ("emotion_anger_threatened", "The old braced retreat is relevant to threat but moves from Anger to Fearful."),
    "Angry / Let down / Resentful": ("emotion_anger_hateful_resentful", "Keep the low sideways withdrawal; Resentful now belongs under Let down."),
    "Angry / Humiliated": ("emotion_fear_humiliated", "Retain the low averted posture for Humiliated, now in the Angry family."),
    "Angry / Humiliated / Disrespected": ("emotion_fear_humiliated_disrespected", "Retain the interrupted approach for Disrespected, now in the Angry family."),
    "Angry / Humiliated / Ridiculed": ("emotion_fear_humiliated_ridiculed", "Retain the double flinch for Ridiculed, now in the Angry family."),
    "Angry / Bitter": ("emotion_anger_hateful", "The deliberate refusing sweep is a candidate for lingering bitterness; meaning changed."),
    "Angry / Bitter / Violated": ("emotion_anger_hateful_violated", "Retain the protective contraction for Violated, now under Bitter."),
    "Angry / Mad / Jealous": ("emotion_anger_threatened_jealous", "Retain the fixed sideways watch for Jealous, now under Mad."),
    "Angry / Frustrated / Annoyed": ("emotion_anger_frustrated_irritated", "Two small neck flicks fit a candidate annoyance cue; the label changes from Irritated."),
    "Disgusted / Disapproving": ("emotion_disgust_disapproval", "Disapproval changes grammatical form to Disapproving; conservative review reset for this branch rename."),
    "Disgusted / Disapproving / Judgmental": ("emotion_disgust_disapproval_judgmental", "Retain the measured inspection; conservative review reset for the Disapproval branch rename."),
    "Disgusted / Repelled": ("emotion_disgust_avoidance", "A sustained averted stance is a candidate for repulsion; Avoidance described an action."),
    "Disgusted / Repelled / Hesitant": ("emotion_disgust_avoidance_hesitant", "Retain the interrupted approach for Hesitant, now under Repelled."),
    "Sad / Hurt": ("emotion_anger_hurt", "The flinch followed by a droop is retained for Hurt, moving from Anger to Sad."),
    "Sad / Hurt / Embarrassed": ("emotion_anger_hurt_embarrassed", "Retain the hesitant averted dip for Embarrassed, moving from Anger to Sad."),
    "Sad / Despair / Grief": ("emotion_anger_hurt_devastated", "A heavy two-stage fold is a candidate for grief; Devastated is not treated as an equivalent label."),
    "Sad / Vulnerable": ("emotion_sad_despair_vulnerable", "The tentative retreat moves from an outer leaf to its own Vulnerable branch."),
    "Sad / Vulnerable / Victimized": ("emotion_sad_abandoned_victimized", "Retain the protective contraction for Victimized, now under Vulnerable."),
    "Happy / Optimistic / Hopeful": ("emotion_happy_peaceful_hopeful", "Retain the gradual upward gaze for Hopeful, now under Optimistic."),
    "Happy / Trusting / Intimate": ("emotion_happy_intimate", "A close lingering approach becomes the Intimate leaf under Trusting, rather than a middle-ring category."),
    "Happy / Trusting / Sensitive": ("emotion_happy_intimate_sensitive", "Retain the careful approach for Sensitive, now under Trusting."),
    "Happy / Accepted / Valued": ("emotion_happy_accepted_fulfilled", "The relaxed open pose is a candidate for feeling valued; Fulfilled expressed a different meaning."),
    "Happy / Content / Joyful": ("emotion_happy_joyful", "Retain the welcoming double bounce for Joyful as a leaf under Content."),
    "Happy / Content / Free": ("emotion_happy_joyful_liberated", "The broad unfolding is a candidate for feeling free, moved from Liberated under Joyful."),
    "Happy / Playful": ("emotion_happy_intimate_playful", "The peek and quick bob become a middle-ring Playful category rather than a leaf under Intimate."),
}


def pose(seconds: float, *angles: float) -> dict:
    """An explicit five-joint relative pose; no generated random variation."""
    if len(angles) != len(JOINTS):
        raise ValueError("A pose requires all five joint angles")
    return {
        "offsets": dict(zip(JOINTS, angles)), "on_limit": "reverse",
        "duration_seconds": seconds,
    }


def hold(seconds: float) -> dict:
    return {"hold_seconds": seconds}


def authored(description: str, reason: str, *steps: dict, recovery: float = 0.8) -> dict:
    return {
        "description": description, "reason": reason,
        "steps": [*steps, {"return_to_start": True, "duration_seconds": recovery}],
    }


# Five values per pose: base yaw, shoulder, elbow, neck yaw, head pitch.
# At zero the lamp looks along -x. Negative shoulder/elbow lean toward that
# subject; positive head pitch lowers the shade. The choices below are authored
# hypotheses, not measured emotional recognition. Long pauses are intentional.
AUTHORED = {
    "Fearful / Scared / Helpless": authored(
        "Try a small escape twice, then fold into a guarded pause.",
        "Helpless emphasizes blocked action, unlike the retained Frightened double flinch.",
        pose(0.35, 0.0, 0.12, -0.22, 0.0, -0.08),
        pose(0.5, -0.14, 0.07, -0.12, -0.1, -0.04), hold(0.22),
        pose(0.45, 0.12, 0.11, -0.2, 0.1, 0.06), hold(0.22),
        pose(0.85, 0.0, 0.2, -0.38, 0.02, 0.3), hold(0.9), recovery=1.0),
    "Fearful / Rejected / Persecuted": authored(
        "Check behind each side and withdraw while keeping the shade watchful.",
        "Persecuted suggests being pursued; repeated rearward checks differ from Excluded's held withdrawal.",
        pose(0.4, -0.18, 0.16, -0.18, -0.28, -0.12), hold(0.3),
        pose(0.55, 0.2, 0.24, -0.28, 0.32, -0.1), hold(0.45),
        pose(0.35, -0.08, 0.3, -0.4, -0.22, 0.05), hold(0.8)),
    "Fearful / Threatened / Nervous": authored(
        "Make three small anticipatory checks, pausing before each change of gaze.",
        "Nervous uses small uneven checking turns rather than the larger sustained exposure retreat.",
        pose(0.4, 0.0, 0.1, -0.18, -0.14, -0.03), hold(0.2),
        pose(0.3, 0.0, 0.11, -0.2, 0.14, -0.07), hold(0.38),
        pose(0.3, 0.0, 0.13, -0.23, -0.08, 0.02), hold(0.65)),
    "Fearful / Threatened / Exposed": authored(
        "Freeze with the shade raised, then tuck inward and turn partly away.",
        "Exposed begins in visible openness before a protective withdrawal, unlike a repeated nervous scan.",
        pose(0.45, 0.0, -0.06, -0.08, 0.0, -0.22), hold(0.65),
        pose(0.5, 0.16, 0.18, -0.32, 0.25, 0.16),
        pose(0.7, 0.2, 0.2, -0.42, 0.3, 0.32), hold(0.8)),
    "Angry / Let down": authored(
        "Offer an expectant approach, stop, and withdraw into a disappointed sideways hold.",
        "Let down needs failed expectation; the original Hurt motion is reserved for Sad / Hurt.",
        pose(0.7, 0.0, -0.12, -0.1, 0.0, -0.16), hold(0.65),
        pose(0.8, 0.0, 0.12, -0.24, 0.0, 0.22),
        pose(0.7, -0.12, 0.14, -0.28, -0.18, 0.28), hold(0.75), recovery=1.0),
    "Angry / Let down / Betrayed": authored(
        "Begin an open approach, recoil, then look back with a rigid sideways stare.",
        "Betrayed juxtaposes initial openness and a sudden guarded look back; Resentful stays withdrawn.",
        pose(0.7, -0.06, -0.18, -0.12, 0.12, -0.18), hold(0.25),
        pose(0.3, 0.18, 0.16, -0.28, 0.28, 0.12), hold(0.4),
        pose(0.65, 0.18, 0.16, -0.28, -0.24, 0.05), hold(1.0)),
    "Angry / Bitter / Indignant": authored(
        "Lift into a firm upright stance and give one measured refusing sweep.",
        "Indignant expresses a firm objection with an upright head rather than Violated's protective contraction.",
        pose(0.5, 0.0, -0.12, -0.14, 0.0, -0.26), hold(0.35),
        pose(0.5, 0.0, -0.12, -0.14, -0.26, -0.2),
        pose(0.6, 0.0, -0.12, -0.14, 0.26, -0.2),
        pose(0.5, 0.0, -0.12, -0.14, 0.0, -0.22), hold(0.7)),
    "Angry / Distant / Numb": authored(
        "Turn away slowly, let the shade settle, and remain still before recovering.",
        "Numb deliberately has low movement and a long still interval; it does not use Suspicious checking behavior.",
        pose(1.2, 0.24, 0.06, -0.12, 0.18, 0.06),
        pose(1.3, 0.24, 0.06, -0.12, 0.18, 0.22), hold(1.4), recovery=1.3),
    "Angry / Critical / Dismissive": authored(
        "Give a short sideways head flick, turn the body away, and break the gaze.",
        "Dismissive ends attention decisively, unlike Skeptical's continuing sideways inspection.",
        pose(0.45, 0.0, -0.06, -0.06, -0.12, -0.12),
        pose(0.3, 0.0, -0.06, -0.06, 0.3, 0.12),
        pose(0.7, 0.28, 0.08, -0.1, 0.38, 0.16), hold(0.9)),
    "Disgusted / Disapproving / Embarrassed": authored(
        "Avert a lowered shade, make a brief reluctant check, and turn away again.",
        "This Embarrassed branch depicts discomfort at what is observed; Sad / Hurt / Embarrassed keeps its inward hesitant bow.",
        pose(0.55, 0.14, 0.12, -0.16, 0.28, 0.16), hold(0.45),
        pose(0.4, 0.14, 0.12, -0.16, -0.06, 0.12), hold(0.2),
        pose(0.5, 0.2, 0.16, -0.2, 0.34, 0.24), hold(0.7)),
    "Disgusted / Disappointed / Appalled": authored(
        "Recoil with the head raised, freeze, then withdraw while turning aside.",
        "Appalled combines disbelief and rejection; the retained Revolted clip has a repeated shudder.",
        pose(0.35, -0.06, 0.24, -0.22, -0.06, -0.3), hold(0.95),
        pose(0.8, -0.24, 0.28, -0.32, -0.3, 0.1), hold(0.6)),
    "Disgusted / Awful / Nauseated": authored(
        "Fold gradually, make a slow shallow head roll from side to side, and settle low.",
        "Nauseated uses a slow uncomfortable sway, not the old Revulsion clip's sharp tremors.",
        pose(1.0, 0.0, 0.12, -0.3, -0.16, 0.32),
        pose(0.85, 0.04, 0.16, -0.35, 0.16, 0.4),
        pose(1.0, 0.0, 0.16, -0.38, -0.06, 0.45), hold(0.65), recovery=1.2),
    "Disgusted / Repelled / Horrified": authored(
        "Retreat abruptly, hold a high watchful head, then avert the whole upper body.",
        "Horrified has a large fast protective retreat and sustained avoidance; Hesitant makes an interrupted approach.",
        pose(0.3, 0.0, 0.3, -0.38, 0.0, -0.28), hold(0.7),
        pose(0.6, 0.32, 0.3, -0.42, 0.36, -0.08), hold(0.9), recovery=1.0),
    "Sad / Hurt / Disappointed": authored(
        "Make a hopeful small lift, wait for a response, and slowly let the pose fall.",
        "Disappointed in Hurt emphasizes unreturned expectation; Disgusted / Disappointed retains its distinct older movement.",
        pose(0.75, 0.0, -0.1, -0.08, 0.04, -0.14), hold(0.9),
        pose(1.2, 0.0, 0.02, -0.2, 0.04, 0.3), hold(0.8), recovery=1.2),
    "Sad / Vulnerable / Fragile": authored(
        "Attempt a delicate lift, falter, and settle into a protected compact pose.",
        "Fragile uses an interrupted low-amplitude lift; Victimized holds a stronger defensive withdrawal.",
        pose(0.8, 0.0, 0.08, -0.2, -0.08, 0.18),
        pose(0.65, 0.0, 0.02, -0.15, -0.04, 0.06), hold(0.3),
        pose(0.85, 0.02, 0.1, -0.28, -0.1, 0.3), hold(1.0), recovery=1.2),
    "Happy / Trusting": authored(
        "Pause after a gentle approach, then move closer and offer one soft acknowledgment.",
        "Trusting is a gradual commitment to approach; Intimate stays close and Sensitive uses smaller tentative movement.",
        pose(0.85, 0.0, -0.12, -0.1, 0.0, -0.1), hold(0.5),
        pose(0.95, 0.0, -0.24, -0.18, 0.04, -0.14),
        pose(0.6, 0.0, -0.24, -0.18, 0.04, 0.04),
        pose(0.6, 0.0, -0.24, -0.18, 0.04, -0.14), hold(0.65), recovery=1.0),
    "Happy / Peaceful / Thankful": authored(
        "Offer a slow appreciative bow, rise, and hold a gentle forward-facing pose.",
        "Thankful centers a deliberate bow and recovery, while Loving remains close and softly dips.",
        pose(0.8, 0.0, -0.08, -0.08, 0.0, 0.18),
        pose(0.75, 0.0, -0.12, -0.14, 0.0, 0.32), hold(0.45),
        pose(1.0, 0.0, -0.14, -0.1, 0.06, -0.12), hold(0.8), recovery=1.0),
    "Happy / Powerful / Creative": authored(
        "Explore two contrasting viewing angles, pause, and rise into a decisive new direction.",
        "Creative uses exploration followed by commitment, unlike Courageous's retreat then direct advance.",
        pose(0.65, -0.14, -0.12, -0.1, 0.18, -0.16), hold(0.25),
        pose(0.75, 0.16, -0.2, -0.14, -0.22, 0.04), hold(0.55),
        pose(0.55, -0.08, -0.26, -0.2, -0.06, -0.28), hold(0.7)),
    "Happy / Proud / Successful": authored(
        "Make a small effortful rise, settle tall, and punctuate it with one clear nod.",
        "Successful shows effort resolving into achievement; Confident starts with steady assured movement.",
        pose(0.55, 0.0, 0.06, -0.14, 0.0, 0.08), hold(0.2),
        pose(0.8, 0.0, -0.2, -0.16, 0.06, -0.28), hold(0.45),
        pose(0.45, 0.0, -0.2, -0.16, 0.06, -0.02),
        pose(0.5, 0.0, -0.2, -0.16, 0.06, -0.28), hold(0.6)),
    "Happy / Interested / Curious": authored(
        "Lean closer, tilt the gaze to one side, and approach again after a thoughtful pause.",
        "Curious focuses on one object with a second approach; Inquisitive's retained clip checks two viewing angles.",
        pose(0.75, 0.0, -0.14, -0.1, 0.12, -0.12), hold(0.5),
        pose(0.65, -0.06, -0.2, -0.16, 0.24, 0.02), hold(0.6),
        pose(0.8, -0.06, -0.28, -0.2, 0.08, -0.12), hold(0.55)),
    "Happy / Content": authored(
        "Settle into a relaxed open pose, breathe through one shallow rise, and rest.",
        "Content is restrained and unhurried; Joyful and Free are its more active distinct leaves.",
        pose(1.2, 0.06, -0.08, -0.06, -0.06, -0.08),
        pose(1.0, 0.06, -0.1, -0.08, -0.06, -0.14),
        pose(1.1, 0.06, -0.08, -0.06, -0.06, -0.08), hold(1.1), recovery=1.2),
    "Happy / Playful / Cheeky": authored(
        "Peek sideways, look back while the body stays turned, and add a quick teasing dip.",
        "Cheeky emphasizes a mismatched body and head direction followed by a teasing beat; no arbitrary jitter.",
        pose(0.5, -0.22, -0.12, -0.12, -0.18, -0.12), hold(0.3),
        pose(0.35, -0.22, -0.12, -0.12, 0.32, -0.14), hold(0.45),
        pose(0.3, -0.22, -0.12, -0.12, 0.32, 0.1),
        pose(0.35, -0.22, -0.12, -0.12, 0.32, -0.14), hold(0.35)),
    "Happy / Playful / Aroused": authored(
        "Wake into an alert lifted pose, sweep toward a stimulus, and lean in with interest.",
        "Aroused is interpreted as heightened activation and attention, expressed by a wake-and-orient sequence.",
        pose(0.55, 0.0, -0.08, -0.1, 0.0, -0.26), hold(0.25),
        pose(0.6, 0.18, -0.16, -0.14, 0.22, -0.22),
        pose(0.65, 0.1, -0.28, -0.2, 0.08, -0.18), hold(0.8)),
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def revision(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def active_bytes(raw: bytes) -> bytes:
    """Keep active files LF-stable across checkouts; archive raw bytes separately."""
    return raw.replace(b"\r\n", b"\n")


def metadata(raw: bytes) -> dict:
    clip = json.loads(raw)
    return {key: clip.get(key, "") for key in ("name", "label", "group")} | {"revision": revision(raw)}


def clip_name(path: str) -> str:
    return "emotion_" + re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")


def catalog_paths(catalog: dict) -> list[str]:
    paths = []
    for family in catalog["families"]:
        paths.append(family["label"])
        for branch in family["branches"]:
            parent = family["label"] + " / " + branch["label"]
            paths.append(parent)
            paths.extend(parent + " / " + word for word in branch["emotions"])
    return paths


def read_baseline(root: Path, *, archive: bool) -> tuple[dict[str, bytes], dict]:
    snapshot = root / "gestures" / "archive" / "six-core-wheel"
    manifest_path = snapshot / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest["catalog_id"] != SOURCE_ID:
            raise ValueError("Archive catalog ID mismatch")
        raws = {}
        for item in manifest["files"]:
            file = snapshot / item["path"]
            if snapshot.resolve() not in file.resolve().parents:
                raise ValueError("Archive path escapes snapshot")
            raw = file.read_bytes()
            if revision(raw) != item["revision"]:
                raise ValueError(f"Frozen archive changed: {item['path']}")
            if item["path"].startswith("clips/"):
                raws[json.loads(raw)["name"]] = raw
        if len(raws) != 123:
            raise ValueError("Frozen archive must contain exactly 123 original clips")
        return raws, manifest
    if not archive:
        raise ValueError("No frozen baseline; use --apply for the initial audited migration")
    if snapshot.exists() and any(snapshot.iterdir()):
        raise ValueError("Incomplete archive exists; inspect it before retrying")
    paths = sorted((root / "gestures").glob("*.json"))
    raws = {json.loads(path.read_bytes())["name"]: path.read_bytes() for path in paths}
    emotions = [json.loads(raw) for raw in raws.values() if json.loads(raw).get("group")]
    if len(raws) != 123 or len(emotions) != 114 or {clip["group"] for clip in emotions} != set(ROOT_ALIASES.values()):
        raise ValueError("Initial archive requires the 123-clip original six-core library")
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "clips").mkdir()
    files = []
    for path in [*paths, root / "gestures" / "emotions.md", root / "gestures" / "README.md"]:
        relative = "clips/" + path.name if path.suffix == ".json" else path.name
        destination = snapshot / relative
        shutil.copyfile(path, destination)
        raw = path.read_bytes()
        if destination.read_bytes() != raw:
            raise ValueError(f"Archive copy failed: {path.name}")
        files.append({"path": relative, "revision": revision(raw), "bytes": len(raw)})
    manifest = {
        "schema_version": 1, "catalog_id": SOURCE_ID, "source_commit": SOURCE_COMMIT,
        "provenance": "Raw active files captured before activating the selected wheel; original baseline and docs are immutable.",
        "counts": {"emotion": 114, "utility": 9, "total": 123}, "files": files,
    }
    manifest_path.write_bytes(json_bytes(manifest))
    return raws, manifest


def build(root: Path, source: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, bytes]]:
    catalog = json.loads((root / "catalogs" / "emotion-wheel.json").read_bytes())
    if catalog["catalog_id"] != TARGET_ID or catalog["counts"] != {
        "core": 6, "middle": 37, "outer": 74, "total": 117, "distinct_words": 114,
    }:
        raise ValueError("The canonical wheel must contain 117 entries excluding Bad")
    paths = catalog_paths(catalog)
    if len(paths) != 117 or len(set(paths)) != 117 or any(path.split(" / ")[0] == "Bad" for path in paths):
        raise ValueError("Invalid target path coverage")
    targets, rows, mappings, added = {}, [], [], []
    consumed, contextual = set(), set()
    for name, raw in sorted(source.items()):
        clip = json.loads(raw)
        if not clip.get("group"):
            targets[name] = active_bytes(raw)
            consumed.add(name)
            mappings.append({"source": metadata(raw), "target": metadata(targets[name]), "preserve_review": True, "reason": "Original utility content and motion retained; active line endings normalized to LF."})
    for path in paths:
        target_name = clip_name(path)
        labels = path.split(" / ")
        old_path = " / ".join([ROOT_ALIASES[labels[0]], *labels[1:]])
        old_name = clip_name(old_path)
        old_raw = source.get(old_name)
        if old_raw is not None:
            clip = json.loads(old_raw)
            if path == old_path:
                kind, reason = "exact", "Full path, meaning, and motion retained; active line endings normalized to LF."
                raw = active_bytes(old_raw)
            else:
                kind, reason = "family_rename", "Only the core-family name changes; branch meaning and all motion steps are retained."
                clip.update(name=target_name, label=path, group=labels[0])
                raw = json_bytes(clip)
            preserve = True
        elif path in REUSES:
            contextual.add(path)
            old_name, reason = REUSES[path]
            old_raw = source[old_name]
            clip = json.loads(old_raw)
            clip.update(name=target_name, label=path, group=labels[0])
            raw, kind, preserve = json_bytes(clip), "contextual_reuse", False
        elif path in AUTHORED:
            contextual.add(path)
            design = AUTHORED[path]
            clip = {
                "$schema": "../gesture.schema.json", "schema_version": 1,
                "name": target_name, "label": path, "group": labels[0],
                "description": design["description"], "units": "radians",
                "steps": deepcopy(design["steps"]),
            }
            raw, kind, preserve = json_bytes(clip), "authored", False
            old_raw, reason = None, design["reason"]
        else:
            raise ValueError(f"Unresolved target: {path}")
        target = metadata(raw)
        targets[target_name] = raw
        row = {
            "path": path, "ring": len(labels), "target": target,
            "classification": kind, "source": metadata(old_raw) if old_raw else None,
            "reason": reason, "preserve_review": preserve,
            "motion_changed": old_raw is None or json.loads(old_raw)["steps"] != clip["steps"],
            "recognition": "requires human review; no automated semantic approval",
            "description": clip["description"],
        }
        rows.append(row)
        if old_raw is not None:
            if old_name in consumed:
                raise ValueError(f"Ambiguous source reuse: {old_name}")
            consumed.add(old_name)
            mappings.append({"source": metadata(old_raw), "target": target, "preserve_review": preserve, "reason": reason})
        else:
            added.append(target | {"reason": reason})
    if contextual != REUSES.keys() | AUTHORED.keys():
        raise ValueError("Unused or conflicting motion authoring decisions")
    if len(targets) != 126:
        raise ValueError("Target library must contain exactly 126 clips")
    retired = [metadata(source[name]) | {"reason": "Old wheel path has no selected target mapping; raw motion and review history remain archived."} for name in sorted(source.keys() - consumed)]
    migration = {
        "schema_version": 1, "source_catalog_id": SOURCE_ID, "target_catalog_id": TARGET_ID,
        "revision_algorithm": "sha256 of the entire raw UTF-8 gesture file, including whitespace and final newline",
        "review_rule": "checked_pass_unchecked_fail",
        "line_endings": "Source hashes identify original raw bytes; active files use LF for stable checkout hashes. Line-ending-only changes do not change motion or meaning.",
        "entries": mappings, "added": added, "retired": retired,
        "counts": {"source": len(source), "target": len(targets), "mapped": len(mappings), "added": len(added), "retired": len(retired), "preserve_review": sum(row["preserve_review"] for row in mappings)},
    }
    signatures = defaultdict(list)
    for row in rows:
        signatures[revision(json_bytes(json.loads(targets[row["target"]["name"]])["steps"]))].append(row["path"])
    duplicates = [names for names in signatures.values() if len(names) > 1]
    audit = {
        "schema_version": 1, "source_catalog_id": SOURCE_ID, "target_catalog_id": TARGET_ID,
        "source_archive": "../gestures/archive/six-core-wheel/manifest.json",
        "coverage": dict(Counter(row["classification"] for row in rows)),
        "counts": catalog["counts"] | {"utility": 9, "library": len(targets)},
        "excluded": catalog["excluded"],
        "neutral": "Separate Neutral / Reset pose action; not an emotion clip. idle_hold holds the current pose and is not reset.",
        "review_policy": "Keep approvals only for utility identity, exact paths, and root-family-only renames with matching source identity and target file revision. Contextual reuses and authored movements begin unchecked. Technical checks never grant human passes.",
        "exact_step_duplicates": duplicates,
        "motion_audit": "Technical trajectory, near-duplicate, and physical tracking measurements are produced separately by tools/audit_emotion_motion.py.",
        "rows": rows, "retired": retired,
    }
    artifacts = {
        "catalogs/emotion-wheel-migration.json": json_bytes(migration),
        "catalogs/emotion-wheel-audit.json": json_bytes(audit),
        "catalogs/emotion-wheel-audit.md": audit_markdown(audit, migration).encode("utf-8"),
        "gestures/emotions.md": catalog_markdown(catalog).encode("utf-8"),
    }
    return targets, artifacts


def audit_markdown(audit: dict, migration: dict) -> str:
    coverage = audit["coverage"]
    lines = [
        "# Emotion wheel coverage and migration audit", "",
        "The user-selected wheel excludes the entire **Bad** branch. This is an authored motion proposal, not a recognition study or a record of human passes.", "",
        "The target has **117 emotion entries** (6 core, 37 middle, 74 outer), 114 distinct words, and nine unchanged utility gestures: **126 clips**. Neutral / Reset pose remains a separate control.", "",
        f"The original 114 emotion entries yield {coverage['exact']} exact paths, {coverage['family_rename']} family-name-only renames, {coverage['contextual_reuse']} explicitly reassigned motions, and {coverage['authored']} newly authored motions. The latter two categories total 52 targets requiring fresh semantic review.", "",
        f"Migration maps {migration['counts']['mapped']} old IDs one-to-one, adds {migration['counts']['added']} IDs without an old equivalent, and archives {migration['counts']['retired']} unmapped old IDs. Only {migration['counts']['preserve_review']} mappings are eligible to preserve a pass, and only with verified identity and target revision.", "",
        "## Frozen evidence and review rules", "",
        "- [Archive manifest](../gestures/archive/six-core-wheel/manifest.json) hashes the byte-preserved 123 original JSON files and original catalog documents.",
        "- [Machine-readable audit](emotion-wheel-audit.json) records every full path, source and target file hash, semantic decision, and reason.",
        "- [Migration manifest](emotion-wheel-migration.json) separates actual mappings, additions, and retired IDs. Matching a leaf word alone is not sufficient.",
        "- Checked means pass; every unchecked clip is failed. New or contextually reassigned clips start unchecked. Existing notes and prior approvals remain in review history.",
        "- Motion steps are preserved exactly for all reused clips. Only new authored clips introduce keyframes. Later user-reviewed failures may be rewritten separately.",
        "- Human review progresses through the six cores, 37 middle entries, then 74 leaves at consistent starting pose, camera, and pace. Technical checks cannot establish emotional recognizability.", "",
        "## Duplicate words and exclusions", "",
        "Inferior appears under Fearful / Insecure and Sad / Depressed. Embarrassed appears under Disgusted / Disapproving and Sad / Hurt. Disappointed appears as a Disgusted branch and a Sad / Hurt leaf. Each full path has its own ID and motion.", "",
        "Removing Bad removes its 13 paths, including Tired, Stressed, Busy, and Bored and their leaves. Fearful / Anxious / Overwhelmed stays: exclusion is by full branch, not by word.", "",
        "## Motion review boundaries", "",
        "The lamp has five joints. Motion descriptions are authored hypotheses and may be confused with nearby feelings. Small differences, duplicate trajectories, and near matches are review candidates, not automatic semantic failures. At joint limits, reverse offsets can alter the intended expression; actual MuJoCo motion may lag or overshoot the bounded command.", "",
        f"Exact step-array duplicates in this candidate emotion library: **{len(audit['exact_step_duplicates'])} groups**. The separate trajectory audit measures neutral/lower/upper starts, stopping, requested motion, physical tracking, and near similarities.", "",
        "## Complete mapping", "",
        "| Target path | Decision | Original path | Pass eligible | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in audit["rows"]:
        source = row["source"]["label"] if row["source"] else "New motion"
        lines.append(f"| {row['path']} | {row['classification']} | {source} | {'Yes' if row['preserve_review'] else 'No'} | {row['reason']} |")
    lines += ["", "## Unmapped original paths", "", "These files remain in the frozen archive; they are no longer in the active catalog.", ""]
    lines.extend(f"- {row['label']} (`{row['name']}`)" for row in audit["retired"])
    return "\n".join(lines) + "\n"


def catalog_markdown(catalog: dict) -> str:
    lines = [
        "# Emotion wheel gestures", "",
        "This catalog follows the user-selected wheel with the entire **Bad** branch excluded: **6 core emotions, 37 middle-ring entries, and 74 outer-ring entries**. The 117 full paths contain 114 distinct words. With nine original utility gestures, the active library has **126 clips**.", "",
        "The core families are **Fearful, Angry, Disgusted, Sad, Happy, and Surprised**. **Neutral / Reset pose** is a separate action, not a seventh emotion clip. `idle_hold` holds the current starting pose; it is not a neutral reset.", "",
        "Use the native viewer's gesture library to select a feeling, play it, and mark **Pass** only if its movement works for you. Every unchecked entry counts as failed. Compare from Reset pose with a consistent camera and pace. Export your review to preserve passes and notes outside the browser.", "",
        "These are authored interpretations, not validated universal emotional signals. Five joints cannot guarantee that nearby feelings are recognizable. All emotion clips return to their starting pose and use radians, bounded control, and reversible offsets near joint limits. Limit reversal can change expression; actual physics may lag or overshoot. Lighting remains under manual control.", "",
        "The original 123 files are byte-preserved in [the archive](archive/six-core-wheel/manifest.json). The [coverage audit](../catalogs/emotion-wheel-audit.md) explains every retained, renamed, reassigned, authored, and retired path; [migration metadata](../catalogs/emotion-wheel-migration.json) protects review provenance.", "",
        "## Complete selected wheel", "",
    ]
    for family in catalog["families"]:
        label = family["label"]
        lines.extend([f"### {label}", "", f"Core gesture: [{label}]({clip_name(label)}.json).", "", "| Middle ring | Outer ring |", "|---|---|"])
        for branch in family["branches"]:
            path = label + " / " + branch["label"]
            leaves = ", ".join(f"[{word}]({clip_name(path + ' / ' + word)}.json)" for word in branch["emotions"])
            lines.append(f"| [{branch['label']}]({clip_name(path)}.json) | {leaves} |")
        lines.append("")
    return "\n".join(lines)


def run(root: Path, *, apply: bool) -> dict:
    source, _ = read_baseline(root, archive=apply)
    targets, artifacts = build(root, source)
    active_paths = {path.stem: path for path in (root / "gestures").glob("*.json")}
    if apply:
        # Do not destroy later human-authored changes. All current files must be
        # either the frozen source or this precise generated target revision.
        for name, path in active_paths.items():
            raw = path.read_bytes()
            if raw not in (source.get(name), targets.get(name)):
                raise ValueError(f"Refusing to overwrite an unknown or edited active clip: {name}")
        # Persist the reviewable, complete audit and mapping BEFORE activation.
        for relative, raw in artifacts.items():
            if relative.startswith("catalogs/"):
                (root / relative).write_bytes(raw)
        for name, raw in targets.items():
            (root / "gestures" / f"{name}.json").write_bytes(raw)
        for name in active_paths.keys() - targets.keys():
            active_paths[name].unlink()
        (root / "gestures" / "emotions.md").write_bytes(artifacts["gestures/emotions.md"])
    else:
        if active_paths.keys() != targets.keys():
            raise ValueError("Active catalog names differ from the generated target")
        for name, raw in targets.items():
            if active_paths[name].read_bytes() != raw:
                raise ValueError(f"Active clip differs from the audited revision: {name}")
        for relative, raw in artifacts.items():
            if (root / relative).read_bytes() != raw:
                raise ValueError(f"Generated artifact is stale: {relative}")
    return json.loads(artifacts["catalogs/emotion-wheel-migration.json"])["counts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true", help="archive, audit, then activate the selected wheel")
    actions.add_argument("--check", action="store_true", help="verify archived and active bytes without changing files (default)")
    args = parser.parse_args()
    counts = run(ROOT, apply=args.apply)
    print(json.dumps({"action": "apply" if args.apply else "check", **counts}, indent=2))


if __name__ == "__main__":
    main()
