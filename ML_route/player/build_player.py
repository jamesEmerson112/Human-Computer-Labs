"""Builds a local HTML page that plays MELD dialogues clip by clip, with the SWE lamp inlined beside the video.

Contents:
- read_rows: reads a label file keyed by clip
- read_word_times: reads Whisper's word timings for one split
- read_audio_fixes: reads the text each audio fix replaced
- load_split: reads one split's lines from MELD.Completed and MELD.Whisper
- read_strict_json: reads a JSON file, rejecting duplicate keys, NaN and Infinity
- load_gestures: reads the SWE gesture library in file-name order
- check_bundle: reads the lamp bundle and fails when it is missing, stale or unsafe to inline
- load_lamp: gathers the URDF, scene, shade mesh and gestures the lamp needs
- fill_template: fills the page's three tokens in one pass
- main: writes ML_route/data/meld-player.html next to MELD.Raw

Testing only: the environment variable LAMP_EMBED names another bundle file to inline, such as a stub,
and skips the stale-bundle hash check (the unsafe-string check still runs). The --output flag writes the
page somewhere other than ML_route/data/meld-player.html. Neither is needed for a normal build.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TEMPLATE = HERE / "player.html"
OUTPUT = DATA / "meld-player.html"
AUDIO_FIXES = HERE.parent / "text_fixes" / "audio_fixes.csv"
SPLITS = ("train", "dev", "test")

REPO = HERE.parent.parent
SIMULATION = REPO / "SWE_route" / "simulation"
URDF = REPO / "SWE_route" / "starter_pack" / "robot" / "dummy_lamp_5dof.urdf"
GESTURES = SIMULATION / "gestures"
SCENE = SIMULATION / "models" / "derived" / "scene.json"
SHADE = SIMULATION / "models" / "derived" / "lamp_shade.stl"
VIEWER = SIMULATION / "viewer"
BUNDLE = VIEWER / "embed-dist" / "lamp-embed.js"
BUNDLE_SOURCES = VIEWER / "embed-dist" / "lamp-embed.sources.json"
REBUILD = (
    "Rebuild it from the repository root with:\n"
    "  npm --prefix SWE_route/simulation/viewer ci\n"
    "  npm --prefix SWE_route/simulation/viewer run build:embed"
)
TOKENS = ("__DATA__", "__LAMP_DATA__", "__LAMP_EMBED__")
# Text that must not appear in the inlined bundle in any case, because it would end the script element
# early or switch the HTML parser into its escaped script state.
UNSAFE = ("<script", "</script", "<!--")
WARN_BYTES = 4_500_000


def read_rows(path):
    """
    Description: Reads a MELD label file into a dict keyed by clip.
    Inputs:
    - path: the label CSV.
    Outputs: {(dialogue, utterance): row} with each row as a dict.
    """
    with path.open(encoding="utf-8", newline="") as f:
        return {
            (int(r["Dialogue_ID"]), int(r["Utterance_ID"])): r
            for r in csv.DictReader(f)
        }


def read_word_times(path):
    """
    Description: Reads Whisper's word timings for every clip of one split.
    Inputs:
    - path: the split's Whisper JSONL file.
    Outputs: {clip: [[start in hundredths of a second, word], ...]}.
    """
    times = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            record = json.loads(raw)
            words = [[round(w["start"] * 100), w["word"].strip()]
                     for w in record.get("words") or []]
            if words:
                times[record["clip"]] = words
    return times


def read_audio_fixes(path):
    """
    Description: Reads the text each audio fix replaced.
    Inputs:
    - path: audio_fixes.csv.
    Outputs: {(split, clip): text before the audio fix}.
    """
    with path.open(encoding="utf-8", newline="") as f:
        return {(r["split"], r["clip"]): r["old"] for r in csv.DictReader(f)}


def load_split(split, audio_fixes):
    """
    Description: Reads one split's lines from MELD.Completed and MELD.Whisper.
    Inputs:
    - split: "train", "dev" or "test".
    - audio_fixes: the lookup from read_audio_fixes.
    Outputs: {dialogue: [[utterance, speaker, emotion, completed, before,
    whisper, words], ...]}, where before is the text an audio fix replaced or
    0, whisper is None when the clip has no transcript, and words holds
    Whisper's word timings or 0.
    """
    name = f"{split}_sent_emo.csv"
    completed = read_rows(DATA / "MELD.Completed" / name)
    whisper = read_rows(DATA / "MELD.Whisper" / name)
    word_times = read_word_times(DATA / "MELD.Whisper" / f"{split}_whisper.jsonl")
    dialogues = {}
    for (dia, utt), row in sorted(completed.items()):
        clip = f"dia{dia}_utt{utt}"
        heard = whisper.get((dia, utt), {}).get("Utterance") or None
        dialogues.setdefault(dia, []).append([
            utt, row["Speaker"], row["Emotion"], row["Utterance"],
            audio_fixes.get((split, clip), 0), heard,
            word_times.get(clip, 0),
        ])
    return dialogues


def read_strict_json(path):
    """
    Description: Reads a JSON file the way lamp_sim/gestures.py load_gesture does.
    Inputs:
    - path: the JSON file.
    Outputs: the decoded value.
    Constraints: raises SystemExit naming the file on a duplicate key, NaN, Infinity or any other error.
    A UTF-8 byte order mark is allowed.
    """
    def unique_keys(pairs):
        """
        Description: Builds a JSON object, refusing a key that appears twice.
        Inputs:
        - pairs: the object's (key, value) pairs in file order.
        Outputs: the object as a dict.
        Constraints: raises ValueError naming the first repeated key.
        """
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key: {name}")
            result[name] = value
        return result

    def reject_constant(value):
        """
        Description: Refuses NaN, Infinity and -Infinity, which strict JSON does not allow.
        Inputs:
        - value: the constant's text.
        Outputs: none; it always raises ValueError.
        """
        raise ValueError(f"{value} is not a valid JSON number")

    try:
        return json.loads(path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"{path}: {error}") from error


def load_gestures(directory):
    """
    Description: Reads every gesture in the library folder, sorted by file name.
    Inputs:
    - directory: SWE_route/simulation/gestures.
    Outputs: the decoded gestures as a list.
    Constraints: the glob is not recursive, so archive/ stays out. Each gesture must be an object with a
    string name and label, and names and labels must be unique, as in lamp_sim/gestures.py load_gestures.
    The lamp bundle validates the steps itself.
    """
    gestures, names, labels = [], set(), set()
    for path in sorted(directory.glob("*.json")):
        gesture = read_strict_json(path)
        if not isinstance(gesture, dict):
            raise SystemExit(f"{path}: a gesture must be a JSON object")
        name, label = gesture.get("name"), gesture.get("label")
        if not isinstance(name, str) or not isinstance(label, str):
            raise SystemExit(f"{path}: a gesture needs a string name and label")
        if name in names:
            raise SystemExit(f"{path.name}: duplicate gesture name {name!r}")
        if label in labels:
            raise SystemExit(f"{path.name}: duplicate gesture label {label!r}")
        names.add(name)
        labels.add(label)
        gestures.append(gesture)
    if not gestures:
        raise SystemExit(f"No gesture JSON files found in {directory}")
    return gestures


def check_bundle():
    """
    Description: Reads the lamp bundle and checks that it matches its sources and is safe to inline.
    Inputs: none; reads LAMP_EMBED, BUNDLE and BUNDLE_SOURCES.
    Outputs: the bundle's text.
    Constraints: raises SystemExit with the npm commands when the bundle or its sources.json is missing,
    or when a listed source file is missing or its sha256 differs. When LAMP_EMBED names a file, that
    file is used and the hash check is skipped, for testing with a stub. Either way it raises SystemExit
    when the bundle contains <script, </script or <!-- in any case, or one of the page tokens.
    """
    override = os.environ.get("LAMP_EMBED")
    path = Path(override) if override else BUNDLE
    if not path.is_file():
        raise SystemExit(f"The lamp bundle {path} is missing.\n{REBUILD}")
    if not override:
        if not BUNDLE_SOURCES.is_file():
            raise SystemExit(f"The lamp bundle's hash list {BUNDLE_SOURCES} is missing.\n{REBUILD}")
        listed = read_strict_json(BUNDLE_SOURCES)
        files = listed.get("files") if isinstance(listed, dict) else None
        if not isinstance(files, dict) or not files:
            raise SystemExit(f"{BUNDLE_SOURCES} lists no source files.\n{REBUILD}")
        for name, digest in sorted(files.items()):
            source = VIEWER / name
            if not source.is_file():
                raise SystemExit(f"The lamp bundle is stale: its source {name} is missing.\n{REBUILD}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise SystemExit(f"The lamp bundle is stale: {name} changed after it was built.\n{REBUILD}")
    bundle = path.read_text(encoding="utf-8")
    lowered = bundle.lower()
    for text in UNSAFE:
        if text in lowered:
            raise SystemExit(f"The lamp bundle {path} contains {text!r}, which cannot be inlined in a script.")
    for token in TOKENS:
        if token in bundle:
            raise SystemExit(f"The lamp bundle {path} contains the page token {token}.")
    return bundle


def load_lamp():
    """
    Description: Gathers everything the inlined lamp needs, as LampEmbed.mount takes it.
    Inputs: none; reads URDF, SCENE, SHADE and GESTURES.
    Outputs: {"urdf", "scene", "mesh_base64", "gestures"}.
    """
    return {
        "urdf": URDF.read_text(encoding="utf-8"),
        "scene": read_strict_json(SCENE),
        "mesh_base64": base64.b64encode(SHADE.read_bytes()).decode("ascii"),
        "gestures": load_gestures(GESTURES),
    }


def fill_template(page, fills):
    """
    Description: Replaces each page token with its text in a single pass.
    Inputs:
    - page: the template text.
    - fills: {token: text} for every token in TOKENS.
    Outputs: the filled page.
    Constraints: raises SystemExit unless each token appears exactly once. The replacement is a function,
    so backslashes in the filled text are kept as they are, and text that was filled in is never
    searched again.
    """
    for token in TOKENS:
        count = page.count(token)
        if count != 1:
            raise SystemExit(f"{TEMPLATE} must contain {token} exactly once, found {count}.")
    pattern = re.compile("|".join(re.escape(t) for t in sorted(TOKENS, key=len, reverse=True)))
    return pattern.sub(lambda match: fills[match.group(0)], page)


def main():
    """
    Description: Writes the player page with every split's lines and the lamp embedded.
    Inputs: none; the command line may give --output, for testing.
    Outputs: none
    """
    parser = argparse.ArgumentParser(description="Build the MELD player page.")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="where to write the page (testing only; default %(default)s)")
    output = parser.parse_args().output
    bundle = check_bundle()
    lamp = load_lamp()
    audio_fixes = read_audio_fixes(AUDIO_FIXES)
    data = {split: load_split(split, audio_fixes) for split in SPLITS}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lamp_json = json.dumps(lamp, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = fill_template(TEMPLATE.read_text(encoding="utf-8"), {
        "__DATA__": payload.replace("</", "<\\/"),
        "__LAMP_DATA__": lamp_json,
        "__LAMP_EMBED__": bundle,
    })
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    lines = sum(len(v) for d in data.values() for v in d.values())
    total = len(page.encode("utf-8"))
    share = len(lamp_json.encode("utf-8")) + len(bundle.encode("utf-8"))
    print(f"wrote {output} with {lines} lines and {len(lamp['gestures'])} lamp gestures")
    print(f"page {total / 1e6:.2f} MB, of which the lamp is {share / 1e6:.2f} MB ({100 * share / total:.0f}%)")
    if total > WARN_BYTES:
        print(f"warning: the page is over {WARN_BYTES / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
