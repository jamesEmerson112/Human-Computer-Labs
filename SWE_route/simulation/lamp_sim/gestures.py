"""Load inert, validated gesture data. JSON never contains executable code."""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Collection

from jsonschema import Draft202012Validator


PROJECT_PATH = Path(__file__).resolve().parents[1]
GESTURE_PATH = PROJECT_PATH / "gestures"


@dataclass(frozen=True)
class GestureStep:
    kind: str
    angles: tuple[tuple[str, float], ...] = ()
    seconds: float = 0.25
    on_limit: str = "clamp"


@dataclass(frozen=True)
class Gesture:
    name: str
    label: str
    description: str
    steps: tuple[GestureStep, ...]
    group: str = ""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads((PROJECT_PATH / "gesture.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def parse_gesture(data: object, joint_names: Collection[str]) -> Gesture:
    """Validate a decoded JSON object before it can reach the motion loop."""
    error = next(_validator().iter_errors(data), None)
    if error is not None:
        location = ".".join(map(str, error.absolute_path)) or "gesture"
        raise ValueError(f"{location}: {error.message}")
    steps = []
    for index, step in enumerate(data["steps"]):
        kind = next(key for key in ("targets", "offsets", "hold_seconds", "return_to_start") if key in step)
        angles = step[kind] if kind in ("targets", "offsets") else {}
        unknown = angles.keys() - set(joint_names)
        if unknown:
            raise ValueError(f"steps.{index}: unknown joints {sorted(unknown)}")
        seconds = step.get("hold_seconds", step.get("duration_seconds", 0.25))
        try:
            finite = all(math.isfinite(value) for value in (*angles.values(), seconds))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"steps.{index}: angles and durations must be finite numbers")
        steps.append(GestureStep(
            kind=kind, angles=tuple((name, float(value)) for name, value in angles.items()),
            seconds=float(seconds), on_limit=step.get("on_limit", "clamp"),
        ))
    return Gesture(data["name"], data["label"], data.get("description", ""), tuple(steps), data.get("group", ""))


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"{value} is not a valid JSON number")


def load_gesture(path: Path, joint_names: Collection[str]) -> Gesture:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
        )
        return parse_gesture(data, joint_names)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"{path.name}: {error}") from error


def load_gestures(
    joint_names: Collection[str], directory: Path = GESTURE_PATH,
) -> dict[str, Gesture]:
    """Read a whole library; the caller swaps it in only when every file is valid."""
    gestures = {}
    labels = set()
    for path in sorted(directory.glob("*.json")):
        gesture = load_gesture(path, joint_names)
        if gesture.name in gestures:
            raise ValueError(f"{path.name}: duplicate gesture name {gesture.name!r}")
        if gesture.label in labels:
            raise ValueError(f"{path.name}: duplicate gesture label {gesture.label!r}")
        gestures[gesture.name] = gesture
        labels.add(gesture.label)
    if not gestures:
        raise ValueError(f"No gesture JSON files found in {directory}")
    return gestures
