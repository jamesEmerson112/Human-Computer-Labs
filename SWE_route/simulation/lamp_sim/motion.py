"""Joint-limited kinematic motion, independent of the browser and rendering."""

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

from .gestures import Gesture


ROBOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "starter_pack/robot/dummy_lamp_5dof.urdf"
)


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float
    velocity: float

    def clamp(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Joint targets must be finite radians")
        return min(self.upper, max(self.lower, value))


def load_limits(path: Path = ROBOT_PATH) -> dict[str, JointLimit]:
    """Use the intersection of hard and soft limits from the supplied model."""
    limits = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        if joint.get("type") != "revolute":
            continue
        hard = joint.find("limit")
        if hard is None:
            raise ValueError(f"Missing limits for {joint.get('name')}")
        lower, upper = float(hard.attrib["lower"]), float(hard.attrib["upper"])
        soft = joint.find("safety_controller")
        if soft is not None:
            lower = max(lower, float(soft.get("soft_lower_limit", lower)))
            upper = min(upper, float(soft.get("soft_upper_limit", upper)))
        velocity = float(hard.attrib["velocity"])
        if not all(map(math.isfinite, (lower, upper, velocity))):
            raise ValueError("Joint limits must be finite")
        if lower > upper or velocity <= 0:
            raise ValueError(f"Invalid limits for {joint.get('name')}")
        limits[joint.attrib["name"]] = JointLimit(lower, upper, velocity)
    if not limits:
        raise ValueError("The model has no revolute joints")
    return limits


@dataclass(frozen=True)
class _Segment:
    targets: dict[str, float]
    seconds: float = 0.25
    hold: bool = False


class MotionController:
    """Single-owner controller; callers enqueue commands from other threads.

    All angles are radians and time is seconds. Cubic easing starts and ends
    each move at rest. Its maximum slope is 1.5, included in the duration so
    no joint exceeds its URDF velocity. This does not simulate forces/contact.
    """

    def __init__(self, limits: Mapping[str, JointLimit]):
        self.limits = dict(limits)
        self.positions = {name: limit.clamp(0.0) for name, limit in limits.items()}
        self.targets = self.positions.copy()
        self._start = self.positions.copy()
        self._elapsed = 0.0
        self._duration = 0.0
        self._pace_limit = math.inf
        self._waypoints: deque[_Segment] = deque()
        self._speed = 1.0
        self.activity = "Ready"

    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, speed: float) -> None:
        """Scale playback time, never above the URDF velocity envelope."""
        if not math.isfinite(speed) or not 0 < speed <= 3:
            raise ValueError("Playback speed must be greater than zero and at most three")
        self._speed = speed

    @property
    def moving(self) -> bool:
        return self._elapsed < self._duration or bool(self._waypoints)

    def _begin(self, segment: _Segment) -> None:
        self._start = self.positions.copy()
        self.targets = segment.targets.copy()
        self._elapsed = 0.0
        minimum_duration = max(
            1.5 * abs(self.targets[n] - self.positions[n]) / lim.velocity
            for n, lim in self.limits.items()
        )
        self._duration = segment.seconds if segment.hold else max(
            0.25, segment.seconds, minimum_duration,
        )
        self._pace_limit = (
            self._duration / minimum_duration if minimum_duration > 0 else math.inf
        )

    def set_targets(self, targets: Mapping[str, float]) -> None:
        """Replace a gesture; unspecified joints retain their existing goals."""
        unknown = targets.keys() - self.limits.keys()
        if unknown:
            raise ValueError(f"Unknown joints: {sorted(unknown)}")
        bounded = self.targets | {
            name: self.limits[name].clamp(value) for name, value in targets.items()
        }
        self._waypoints.clear()
        self._begin(_Segment(bounded))
        self.activity = "Moving"

    def reset(self) -> None:
        self.set_targets({name: 0.0 for name in self.limits})
        self.activity = "Resetting"

    def synchronize(self, positions: Mapping[str, float]) -> None:
        """Start a new command from an observed pose; clear obsolete waypoints."""
        bounded = {name: limit.clamp(positions[name]) for name, limit in self.limits.items()}
        self.positions = bounded
        self.stop()

    def play(self, gesture: Gesture) -> None:
        """Compile validated JSON steps against the current pose, then play.

        Offsets always refer to this starting pose, not the previous waypoint.
        Unspecified joints keep the preceding waypoint's angle. Compilation
        finishes before changing playback, so invalid data cannot partly play.
        """
        origin = self.positions.copy()
        pose = origin.copy()
        segments = []
        for step in gesture.steps:
            if step.kind == "return_to_start":
                pose = origin.copy()
            elif step.kind in ("targets", "offsets"):
                for name, angle in step.angles:
                    if name not in self.limits:
                        raise ValueError(f"Unknown joint: {name}")
                    limit = self.limits[name]
                    target = angle
                    if step.kind == "offsets":
                        target = origin[name] + angle
                        if step.on_limit == "reverse" and not limit.lower <= target <= limit.upper:
                            target = origin[name] - angle
                    pose[name] = limit.clamp(target)
            elif step.kind != "hold_seconds":
                raise ValueError(f"Unknown gesture step: {step.kind}")
            if not math.isfinite(step.seconds) or not 0 < step.seconds <= 60:
                raise ValueError("Step duration must be finite and between 0 and 60 seconds")
            segments.append(_Segment(pose.copy(), step.seconds, step.kind == "hold_seconds"))
        if not segments:
            raise ValueError("A gesture needs at least one step")
        self._waypoints = deque(segments[1:])
        self._begin(segments[0])
        self.activity = f"Playing {gesture.label}"

    def stop(self) -> None:
        """Freeze at the current pose and discard all queued gesture motion."""
        self._waypoints.clear()
        self.targets = self.positions.copy()
        self._start = self.positions.copy()
        self._elapsed = self._duration = 0.0
        self.activity = "Stopped"

    def step(self, dt: float) -> dict[str, float]:
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        # Preserve the original timeline arithmetic at ordinary/slower pace.
        # Faster playback consumes wall time per segment because each segment
        # has different velocity headroom; holds can always use the full pace.
        dt *= min(self._speed, 1.0)
        while dt > 0 and self.moving:
            scale = min(self._speed, self._pace_limit) if self._speed > 1 else 1.0
            remaining = self._duration - self._elapsed
            used = min(dt * scale, remaining)
            self._elapsed += used
            dt -= used / scale
            fraction = min(1.0, self._elapsed / self._duration)
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            self.positions = {
                name: self._start[name] + (value - self._start[name]) * eased
                for name, value in self.targets.items()
            }
            if self._elapsed >= self._duration:
                self.positions = self.targets.copy()
                if self._waypoints:
                    self._begin(self._waypoints.popleft())
                else:
                    self.activity = "Ready"
        return self.positions.copy()
