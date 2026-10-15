# Editing lamp gestures

Each top-level `.json` file in this folder defines one gesture. The native viewer
lists its `label` and optional emotion family. The Python reference viewer creates
a button or an entry in its optional `group` folder. Archive subfolders are not
loaded. Edit or add a file, save it, then click **Reload gestures** in the Python
reference viewer, or restart the native controller and refresh its viewer.
The whole folder is validated before replacing the loaded library. A failed
reload shows the filename and error and keeps the previous working gestures.
An already-playing gesture finishes its original sequence unless interrupted.

Start by changing the angle or pause in [`nod.json`](nod.json):

```json
{
  "$schema": "../gesture.schema.json",
  "schema_version": 1,
  "name": "nod",
  "label": "Nod",
  "description": "Tilt the head, pause, and return.",
  "units": "radians",
  "steps": [
    {"offsets": {"head_pitch_joint": 0.3}, "on_limit": "reverse"},
    {"hold_seconds": 0.25},
    {"return_to_start": true}
  ]
}
```

`0.3` radians is about 17 degrees. Try `0.15` for a smaller nod or increase
`hold_seconds` to `0.75` for a longer pause. Select Slow pace to inspect it.
The schema reference enables validation/completion in editors with JSON Schema
support. The application always validates against its local schema.

## Step types

Each step has exactly one action:

| Action | Meaning | Example |
|---|---|---|
| `targets` | Absolute joint angles. | `{"targets": {"head_pitch_joint": 0.2}}` |
| `offsets` | Add angles to the pose captured when the gesture starts. | `{"offsets": {"neck_yaw_joint": -0.3}}` |
| `hold_seconds` | Hold the previous pose for the specified playback time. | `{"hold_seconds": 0.5}` |
| `return_to_start` | Move all joints back to that captured starting pose. | `{"return_to_start": true}` |

Offsets always reference the **gesture's starting pose**, never the previous
step. For example, starting at 0.1 radians, offsets `0.2` then `-0.2` produce
targets 0.3 then −0.1. Joints omitted from a move keep their previous waypoint's
angles. A gesture without a return step stays at its final pose.

Move and return steps optionally accept `duration_seconds`:

```json
{"targets": {"base_yaw_joint": 0.4}, "duration_seconds": 1.0}
```

This is a minimum duration at Full speed. The player lengthens it if necessary
to respect joint velocity limits and uses at least 0.25 seconds per move.
Explicit holds use the requested duration exactly. All durations are **playback
seconds**: Normal pace takes twice as long and Slow takes four times as long.
Thus a 0.25-second hold lasts 0.5 real seconds at Normal pace. Loop stalls can
extend wall-clock time further. Durations must be greater than zero and no more
than 60 seconds; each file can contain up to 128 steps.

Targets are clamped to the robot's soft position limits. On offset steps,
`"on_limit": "reverse"` tries the opposite offset for each joint whose target
would exceed its limit, then clamps if necessary. Without that option, the
offset is simply clamped. This is how Nod changes direction near the upper limit.
The controller handles interpolation, clamping, and speed limits for requested
angles. MuJoCo's actual angles can lag or overshoot; the viewer and benchmark
show both. Stop requests a physical hold rather than freezing the simulated body.

## Gesture comparison suite

The library contains **126 gestures**: the original nine below and
[117 emotion-wheel gestures](emotions.md) spanning the selected wheel's three
rings with the entire **Bad** branch excluded. The six families are Fearful,
Angry, Disgusted, Sad, Happy, and Surprised. Repeated words retain their complete
branch context. Choose a family and emotion in the viewer, then play the clip.
Every emotion clip returns to its starting pose. **Neutral / Reset pose** is a
separate action; `idle_hold` holds the current pose and does not reset to neutral.

The native review checklist uses **checked = pass; unchecked = fail**. The
[coverage audit](../catalogs/emotion-wheel-audit.md) explains the migration.
Original clips and catalog documents are byte-preserved in
[the archive](archive/six-core-wheel/manifest.json). Approvals transfer only for
unchanged meaning and verified motion revisions; reassigned and authored clips
need a fresh human review. Technical checks do not establish recognition.

| File | What it exercises |
|---|---|
| `nod.json`, `shake.json` | Single-axis pitch and yaw, pauses, reversals. |
| `curious.json`, `bow.json` | Coordinated arm/head expressions. |
| `scan.json` | Base and neck coordination while looking around. |
| `excited.json` | Repeated direction changes and transient tracking error. |
| `idle_hold.json` | Gravity sag during a two-playback-second hold. |
| `reach_hold.json` | Extended arm load and settling error. |
| `limit_sweep.json` | Each joint near its soft limits; overshoot diagnostics. |

Run `uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5`
from the simulation folder to log every gesture across all four backends.
Each trial starts at zero, uses the same requested trajectory, and includes one
additional simulated second to settle. New JSON files join the suite automatically.
See [benchmark methodology](../benchmarks/README.md) before comparing runs.

## Joint names and metadata

Use these exact keys:

- `base_yaw_joint`: turn the whole arm.
- `shoulder_pitch_joint`: lean the arm.
- `elbow_pitch_joint`: fold the upper arm.
- `neck_yaw_joint`: swivel the head.
- `head_pitch_joint`: tilt the shade; positive angles aim it downward at zero arm pose.

Angles must be finite numbers in radians. Misspelled joints, unknown fields,
duplicate JSON keys, mixed actions, and invalid values are rejected. Use JSON
syntax with double quotes, no comments, and no trailing commas.

When copying a gesture, give it a unique `name` and `label`; use a name such as
`small_nod` and a matching filename for clarity. `schema_version` must be `1`,
`units` must be `"radians"`, and `description` is optional hover text (shown below
the selector for grouped gestures). An optional `group`, such as `"Happy"`,
places a gesture in a collapsed folder with a selector and **Play emotion**
button. Labels must remain unique across the whole library; emotion labels
include their wheel path. Groups sort alphabetically and entries follow filename
order. Adding a file requires no Python changes.

## Validate without starting the viewer

From `SWE_route/simulation/`:

```powershell
uv run --locked python -m lamp_sim --check-gestures
uv run --locked python tools/build_emotion_library.py --check
```

The second command verifies the frozen archive, active file hashes, and generated
audit/migration metadata. It does not modify files. `--apply` is the explicit
archive-and-migrate command; it refuses to overwrite edited or unknown active
clips. For later animation revisions, update the authoring decision and migration
metadata deliberately so a previous pass cannot silently approve a changed motion.

For Python integrations, load and validate the same data before playback:

```python
from lamp_sim.gestures import load_gestures, parse_gesture
from lamp_sim.motion import MotionController, load_limits

motion = MotionController(load_limits())
library = load_gestures(motion.limits)
motion.play(library["nod"])
motion.step(1 / 30)  # Advance repeatedly from your existing update loop.

# Or validate a decoded JSON object: parse_gesture(data, motion.limits).
```

This defines movement data and an in-process player; it does not add a network
command endpoint. Stop, Reset, a manual joint target, or another gesture can
interrupt playback through the existing controller.
