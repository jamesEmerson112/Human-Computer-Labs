# Lamp playground

A local browser playground for the supplied five-joint lamp. A Rust controller
drives motion, MuJoCo simulates gravity, motors, and contacts, and Three.js draws
the result in your browser. Move joints, play gestures, and aim its light while
learning how the robot fits together. The Python/viser reference and its original
direct playback remain available for comparison. Blender is optional and is not
needed to run this playground.

## Rust controller and Three.js viewer

The native stack lives alongside the Python reference:
**Three.js controls → WebSocket → Rust motion controller → MuJoCo 3.14.0 → observed body poses → Three.js**.
Rust owns motion and physics in one process; the browser interpolates observed
poses for display. The same 123 JSON gestures and supplied robot assets are used.
The Python launch command below remains available for comparison.

Prerequisites: Rust 1.92 or newer, Node.js 22.12 or newer, and a C compiler
(Visual Studio Build Tools with Desktop development with C++ on Windows;
`build-essential` on Ubuntu). Python/uv prepare and validate the model and supply
the pinned MuJoCo SDK. They are not invoked by the running native application.

From `SWE_route/simulation/` in PowerShell:

```powershell
uv sync --locked
uv run --locked python tools/export_native_model.py --check
cargo build --release --locked --manifest-path controller/Cargo.toml --bins
npm --prefix viewer ci
npm --prefix viewer run build
controller/target/release/lamp-controller.exe serve --port 8082
```

Open **http://127.0.0.1:8082**. Keep the server running; Ctrl+C stops it.
The frontend bundles Three.js locally, with no CDN requests. The server binds
to loopback, checks browser origins, and does not request camera/microphone access.
Use the gesture search/family filter or the five joint sliders; **Stop**
cancels queued motion and requests a hold at the observed pose. Physical motors
can still settle. Light on/off and brightness affect rendering. **Manual pose**
shows actual/requested angles; **Session details** includes contact diagnostics,
timing, and a mode selector.

The native modes are `mujoco`, `mujoco-source`, and `mujoco-kinematic`. The original
yourdfpy `kinematic` mode and teaching controls remain in the Python viewer.
Both viewers read the same gesture directory; **Reload gesture files** replaces the
catalog only when every file is valid. Commands sent while disconnected are
not replayed when a browser reconnects.

### Review the animations

Each animation has a checkbox: **checked means pass; unchecked means fail**.
Select an animation and press **Play**, then check it if it passes. **Next**
selects the next visible animation without starting playback. Family, search,
and Passed/Failed filters help you work through the list; the counts always cover
the whole library. Add optional notes under **Notes for this animation** to explain
what needs changing.

Checks and notes save automatically in this browser and survive refreshes.
Use the same address and browser profile to resume (for example,
`http://127.0.0.1:8082`); another hostname, port, browser, or cleared browser data
has separate storage. If browser storage is unavailable, the status message asks
you to export so that your work is retained.

When finished, click **Export review** to download `lamp-animation-review-*.json`.
It contains every current animation, its stable name, pass/fail result, optional
notes, and timestamps, including animations hidden by filters. Share that file
for the next revision pass; entries with `status: "fail"` identify the animations
to revise. Exporting and checking boxes do not change gesture JSON or start motion.

**Import review** restores a saved file, replacing the current checks and notes.
Unknown names are skipped and reported; animations missing from the imported
file are unchecked. Invalid or unrelated files leave the existing review intact.
Reviews follow animation names; if you manually replace a passed animation's
motion, uncheck it until you have reviewed the replacement.

### Ubuntu and standalone deployment

On Ubuntu 24.04, use the same preparation/build commands and launch
`controller/target/release/lamp-controller serve --port 8082`. For a standalone
SDK, set `MUJOCO_DIR` to the MuJoCo 3.14.0 directory before building; it must contain
the matching headers and shared library. `MUJOCO_LIBRARY` can override the runtime
library path. Deploy the executable with `viewer/dist/`, `models/derived/`,
`gestures/`, and that shared library, and pass `--root /path/to/simulation`.
The bridge rejects mismatched MuJoCo headers/libraries before accessing native
structures. A small C bridge avoids duplicating MuJoCo's versioned struct layouts
in Rust; motion, scheduling, validation, networking, and ownership are Rust.

### Timing and model provenance

Native control runs at 30 Hz with eight 240 Hz physics steps per control step.
The scheduler accumulates elapsed time, catches up at most eight control steps
per iteration, and reports discarded time if severely delayed. This differs
from the Python viewer's elapsed-time cap, which can slow simulation under load.
The browser uses a 33.3 ms display buffer and quaternion interpolation; this
smooths rendering while adding display delay. It does not alter physics or hide
physical soft-limit overshoot.

`models/derived/manifest.json` records source and generated-asset hashes.
Regenerate intentionally with `uv run --locked python tools/export_native_model.py`
after changing the model or conversion code, then rerun parity checks. The exporter
preserves the exact timestep and checks serialized physics against the original.
It never edits the supplied URDF/STL. The default collision exclusion and servo
assumptions described below are identical in both implementations.

### Native validation

```powershell
cargo test --locked --manifest-path controller/Cargo.toml
npm --prefix viewer test
uv run --locked python tools/verify_native_protocol.py --output .verification/native-protocol.json
uv run --locked python tools/verify_animation_review.py --output .verification/animation-review
uv run --locked python benchmarks/compare_native.py --output benchmarks/my-native/headless
uv run --locked python benchmarks/measure_native_viewer.py --output benchmarks/my-native/browser
```

The Rust parity test invokes the locked Python reference to compare all gestures
from three starting poses, interruptions, and mixed commands. Set `LAMP_PYTHON`
to an alternate prepared Python interpreter when testing outside the default
`.venv`. Benchmark commands require release binaries and the built frontend;
choose new output paths to preserve earlier evidence. See
[benchmark methodology](benchmarks/README.md) for comparison limits.

The [versioned command schema](protocol/commands.schema.json) describes commands.
Every command includes `v: 1` and a client-generated `id`; acknowledgments report
acceptance or rejection. Connections receive a complete catalog and current
state. State packets carry monotonic server timestamps, simulation time, requested
and actual radians, world-space body transforms (z-up, quaternion order xyzw),
light settings, and diagnostics. A bounded command queue and a latest-state
channel prevent obsolete visual snapshots from accumulating.

## Python reference: Start on Windows

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a browser with WebGL enabled (Chrome, Edge, or Firefox).

From the repository root in PowerShell:

```powershell
cd SWE_route/simulation
uv sync --locked
uv run --locked python -m lamp_sim
```

Open the address printed in the terminal, normally **http://127.0.0.1:8080**.
Keep that terminal running. Press **Ctrl+C** in it to stop the server.
The first sync downloads packages into this folder's `.venv`; it does not
install them into your global Python environment. `uv.lock` records versions.

If port 8080 is occupied, choose another:

```powershell
uv run --locked python -m lamp_sim --port 8081
```

The server binds only to this computer. It does not request camera/microphone
access or call model services. The browser and Python server communicate locally.

## Python reference: Your first five minutes

A **link** is a rigid part of the robot; a **joint** connects parts and lets them
move relative to each other. This lamp has five degrees of freedom (five-DOF):
five independent joint angles you can control. Turning the view changes where
you look from; changing a joint angle changes the robot itself.

Use **Learn the controls → Practice step** for instructions inside the viewer.
Selecting a lesson changes the explanation; you perform the movements yourself.

1. **Explore the view.** Drag to orbit, scroll to zoom, and right-drag to pan.
   **Reset view** brings the robot close; **Light view** includes the wall.
2. **Turn one joint.** Set **Base turn** to 20°, then **Neck turn** to 20°.
   The base turns the whole arm; the neck swivels only the head. Reset the pose.
3. **Fold the arm.** Set **Shoulder tilt** to 20° and **Elbow bend** to −30°.
   The elbow moves with the shoulder. Reset the pose before the next step.
4. **Aim the light.** Click **Light view**, then set **Head tilt** to 25°.
   Watch the light move down the wall. Toggle **Light on** and try **Brightness**.
5. **Try expression.** Reset the pose, choose **Slow** pace, and compare the
   gestures below. Try **Stop motion** during a gesture.

| Gesture | What moves | What to notice |
|---|---|---|
| Nod | Head pitch | A small tilt, a pause, and a return; reverses direction near the upper limit. |
| Shake head | Neck yaw | A side-to-side motion while the lower arm stays still. |
| Curious lean | Shoulder, elbow, and head pitch | Several joints move together, pause, and return. |
| Look around | Base and neck yaw | Coordinated scanning and direction reversals. |
| Bow | Shoulder, elbow, and head | A slower compound expression. |
| Excited | Head, neck, and elbow | Repeated changes of direction. |
| Hold still | All joints hold | Gravity sag and steady tracking error. |
| Reach and hold | Arm and head | Tracking under an extended load. |
| Limit sweep (diagnostic) | Each joint separately | Near-limit behavior and physical overshoot. |

Expressive gestures request a return to their starting pose, unless interrupted;
Hold still keeps its initial target. Physical tracking can have a small error.
**Pace** affects all movement: Slow is 25%, Normal (the default) is 50%, and
Full speed is 100% of the controller's allowed playback rate. Changing pace
during a move keeps the requested trajectory continuous and within its speed limit.
Actual physics can lag or overshoot that request.

The sliders show **target angles in degrees**. During a gesture they show its
current waypoint. **Current pose** shows the angles
the lamp has actually reached, alongside the requested angles. **Stop motion**
cancels the gesture and requests a hold at the observed pose; physical motors
may still settle. **Reset pose** smoothly requests zero.
**Reset view** restores your browser's camera without changing the robot.
**Show light direction** displays axes at the emitter; red points out of the shade.

## Edit movements as JSON

The library loads **123 gestures** from [`gestures/`](gestures/README.md): the
original nine above plus [114 emotion-wheel entries](gestures/emotions.md).
In **Try a movement > Gestures**, expand Anger, Disgust, Sad, Happy, Surprise,
or Fear, choose an **Emotion**, and click **Play emotion**. The new clips cover
all three rings of the supplied wheel, including context-specific variants of
repeated words. Each has a distinct motion sequence and returns to its starting
pose. These are expressive interpretations that need human evaluation for
recognizability. Restart an older running viewer to load the grouped controls.

Edit a file and click **Reload gestures**.
Add a JSON file with a unique name and label to get another button, or use the
optional `group` field to put it in a family selector. An invalid
file displays an error and leaves the previous working library available.

Files describe joint targets or starting-pose offsets, explicit pauses, and
returns to the starting pose. Python still enforces motion limits and timing.
See the [gesture authoring guide](gestures/README.md) for the complete format
and examples. Validate files before playback with:

```powershell
uv run --locked python -m lamp_sim --check-gestures
```

## Simulation modes and debugging

Use **Simulation → Mode** to compare the same JSON movement:

| Mode | CLI backend | Purpose |
|---|---|---|
| MuJoCo physics (default) | `mujoco` | Gravity, effort-limited position motors, and contacts. |
| Direct playback (original) | `kinematic` | Original interpolation and forward kinematics; assigns angles directly. |
| MuJoCo pose check | `mujoco-kinematic` | Assign the same angles in MuJoCo to check model conversion. |
| MuJoCo source contacts | `mujoco-source` | Physics with the original self-contact geometry, for comparison. |

The controller requests bounded, cubic-eased angles at 30 Hz. Physics runs at
240 Hz using URDF masses, inertia, damping, hard limits, and effort limits.
Position-servo gains and rotor inertia are explicit assumptions in
`lamp_sim/backends.py`, not measurements of a real robot. A stalled viewer slows
simulation time rather than jumping forward. Switching modes transfers the
current bounded pose and resets velocity.

The supplied arm/head collision proxies overlap by **1 mm at zero pose**.
Default physics excludes only `upper_arm_link / lamp_head_link`; all other
self/table contacts remain enabled. This exclusion can hide genuine contact
between those two parts, so the source-contact mode remains available. Neither
mode changes the supplied URDF or STL. Physical motion can overshoot requested
soft limits; the benchmark records this instead of clamping observed physics.

The browser displays tracking error, contact count, and penetration. To also
open MuJoCo's native contact/force inspector:

```powershell
uv run --locked python -m lamp_sim --native-viewer --backend mujoco-source
```

Use the browser to issue movements and the native window to inspect physics.
Backend switching is disabled while the native inspector is attached. Brightness
remains a visual effect, not a calibrated optical measurement.

The supplied URDF and STL remain in `../starter_pack/robot/`. Paths are resolved
relative to the Python source, and the base frame stays fixed. The light follows
the emitter's computed transform, including the head joint's 180-degree yaw.
The URDF's camera frame is only a marker: moving the virtual head would not move
the laptop's real camera.

## Benchmarks and future comparison

See [benchmark methodology and upgrade evidence](benchmarks/README.md).
Record every installed gesture in each of the four modes:

```powershell
uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5
```

Each run creates a new timestamped directory under `benchmarks/` with a report,
summary CSV/JSON, per-trial metrics, compressed per-frame traces, package versions,
and source hashes. Keep existing runs unchanged. Add a gesture JSON, validate it,
then rerun with the same settings to compare. These are headless measurements;
they do not measure browser FPS or establish Ubuntu performance.
The preserved October 1 baseline covers the original nine gestures only.
With the full emotion library, three repeats across four modes produce 1,476
measured trials plus warmups, so allow more time for a complete run.

The [October 13 baseline](benchmarks/2026-10-13-current-setup/README.md) records
the current Python/MuJoCo/viser setup before any controller-language or frontend
changes. It includes all 123 gestures plus a separate browser workload.
To repeat both measurements, choose new output directories and run sequentially:

```powershell
uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5 --output benchmarks/my-baseline/headless
uv run --locked python benchmarks/measure_viewer.py --output benchmarks/my-baseline/browser
```

The browser harness starts its own temporary server and headless Chrome profile,
then stops both. Use `--chrome` to supply another Chrome executable path. It
records three 15-second idle trials and three 30-second active trials, along
with server tick timing, browser draw cadence, command acknowledgment latency,
and sampled CPU/memory use. Draw cadence counts frames submitting WebGL work;
it does not measure completed GPU presentation. CPU percentages use one logical
core as 100%, and summed browser working sets may double-count shared memory.
The harness uses the existing locked dependencies and leaves application code
unchanged. See the saved baseline for hardware and measurement limitations.

## Development and validation

```powershell
uv run --locked pytest -q
```

Tests cover joint bounds, velocity, gesture interruption, Stop, Reset, invalid
inputs, mesh loading, and the light frame's orientation. For a visual check,
exercise all five sliders, verify the shade appears, click Nod then Stop, toggle
the light, and reload the browser while a pose is held.

The tests also cover MuJoCo/original frame parity, physical tracking, effort
limits, contact filtering, fractional timesteps, and benchmark trace semantics.
Ubuntu device and performance validation remains outstanding.

- `gestures/*.json`: editable gesture definitions; see `gesture.schema.json`.
- `lamp_sim/gestures.py`: JSON loading and validation.
- `lamp_sim/motion.py`: joint limits and generic gesture playback; radians and seconds.
- `lamp_sim/viewer.py`: scene, degree-based controls, and the update loop.
- `lamp_sim/backends.py`: source-model conversion, direct playback, physics, and diagnostics.
- `lamp_sim/benchmark.py`: repeatable comparisons and trace logging.
- `lamp_sim/__main__.py`: command-line entry point.
- `tests/test_motion.py`: renderer-independent checks.
- `tests/test_gestures.py`: JSON validation, step semantics, and reload checks.
- `tests/test_backends.py`, `tests/test_benchmark.py`: physics and logging checks.

Use four-space indentation, `snake_case` functions, and type hints. Future
perception/speech code should send commands through the viewer's queue rather
than mutate motion from background threads. Manual controls and future behavior
commands must share the same limit-enforcing controller.

## Ubuntu 24.04

With Python 3.12 and uv installed, the same commands work from a checkout:

```bash
cd SWE_route/simulation
uv sync --locked
uv run --locked python -m lamp_sim
uv run --locked pytest -q
```

The viewer needs a graphical browser and working WebGL, with integrated graphics
or software rendering. CUDA and a discrete GPU are not required. Windows checks
do not establish performance on the challenge's four-core, 8 GB Ubuntu laptop;
validate that target separately, including browser CPU and memory usage.

If the page is blank, check WebGL availability and the terminal for mesh or
dependency errors. If the page disconnects, make sure the Python process is
still running and use its printed port.
