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
poses for display. Both runtimes use the same 126 JSON gestures and supplied robot assets.
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

**Pace** starts at **1.0×**, with a maximum of **3.0×**. Joint angle and velocity
limits still apply: faster playback shortens holds and slower authored motions,
but cannot accelerate a move already at its joint speed limit. Speech timing is fixed.

The native modes are `mujoco`, `mujoco-source`, and `mujoco-kinematic`. The original
yourdfpy `kinematic` mode and teaching controls remain in the Python viewer.
Both viewers read the same gesture directory; **Reload gesture files** replaces the
catalog only when every file is valid. Commands sent while disconnected are
not replayed when a browser reconnects.

### Hear the lamp perform a line

The **A speaking moment** panel provides nine hardcoded performances. Compare
"Oh! You're back. I missed you." across Happy, Sad, Angry, Fearful, Disgusted,
Surprised, and Neutral, or try Happy with a question or reassurance. Select a
preset and press **Perform**. The line is spoken exactly as shown, while the lamp
anticipates, expresses the phrases, settles, and maintains a small attentive idle.
Repeat a preset to compare it from another view; use Reset pose first for a
consistent starting position.

The two logical inputs are the supplied emotion and the lamp's speech text.
For this first demo, a scene ID selects fixed text and prerecorded audio from
[`performances/demo.json`](performances/demo.json), and intensity is fixed at
0.7. All emotions share the same local voice recording so movement differences
can be compared. The [speech fixture notes](performances/README.md) record the
voice, measured phrase durations, file hashes, and regeneration command. Playback
needs no speech service, downloaded model, microphone, or external connection.
Arbitrary text synthesis, emotion mixtures, and word alignment are future work.

The Rust performance coordinator samples a continuous emotional pose and short
phrase accents against server wall time, then sends bounded requests through the
existing controller and MuJoCo. The browser prepares the audio before requesting
a performance and schedules it against the same announced start time. These are
new performance profiles; existing gesture files and their review decisions stay
independent. A source-clip approval does not approve the acting in this demo.

Only the browser that starts a performance plays its audio. Other tabs observe
the shared lamp. **Stop** cancels speech, future accents, and idle movement and
requests a hold; physical motors can still settle. Manual joint changes, gesture
playback, Reset pose, mode changes, and gesture reload cancel the performance.
The pace control is disabled while a performance is active, including its idle
phase, to preserve speech timing. A new prepared performance replaces the current
one. Closing or disconnecting its owner cancels it; reconnecting never replays it.
Light and camera controls remain available.

Rebuild the frontend and Rust executable using the commands above after an
update. The standalone application also needs the `performances/` directory.
Verify with:

```powershell
uv run --locked pytest -q tests/test_performance_speech.py
cargo test --locked --manifest-path controller/Cargo.toml
npm --prefix viewer test
uv run --locked python tools/verify_performance.py --output .verification/performance-demo
```

The browser check records actual Web Audio scheduling and cancellation, native
motion states, screenshots, and a small resource sample in a fresh output folder.
It uses its own browser profile. It does not record sound from the speakers or
assign human approval; listen and judge the delivery in the viewer.

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
Version 2 reviews include each animation's revision. Editing a gesture invalidates
its earlier pass while retaining notes and the previous decision in exportable
history. Version 1 reports can migrate through the explicit old-to-new catalog
mapping; because they have no revision hashes, this assumes they describe the
archived original library. Changed or reassigned motions require a new check.
The original browser storage is retained as a backup.

### Audit the selected emotion wheel

The selected wheel contains **Fearful, Angry, Disgusted, Sad, Happy, and
Surprised**. Its complete **Bad** branch is excluded. There are **117 emotion
entries** (6 core, 37 middle, 74 outer), plus the nine original utility clips.
**Neutral / Reset pose** is a separate control that smoothly requests zero joint
angles; `idle_hold` holds the current pose and is not a reset animation.

Use the ring filter to review core emotions first, then middle and outer rings.
For each comparison, use Reset pose, the same camera, and Normal pace. Judge the
movement's meaning, intensity, timing, and distinction from neighboring labels.
Check accepted motions and note what should change on failed ones. Repeated
words retain their full branch path. No automated check assigns human approval.

The [catalog audit](catalogs/emotion-wheel-audit.md) records retained, renamed,
reassigned, added, and retired entries. The [migration manifest](catalogs/emotion-wheel-migration.json)
records which reviews can transfer; the original files are preserved under
[`gestures/archive/six-core-wheel/`](gestures/archive/six-core-wheel/).
Archived documents retain their original bytes and relative links; use the
active [emotion catalog](gestures/emotions.md) to browse the current files.

To record a new technical audit, first generate a matching physics reference,
then choose a fresh audit output directory:

```powershell
uv run --locked python -m lamp_sim.benchmark --repeats 1 --speed 0.5 --output benchmarks/my-wheel/reference
uv run --locked python tools/audit_emotion_motion.py --physics-reference benchmarks/my-wheel/reference --output benchmarks/my-wheel/motion-audit
```

The audit checks requested limits, velocity, completion, interruption, return
poses, and reset. It also reports actual MuJoCo tracking and contact observations
from neutral starts, plus close trajectory pairs to inspect visually. Similarity
flags are review prompts, not proof that an expression is right or wrong.
The [recorded wheel audit](benchmarks/2026-10-14-wheel-audit/README.md) includes
the latest technical results and the ten similar pairs to prioritize in review.

### Ubuntu and standalone deployment

On Ubuntu 24.04, use the same preparation/build commands and launch
`controller/target/release/lamp-controller serve --port 8082`. For a standalone
SDK, set `MUJOCO_DIR` to the MuJoCo 3.14.0 directory before building; it must contain
the matching headers and shared library. `MUJOCO_LIBRARY` can override the runtime
library path. Deploy the executable with `viewer/dist/`, `models/derived/`,
`gestures/`, `performances/`, and that shared library, and pass `--root /path/to/simulation`.
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
**Pace** defaults to **1.0×** and goes up to **3.0×** for gestures and manual
movements. Each segment speeds up only as far as its joint velocity limits allow;
holds and slower authored moves can reach the full 3.0× rate. Changing pace
during a move keeps the requested trajectory continuous and within its speed limit.
Speaking performances use their fixed audio timeline and disable this control.
Actual physics can lag or overshoot that request.

The sliders show **target angles in degrees**. During a gesture they show its
current waypoint. **Current pose** shows the angles
the lamp has actually reached, alongside the requested angles. **Stop motion**
cancels the gesture and requests a hold at the observed pose; physical motors
may still settle. **Reset pose** smoothly requests zero.
**Reset view** restores your browser's camera without changing the robot.
**Show light direction** displays axes at the emitter; red points out of the shade.

## Edit movements as JSON

The library loads **126 gestures** from [`gestures/`](gestures/README.md): the
original nine above plus [117 emotion-wheel entries](gestures/emotions.md).
In the Python viewer's **Try a movement > Gestures**, expand Angry, Disgusted,
Sad, Happy, Surprised, or Fearful, choose an **Emotion**, and click **Play emotion**.
The clips cover all three rings of the selected wheel except its Bad branch,
including context-specific variants of repeated words. Each returns to its starting
pose. These are expressive interpretations that need human evaluation for
recognizability. Restart an older running viewer to load the grouped controls.

Edit a file and click **Reload gestures**.
Add a JSON file with a unique name and label to get another button, or use the
optional `group` field to put it in a family selector. An invalid
file displays an error and leaves the previous working library available.

The same library also drives the desk lamp in the MELD player page,
`ML_route/data/meld-player.html`. For each line it picks a clip from the
`group` family that matches the line's emotion, or a calm utility clip such as
`nod` for neutral. That page copies the gestures in when it is
built, so an edit reaches it only after a rebuild with
`ML_route/player/build_player.py`. The root
[README](../../README.md#8-run-the-live-player) gives the build commands.

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
With the full emotion library, three repeats across four modes produce 1,512
measured trials plus warmups, so allow more time for a complete run.

The [October 13 baseline](benchmarks/2026-10-13-current-setup/README.md) records
the previous Python/MuJoCo/viser setup before any controller-language or frontend
changes. It includes the original 123 gestures plus a separate browser workload.
It is a historical workload, not a matching reference for the current 126 clips.
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
