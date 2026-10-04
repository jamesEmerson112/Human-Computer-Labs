# Lamp playground

A local browser playground for the supplied five-joint lamp. Move joints, play a
gestures, and aim its light while learning how the robot fits together. MuJoCo
simulates gravity, motors, and contacts; viser draws the result in your browser.
The original direct playback remains available for comparison. Blender is optional and is not needed
to run this playground.

## Start on Windows

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

## Your first five minutes

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

The nine gesture buttons load their movements from [`gestures/`](gestures/README.md).
Edit a file and click **Reload gestures**.
Add a JSON file with a unique name and label to get another button. An invalid
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
Record all nine gestures in each of the four modes:

```powershell
uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5
```

Each run creates a new timestamped directory under `benchmarks/` with a report,
summary CSV/JSON, per-trial metrics, compressed per-frame traces, package versions,
and source hashes. Keep existing runs unchanged. Add a gesture JSON, validate it,
then rerun with the same settings to compare. These are headless measurements;
they do not measure browser FPS or establish Ubuntu performance.

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
