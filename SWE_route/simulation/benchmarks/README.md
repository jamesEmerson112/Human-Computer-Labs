# Simulation benchmarks and upgrade decision

## Why add MuJoCo?

Keep the browser and JSON authoring workflow, but use MuJoCo for physical
debugging. Direct playback can show a gesture's shape; it assigns the requested
angles exactly and cannot expose gravity sag, motor tracking, effort saturation,
or contact forces. MuJoCo adds these observations at a measured computational
cost. Zero kinematic tracking error is a consequence of assignment, not evidence
that a real motor follows perfectly.

The supplied fictional five-joint lamp remains the source model. Its mesh is
converted to binary STL in memory for import; visual/fixed frames are retained,
and the base is anchored. No company motion recordings or implementation code
were copied. Servo gains and rotor inertia are simulation assumptions.

## Recorded baseline

[Windows results and measured table](2026-10-01-windows/report.md) include
108 measured trials: nine gestures, four backends, three repeats. The folder is
named for the local date; the manifest records UTC. The host is a Ryzen 5 2600X
(6 cores/12 threads), 32 GiB RAM, Windows 11, with a GTX 1070 Ti available for
graphics. The headless timing benchmark does not use GPU rendering.

| Measured result | Original playback | Default MuJoCo physics |
|---|---:|---:|
| Worst trial p95 compute | 0.791 ms | 1.274 ms |
| Median real-time factor | 91.9x | 58.1x |
| Peak worker memory | 108.9 MiB | 125.1 MiB |
| Median CPU time / simulated second | 11.30 ms | 17.05 ms |
| Frames exceeding 33.3 ms | 0 | 0 |

The measured overhead leaves ample headless control-loop capacity on this host.
The reason to upgrade is the additional physics feedback. With source contacts,
maximum tracking error was **31.79 degrees**; excluding the overlapping pair
reduced it to **4.43 degrees** across this suite. This is a contact-policy
comparison, not proof of real-world accuracy. Default physics had **1.13-degree
tracking RMSE**, **1.03-degree maximum soft-limit overshoot**, and **zero solver
warnings**. Overshoot remains a documented tuning issue for future comparisons.

Validation accompanying this run: 79 pytest checks passed; Chrome automation
exercised all four modes, gesture loading, playback, Stop, and Reset. MuJoCo's
offscreen OpenGL renderer produced a frame successfully. The optional native
inspector's interactive controls and Ubuntu deployment were not exercised.

## Reproduce or extend

From `SWE_route/simulation/`:

```powershell
uv sync --locked
uv run --locked python -m lamp_sim --check-gestures
uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5
```

By default, output goes to a new UTC timestamp directory. `--output benchmarks/my-run`
selects another new directory; existing directories are rejected to preserve
history. Close the viewer and avoid other heavy workloads while measuring.
Add gesture JSON files to extend the suite; they are discovered automatically.

The backends run sequentially in separate fresh Python processes:

1. `kinematic`: original trajectory controller plus yourdfpy forward kinematics.
2. `mujoco-kinematic`: identical requested angles assigned directly in MuJoCo;
   useful for conversion/FK comparisons, without physical force interpretation.
3. `mujoco-source`: dynamics with the original collision shapes.
4. `mujoco`: the same dynamics with one documented contact-pair exclusion.

Every gesture warms up once per backend, then resets to zero for each measured
trial. Pace is 0.5, control is 30 Hz, physics is 240 Hz, and each trial includes
one simulated second after the command ends. Timing uses a monotonic clock with
no real-time sleeping. All modes receive the same desired trajectory.

## Metrics and artifacts

| Artifact or metric | Interpretation |
|---|---|
| `manifest.json` | Host, packages, rates, gains, contact policy, and SHA-256 hashes of source, model, gestures, schema, and lockfile. |
| `summary.csv`, `summary.json`, `report.md` | Comparable aggregate results. |
| `<backend>.json` | Every trial's timing, CPU time, tracking/settling error, contacts, effort, speed ratio, soft-limit overshoot, and warnings. |
| `<backend>.csv.gz` | Every frame's desired/actual radians, velocities, simulation time, compute time, contacts, penetration, force, and effort. Blank values mean unavailable. |
| Worst trial p95 compute | Largest per-trial 95th percentile for trajectory generation plus backend advance; compare with the 33.3 ms control budget. |
| Median real-time factor | Median simulated seconds / measured loop seconds across trials. Includes in-memory logging; excludes CSV serialization. |
| Peak process RSS | Whole worker process including imports and trace buffers; Windows peak working set. On platforms without `peak_wset`, final RSS is the fallback; per-trial sampled peaks are also recorded. |
| Tracking RMSE | Error across all joints/frames, weighted by trial frame count; longer gestures contribute more. |
| CPU ms / simulated second | Process CPU time divided by simulated duration, not whole-machine CPU utilization. |

Per-frame contact counts, penetration, force, and effort are maxima over physics
substeps; observed angles are sampled at the control-frame end. Physics warning
counts are cumulative within each trial. Browser drawing, networking, camera/
microphone processing, and speech/model inference are outside this benchmark.

## Contact policy and limits of the evidence

The source proxies for `upper_arm_link / lamp_head_link` overlap by **1 mm at
zero pose**. The adjusted mode excludes only that pair; other robot/table
contacts remain enabled. Comparing source and adjusted dynamics isolates the
effect of this exclusion. It can also conceal genuine contact between those
parts, so retain the source mode for audits and improve the collision geometry
before claiming physical fidelity.

Requested positions and speeds are bounded. Actual physics can overshoot the
soft range, which is recorded rather than hidden by overwriting physical state.
URDF effort limits are applied, but servo gains, rotor inertia, friction, and
collision approximations have not been calibrated against hardware. A clear
contact log is only evidence for these trajectories and this contact policy.

Compare future runs using matching hardware, versions, pace, rates, and source
hashes. When changing a gesture, report it explicitly; use unchanged gestures
for an apples-to-apples regression comparison. These Windows/headless results
do not establish browser FPS, hardware accuracy, or performance on the challenge's
four-core, 8 GB Ubuntu laptop. Test that deployment separately.

MuJoCo reference: [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
and [modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html).
