# Simulation benchmarks and upgrade decision

## Rust / MuJoCo / Three.js comparison

Build the native application using the [simulation instructions](../README.md#rust-controller-and-threejs-viewer),
then run these sequentially from `simulation/` with new output directories:

```powershell
uv run --locked python benchmarks/compare_native.py --output benchmarks/my-native/headless
uv run --locked python benchmarks/measure_viewer.py --output benchmarks/my-native/python-browser
uv run --locked python benchmarks/measure_native_viewer.py --output benchmarks/my-native/rust-browser
uv run --locked python benchmarks/measure_native_viewer.py --preset default --output benchmarks/my-native/rust-browser-default
```

`compare_native.py` first remeasures Python default physics, then measures all
three native MuJoCo modes in fresh processes. Each uses all 123 gestures, three
repeats, pace 0.5, one warmup per gesture, and the original settling-frame rule.
It compares every native requested/actual/velocity trace against the preserved
October 13 traces, records source/executable hashes, process memory and host load,
and retains compressed frame logs. The original yourdfpy-only mode remains in
Python; it has no native counterpart. Headless compute excludes networking and
rendering. Whole-loop throughput includes each implementation's observation and
in-memory logging overhead, so it is a pipeline measurement rather than an
isolated language benchmark.

### October 14 native migration results

[Audited headless results](2026-10-14-native-stack/analysis.json) compare 369
current Python default-physics trials with 1,107 native trials across the three
MuJoCo modes. All **343,746 native frames** match the historical requested angles,
actual angles, and velocities within `1e-9`; repetition completeness and duplicate
rows were checked separately. A further 123 Ubuntu 24.04 WSL default-physics
trials match the Windows reference within `5.11e-15`.

| Default-physics headless measurement | Current Python | Rust |
|---|---:|---:|
| Worst trial p95 compute | 2.512 ms | 0.328 ms |
| Median real-time factor | 54.7× | 437.4× |
| Pooled CPU ms / simulated second | 18.69 | 2.36 |
| Peak worker memory | 124.7 MiB | 11.4 MiB |
| Tracking RMSE | 1.034° | 1.034° |
| Maximum soft-limit overshoot | 1.027° | 1.027° |

Both default-physics runs had zero over-budget frames, solver warnings, and
penetration. This is the Ryzen 5 2600X Windows host; mean whole-host CPU load was
22.6% during Python and 23.2% during Rust, including other desktop work. The
clock and rendering are outside the headless measurement. Ubuntu WSL establishes
native portability, not performance on a physical four-core, 8 GB Ubuntu laptop.

The [validation manifest](2026-10-14-native-stack/validation.json),
[live protocol checks](2026-10-14-native-stack/protocol-04.json), and
[browser interaction checks](2026-10-14-native-stack/visual-04/result.json)
record the accepted evidence. Inspect the
[viewer screenshot](2026-10-14-native-stack/visual-04/viewer-ready.png) and
[light view](2026-10-14-native-stack/visual-04/viewer-light-view.png).
The original Python tests still pass (538); native tests pass on Windows and
Ubuntu (12 each), and seven frontend tests plus the TypeScript build pass.
The [development-proxy checks](2026-10-14-native-stack/dev-proxy/result.json)
verify local HTTP/WebSocket forwarding and rejection of foreign origins.

Use `analysis.json` for headless resource conclusions. The first generated
`headless/summary.json` sampled the small Windows Python launcher rather than its
child for RSS; the audit uses the Python worker's own recorded peak instead.
The executed harness is preserved as `headless/compare_native.measured.py`, and
the reusable harness now applies that correction. Pooled CPU avoids misleading
zero medians caused by Windows CPU-time granularity on short native trials.
The headless run predates a transport-only change enabling TCP_NODELAY on accepted
connections; motion and physics code are unchanged. The
[archived transport source](2026-10-14-native-stack/pre-transport/provenance.json)
matches the initial measurements. Final live runs identify the updated executable
and built frontend by hash.

### Live browser comparison

The fresh [Python run](2026-10-14-native-stack/python-browser-02/summary.json)
and final [native matched run](2026-10-14-native-stack/rust-browser-02/summary.json)
use Chrome 155 on the same Windows host and GTX 1070 Ti. Each has three 30-second
active phases following warmup. Rates and CPU below are phase medians, memory is
the maximum sampled active-phase peak, and latency pools all 18 commands.

| Active workload | Python / viser | Rust / Three.js |
|---|---:|---:|
| Server control rate | 29.24 Hz | 30.00 Hz |
| Simulated seconds / wall second | 0.925 | 1.000 |
| Server CPU, one-core percent | 14.51% | 1.72% |
| Server peak sampled memory | 154.7 MiB | 19.0 MiB |
| Browser process-tree CPU, one-core percent | 35.21% | 32.08% |
| Browser process-tree peak sampled memory | 850.6 MiB | 785.9 MiB |
| Submitted WebGL draw frames / second | 74.96 | 74.96 |
| Click-to-status p50 / p95 | 67.9 / 132.1 ms | 49.0 / 109.6 ms |
| Median phase p95 server compute | 5.569 ms | 0.307 ms |

Native simulation dropped no time in these six measured phases. Both renderers
were near the browser's 75 Hz animation cadence; this establishes no FPS gain.
Whole-host active-phase CPU averages ranged from 17.5-19.1% for Python and
10.1-11.5% for native, so latency and renderer differences are observations under
different background load, not an isolated language or transport experiment.
The initial native matched run had 165.4 ms p95 click-to-status latency and a
348.3 ms maximum. After enabling TCP_NODELAY, the final run measured 109.6 ms
p95 and 120.4 ms maximum; the earlier evidence remains in `rust-browser/`.

The native browser harness uses the original Chrome, WebGL draw, and process
resource probes. `?benchmark=1` fixes the canvas to 940×805, DPR 1, camera position
and target, 75-degree field of view, scene dimensions, and baseline light settings.
Both browser workloads play the same six core emotions at five-second intervals
after warmup; three idle and active phases last 15 and 30 seconds respectively.
Frontend implementation and operating-system load still differ. The native
renderer buffers one 33.3 ms control frame for interpolation; click-to-status
latency does not include that display delay or measure completed GPU presentation.
Use `--preset default` to record the delivered camera, lighting, and shadows
separately; that run is not a matched rendering-cost comparison.

The final [normal-view run](2026-10-14-native-stack/rust-browser-default-02/summary.json)
used a 924-by-641 canvas with shadows. Its active-phase medians were 30.00 Hz
control, 1.000 simulated seconds per wall second, 74.97 submitted draw frames/s,
2.39% server CPU, and 34.27% browser CPU. Active-phase sampled peaks were 19.1 MiB
for the server and 803.4 MiB for the browser tree. Click-to-status p50/p95 was
64.1/119.8 ms; no time was dropped and no browser long tasks were observed in
the six phases. Whole-host active-phase means ranged from 9.7-12.0%.
The earlier `rust-browser-default/` run is retained as a separate loaded-host
observation: whole-host CPU reached 100%, and two idle phases discarded a total
of 1.0 simulation second through the documented catch-up limit. It is excluded
from the final normal-view figures above. Rebuild the browser aggregates with
`uv run --locked python benchmarks/2026-10-14-native-stack/summarize_browser.py`.

Native `server-ticks.csv` includes controller, physics, body transforms, snapshot
serialization, and publication; disk-log writes are outside compute time. State
packets separately report controller/physics compute only. Catch-up batches are
reported as both step counts and batch time; summaries divide batch time by its
step count. Dropped simulation time remains visible. Browser CPU/RSS cover the
whole process tree, with 100% CPU meaning one logical core and shared memory
potentially counted more than once. Do not compare unmatched camera/load runs as
evidence of a Rust or renderer speedup.

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

The [October 13 baseline](2026-10-13-current-setup/README.md) captures all 123
gestures and a separate Chrome workload before changes to the controller
language or frontend. It records source hashes and the starting Git revision.
The older nine-gesture results below remain historical evidence; their workload
differs from the expanded library.

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

The library now has 123 gestures, including 114 entries from the supplied
emotion wheel. The [emotion-library smoke check](2026-10-04-emotions-smoke/report.md)
records functional validation and one default-MuJoCo trial per new clip.
The original nine-gesture baseline above remains unchanged; a full run now
includes 1,476 measured trials at three repeats across four backends.

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
