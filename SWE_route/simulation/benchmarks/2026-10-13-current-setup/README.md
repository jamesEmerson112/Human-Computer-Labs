# Current setup performance baseline

Recorded on **October 13, 2026, America/Los_Angeles** (October 14 UTC), before
changing the controller language or browser frontend. The application remains
Python + MuJoCo + viser, with 123 JSON gestures. No application, gesture, or
dependency changes were made for this measurement.

Starting Git revision: `46ea0d26321f976d9340dc5406e149f4da2db64c` on `main`.
The working tree was clean and no lamp process was running at the start.
[Context and commands](context.json) and both run manifests record provenance.

## Machine and software

| Item | Recorded configuration |
|---|---|
| Operating system | Windows 11, build 26200 |
| CPU | AMD Ryzen 5 2600X, 6 physical / 12 logical cores |
| Memory | 31.94 GiB; 16.63 GiB available at headless start |
| GPU | NVIDIA GeForce GTX 1070 Ti; Windows driver 32.0.15.8266 |
| Python | 3.12.10 |
| Packages | MuJoCo 3.14.0, viser 1.1.1, yourdfpy 0.0.60, NumPy 2.5.3, psutil 7.2.2 |
| Browser | Chrome 155.0.8059.41, isolated profile, headless mode |
| Browser rendering | ANGLE / D3D11 on the GTX 1070 Ti |
| Actual viewport / canvas | 1264 x 805 / 940 x 805 pixels; device pixel ratio 1 |
| Simulation settings | Normal pace (0.5), requested 30 Hz control, 240 Hz physics |

This was a desktop measurement with normal background applications left alone.
During browser trials, the median of phase-average **whole-machine CPU load**
was 40.7% idle and 45.8% active, including all processes. Consequently this is an
observed baseline under background load, not an otherwise-idle-machine capacity
test. Do not attribute differences in a later run solely to language or frontend
changes without also matching or controlling that load.

## Headless backend measurements

All 123 gestures ran through each of four backends with one warmup and three
measured repeats: **1,476 measured trials, 458,328 recorded control frames**.
Every trial started at zero and included one simulated second to settle.
The modes ran sequentially in fresh worker processes, before the browser test.

| Backend | Worst trial p95 compute | Median simulation throughput | Peak worker RSS | Median CPU ms / simulated second |
|---|---:|---:|---:|---:|
| Original kinematic | 0.705 ms | 100.1x real time | 110.3 MiB | 10.12 |
| MuJoCo pose check | 0.308 ms | 257.5x real time | 126.1 MiB | 3.88 |
| MuJoCo source contacts | 2.539 ms | 32.5x real time | 126.1 MiB | 30.62 |
| **Default MuJoCo** | **2.521 ms** | **46.2x real time** | **126.3 MiB** | **21.19** |

Compute time includes trajectory generation and backend advance. It excludes
the viewer, networking, serialization, and real-time sleeping. Throughput also
includes in-memory metric collection. These rates are not browser FPS.

Default MuJoCo had one frame above the 33.3 ms compute budget out of 114,582:
**34.234 ms**, in `limit_sweep`, repeat 1. It had zero solver warnings, a
frame-weighted tracking RMSE of **1.034 degrees**, maximum tracking error of
**4.430 degrees**, and maximum observed soft-limit overshoot of **1.027 degrees**.
Source contacts produced a maximum tracking error of 32.376 degrees and 1 mm
penetration. The existing exclusion of the overlapping upper-arm/head pair
remains in place for default physics.

See the [headless report](headless/report.md), [summary](headless/summary.json),
[manifest](headless/manifest.json), per-backend trial JSON, and compressed frame
CSVs in [headless/](headless/). Counts in each trace were checked against its
trial records. Source hashes match the browser run and the current application.

## Live viewer measurements

The harness runs the existing viewer with a timing wrapper around `tick` in a
separate process. After a five-second browser warmup, each of three repetitions
contains 15 seconds idle and 30 seconds active. Each phase starts with Reset
and a 1.5-second settle. Active phases issue the six core emotions in this order:
Anger, Disgust, Fear, Happy, Sad, Surprise, one every five seconds. This workload
intentionally includes interruptions of longer clips. The full gesture corpus
is covered by the headless test, not the browser sample.

Values below are **medians across the three trials**, except memory, which is
the highest sampled process-tree total in that phase type.

| Metric | Idle | Active gestures |
|---|---:|---:|
| Server update rate | 29.11 Hz | 29.15 Hz |
| Server tick compute p95 | 8.22 ms | 8.44 ms |
| Server tick interval p95 | 41.18 ms | 41.43 ms |
| Simulated seconds / wall second | 0.922 | 0.924 |
| Browser animation callback rate | 74.99 Hz | 74.99 Hz |
| Browser frames submitting WebGL drawing work | 66.96 / s | 74.96 / s |
| Browser draw-frame interval p95 | 13.40 ms | 13.40 ms |
| Server CPU, one core = 100% | 18.9% | 20.5% |
| Browser process-tree CPU, one core = 100% | 40.3% | 52.9% |
| Peak sampled server process-tree working set | 155.5 MiB | 155.8 MiB |
| Peak sampled browser process-tree working set | 846.3 MiB | 857.1 MiB |

The active server and browser CPU medians correspond to approximately 1.7% and
4.4% of this machine's 12-logical-core capacity. Summed working sets can count
shared pages more than once; they are not private allocated memory or VRAM.
CPU and memory were sampled about every 0.5 seconds, so brief peaks can be missed.

Across 18 Play commands, **click-to-status acknowledgment** was **86.0 ms median**,
**130.6 ms p95**, and **141.4 ms maximum**. The timer starts at the browser button
click and ends when the returned `Playing <emotion>` status appears in the DOM.
The existing status display updates at about 10 Hz. This measurement is not
click-to-first-motion or click-to-visible-pixel latency.

Observed startup was 3.17 seconds from starting the viewer subprocess to HTTP
availability, and 1.50 seconds from browser navigation to all controls being
available. Each was measured once, with readiness polling, so treat them as
approximate observations rather than percentile results. All six phases
recorded zero browser long tasks of at least 50 ms.

The current loop caps each simulated step at 1/30 second, discarding catch-up
time after late ticks. It advanced simulation by about 0.92 seconds per wall
second in this run. This is an existing pacing limitation to preserve in the
baseline; the application was not optimized during measurement.

Browser draw rate counts animation-frame intervals containing WebGL draw calls.
It does not prove GPU completion, monitor presentation, or 75 distinct physics
poses per second. Physics state arrives at the lower server update rate. The
probe itself and DevTools add measurement overhead. Idle drawing varied from
53.46 to 67.46 frames/s; use per-trial records when comparing it.

See [browser summary](browser-02/summary.json), [raw measurements](browser-02/measurements.json),
[server tick trace](browser-02/server-ticks.csv), and [browser manifest](browser-02/manifest.json).
The [ready screenshot](browser-02/viewer-ready.png) confirms that the lamp, mesh,
controls, and lighting rendered. The earlier `browser/` attempt failed while
setting up its timing probe, before any measured phase; it is marked incomplete
and is excluded from every figure above.

## Reproduction and validation

From `SWE_route/simulation/`, choose fresh output directories and run sequentially:

```powershell
uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5 --output benchmarks/my-baseline/headless
uv run --locked python benchmarks/measure_viewer.py --output benchmarks/my-baseline/browser
uv run --locked pytest -q
```

The [viewer harness](../measure_viewer.py) uses existing locked dependencies and
an installed Chrome executable. Override its default Windows Chrome path with
`--chrome`. It creates an isolated temporary browser profile and server, and
stops its own processes afterward. Preserve the captured workload, pace, viewport,
GPU/driver, package versions, contact policy, and comparable background load when
measuring a replacement. If application code or gesture data changes, compare
the recorded hashes and state that change explicitly.

The current simulation suite passed **538 tests** after the measurements.
The original October 1 and October 4 reports were preserved. This baseline
does not include camera/microphone processing, speech/model inference, network
services, real robot hardware, or validation on the four-core, 8 GB Ubuntu target.
