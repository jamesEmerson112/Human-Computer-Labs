# Lamp backend benchmark

Measured at 2026-10-02T03:34:25.731474+00:00. 3 measured trials per gesture and backend; one warmup each. Pace: 0.5.

Headless backend + trajectory generation; browser rendering, network transport, and disk writes excluded from per-frame compute times. Loop throughput includes in-memory metric collection. No real-time sleeping.

| Backend | Median real-time factor | Worst trial p95 compute (ms) | Peak process RSS (MiB) | Tracking RMSE (deg) | Max error (deg) | Max penetration (mm) | Over 33.3 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| kinematic | 91.9x | 0.791 | 108.9 | 0.000 | 0.000 | not measured | 0 |
| mujoco-kinematic | 261.2x | 0.493 | 125.2 | 0.000 | 0.000 | 0.000 | 0 |
| mujoco-source | 33.9x | 2.075 | 125.7 | 2.765 | 31.792 | 1.000 | 0 |
| mujoco | 58.1x | 1.274 | 125.1 | 1.135 | 4.430 | 0.000 | 0 |

Kinematic zero tracking error is by construction: poses are assigned directly. It is not evidence of motor accuracy. Physics errors and contacts are additional diagnostic capabilities, not a regression against a physical baseline.

`mujoco-source` retains the source self-contact geometry. `mujoco` excludes only upper_arm_link / lamp_head_link because those proxies overlap at zero pose. Other self/table contacts remain enabled. The exclusion can also hide genuine contact between that pair; keep the source mode for audits.

Each raw .csv.gz row contains simulation time, desired/actual joint angles, velocities, contact diagnostics, effort, and compute time. JSON files include per-gesture trials, initialization cost, settling error, warnings, and soft-limit violations. None of these results establish real-hardware accuracy or Ubuntu performance.
