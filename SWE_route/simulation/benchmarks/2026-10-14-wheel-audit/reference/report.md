# Lamp backend benchmark

Measured at 2026-10-15T02:06:16.865011+00:00. 1 measured trials per gesture and backend; one warmup each. Pace: 0.5.

Headless backend + trajectory generation; browser rendering, network transport, and disk writes excluded from per-frame compute times. Loop throughput includes in-memory metric collection. No real-time sleeping.

| Backend | Median real-time factor | Worst trial p95 compute (ms) | Peak process RSS (MiB) | Tracking RMSE (deg) | Max error (deg) | Max penetration (mm) | Over 33.3 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| kinematic | 106.3x | 0.518 | 108.9 | 0.000 | 0.000 | not measured | 0 |
| mujoco-kinematic | 269.3x | 0.265 | 124.3 | 0.000 | 0.000 | 0.000 | 0 |
| mujoco-source | 34.8x | 2.154 | 124.7 | 4.754 | 32.376 | 1.000 | 0 |
| mujoco | 60.1x | 1.330 | 124.6 | 1.036 | 4.430 | 0.000 | 0 |

Kinematic zero tracking error is by construction: poses are assigned directly. It is not evidence of motor accuracy. Physics errors and contacts are additional diagnostic capabilities, not a regression against a physical baseline.

`mujoco-source` retains the source self-contact geometry. `mujoco` excludes only upper_arm_link / lamp_head_link because those proxies overlap at zero pose. Other self/table contacts remain enabled. The exclusion can also hide genuine contact between that pair; keep the source mode for audits.

Each raw .csv.gz row contains simulation time, desired/actual joint angles, velocities, contact diagnostics, effort, and compute time. JSON files include per-gesture trials, initialization cost, settling error, warnings, and soft-limit violations. None of these results establish real-hardware accuracy or Ubuntu performance.
