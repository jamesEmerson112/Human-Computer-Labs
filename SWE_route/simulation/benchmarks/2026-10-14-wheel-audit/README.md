# Selected emotion wheel audit

The selected image has six retained families: Fearful, Angry, Disgusted, Sad,
Happy, and Surprised. The complete Bad branch is excluded. The active library
contains **117 emotions** (6 core, 37 middle, 74 outer) and **9 utility clips**.
Neutral / Reset pose is a separate command, not an idle-hold animation.

## Coverage and review decisions

The [coverage report](../../catalogs/emotion-wheel-audit.md) maps every target:
25 exact paths, 40 core-family renames, 29 contextual reuses, and 23 authored
motions. Original motion steps remain unchanged for reused clips. Twenty old
IDs are retired. The [original archive](../../gestures/archive/six-core-wheel/manifest.json)
preserves all 123 raw source files and both original documents. Active files use
LF line endings; archived bytes and this measured evidence are exempt from Git
newline conversion so their hashes remain verifiable.

The [review migration](../../catalogs/emotion-wheel-migration.json) permits
approval transfer for 65 equivalent emotion paths and nine utilities. All 52
new or contextually reassigned entries require a fresh check. Version 2 reports
bind decisions to file revisions and preserve earlier reviews in history.
Version 1 imports assume the explicit archived baseline because they contain no
motion hashes; the original browser storage key remains an untouched backup.

Checked means pass; unchecked means fail. These artifacts do not record human
approval. All review files under `review/` and `review-final/` contain synthetic
test decisions and must not be used to choose real animation revisions.

## Technical results

- **569 Python tests**, **15 Rust tests**, and **24 viewer tests** passed. The
  production frontend built successfully and model export verification passed.
- [Motion audit](motion-audit/report.md): **756 controller cases** covering all
  126 clips from neutral and both near-limit poses at Normal and Full pace.
  No requested limit/speed, completion, interruption, emotion return, or reset
  failures occurred. Near-limit testing here covers requested motion.
- Rust/Python controller parity covered **378 start cases**, **103,843 frames**,
  and **378 interruptions**.
- [Fresh physics reference](reference/report.md): one measured repeat per clip
  in each of four Python modes, after one warmup per clip: **504 trials**.
- [Native comparison](native/summary.json): **115,164 frames** across three
  native modes matched the same current Python workload within `1e-9`, with
  complete repetitions and no duplicate rows. This replaces neither the
  original October 13 results nor their workload.
- Default MuJoCo's 126 neutral-start trials recorded **zero contacts and solver
  warnings**. Maximum tracking error was **4.430200 degrees** and actual
  soft-limit overshoot was **1.027361 degrees**. Requested-angle limits and
  actual physics observations are separate; actual overshoot is not hidden.
- [Final review checks](review-final/result.json): **11 groups** covering
  checkbox state, rings, persistence, v1 migration, v2 import/export, archived
  decisions, invalid imports, and durable cross-tab history.
- [Final viewer regression](viewer-final/result.json): **9 groups** covering
  playback, Stop, reset, modes, lighting, camera, reconnect, and mobile layout.
- [Protocol checks](protocol.json): **9 groups**; [development proxy](dev-proxy/result.json):
  **5 groups**, including revision hashes, migration metadata, and local origin guards.

The [short Python browser run](python-browser-smoke/manifest.json) confirms the
retained viewer loads 126 clips and plays all six renamed families. It uses one
2-second idle phase and one 12-second active phase. It ran alongside functional
checks and is a smoke test, not a comparable performance measurement.

Headless measurements ran on the existing Windows Ryzen 5 2600X host with
normal background applications. There was one measured repeat per clip, no
rendering, and no simultaneous agent test/build workload during the headless
runs. Native default physics recorded a worst per-trial p95 compute of
**0.292 ms** and sampled peak process RSS of **10.83 MiB**; these observations
do not establish browser latency or performance on the target Ubuntu laptop.
The raw manifests record versions, workload, host load, source hashes, and
executable hashes. Model settings and contact exclusions were unchanged.

## Human review queue

The trajectory audit found **zero identical step groups** and **10 similar
pairs**. Start with [the pair table](motion-audit/report.md#similarity-review-candidates),
especially Sad / Despair versus Sad / Despair / Grief. These are prompts to
inspect meaning and timing, not automatic failures or passes. The heuristic
resamples phase and cannot establish emotional recognition.

Open the viewer, choose **Core emotions** in the review-round filter, then work
through middle and outer emotions. Use a consistent camera, Neutral / Reset pose,
and Normal pace. Check accepted movements; add notes to failures. Export the
review when finished so subsequent work can revise the actual failed list.

## Reproduce

From `SWE_route/simulation/`, use fresh output paths:

```powershell
uv run --locked python tools/build_emotion_library.py --check
uv run --locked pytest -q
cargo test --locked --manifest-path controller/Cargo.toml
npm.cmd --prefix viewer test
npm.cmd --prefix viewer run build
uv run --locked python -m lamp_sim.benchmark --repeats 1 --speed 0.5 --output benchmarks/my-wheel/reference
uv run --locked python benchmarks/compare_native.py --reference benchmarks/my-wheel/reference --repeats 1 --output benchmarks/my-wheel/native
uv run --locked python tools/audit_emotion_motion.py --physics-reference benchmarks/my-wheel/reference --output benchmarks/my-wheel/motion-audit
uv run --locked python tools/verify_animation_review.py --output benchmarks/my-wheel/review
uv run --locked python tools/verify_native_viewer.py --output benchmarks/my-wheel/viewer
```

The generator's default check makes no edits. Its explicit apply mode refuses
to overwrite gestures edited since the audited source/target versions. Historical
benchmark folders remain intact. Earlier `review/` and `viewer/` runs in this
folder preceded the final review-transaction safeguards; final acceptance uses
`review-final/` and `viewer-final/` with their recorded bundle hashes.
