# Speaking performance demonstration

The lamp accepts a supplied emotion and speaks an exact prepared line. Nine
presets cover the greeting across six core emotions plus Neutral, and Happy with
a question and reassurance. The three lines use fixed prerecorded local speech;
movement carries the emotional variation. See [speech provenance](../../performances/README.md)
and [usage](../../README.md#hear-the-lamp-perform-a-line).

The Rust coordinator supplies emotional posture, phrase accents, smooth entry
and settling, and occasional small idle movement. Requested joint position,
velocity, and acceleration limits remain separate from actual MuJoCo tracking.
It uses monotonic wall time so delayed frames skip obsolete cues. The browser
decodes the prepared WAVs before submitting the performance and schedules one
audio buffer against the announced onset, with 180 ms gaps between phrases.
Only the initiating browser plays speech; other tabs observe the shared lamp.

## Validation

- **578 Python tests**, **24 Rust tests**, and **32 viewer tests** passed; native
  release and production frontend builds succeeded. Rust coverage includes
  existing Python gesture parity and 37,800 performance samples across seven
  emotions, three scenes, three starting poses, and all performance phases.
- [Final performance browser run](performance/result.json): **12 groups** with
  real Web Audio buffers, native physics, and ordinary browser audio policy.
  Trusted browser clicks start playback; no autoplay bypass is used. Checks cover
  preparation cancellation, complete delivery, different emotion/text motion,
  replacement, Stop before speech, Reset/manual overrides, audio suspension,
  two tabs, disconnect/reconnect, gesture overrides, and mobile layout.
- [Native protocol](protocol.json): **9 groups**, 41 commands, 230 states, and
  24 forward-kinematics samples.
- [Existing viewer](viewer-regression/result.json): **9 groups** for gestures,
  manual controls, lighting, camera, modes, reconnect, and mobile layout.
- [Animation reviews](review-regression/result.json): **11 groups** for checkbox
  persistence, import/export, migration, and cross-tab history. These synthetic
  test decisions are not user approvals.
- [Development proxy](dev-proxy/result.json): **5 groups**, including the new
  speech manifest/WAV route and existing local-origin guards.

The [validation manifest](validation.json) pins the evidence and executable
hashes. `performance/result.json` records the final frontend bundle; the separate
regression artifacts retain the hashes of their tested bundles. Original gesture
files and historical benchmark folders were not changed.

## Observed sample and limits

Five complete performances ran once each: Happy, Sad, and Angry greeting; Happy
question; Happy reassurance. All reached anticipation, speaking, settling, and
attentive idle. Recorded requested trajectories differed with both emotion and
text. Actual physics advanced and moved, with **zero contacts, zero solver
warnings, and zero observed actual soft-limit overshoot** in these five runs.
Maximum observed tracking error was **2.691 degrees**.

The run used Windows 11 build 26200, a Ryzen 5 2600X, Chrome 155.0.8059.41, and
normal background desktop applications. Sampled native server peak RSS was
**13.11 MiB**; mean native CPU use was **0.0266 cores**. These exclude the browser
and do not establish total application resource use or target Ubuntu performance.

The controller deliberately announces speech 800 ms ahead to allow anticipation
and browser scheduling. Across five samples, command-to-scheduled-audio-processing
onset was **753 ms median / 760 ms p95**. Audio-output timestamp compensation
schedules processing ahead of expected speaker output. These figures exclude
audio fetch/decode and are not microphone-measured audible latency. Phrase timing
includes silence in the generated recordings; no word alignment is claimed.

This is a small functional sample, not a matched performance comparison. Sound
quality, emotional recognizability, and whether the character feels alive still
need human listening and viewing. The automated run checks nonempty PCM content,
running audio contexts, source connections, scheduling, cancellation, actual
physics motion, and separate requested-versus-observed diagnostics.

## Inspect and reproduce

See the [desktop performance](performance/speaking.png) and
[mobile controls](performance/mobile.png). Use fresh output paths from
`SWE_route/simulation/`:

```powershell
uv run --locked pytest -q
cargo test --locked --manifest-path controller/Cargo.toml
npm --prefix viewer test
npm --prefix viewer run build
cargo build --release --locked --manifest-path controller/Cargo.toml --bins
uv run --locked python tools/verify_performance.py --output .verification/my-performance
uv run --locked python tools/verify_native_protocol.py --output .verification/my-protocol.json
uv run --locked python tools/verify_native_viewer.py --output .verification/my-viewer
uv run --locked python tools/verify_animation_review.py --output .verification/my-reviews
uv run --locked python tools/verify_dev_proxy.py --output .verification/my-dev-proxy
```

Windows requires the running release server to exit before its executable can
be rebuilt. The development-proxy check temporarily owns ports 8082 and 5173;
other verification commands start isolated servers on available loopback ports.
