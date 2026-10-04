# Emotion library smoke check

The library now includes 114 emotion-wheel entries (6 core, 36 middle, 72 outer),
alongside the original nine gestures. Each wheel entry has a distinct sequence;
the four repeated emotion words retain separate branch-specific gestures.

## Validation

From `SWE_route/simulation/`:

```powershell
uv run --locked python -m lamp_sim --check-gestures
uv run --locked pytest -q
```

All 123 files validated; **538 tests passed**. Every new clip was exercised at
Normal pace from neutral and near both sets of joint limits, checking requested
position and velocity bounds, visible movement, completion, and exact requested
return to the starting pose. Every new clip was also interrupted with Stop.

Chrome checked selection and playback in each of the six family folders,
including repeated words in different branches, plus Stop, Reload, and Reset.
The lamp, shade mesh, and light spot rendered. See the
[browser screenshot](emotion-controls.png). This is a functional check, not a
human study of whether viewers recognize all 110 distinct emotion words.

## MuJoCo smoke measurements

The [per-clip results](trials.json) record one trial per new gesture at Normal
pace, starting from zero, with a one-second settling tail. There were no warmups.
The [manifest](manifest.json) records platform, versions, and source hashes.
This uses the existing `lamp_sim.benchmark.run_trial` measurement function and
the default MuJoCo contact policy. It is a Windows headless smoke check, not a
replacement for the preserved four-backend, three-repeat baseline.

| Metric | Observed |
|---|---:|
| Completed new gestures | 114 |
| Worst trial p95 controller/backend compute | 1.515 ms |
| Peak sampled worker RSS | 123.9 MiB |
| CPU time per simulated second | 21.84 ms |
| Maximum tracking error | 3.88 degrees |
| Maximum observed soft-limit overshoot | 0 degrees |
| Maximum penetration | 0 mm |
| Solver warnings | 0 |
| Frames exceeding 33.3 ms compute | 0 |

Browser rendering and perception are outside these timings. The viewer was
also open during this smoke check, so these timings are not a controlled
performance comparison with the earlier baseline. Zero overshoot/contact here
applies only to the tested trajectories from zero with the existing excluded
arm/head contact pair. The controller tests near limits do not establish
physical collision clearance there. Ubuntu validation remains outstanding.

To record a full fresh benchmark, including all 123 gestures, use:

```powershell
uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5
```

This produces 1,476 measured trials across four backends, plus warmups.
