# Repository Guidelines

## Project Structure & Module Organization

This repository contains two challenge briefs, robot assets, and a SWE simulation playground.

- `ML_route/README.md`: multimodal emotion challenge requirements.
- `SWE_route/starter_pack/CHALLENGE.md` and `SUBMISSION.md`: authoritative software challenge and deliverables. The adjacent `README.md` contains outdated details; prefer these two documents.
- `SWE_route/starter_pack/robot/`: five-DOF URDF, preview image, and `assets/lamp_shade.stl`. Preserve relative asset paths.
- `SWE_route/simulation/`: MuJoCo physics, browser viewer, JSON gestures, tests, and locked dependencies; `benchmarks/` holds reproducible comparison logs.
- `docs/research/`: local research notes.
- `parameter_golf/`: local reference trainers, data scripts, and plotting tools, separate from both challenges.
- `CLAUDE.md`: additional repository context and robot integration details.

Target SWE simulation and preserve unrelated local work.

## Build, Test, and Development Commands

From `SWE_route/simulation/`, run `uv sync --locked` to install dependencies, `uv run --locked python -m lamp_sim` to launch the viewer, and `uv run --locked pytest -q` to test. Python 3.12 is required. No formatter or linter is configured. From the repository root:

- `git status --short`: review modified and untracked files.
- `git diff --check`: check tracked changes for whitespace errors.
- `git diff -- <path>`: inspect a specific tracked change before committing.

Run `uv run --locked python -m lamp_sim.benchmark --repeats 3 --speed 0.5` for a new benchmark. Preserve historical logs and model fixtures. Document commands in the route's README.

## Coding Style & Naming Conventions

Use descriptive Markdown headings, concise instructions, and relative links. Python uses four-space indentation, `snake_case`, and type hints. Keep motion angles in radians internally. Preserve two-space indentation in URDF/XML and existing asset names. `.gitattributes` normalizes text to LF on commit. Do not add virtual environments, `.DS_Store`, or `__MACOSX` extraction artifacts.

## Testing Guidelines

Simulation tests use pytest in `tests/test_*.py`; no coverage threshold is set. Check motion limits, stopping, gestures, and model loading, then visually verify the browser controls and lighting. Verify documentation links and paths. Report observed latency and resource use for demonstrations. ML inference must remain local and within six billion total learned parameters; SWE deployment targets Ubuntu 24.04, four CPU cores, 8 GB RAM, and no discrete GPU.

## Commit & Pull Request Guidelines

History contains only `Initial commit` and `init`; no commit convention is established. Use short, imperative, descriptive subjects, for example `Document ML inference setup`. Keep commits focused. PRs should identify the route, explain changes, list validation and limitations, link relevant issues, and include screenshots or recordings for visual or interaction changes.
