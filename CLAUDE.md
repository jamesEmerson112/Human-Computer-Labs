# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository holds two take-home challenge briefs and a SWE lamp playground in `SWE_route/simulation/`. Current implementation work targets SWE simulation. From that directory, run `uv sync --locked`, `uv run --locked python -m lamp_sim`, and `uv run --locked pytest -q` with Python 3.12. MuJoCo physics is the default; the viser browser retains original kinematic playback as a comparison mode. The library contains 123 JSON gestures: nine original clips and 114 entries from all three rings of the supplied emotion wheel, grouped into six family selectors. See `SWE_route/simulation/gestures/emotions.md` for the complete catalog. A headless benchmark logs requested/actual motion, timing, memory, and contacts; the preserved October 1 baseline covers only the original nine clips. See `SWE_route/simulation/benchmarks/README.md` for evidence and assumptions. No formatter or linter is configured. See the simulation README for controls, limitations, and Ubuntu instructions. `parameter_golf/` is reference code copied unchanged from the user's earlier `DL-Team-Proposal` repo (small-language-model trainers, data download scripts, and plotting tools). It is not part of either challenge, and several of its trainers import the Unix-only `fcntl` module, so they do not run natively on Windows.

## The two routes

`ML_route/README.md` is a real-time multimodal emotion challenge built on the MELD dataset. The user picks one track, either text plus audio or text plus vision, and produces a MELD emotion category plus short grounded response text from input that arrives over time. The whole local inference path may hold at most 6 billion total learned parameters, counting every encoder, backbone, adapter, projector, head, and auxiliary generator. Remote inference does not satisfy the core task. The timebox is two to three days, and the submission must report hardware and observed resource use.

`SWE_route/starter_pack/` is a live character challenge around a simulated five-DOF desk lamp. The laptop camera, microphone, and speaker act as the robot's senses and voice, and the URDF provides the simulated body for motion and light. Cloud APIs are allowed for perception and speech, but the submission must state what data is sent, where decisions are made, and why. The timebox is six to eight hours. The deployment target is a clean Ubuntu 24.04 laptop with four CPU cores, 8 GB of RAM, no discrete GPU, and standard Linux audio. Development may happen on another operating system, but the deployment story must still address that Ubuntu target.

The two routes disagree on remote inference, so check which route applies before proposing a cloud model.

## ML route: compute, data, and models

The ML brief sets no hardware limit. It asks only that the submission report the hardware used and the resources observed. The user's own PC has a GTX 1070 Ti (8 GB, Pascal, no bf16), so on 2026-10-14 the user chose Runpod for all ML work. Training runs on a pod, and the demo also runs entirely on one pod. MELD scenes are fed from disk at real-time speed, so no input or output crosses the network during inference. The pod is the local machine for this project. Every model in the inference path runs on it, and none is called over a network, so the demo is not remote inference. Do not raise the pod as a remote-inference risk. The submission reports the pod's GPU as its hardware. The planned GPU is an A40 with 48 GB. Never create, start, or stop a pod unless the user asks in that conversation. Pod sizing and prices are in `docs/research/runpod-pod-sizing.md`.

MELD is unpacked in `ML_route/data/MELD.Raw/`, which is gitignored. It came from https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz. The label files are `train_sent_emo.csv`, `dev_sent_emo.csv`, and `test_sent_emo.csv`. The clips are `train_splits/` (9,989 mp4 files), `dev_splits_complete/` (1,112), and `output_repeated_splits_test/` (2,747). The test folder also holds 2,060 macOS `._*` stub files, which a loader must skip. MELD has no separate audio files. The sound is an AAC audio stream inside each mp4. Of three clips checked on 2026-10-14, all were 48 kHz, two were stereo and one was 6-channel. The audio therefore needs extracting with ffmpeg and downmixing to mono.

MELD labels each line with one of seven emotions: neutral, joy, surprise, anger, sadness, disgust, and fear. The user's reference emotion wheel is `docs/images/emotion-wheel.png`. Its seven core emotions are Happy, Sad, Angry, Fearful, Surprised, Disgusted, and Bad. The first six map to MELD's joy, sadness, anger, fear, surprise, and disgust. The wheel's Bad and MELD's neutral have no match. This wheel differs from the one transcribed in `SWE_route/simulation/gestures/emotions.md`, which has six core emotions and different outer rings.

The candidate models live in `docs/research/ml-model-catalog.json`, which also feeds the explainer page `docs/explainers/meld-head-training.html`. On 2026-10-14 the user removed every model that a newer successor has replaced, usually a newer version from the same maker. CLIP counts as replaced by SigLIP 2. The catalog's `removed` list names each one and its replacement, so do not suggest those models again.

The user wants to train in Rust, which is their learning track. `ML_route/hello_world/` is the user's first Cargo crate. Run `cargo run` from that folder. It prints a greeting, then one training dialogue as a table, read with the `csv` crate, and the dialogue defaults to ID 1036. Candle 0.11 has Whisper, ModernBERT, DeBERTa-v2, XLM-RoBERTa, Granite, Qwen3, and Gemma 3 modules, but no WavLM, HuBERT, wav2vec 2.0, or emotion2vec. A voice encoder in Rust therefore needs a port, ONNX Runtime, or features extracted outside Rust.

## SWE route: which documents are authoritative

`CHALLENGE.md` and `SUBMISSION.md` are the current specification. `README.md` in the same folder is a stale earlier draft. It refers to a six-DOF URDF and a Google Drive link, and it lists only four demonstration moments. `CHALLENGE.md` adds a fifth moment, goal-directed action, in which a spoken goal about the live scene drives a sequence of lamp actions and the scene is observed again before the goal is complete. The shipped URDF is five-DOF, which confirms that `CHALLENGE.md` is current.

`SUBMISSION.md` defines the deliverables and the scoring rubric. The deliverables are source code with dependency declarations, Ubuntu setup and run instructions, and a technical note of at most two pages. The note must contain one architecture and data-flow diagram, must cover protocol, model-to-action, simulation, deployment, and key design choices, and must report engagement reliability, response latency, and CPU and memory measurements. The rubric weights architecture and protocol at 25%, character integration at 20%, perception, action, and memory at 20%, deployment at 15%, physical reasoning at 10%, and evidence and communication at 10%.

## SWE route: URDF facts that take a careful read

The file is `SWE_route/starter_pack/robot/dummy_lamp_5dof.urdf`. The actuated chain from base to head is `base_yaw_joint` (z axis), `shoulder_pitch_joint` (y), `elbow_pitch_joint` (y), `neck_yaw_joint` (z), and `head_pitch_joint` (y). Each joint has a hard `<limit>` and a tighter `<safety_controller>` soft range. Control code should clamp to the soft range.

The URDF deliberately contains no fixed world joint, no simulator plugins, no transmissions, and no controllers. Whatever simulator or visualizer is chosen must anchor `base_link` itself. The mesh path `assets/lamp_shade.stl` is relative to the URDF's own directory, so loaders must resolve it from there.

The origin of `head_pitch_joint` carries a 180° yaw. At zero configuration, the lamp head therefore faces the base frame's −x direction, opposite to the direction in which positive shoulder and elbow pitch lean the arm. Positive head pitch tilts the shade downward. Anyone computing where the lamp is looking, or mapping a target in the scene to joint angles, needs to account for this flip.

`light_emitter_link`, `camera_link`, and `speaker_link` are semantic frames with no behavior attached. The software adapter decides what they do. `camera_link` rides on the lamp head, but the challenge says the laptop camera does the seeing, so moving the simulated head does not move the real camera view. That mismatch shapes how the "observe the scene again" step can work, and it is the kind of physical-reasoning point the rubric scores.

## Repository hygiene

`SWE_route/__MACOSX/` and the `.DS_Store` files are macOS zip-extraction artifacts, not challenge content. The root `.gitignore` excludes them, along with Python caches, `.env` files, downloaded datasets, and `*.pt` weights, so a trained head that belongs in the submission needs `git add -f`. `.gitattributes` sets `* text=auto`, so line endings are normalized to LF on commit.
