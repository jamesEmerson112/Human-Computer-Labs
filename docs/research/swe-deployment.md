# SWE route: packaging, devices, measurement, and test environments

Researched 2026-10-01. Figures marked "estimate" are my own judgement, not taken from a source.

## Bottom line

This route does not need a Runpod machine. A 4-vCPU, 8 GB CPU pod can confirm that the install works from scratch and give a rough CPU and memory envelope. It cannot exercise the camera, microphone, speaker, or acoustic echo, which the engagement-reliability and latency numbers depend on. Honest measurements have to come from real Ubuntu hardware with cgroup limits (kernel resource caps) applied, and free alternatives cover the install test as well as a pod would.

## 1. Packaging for a clean Ubuntu 24.04 laptop

Ubuntu 24.04 ships Python 3.12 and marks the system interpreter as externally managed under [PEP 668](https://peps.python.org/pep-0668/), so a bare `pip install` is refused and a virtual environment is required. The simplest reproducible path is uv, whose `uv.lock` "is portable across platforms" ([uv docs](https://docs.astral.sh/uv/concepts/resolution/)), so a lock made on Windows also pins the Linux wheels. `uv sync --locked` fails rather than re-resolving when the lock is stale ([uv sync docs](https://docs.astral.sh/uv/concepts/projects/sync/)). The only apt packages needed are `libportaudio2` and, for camera diagnosis, `v4l-utils`.

The heavy dependencies set the disk footprint. On PyPI (checked 2026-10-01), the x86_64 wheel for `opencv-python-headless` 5.0.0.93 is 56.6 MB, `onnxruntime` 1.30.0 is 23.6 MB, and `mediapipe` 1.0.1 is 37.9 MB. Wheels unpack to roughly two to three times their download size (estimate). The `python:3.12-slim` Docker base image is 43.2 MB compressed ([Docker Hub](https://hub.docker.com/_/python)).

| Option | Setup on a clean target | Reproducibility | Disk | Device access |
|---|---|---|---|---|
| uv and a venv | 3 to 5 minutes (estimate) | Lockfile pins Python packages, not apt | 0.5 to 1 GB (estimate) | Native |
| Docker or Podman | About 10 minutes with engine install (estimate) | Image digest pins everything | 0.6 to 1.5 GB (estimate) | Needs `--device` and socket mounts |
| Snap | Hours to author (estimate) | Strong | Similar to Docker | `audio-record` and `camera` not auto-connected ([Snap docs](https://snapcraft.io/docs/reference/interfaces/audio-record-interface/)) |
| AppImage | Hours to author (estimate) | Moderate | Similar | Native, but bundling Python is fragile |

A container shares the host kernel and is capped by the same cgroups a native process would use, so its runtime CPU and memory overhead is negligible; the real costs are image disk and the device and socket mounts (estimate). For a six-to-eight-hour timebox, a native uv install with a short `setup.sh` is the primary path, and a Dockerfile is a useful secondary path for reproducible measurement. To autostart the character, use a systemd user unit, because it runs in the login session and inherits `XDG_RUNTIME_DIR`, the per-user directory that holds the PipeWire socket. A system service would not see that socket.

## 2. Linux audio on Ubuntu 24.04

Ubuntu 24.04 ships PipeWire 1.0.5 ([Launchpad](https://launchpad.net/ubuntu/noble/+source/pipewire)), and PulseAudio is not installed by default. The `pipewire-audio` metapackage pulls in `pipewire-pulse`, which serves the PulseAudio protocol on `$XDG_RUNTIME_DIR/pulse/native`, and `pipewire-alsa`, which routes ALSA clients into PipeWire ([Debian wiki](https://wiki.debian.org/PipeWire)). Check the target with `pactl info`, which should name PipeWire as the server, and with `aplay -L`, which should list `pipewire`, `pulse`, or `default` devices.

Both Python audio libraries sit on PortAudio. sounddevice does not bundle PortAudio on Linux, so `libportaudio2` must come from apt ([sounddevice docs](https://python-sounddevice.readthedocs.io/en/latest/installation.html)). Ubuntu 24.04 packages PortAudio 19.6.0 ([Launchpad](https://launchpad.net/ubuntu/noble/+source/portaudio19)), which has no PulseAudio or PipeWire backend; only unreleased PortAudio code adds one ([sounddevice #609](https://github.com/spatialaudio/python-sounddevice/issues/609), November 2025). sounddevice therefore talks ALSA, and the app should open the ALSA `pipewire` or `default` device rather than a raw `hw:` device that PipeWire may already hold. [PyAudio 0.2.14](https://pypi.org/project/PyAudio/#files) publishes only Windows wheels, so on Linux it compiles from source, which is one more reason to prefer sounddevice.

Inside a container, mount the host's PulseAudio socket, point `PULSE_SERVER` at it, and run as the host user's UID ([x11docker wiki](https://github.com/mviereck/x11docker/wiki/Container-sound:-ALSA-or-Pulseaudio), edited December 2025). The image also needs the ALSA pulse plugin (`libasound2-plugins`) for PortAudio to reach the socket. Passing `--device /dev/snd` bypasses the sound server and breaks when the host's PipeWire holds the card.

There are three ways to stop the character hearing its own voice. PipeWire's echo-cancel module uses WebRTC's AEC (acoustic echo canceller) by default and is documented for this release in the [noble man page](https://manpages.ubuntu.com/manpages/noble/man7/libpipewire-module-echo-cancel.7.html). It creates a virtual source and sink, and `pactl set-default-source` and `set-default-sink` route the ALSA `default` device through them. LiveKit's `AudioProcessingModule` does the same in-process, taking the speaker signal through `process_reverse_stream` in 10 ms frames ([LiveKit API](https://docs.livekit.io/reference/python/livekit/rtc/apm.html)). Half-duplex gating ignores the microphone while speaking plus a short tail. Gating is the most robust choice in the timebox, but it gives up barge-in, meaning the user cannot interrupt. WebRTC AEC should cost a few percent of one core (estimate).

## 3. Camera through V4L2

V4L2 (Video4Linux2) exposes the camera as `/dev/video0`. On Ubuntu desktops the logged-in user normally gets access through udev's `uaccess` tag, and otherwise through the `video` group. UVC cameras often create a second metadata node, so `v4l2-ctl --list-devices` and `--list-formats-ext` should drive device selection rather than a hard-coded path. In a container, pass `--device /dev/video0` and add the host's `video` group ID.

For low CPU, open the camera with `cv2.CAP_V4L2` explicitly, request 640x480 at 15 fps, and log the format the driver actually negotiated, because V4L2 can silently change it ([wendylabs PR #109](https://github.com/wendylabsinc/templates/pull/109), September 2026). That PR notes that raw YUYV at 720p30 needs about 440 Mbit/s, too much for USB 2.0. At 640x480 and 15 fps, YUYV needs about 74 Mbit/s, which fits and avoids JPEG decoding on the CPU (my arithmetic). Run capture in its own thread that keeps only the newest frame, run detection at 5 to 10 Hz on a downscaled copy, and cap OpenCV's internal threads with `cv2.setNumThreads(2)` so it does not claim every core.

## 4. Measuring CPU, memory, and latency honestly

psutil's `Process.cpu_percent()` returns a meaningless 0.0 on its first call and can exceed 100% for multithreaded processes ([psutil API](https://psutil.io/api/)), so state whether a figure is percent of one core or of four. For memory, report PSS (proportional set size, which splits shared pages among their users) or USS (memory unique to the process) alongside RSS. `psutil.cpu_count()` reports logical CPUs, while `len(Process().cpu_affinity())` gives the usable count. For time series, `pidstat -u -r -h -t -p <pid> 1` from sysstat 12.6.1 logs per-thread CPU and memory each second ([man page](https://manpages.ubuntu.com/manpages/noble/man1/pidstat.1.html)). `/usr/bin/time -v` reports peak "Maximum resident set size" for the whole run, and the bash builtin `time` lacks that option ([man page](https://manpages.ubuntu.com/manpages/noble/man1/time.1.html)).

To emulate the target on a larger Linux machine, combine `taskset -c 0-3` with `systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=8G -p MemorySwapMax=0`. Since systemd 252 the cpu controller is delegated to user managers by default ([v252 release notes](https://github.com/systemd/systemd/releases/tag/v252-rc1)), and Ubuntu 24.04 ships systemd 255 ([Launchpad](https://launchpad.net/ubuntu/noble/+source/systemd)). The cpuset controller is not delegated to users by default ([rootlesscontaine.rs](https://rootlesscontaine.rs/getting-started/common/cgroup2/)), so `taskset` sets the core count. A 400% quota alone would still spread work over every host core. With Docker, use `--cpus=4 --cpuset-cpus=0-3 --memory=8g --memory-swap=8g`. The Docker docs warn that `free` inside a container reports the host's swap ([Docker docs](https://docs.docker.com/engine/containers/resource_constraints/)), and total memory is likely host-wide too, so read `/sys/fs/cgroup/memory.max` and `cpu.max` instead. Remember that GNOME and PipeWire on the real laptop already use part of the 8 GB, likely 1.5 to 2.5 GB (estimate).

For latency, stamp each stage with `time.monotonic_ns()`, from end of user speech through final transcript, first model token, and first synthesized audio to first sample at the speaker. sounddevice's callback reports that last moment as `outputBufferDacTime` ([sounddevice API](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)). Report the median and 95th percentile over at least 20 scripted trials, and check end to end by recording the session with a phone and measuring the silent gap.

## 5. Can a Runpod CPU pod stand in for the target?

Runpod's CPU-type table, written for its Flash serverless product but using the same flavor names the pod API accepts, lists sizes that match the target exactly ([Runpod CPU types](https://docs.runpod.io/flash/configuration/cpu-types)).

| Flavor | vCPU | RAM |
|---|---|---|
| `cpu5c` (5th gen, compute-optimized) | 4 | 8 GB |
| `cpu3c` (3rd gen, compute-optimized) | 4 | 8 GB |
| `cpu3g` (3rd gen, general purpose) | 4 | 16 GB |

The pod API takes a flavor and a power-of-two `vcpuCount`, and its container disk defaults to 50 GB, wiped on restart ([create-pod API](https://docs.runpod.io/api-reference/pods/POST/pods)). A 20 to 30 GB disk is enough (estimate). The docs list no CPU pod prices; container disk costs $0.10 per GB per month ([pricing](https://docs.runpod.io/pods/pricing)).

A pod is a Docker container, not a VM. Runpod moved CPU pods from Kata to the Docker runtime, and "Docker-in-Docker ... is no longer available" ([Runpod blog](https://www.runpod.io/blog/enhanced-cpu-pods-docker-network), updated September 2026). The pod docs say you cannot run your own Docker or Compose, and that pods support only TCP and HTTP, not UDP ([pods overview](https://docs.runpod.io/pods/overview)). The pod API has no field for privileged mode, extra capabilities, or devices, so the public docs offer no way to load kernel modules. Treat v4l2loopback (a virtual camera) and snd-aloop (a virtual sound card) as unavailable. A pod cannot run your Dockerfile with `--device` flags or a WebRTC-over-UDP link to a cloud speech API, though WebSocket works. Because a pod starts as a container image rather than a booted system, it normally has no systemd user session or desktop PipeWire either. Pod vCPUs are server hyperthreads without laptop thermal throttling, so CPU figures are indicative only (estimate). Check `/sys/fs/cgroup/cpu.max` on the pod, because `os.cpu_count()` may report the host's count (unverified on Runpod).

The workarounds are partial. Replay recorded video through `cv2.VideoCapture(path)` throttled to real time, and record the clips as MJPEG or raw, because H.264 decoding adds CPU that a YUYV camera would not. For audio, pipewire-pulse supports `module-null-sink` and `module-pipe-source`, which turns a named pipe into a fake microphone in user space ([pipewire-pulse modules](https://docs.pipewire.org/page_man_pipewire-pulse-modules_7.html)). On a pod the PipeWire daemons would have to be started by hand, which I have not verified. None of this tests echo, room acoustics, or driver behavior.

Cheaper stand-ins exist. A GitHub Actions `ubuntu-24.04` runner gives 4 CPUs, 16 GB RAM, and 14 GB SSD for public repositories, or 2 CPUs and 8 GB for private ones, at no cost ([GitHub docs](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)). Its image carries many preinstalled tools, so it is less clean than a fresh laptop (estimate). The only option that tests real devices is booting Ubuntu 24.04 from a USB stick on the development laptop and applying the limits above (suggestion).

## 6. Developing on Windows 11

Develop the application natively on Windows behind a device abstraction, since sounddevice bundles PortAudio there, and use WSL2 for the Linux install path. Setting `processors=4`, `memory=8GB`, and `swap=0` in `.wslconfig` sizes the WSL VM like the target ([Microsoft docs](https://learn.microsoft.com/en-us/windows/wsl/wsl-config), updated September 2026). WSLg provides microphone and speaker through its own PulseAudio server ([WSLg README](https://github.com/microsoft/wslg)), so WSL tests PulseAudio rather than PipeWire. An open report describes crackling from continuous underruns on Ubuntu 24.04 under WSL 2.6.3 ([wslg #1429](https://github.com/microsoft/wslg/issues/1429), March 2026).

Cameras are the bigger pitfall. usbipd-win can attach USB devices to WSL ([Microsoft docs](https://learn.microsoft.com/en-us/windows/wsl/connect-usb), updated June 2026), but the default WSL kernel has no UVC webcam driver ([WSL #8655](https://github.com/microsoft/WSL/issues/8655), open since July 2022). Even with a custom kernel, one user's stream failed with USB/IP connection resets ([InternLM #487](https://github.com/InternLM/InternLM-XComposer/issues/487), May 2025). Run `modinfo uvcvideo` after `wsl --update` before relying on it.
