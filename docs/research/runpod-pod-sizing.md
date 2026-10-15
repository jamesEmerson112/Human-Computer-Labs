# Runpod pod sizing for the two routes

This file turns the six research notes in this folder into pod sizes. GPU and CPU prices and stock come from a live read of the Runpod catalog on 2026-10-01, using Secure Cloud on-demand prices unless a line says Community Cloud. Storage prices come from https://docs.runpod.io/pods/pricing. Stock levels change hourly, so re-check them before creating anything.

## Short answer

The SWE route needs no GPU at all. All three SWE notes reach that conclusion independently, and the target laptop has no GPU anyway. A CPU pod with 4 vCPUs and 8 GB of RAM mirrors the target exactly for $0.12 per hour, but it cannot test the camera, microphone or speaker, so it is optional.

The ML route needs a GPU for fine-tuning and for the demo. On 2026-10-14 the user chose to run both on a pod, with the demo playing MELD scenes from the pod's own disk. Fine-tuning needs 24 GB of GPU memory at minimum and 48 GB to be comfortable. In today's catalog the cheapest comfortable option is an A40 with 48 GB at $0.49 per hour, which also has the best stock of any GPU listed. That is less than your existing RTX PRO 4000 Blackwell pods cost ($0.57 per hour), and it has twice the memory.

## Scenario 1: an ML pod for data preparation and fine-tuning

The numbers below come from `ml-training-compute.md`, which cites its sources. Where that note marks a figure as an estimate, it is an estimate here too.

| Resource | Minimum | Comfortable | Why |
|---|---|---|---|
| GPU memory | 24 GB | 48 GB | Published MELD work fine-tuned 4B and 8B models with 4-bit QLoRA on 24 GB cards. Plain bf16 LoRA on a 4B model at batch 4 needs about 45 GB (estimate). |
| System RAM | 32 GB | 64 GB | Loading a 4B checkpoint passes about 8 GB through host memory, and video-decoding data loaders hold decoded frames (estimates). |
| vCPUs | 8 | 16 | Audio extraction and frame sampling with ffmpeg scale with cores, and about 10 minutes across 8 processes covers the dataset (estimate). |
| Disk | 100 GB | 150 GB | The 10.9 GB MELD archive peaks at about 33 GB during extraction. Model weights take about 12 GB, a PyTorch environment about 10 GB, and saved features and checkpoints 5 to 50 GB. The total is roughly 75 to 115 GB (estimate). |

The GPU options that fit, from the live catalog, are below. "Low" stock means a pod may not be available in every data center at the moment you ask.

| GPU | Memory | Secure price per hour | Community price per hour | Stock |
|---|---:|---:|---:|---|
| RTX A5000 | 24 GB | $0.27 | $0.16 | Low |
| L4 | 24 GB | $0.49 | $0.44 | Low |
| RTX 3090 | 24 GB | $0.50 | $0.22 | Low |
| RTX PRO 4000 Blackwell (your existing pods) | 24 GB | $0.57 | — | Low, EU-RO-1 only |
| **A40** | **48 GB** | **$0.49** | **$0.35** | **Medium** |
| RTX A6000 | 48 GB | $0.53 | $0.33 | Low |
| RTX 6000 Ada | 48 GB | $0.84 | $0.74 | Low |
| L40S | 48 GB | $1.09 | $0.79 | Low |

An 80 GB card such as the A100 ($1.59 per hour) buys nothing this workload needs. As a cost example, 20 GPU hours on an A40 comes to $9.80. A 100 GB volume disk adds $0.10 per GB per month while the pod runs and $0.20 per GB per month while it is stopped, billed per second. That is $10 a month running or $20 a month stopped.

The pod's RAM and vCPU count depend on the GPU type and host, and the catalog does not list them. Check them on the deploy screen against the table above.

## Scenario 2: an ML pod as the demo machine

The user chose this setup on 2026-10-14. The numbers come from `ml-models.md` and `ml-realtime-inference.md`.

Every example inference path in `ml-models.md` fits under 12 GB of weights at fp16. A full path near the 6B cap needs about 13.5 to 15 GB of GPU memory at bf16 once runtime overhead is included, or 5 to 7 GB with a 4-bit generator (estimates). Any 24 GB card therefore covers it, including your existing pods and the RTX A5000 at $0.27 per hour. A 16 GB card is tight. The demo also needs about 32 GB of RAM, 8 vCPUs and 25 GB of disk (estimates).

Latency is the reason to use a GPU at all for the demo. A 3B model at 4-bit decodes at 335 tokens per second on an RTX 4090 but 12 to 24 tokens per second on CPUs, and its first token arrives in about 40 ms on the GPU versus several seconds on a CPU (estimate).

The demo reads MELD scenes from the pod's own disk, so no audio or video streams over the network. Live input from a laptop camera or microphone would have to stream to the pod instead. Runpod pods expose only TCP and HTTP ports, not UDP (`swe-deployment.md`), which rules out the usual WebRTC media path unless it is tunneled over TCP.

## Scenario 3: the SWE route

No component needs a GPU. Perception and speech fit the 4-core, 8 GB target, using about 3 to 4 GB of RAM (estimate, `swe-perception-speech.md`). The simulation options need under one core and a few hundred megabytes (`swe-simulation.md`). The language and decision layer is better served by a hosted API than by a rented GPU, at roughly 0.6 to 1 second to the first token.

A CPU pod can mirror the target for install and resource testing. The Compute-Optimized `cpu3c` type gives 2 GB of RAM per vCPU, so 4 vCPUs come with exactly 8 GB for $0.12 per hour, and stock is high in every listed data center. The `cpu5c` type gives the same shape for $0.14 per hour. Such a pod cannot test the camera, microphone, speaker or echo cancellation. Its API offers no privileged mode or device access, so virtual devices such as v4l2loopback are unavailable, and nested Docker does not work (`swe-deployment.md`).

Two free alternatives cover most of the same ground. A GitHub Actions `ubuntu-24.04` runner gives 4 CPUs and 16 GB to public repositories. On any Linux machine, `systemd-run --user -p CPUQuota=400% -p MemoryMax=8G` with `taskset -c 0-3` emulates the target's limits. Only the real laptop, or Ubuntu booted from a USB stick, tests the devices.

## Your existing pods

The two stopped pods named `disciplinary_olive_hyena` and `disciplinary_olive_hyena-migration` each have one RTX PRO 4000 Blackwell with 24 GB of GPU memory, 31 GB of RAM, 12 vCPUs, a 20 GB container disk and a 20 GB volume at `/workspace`. On paper they cover the 24 GB tier of scenario 1 and all of scenario 2. Four things stand in the way.

The first is disk. A 20 GB volume cannot hold MELD's 33 GB extraction peak, so the volume would need to grow to about 100 GB.

The second is software. Both pods run the `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` image. PyTorch added Blackwell support in version 2.7 with CUDA 12.8 builds, so PyTorch 2.4 is unlikely to run on these GPUs. They would need a newer image.

The third is price and stock. These GPUs are listed only in EU-RO-1 with low stock, and at $0.57 per hour they cost more than an A40 with twice the memory.

The fourth is storage cost while idle. Each pod's 20 GB volume costs $0.20 per GB per month while stopped, which is $4 a month per pod and $8 a month for both.

The two CPU pods stopped on 2026-10-01 had no volume disk. Runpod's pricing page says container disk is not charged while a pod is stopped, and that it is erased when the pod stops.
