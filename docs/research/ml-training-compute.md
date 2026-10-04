# Compute for MELD data preparation and fine-tuning

Researched 2026-10-01. Figures marked "estimate" are arithmetic, not measurements, and the arithmetic is shown.

## 1. Dataset size on disk

The raw archive `MELD.Raw.tar.gz` is 10,878,146,150 bytes (10.9 GB). I read that size from the HTTP headers of both official mirrors: the University of Michigan server (http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz, last modified 2018-11-03) and the Hugging Face dataset repo (https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz). Both links come from the official repo (https://github.com/declare-lab/MELD). A plain `wget` works with no login, and the Hugging Face copy is served from a CDN (content delivery network), so it is faster.

The archive is nested. Its first tar header shows `MELD.Raw/train.tar.gz` at 7,863,231,551 bytes, so dev, test, and the CSVs make up the other 3.0 GB. The clips are already-compressed MP4 and gzip shrank them by under 1% in my sample, so the extracted clips also total about 10.9 GB. Keeping the outer archive, the inner tarballs, and the clips at once peaks at about 33 GB (10.9 × 3, estimate). Streaming the download into `tar` and deleting each inner tarball after extraction keeps the peak near 22 GB.

The second release, `MELD.Features.Models.tar.gz`, is 922,694,594 bytes (0.92 GB, same mirrors). It holds 2018 baseline pickles and is not useful for a modern encoder pipeline.

Four clips probed with `ffprobe` were all 1280×720 H.264 at about 24 fps, with AAC audio at 48 kHz in 2 or 6 channels. The repo gives a mean utterance length of about 3.59 s. Some clips are badly cut: GitHub issue #20 (https://github.com/declare-lab/MELD/issues/20) reports test clips dia220_utt0 and dia38_utt4 running 4 and 5 minutes, and issue #9 (https://github.com/declare-lab/MELD/issues/9) reports text and video out of sync. Totals below that use 13,708 × 3.59 s ≈ 49,200 s (13.7 h) of audio are therefore lower bounds.

## 2. Preprocessing cost

I timed ffmpeg on four real MELD clips on a 2018 six-core Ryzen 5 2600X under Windows, one ffmpeg process per file. These are conservative baselines, and a Linux pod will be faster.

| Step | Measured rate | Full-dataset time (estimate) | Output size (estimate) |
|---|---|---|---|
| Audio to 16 kHz mono WAV | 0.12 s per clip | 13,708 × 0.12 s ≈ 27 min on one process, about 4 min across 8 processes | 49,200 s × 32 KB/s ≈ 1.6 GB (measured 31.3 KB/s) |
| Frames at 2 fps, resized to 224×224 JPEG | 0.17 s per clip | 13,708 × 0.17 s ≈ 39 min on one process, about 5 min across 8 processes | 49,200 s × 2 fps ≈ 98,400 frames × 7.1 KB ≈ 0.7 GB |
| Face detection with MTCNN on those frames | 20 frames/s at 720p on a P100 (https://github.com/timesler/facenet-pytorch README, undated, old GPU) | 98,400 / 20 ≈ 82 min on a P100, likely under an hour on a current GPU | Similar to the frame row |

Download and two layers of gunzip add a few minutes at datacenter bandwidth (estimate).

Feature pre-computation costs little GPU time. The timm benchmark files (https://github.com/huggingface/pytorch-image-models/tree/main/results, 2025 runs) report CLIP ViT-B/16 inference at 3,356 images/s on an RTX 4090 and 4,671 on an RTX PRO 6000 Max-Q, and CLIP ViT-L/14 at 785 images/s on the 4090. All 98,400 frames take about 30 s with ViT-B or 2 min with ViT-L (estimate), so CPU video decoding is the bottleneck.

Feature storage is the main disk variable, and these sizes are estimates. Pooled vectors, one per utterance, are tiny at 13,708 × 1,024 × 4 bytes = 56 MB. One WavLM-large layer at 50 frames/s in fp16 is 49,200 × 50 × 1,024 × 2 bytes ≈ 5.0 GB, and saving all 25 layers would be about 126 GB. CLIP ViT-B patch tokens for every frame would be 98,400 × 197 × 768 × 2 bytes ≈ 30 GB, against 0.15 GB for the class token alone.

## 3. GPU memory for fine-tuning

Hugging Face's memory guide (https://huggingface.co/docs/transformers/model_memory_anatomy, undated, accessed 2026-10-01) gives 18 bytes per parameter for full mixed-precision training with AdamW: 6 for weights, 8 for optimizer state, and 4 for gradients. Activations come on top and scale with batch and sequence length. For activations I used the per-layer formula from Korthikanti et al. (https://arxiv.org/abs/2205.05198, 2022), which is s·b·h·(34 + 5·a·s/h) bytes, where s is sequence length, b is batch, h is hidden size, and a is head count.

| Setup | VRAM | Basis |
|---|---|---|
| RoBERTa-large (355M), full fine-tune | Fits a 16 GB V100 at batch 4 to 16 with mixed precision | EmoBERTa appendix (https://arxiv.org/abs/2108.12009, 2021) |
| RoBERTa-large, batch 32, 128 tokens | About 11 GB | Estimate: 6.4 GB states + 24 layers × 184 MB activations |
| WavLM-base+ (95M), batch 32, 5 s clips, CNN frozen | About 6 GB | Estimate: 1.7 GB states + 12 × 329 MB |
| WavLM-large (316M), same settings | About 16 GB | Estimate: 5.7 GB states + 24 × 438 MB; Diatlova et al. trained this setup on one A100 (https://www.isca-archive.org/odyssey_2024/diatlova24_odyssey.pdf, 2024) |
| ViT-B/16 (86M) image training with AMP | Batch 192 fits in 24 GB on an RTX 3090 | timm train benchmark (2022 run) |
| 3B LLM, QLoRA / LoRA | 3.5 GB / 8 GB minimum | Unsloth requirements (https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements, undated, accessed 2026-10-01) |
| 3B (T0_3B), LoRA versus full | 14.4 GB versus 47.1 GB on an A100 80 GB | PEFT README (https://github.com/huggingface/peft, accessed 2026-10-01) |
| Any size, rule of thumb | LoRA ≈ 2 GB per billion parameters, 4-bit QLoRA ≈ 0.5 GB per billion, full ≈ 18 GB per billion | LLaMA-Factory table, marked estimated (https://github.com/hiyouga/LLaMA-Factory, accessed 2026-10-01) |
| Qwen3-4B, bf16 LoRA, 2,048 tokens, no gradient checkpointing | About 17 GB at batch 1, about 45 GB at batch 4 | Estimate below |

The Qwen3-4B row uses its config (4.02B parameters, 36 layers, hidden size 2,560, vocabulary 151,936). The bf16 weights take 8.0 GB. Activations take 34 × 2,048 × 2,560 bytes ≈ 178 MB per layer, or 6.4 GB over 36 layers, assuming fused attention removes the quadratic term. The fp32 logits and their gradient add 2 × 2,048 × 151,936 × 4 bytes ≈ 2.5 GB, so each sample costs about 9 GB. Gradient checkpointing (recomputing activations during the backward pass instead of storing them) or a shorter prompt cuts this sharply, and a ten-utterance MELD context is a few hundred tokens.

Published MELD systems show that 24 GB is enough. TelME (https://arxiv.org/abs/2401.12987, NAACL 2024) fine-tuned RoBERTa-large, data2vec-audio, and TimeSformer on one RTX 3090. DiaRelay (https://arxiv.org/abs/2608.22745, August 2026) trained Qwen3-4B and Qwen3-8B with 4-bit NF4 QLoRA at 2,048 tokens, micro-batch 1, and no gradient checkpointing on one RTX 3090. A GRPO-plus-LoRA ERC paper (https://aclanthology.org/2025.findings-emnlp.1059.pdf, EMNLP Findings 2025) used two 24 GB RTX 4090s.

## 4. Reported training times

No paper reports MELD epoch times on H100, L40S, or RTX PRO cards. The anchors below are on older or mid-range cards, and they show every component training in minutes per epoch.

SpeechBrain's IEMOCAP recipe (https://github.com/speechbrain/speechbrain/tree/develop/recipes/IEMOCAP, 2021) fine-tunes wav2vec2-base at 3 min 14 s per epoch on a V100 over roughly 7 h of audio. Scaled to MELD's roughly 10 h of training audio, that is about 4.6 min per epoch (estimate). EmoBERTa reports one five-epoch RoBERTa-large run taking about 45 minutes on a V100, or about 9 min per epoch, though the appendix does not say whether that was MELD or IEMOCAP. Li et al. (https://arxiv.org/abs/2511.13731, November 2025) report 30 to 50 minutes of total MELD training on an RTX 3090, excluding feature extraction.

For LLMs, MiSTER-E (https://arxiv.org/abs/2602.23300, February 2026) reports about 10 min per epoch for LLaMA-3.1-8B with rank-8 LoRA and 20 min per epoch for the SALMONN-7B speech LLM on a 48 GB RTX A6000, without naming which of its three datasets. DialogueLLM (https://arxiv.org/abs/2310.11374, 2023) trained a 7B LoRA model for 10 epochs over 24,304 utterances in 5 hours on a 40 GB A100, which scales to about 12 min for a MELD-sized epoch (estimate). For vision, timm's 391 images/s for ViT-B/16 training on an RTX 3090 means about 3 min per epoch over roughly 71,700 training frames (estimate).

## 5. Recommendation

The GPU should have 48 GB (L40S, RTX A6000, RTX 6000 Ada, or similar). 24 GB is the proven floor, because the three MELD systems cited in section 3, including DiaRelay's 4B and 8B QLoRA runs, all ran on 24 GB cards. 48 GB is comfortable because it fits bf16 LoRA on a 4B model at batch 4 without gradient checkpointing (about 45 GB, estimate), and it lets an audio encoder train jointly with the LLM without spending timebox hours on memory tuning. 16 GB covers the encoders and small QLoRA runs with no headroom, and 80 GB buys nothing this workload needs.

System RAM should be 64 GB, with 32 GB as the workable minimum. A 4B bf16 checkpoint passes through about 8 GB of host memory on load, video-decoding dataloader workers each hold decoded 720p frames, and frame-level audio features add about 5 GB (estimates). PEFT measured only 3 to 4 GB of host RAM for a plain LoRA run, so 64 GB is headroom.

The CPU should have 8 to 16 vCPUs. Preprocessing is ffmpeg decoding, which scales with cores, and my measured rates give about 10 minutes for audio plus frames across 8 processes (estimate). 16 vCPUs also keep video dataloaders from starving the GPU.

Disk should be 150 GB, with 100 GB workable if you stream-extract and keep only pooled features. The estimated budget is 33 GB for the extraction peak, 2.3 GB for WAVs and frames, and 5 to 35 GB for features. The model cache takes about 12 GB (Qwen3-4B bf16 at 8 GB plus RoBERTa-large, WavLM-large, and CLIP), plus 8 GB per extra LLM compared. A PyTorch and CUDA environment takes about 10 GB, and three full encoder checkpoints with optimizer state take about 13 GB (355M × 12 bytes each). That totals roughly 75 to 105 GB, or 85 to 115 GB with a second LLM.
