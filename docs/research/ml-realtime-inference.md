# Real-time local inference path: sizing research

Researched on 2026-10-01 for the ML route (MELD emotion prototype, 6B total-parameter cap, local inference only). Each figure names the hardware it was measured on and links its source. Figures I derived rather than read from a source are marked "estimate".

## 1. Speech recognition and voice activity detection

Voice activity detection (VAD, deciding which audio frames contain speech) costs almost nothing. Silero VAD reads one 32 ms window in under 1 ms on one thread of an AMD Threadripper 3960X, and the model file is about 2 MB ([README](https://github.com/snakers4/silero-vad), [metrics](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics)). Smart Turn v3, an end-of-turn classifier with about 8M parameters, runs in about 10 ms on some CPUs and about 65 ms on a standard Pipecat Cloud instance ([repo](https://github.com/pipecat-ai/smart-turn)).

Whisper works on fixed 30-second windows and caches nothing between calls, so its latency after the user stops talking is the whole transcription time. Streaming models do most of the work while speech is still arriving. The table lists latency from end of speech to final transcript.

| Model | Params | Latency after speech ends | Hardware | Source |
|---|---|---|---|---|
| Moonshine Tiny Streaming | 34M | 18 ms; 69 ms | MacBook Pro; unnamed Linux x86 CPU | [Moonshine docs](https://moonshine-voice.readthedocs.io/en/latest/moonshine-vs-whisper/) |
| Moonshine Medium Streaming | 245M | 59 ms; 269 ms; 802 ms | MacBook Pro; Linux x86; Raspberry Pi 5 | same |
| Whisper Small | 244M | 1,940 ms; 3,425 ms | MacBook Pro; Linux x86 | same |
| Whisper Large v3 | 1.5B | 11,286 ms; 16,919 ms | MacBook Pro; Linux x86 | same |
| Nemotron Speech Streaming EN 0.6B | 600M | 24 ms median | Pipecat pipeline on Modal, DGX Spark and RTX 5090 | [NVIDIA blog, Jan 2026](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents) |
| Whisper large-v2 (Whisper-Streaming) | 1.55B | 3.1 s average | NVIDIA A40 | [Macháček et al., 2023](https://arxiv.org/abs/2307.14743) |

The Moonshine figures come from its current docs, and the model family is described in the Moonshine v2 paper (Feb 2026, [arXiv 2602.12241](https://arxiv.org/abs/2602.12241)). The docs say the Linux x86 and Pi columns predate a build fix and read pessimistically. Nemotron Speech Streaming accepts 80, 160, 560 or 1,120 ms chunks, and its average WER improves from 8.43% at 80 ms to 6.93% at 1.12 s. It runs in NeMo and was tested on V100, A100, A6000 and DGX Spark ([model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)). Parakeet TDT 0.6B v3 has 600M parameters, needs at least 2 GB of RAM to load, and supports chunked streaming ([card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)).

For Whisper memory, faster-whisper with large-v2 on an RTX 3070 Ti 8 GB used 4,525 MB of VRAM at fp16 and 2,926 MB at int8 ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)). Estimate: a 0.6B streaming ASR model at fp16 needs about 1.2 GB for weights and 2 to 3 GB of VRAM with runtime overhead.

## 2. Small language model runtimes

Parameter counts for the language and multimodal models in this file come from the safetensors totals that the Hugging Face model API returns (`https://huggingface.co/api/models/<model-id>`). They are 2.03B for Qwen3-1.7B, 3.21B for Llama-3.2-3B, 4.02B for Qwen3-4B, and 4.30B for Gemma-3-4B-it (vision encoder included).

Qwen's official benchmark, published with Qwen3 in 2025, ran on an NVIDIA H20 96 GB at batch 1 with Hugging Face Transformers ([Qwen speed benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)). With a 1-token prompt, Qwen3-4B used 7,973 MB in BF16 at 45.9 tokens/s and 2,915 MB with AWQ 4-bit weights at 51.6 tokens/s. Qwen3-1.7B used 3,412 MB in BF16. Under SGLang, a dedicated serving engine, Qwen3-4B decoded at 133 tokens/s in BF16 and 200 tokens/s in AWQ 4-bit on the same GPU.

llama.cpp numbers for Llama 3.2 3B at Q4_K_M (a 1.87 GiB file) come from [geerlingguy/ai-benchmarks](https://github.com/geerlingguy/ai-benchmarks). Prompt processing ("pp512", tokens/s while ingesting a 512-token prompt) was 10,640 on an RTX 4090 and 3,409 on an RTX 3060 12 GB (Dec 2025, [#43](https://github.com/geerlingguy/ai-benchmarks/issues/43), [#40](https://github.com/geerlingguy/ai-benchmarks/issues/40)). Those runs used llama.cpp's Vulkan backend with the GPU attached to a Raspberry Pi CM5 host, so a CUDA build on a desktop host should be at least as fast. Ollama decode on an Intel 265K desktop host was 335 tokens/s with the RTX 4090 and 123 tokens/s with the RTX 3060. A CPU-only Radxa Dragon Q8B Arm board with 8 threads reached pp512 72 and decode 12.3 tokens/s (Sep 2026, [#54](https://github.com/geerlingguy/ai-benchmarks/issues/54)), and a Framework 13 with a Ryzen AI 5 340 decoded at 23.8 tokens/s on CPU (Nov 2025, [#33](https://github.com/geerlingguy/ai-benchmarks/issues/33)). The [llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013) (opened Aug 2025) for a larger Llama 2 7B Q4_0 gives decode rates of 186 tokens/s on an RTX 4090, 152 on an L40, 130 on an RTX A5000, 84 on an RTX A4000 and 46 on a Tesla T4.

Time to first token (TTFT) is roughly prompt length divided by prompt-processing speed, plus one decode step. Estimate: a 400-token prompt costs about 40 ms on an RTX 4090, 120 ms on an RTX 3060, and 5.5 s on the Arm CPU above. llama.cpp can reuse the cached prefix of an unchanged system prompt, so only new transcript tokens need processing.

vLLM pre-allocates 92% of GPU memory by default (`gpu_memory_utilization`, [cache.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)), so that setting must be lowered when ASR and vision models share the GPU. llama.cpp and Ollama allocate only weights plus the KV cache, which is the stored attention state for past tokens. From the Qwen3-4B config (36 layers, 8 KV heads, head dimension 128), the KV cache costs 144 KiB per token in BF16, so a 4,096-token context adds about 0.6 GB (estimate).

## 3. Streaming pipeline structure and latency budgets

In human conversation, the gap between turns most often falls within 200 ms, across 10 languages ([Stivers et al., PNAS 2009](https://www.pnas.org/doi/10.1073/pnas.0903616106)). Preparing even one word takes over 600 ms, so listeners plan their reply before the turn ends ([Levinson & Torreira, 2015](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00731/full)). Spoken dialogue systems typically wait 700 to 1,000 ms of silence before taking the turn, which makes them feel slow ([Skantze, 2021](https://www.sciencedirect.com/science/article/pii/S088523082030111X)). Moshi, a 7B full-duplex model, reaches 200 ms practical latency on an L4 GPU using 80 ms frames ([repo](https://github.com/kyutai-labs/moshi)). NVIDIA reports a fully local cascaded voice agent finishing under 900 ms from voice in to voice out (blog above).

MELD utterances average 3.59 s and about 8 words, and each carries one label ([MELD](https://github.com/declare-lab/MELD)). The authoritative label should therefore come at utterance end, and anything earlier should be flagged provisional. The following structure is my recommendation, not a sourced result. Silero runs on 32 ms windows. ASR emits partial transcripts every 0.5 s, which is Moonshine's default interval, or uses 160 to 560 ms Nemotron chunks. An audio emotion encoder refreshes a provisional label every 0.5 to 1 s. At end of turn (200 to 300 ms of silence, confirmed by Smart Turn), the system fuses the final transcript with the whole-utterance audio, emits the final label, and starts the response text. Estimate: speech end to final label takes 0.4 to 0.6 s on a GPU, with the first response token about 0.1 s later. Getting closer to human timing needs turn-end prediction, which cut average latency by 505 ms in one 2026 study ([arXiv 2606.13450](https://arxiv.org/abs/2606.13450)).

## 4. Face and vision processing per frame

The YuNet face detector took 0.69 ms per frame at 160×120 on an Intel i7-12700K CPU, and a MobileFaceNet facial-expression model took 1.79 ms at 112×112 on the same CPU (OpenCV 4.10, 2024, [OpenCV Zoo benchmark](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)). A DINOv3 ViT-B/16 encoder at 256×256 took 2.18 ms with 362 MB peak memory on an RTX 4090 (BF16, batch 1), against 104 ms on a 16-thread Ryzen 9 7950X. ViT-S/16 took 2.16 ms and 20.2 ms on the same two machines (Mar 2026, [CanViT, arXiv 2603.22570, Appendix F](https://arxiv.org/html/2603.22570)). VLM-scale encoders cost far more. On an M1 Max, ViT-L/14 at 336 px took 127 ms and SigLIP-SO400M took 272 ms, and their 576 visual tokens also lengthen LLM prefill ([FastVLM, CVPR 2025](https://arxiv.org/html/2412.13303)).

Estimate: face detection plus a ViT-B encoder costs 3 to 5 ms per frame on a GPU, so 10 frames per second uses under 5% of GPU time. On a CPU, budget 20 to 110 ms per frame and sample 2 to 5 frames per second.

## 5. Total for a near-6B path

Example compositions follow. Counts come from the Hugging Face model API described in Section 2, except emotion2vec+ large, whose [model card](https://huggingface.co/emotion2vec/emotion2vec_plus_large) states about 300M.

| Path | Components | Total params |
|---|---|---|
| Text+audio cascade | Nemotron 0.6B ASR, emotion2vec+ large (about 300M), Qwen3-4B, Smart Turn, Silero | about 4.93B |
| Single omni model | Phi-4-multimodal-instruct (text, vision, speech) | 5.57B |
| Single omni model | Qwen2.5-Omni-3B, including its speech generator | 5.54B |
| Single omni model | Gemma 3n E2B (E4B is 7.85B, over the cap) | 5.44B |
| Three-modality extension | Nemotron 0.6B, emotion2vec+ large, Qwen3-VL-4B (4.44B), YuNet | about 5.34B |

All memory figures in this section are estimates. BF16 stores 2 bytes per parameter, so 6B parameters need about 12 GB for weights. Qwen3-4B's measured 7,973 MB is only about 300 MiB above its raw weights, so runtime overhead is small at short context. Adding CUDA contexts (about 0.3 to 0.5 GB per process), a 4K KV cache and encoder activations gives about 13.5 to 15 GB of VRAM for a full BF16 path. Quantizing the LLM to 4-bit cuts the cascade to about 5 to 7 GB.

A 24 GB GPU (L4, RTX 4090, RTX 3090, RTX A5000) holds the full BF16 path with headroom, and a 16 GB GPU fits it tightly. Laptop GPUs ship with 6 GB (RTX 4050), 8 GB (RTX 4060, 4070, 5060, 5050, and some 5070), 12 GB (RTX 4080, 5070 Ti), 16 GB (RTX 4090, 5080) or 24 GB (RTX 5090) ([NVIDIA laptop specs](https://www.nvidia.com/en-us/geforce/laptops/compare/)). An 8 GB laptop therefore needs a 4-bit LLM, while a 16 to 24 GB laptop can run BF16. Laptop parts decode at roughly half desktop speed: the scoreboard shows 156 tokens/s for the RTX 5090 Laptop against 290 for the desktop RTX 5090.

System RAM should be 32 GB, with 16 GB as a floor (estimate), because BF16 checkpoints pass through host memory while loading and decoded video adds buffers. Plan on 8 vCPUs (estimate) for audio capture, VAD, video decoding, face detection and tokenization; the Whisper-Streaming A40 runs also reserved 8 CPU cores. The MELD raw archive is 10.9 GB by its HTTP Content-Length ([link](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)), so plan about 25 GB of disk for the archive and its extracted copy, plus model files.

CPU-only operation works for a reduced path of Moonshine streaming, small emotion encoders, and a 1B to 3B LLM at 4-bit. It is slow, though. Decode runs at 12 to 24 tokens/s and prefill at about 72 tokens/s on the CPUs above, so TTFT runs to seconds unless the prompt prefix is cached. A full BF16 6B path on a CPU is impractical. Estimate: decode is limited by memory bandwidth, and reading about 12 GB of weights per token over 60 to 90 GB/s of DDR5 caps decode near 5 to 7 tokens/s.
