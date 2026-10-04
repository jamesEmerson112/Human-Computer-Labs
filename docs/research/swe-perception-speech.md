# Perception, speech and decision components for a 4-core, 8 GB laptop

Researched 2026-10-01 for the SWE route (simulated five-joint desk lamp). The target is a clean Ubuntu 24.04 laptop with 4 CPU cores, 8 GB of RAM, no discrete GPU, and Wi-Fi.

Nothing in this route needs a GPU. Every perception and speech component has a CPU option with a published latency that fits a live demo. The language and vision decision layer does not fit well on 4 cores, and a hosted API serves it better than a rented GPU would. A Runpod machine would only make sense for self-hosting a large vision-language model, which the challenge does not require. Each number names the hardware it was measured on. Anything marked "estimate" is my own arithmetic.

## 1. Engagement detection

OpenCV's YuNet face detector runs in 0.69 ms per frame at a 160×120 input on an Intel i7-12700K (a 12-core desktop chip). It takes 6.23 ms on a Raspberry Pi 4B, whose four ARM cores give a pessimistic bound for this laptop ([opencv_zoo benchmark](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)). Its WIDER Face AP (average precision on the standard face benchmark) is 0.88, 0.87 and 0.75 on the easy, medium and hard splits ([model page](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).

MediaPipe Face Landmarker returns 478 landmarks, 52 blendshape scores, and a facial transformation matrix ([docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)). That matrix is a 3D head pose, so no second model is needed for yaw and pitch. Google publishes CPU latency only for its BlazeFace short-range detector, at 2.94 ms on a Pixel 6 phone ([face detector docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)). Google publishes no x86 number for the full landmarker. My estimate is 10 to 20 ms per frame on one laptop core. The [Face Mesh V2 model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf) (dated 2022-09-15) says the model is unsuitable for faces turned more than 80° away from the camera and tolerates only 8° of roll. That limit is useful here, because losing the face mesh past about 80° of yaw is itself a disengagement signal.

Dedicated gaze networks cost far more. L2CS-Net (ResNet-50, 3.92° error on MPIIGaze, [paper](https://arxiv.org/abs/2203.03339)) and 6DRepNet (RepVGG, 3.47° on AFLW2000, [repo](https://github.com/thohemp/6DRepNet)) publish no CPU latency. My estimate is 50 to 150 ms per face on 4 cores. Gaze4HRI ([arXiv 2605.04770](https://arxiv.org/abs/2605.04770), May 2026) found that every evaluated gaze model fails in some condition, and steeply downward gaze defeats all of them, which matters for someone looking at a desk. The practical choice is YuNet or BlazeFace plus Face Landmarker head pose at 10 to 15 frames per second, with hysteresis (separate thresholds and hold times for engaging and disengaging).

## 2. Object recognition for scene memory

| Option | Published latency and hardware | Memory | Notes |
|---|---|---|---|
| YOLO26n (ONNX, 640 px) | 38.9 ms on an Intel Xeon at 2.0 GHz ([docs](https://docs.ultralytics.com/models/yolo26/), Jan 2026) | Estimate 200–300 MB | COCO mAP 40.9, 2.4M parameters |
| YOLO11n (ONNX, 640 px) | 56.1 ms on an Amazon EC2 P4d instance's CPU ([docs](https://docs.ultralytics.com/models/yolo11/)) | Estimate 200–300 MB | COCO mAP 39.5 |
| YOLOE-26n (text-prompted) | Not published | Not published | Open-vocabulary, LVIS mAP 24.7 at 6.1 GFLOPs ([docs](https://docs.ultralytics.com/models/yoloe/)) |
| Moondream 0.5B (int4) | Not published | 816 MiB at runtime ([blog](https://moondream.ai/blog/introducing-moondream-0-5b), Dec 2024) | Small VLM for captions and questions |
| SmolVLM 256M / 500M / 2.2B | No x86 CPU figure | Under 1 GB / 1.2 GB / 4.9 GB, measured on GPU ([paper](https://arxiv.org/abs/2504.05299)) | Only on-device figure is 80 tokens/s on an M4 Max |

mAP is mean average precision on the named benchmark. Both YOLO figures come from server CPUs. My estimate is 80 to 200 ms per frame at 640 px on a 4-core laptop, and roughly a quarter of that at 320 px. Grounding DINO, OWLv2, Florence-2 and Qwen2.5-VL-3B have no primary CPU benchmarks I could find. My estimate is several seconds to tens of seconds per image on 4 cores, which is too slow to feel live.

The cloud vision APIs take one webcam frame as an image block. Claude bills ⌈width/28⌉×⌈height/28⌉ visual tokens, so a 640×480 frame costs 414 tokens. It accepts up to 10 MB per base64 image, and Anthropic states that uploaded images are deleted after processing ([vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)). OpenAI bills ⌈w/32⌉×⌈h/32⌉ patches times about 1.2 ([guide](https://developers.openai.com/api/docs/guides/images-vision)). Gemini bills 258 tokens per 768-px tile and returns bounding boxes normalized to 0–1000 ([guide](https://ai.google.dev/gemini-api/docs/image-understanding)). The vendor latencies in section 5 come from text-only prompts. My estimate for an image round trip with a short JSON answer is 1 to 2 seconds over Wi-Fi.

## 3. Speech to text and voice activity detection

Silero VAD (voice activity detection, which decides when someone is speaking) processes a 30 ms chunk in under 1 ms on one CPU thread. Its model weighs about 2 MB ([repo](https://github.com/snakers4/silero-vad)).

Whisper throughput benchmarks are misleading for a live character. faster-whisper transcribes 13 minutes of audio with the small model at int8 in 1 minute 42 seconds, using 1,477 MB of RAM, on 8 threads of an i7-12700K ([repo](https://github.com/SYSTRAN/faster-whisper)). That is a real-time factor (processing time over audio length) of about 0.13. But Whisper pads every utterance to 30 seconds, so per-phrase latency stays high. The Moonshine v2 paper ([arXiv 2602.12241](https://arxiv.org/abs/2602.12241), Feb 2026) measured time to first token at 1 second of audio on a MacBook Pro M3. Whisper Small took 401.7 ms at 8.6% WER (word error rate), and Moonshine v2 Small took 65.1 ms at 7.8% WER.

Moonshine's README also reports the delay from the VAD's end-of-phrase decision to the final transcript on a Linux x86 machine whose CPU it does not name. Tiny Streaming took 69 ms, Small Streaming 165 ms and Medium Streaming 269 ms, while Whisper Small on faster-whisper took 3,425 ms. Upstream has since moved this table, so I read it from a [fork of the README](https://github.com/nxpatterns/moonshine-voice) of the [upstream repo](https://github.com/moonshine-ai/moonshine). The v2 models are English-only. Vosk's small English model is 40 MB, needs about 300 MB at runtime, and scores 9.85% WER on LibriSpeech clean ([models page](https://alphacephei.com/vosk/models)).

In the cloud, AssemblyAI reports a median word-emission latency of about 300 ms for Universal-Streaming, against 516 ms for Deepgram Nova-3 in its own test ([vendor blog](https://www.assemblyai.com/blog/introducing-universal-streaming), June 2025). Streaming STT sends microphone audio off the machine while the stream is open, so a local VAD gate is worth keeping even in a cloud design.

## 4. Text to speech

Piper (now OHF-Voice/piper1-gpl) reached a real-time factor of 0.06 with 150 MB of RAM, a 75 MB model, and a 400 ms cold start on a 48-core Xeon E5-2650 v4, per the [piper-plus fork's README](https://github.com/ayutaz/piper-plus/blob/dev/README_EN.md). Picovoice measured 1,720 ms to first speech and 2.6 GB peak for Piper on a Ryzen 7 5700X ([blog](https://picovoice.ai/blog/on-device-tts/), July 2026). Picovoice sells a competing engine, and its peak probably counts more than the ONNX runtime, so I treat 150 MB as realistic.

Kokoro-82M on ONNX Runtime ran on 4 cores of an AMD EPYC 7763 at a mean real-time factor of 0.51. A 59-character sentence took 1.81 s ([heyneo benchmark](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts), May 2026), and Picovoice measured 2.0 GB peak. Kokoro therefore needs sentence-by-sentence streaming on this laptop, so Piper is the practical default and Kokoro the higher-quality option. ElevenLabs quotes about 75 ms for Flash v2.5, excluding network time ([models page](https://elevenlabs.io/docs/overview/models)). My estimate for its time to first audio over Wi-Fi is 200 to 400 ms.

## 5. Language and decision layer

Jeff Geerling ran Ollama's default model tags, which are Q4_K_M (a 4-bit quantization), on an Intel N150, a 4-core, 4-thread mini PC. Llama 3.2 3B generated 9.06 tokens per second, and DeepSeek-R1 1.5B (a Qwen2.5-1.5B distillation) generated 17.99 tokens per second ([issue #12](https://github.com/geerlingguy/ai-benchmarks/issues/12), Feb 2025). A 6-core Ryzen AI 5 340 laptop reached 23.81 tokens per second on the 3B model ([table](https://github.com/geerlingguy/ai-benchmarks)). On a Raspberry Pi 5, models up to 1.5B parameters ran at 5 to 15 tokens per second and 3B models at 2 to 5 ([arXiv 2511.07425](https://arxiv.org/abs/2511.07425), Oct 2025).

Generation speed is not the blocker; prompt processing is. A tool-calling prompt with the tool schema, scene facts and recent dialogue runs 1,000 to 2,000 tokens. No primary source measures prompt processing for a 3B model on 4 laptop cores. My estimate is 30 to 60 tokens per second, from about 6 GFLOPs per token and 100 to 200 GFLOPS of usable CPU throughput, which puts a cold turn at 20 to 60 seconds. llama.cpp can reuse the cached state of a fixed system prompt, but a turn still takes an estimated 10 to 15 seconds once a 60-token JSON plan is generated. The model also needs 2 to 2.5 GB of RAM and competes with vision and speech for the same four cores. This is why the decision layer belongs in the cloud.

Hosted models answer in under a second when reasoning is off. Artificial Analysis measures time to first token on text prompts. Claude Haiku 4.5 starts in 0.59 s at 88.5 tokens per second and costs $1 and $5 per million input and output tokens ([page](https://artificialanalysis.ai/models/claude-4-5-haiku)). GPT-5.4 nano (non-reasoning) starts in 0.76 s at 160.6 tokens per second and costs $0.20 and $1.25 ([page](https://artificialanalysis.ai/models/gpt-5-4-nano-non-reasoning)). Gemini 3.5 Flash with minimal thinking starts in 0.97 s ([page](https://artificialanalysis.ai/models/gemini-3-5-flash-minimal)). Gemini 3.5 Flash-Lite takes 9.05 s because it reasons by default ([page](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)), so set its thinking level to minimal. Google now limits Gemini 2.5 to existing users ([models page](https://ai.google.dev/gemini-api/docs/models)).

Published VLA models do not fit a six-to-eight-hour build. SmolVLA ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844), June 2025) has 450M parameters and runs on CPU, but it was trained on real SO-100 arm data and would need fine-tuning on lamp demonstrations. Gemini Robotics-ER 2 plans and orchestrates robot APIs through function calling, but it is still a preview ([docs](https://ai.google.dev/gemini-api/docs/robotics-overview)). The pattern that fits is the one from SayCan ([arXiv 2204.01691](https://arxiv.org/abs/2204.01691)) and Code as Policies ([arXiv 2209.07753](https://arxiv.org/abs/2209.07753)). A cloud vision-language model receives the frame, the goal and a few lamp-action tools. It emits a short action sequence, local code clamps it to joint limits and executes it, and a fresh frame goes back before the model declares the goal done.

## 6. Recommended split and headroom

| Component | Where | CPU (estimate) | RAM (estimate) |
|---|---|---|---|
| Ubuntu 24.04 desktop, Python, OpenCV capture | Local | Background | 1.5–2 GB |
| YuNet or BlazeFace plus Face Landmarker at 10–15 fps | Local | About 0.5 core | 200–300 MB |
| YOLO26n at 320–640 px, on demand or about 1 Hz | Local | Bursts of 1–2 cores | 200–300 MB |
| Silero VAD | Local | Under 5% of a core | About 100 MB |
| Moonshine Small Streaming | Local | Under 1 core while speaking | About 500 MB |
| Piper | Local | Bursts of 1 core | About 150 MB |
| Lamp simulation and viewer | Local | 0.5–1 core | 300–500 MB |
| Scene description and action planning (Claude, OpenAI or Gemini) | Cloud | Network only | Negligible |

The local total is roughly 3 to 4 GB, which leaves about 4 GB of RAM and one to two cores of headroom (estimate). An all-local design would add a 3B LLM (2 to 2.5 GB), a small VLM (at least 1 GB) and possibly Kokoro, which approaches the 8 GB ceiling and saturates all four cores. The estimated voice-to-voice latency for the recommended split is 1.7 to 2.6 seconds: 300 to 500 ms of VAD silence hangover, about 165 ms of STT, 1 to 1.5 s for the cloud LLM to start and finish a short tool call, and 200 to 400 ms to Piper's first audio. Only one frame and the transcript leave the machine, at question or goal time. The camera feed and ambient audio are never streamed.
