# Text and audio, text and vision, or all three, plus Gemma 4

Researched 2026-10-14. It adds to `ml-models.md`, which covers parameter counts and example paths. Every score here is the weighted F1 on MELD's seven-class test set, written the way papers write it, so 66.57 means 0.6657. Weighted F1 averages each emotion's F1 score, weighting each emotion by how many test lines it has.

## What each extra input adds on MELD

Gains over text alone are in parentheses.

| Paper (year) | Text only | Text and audio | Text and vision | All three |
|---|---|---|---|---|
| MELD, DialogueRNN (2019) | 57.03 | 60.25 (+3.22) | not reported | not reported |
| MMGCN (2021) | 57.72 | 58.02 (+0.30) | 57.92 (+0.20) | 58.65 (+0.93) |
| MM-DFN (2022) | 56.95 | 58.34 (+1.39) | 58.49 (+1.54) | 59.46 (+2.51) |
| M2FNet (2022) | 66.23 | 66.32 (+0.09) | 66.35 (+0.12) | 66.71 (+0.48) |
| MultiEMO (2023) | 61.23 | 64.21 (+2.98) | 63.78 (+2.55) | 66.74 (+5.51) |
| SDT (2023) | 65.52 | 66.24 (+0.72) | 66.18 (+0.66) | 66.60 (+1.08) |
| FacialMMT (2023) | 63.98 | 65.55 (+1.57) | 66.20 (+2.22) | 66.58 (+2.60) |
| TelME (2024) | 66.57 | 67.19 (+0.62) | 67.05 (+0.48) | 67.37 (+0.80) |

Sources: MELD https://arxiv.org/abs/1810.02508, MMGCN https://arxiv.org/abs/2107.06779, MM-DFN https://arxiv.org/abs/2203.02385, M2FNet https://arxiv.org/abs/2206.02187, MultiEMO https://aclanthology.org/2023.acl-long.824/, SDT https://arxiv.org/abs/2310.20494, FacialMMT https://aclanthology.org/2023.acl-long.861/, TelME https://arxiv.org/abs/2401.12987.

The words carry most of MELD's signal. Adding either audio or vision gains between 0.1 and 3 points, and the two are within half a point of each other in six of the seven papers that report both. Adding the third input gains another 0.2 to 2.5 points over the better pair. The gains are larger when the text model is weak, as in MMGCN and MM-DFN, which use a small TextCNN, than when it is RoBERTa-class. On their own, audio scores 33.55 to 46.60 and vision 32.01 to 36.72. Predicting neutral for every line scores 31.26 (MELD-FAIR, https://arxiv.org/abs/2211.15377). TelME trains its audio and vision models to imitate its text model, so its non-text scores are not purely single-input.

Frozen modern voice encoders do better than the openSMILE features that most of those fusion papers used. With a frozen encoder and a small trained head on MELD alone, emotion2vec scored 48.70, data2vec 2.0 45.80, data2vec 43.59 and WavLM-base 35.16 (https://arxiv.org/abs/2312.15185). emotion2vec's unlabeled pretraining audio included MELD, which flatters its score.

## Why vision is hard on MELD but easy on a webcam

The MELD authors left video out of their baselines because "video-based speaker identification and localization is an open problem." Friends scenes often show several people, and the system must work out which face belongs to the speaker. MELD-FAIR lists four obstacles: hidden faces, listeners who move their mouths, laugh tracks, and a speaker who is out of focus or smaller than the others. Its realignment found the speaker's face in 96.92% of lines. FacialMMT combined a speaker detector, face clustering and hand-labelled faces of the six lead characters, which raised vision-only F1 from 32.27 to 36.48. FacialMMT's code and MELD-FAIR's realignment data are public.

A live webcam usually shows one face, so the hardest step disappears in the demo. The vision track is therefore hard to train on MELD and easy to run live. The audio track is simple to train, because the voice in each clip is the speaker's.

Published pipelines run a face detector on 15 to 20 frames per line, encode each face, and pool the results. MediaPipe's BlazeFace short-range detector runs in 2.94 ms on a Pixel 6 CPU (https://developers.google.com/edge/mediapipe/solutions/vision/face_detector). HSEmotion's EfficientNet-B0 takes 54.8 ms per face on an i7-8750H laptop CPU (https://arxiv.org/abs/2103.17107). Frames should therefore be sampled while the person is still speaking. No MELD paper found reports end-to-end live latency.

In live use, the vision track still needs the words. If the person speaks, the system needs the microphone and a speech-to-text model anyway, so adding a voice encoder on top costs about 95M parameters and little extra wiring.

## Gemma 4

Google announced Gemma 4 on 2026-04-02 (https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/). The 12B Unified size followed on 2026-06-03, and the technical report on 2026-07-02 (https://arxiv.org/abs/2607.02770). All sizes are Apache 2.0 and not gated. Totals come from `https://huggingface.co/api/models/google/<id>-it`.

| Size | Total parameters | Inputs | Under 6B? |
|---|---|---|---|
| E2B | 5,123,178,051 | text, image, video as frames, audio | Yes, about 0.88B spare |
| E4B | 7,996,156,490 | same | No |
| 12B | 11,959,730,224 | same | No |
| 26B A4B | 25,805,936,206 | text, image, video | No, although only 3.8B run per token |
| 31B | 31,273,088,876 | text, image, video | No |

The names give effective sizes. E2B is advertised as 2.3B effective because its per-layer embeddings, which are lookup tables that give each layer its own small vector per token, hold about 2.34B parameters. The brief counts total parameters, so E2B counts as 5.12B. Its total already includes an audio encoder of about 305M and a vision encoder of about 150M. Audio input is capped at 30 seconds and video at 60 seconds at one frame per second.

Google's memory table gives E2B 11.4 GB at 16-bit, 5.7 GB at 8-bit and 2.9 GB at 4-bit, weights plus 20% overhead (https://ai.google.dev/gemma/docs/core). On 2026-10-14 the user moved ML work from the GTX 1070 Ti to a Runpod A40 (48 GB, Ampere), which runs the bf16 checkpoints at full 16-bit precision with room to spare. llama.cpp has supported Gemma 4 audio since 2026-04-12 (https://github.com/ggml-org/llama.cpp/pull/21421).

Gemma 4 has no published emotion results, and no Gemma 4 MELD model was found on the Hub. Unsloth publishes E2B audio fine-tuning notebooks and quotes 8 GB of VRAM for E2B QLoRA (https://unsloth.ai/docs/models/gemma-4/train).

## All-in-one models against a trained head

Untuned all-in-one models score poorly on MELD. Phi-4-multimodal scored 43.4 weighted F1 zero-shot (OmniVox, https://arxiv.org/abs/2503.21480). Qwen2.5-Omni-3B reports 0.558 on MELD, most likely accuracy rather than weighted F1 (https://arxiv.org/abs/2503.20215). Fine-tuned encoder stacks under 1B parameters reach 66 to 67. An all-in-one model also writes its emotion as text, which must be parsed into one of seven labels and can drift. A trained head outputs a score for each of the seven labels directly, trains in minutes on cached features, and leaves budget for a separate reply model. The pipeline's weakness is that its reply model sees only the transcript and the label, not the voice or the face.
