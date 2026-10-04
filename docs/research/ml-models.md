# Candidate models under the 6B cap, and MELD benchmarks

Researched 2026-10-01 to size a Runpod pod for the MELD emotion prototype. The cap counts every learned parameter on the local inference path, so this note lists total parameters, not active or trainable ones.

## How the numbers were checked

Most parameter counts come from the safetensors header that the Hugging Face Hub reports for each checkpoint (readable at `https://huggingface.co/api/models/<repo-id>`, field `safetensors.total`). The rest come from the model card or paper named in the table. VRAM marked "doc" is a figure the vendor published. VRAM marked "est" is weights only, computed as 2 bytes per parameter at fp16/bf16, 1 byte at int8, and about 0.5 bytes at int4. Weight-only estimates leave out the KV cache (the per-token attention memory a generator keeps), activations, and the roughly 0.5 to 1 GB CUDA context.

Qwen3-0.6B and Qwen3-1.7B tie their input and output embeddings, but the published checkpoints store that matrix twice. Their cards give 0.6B and 1.7B, while the safetensors headers give 0.75B and 2.03B. This note uses the card figure as the unique learned-parameter count and quotes the larger figure as a conservative upper bound. No path verdict below changes either way.

## Encoders

| Model | Role | Params | fp16 weights | Source |
|---|---|---|---|---|
| RoBERTa-base | text | 124.7M | 0.25 GB est | https://huggingface.co/FacebookAI/roberta-base |
| RoBERTa-large | text (EmoBERTa, TelME) | 355.4M | 0.71 GB est | https://huggingface.co/FacebookAI/roberta-large |
| DeBERTa-v3-large | text | 304M backbone + 131M embedding = 435M | 0.87 GB est | https://huggingface.co/microsoft/deberta-v3-large |
| ModernBERT-large (Dec 2024) | text | 395.9M | 0.79 GB est | https://huggingface.co/answerdotai/ModernBERT-large |
| all-mpnet-base-v2 | text | 109.5M | 0.22 GB est | https://huggingface.co/sentence-transformers/all-mpnet-base-v2 |
| emotion2vec (base) | speech emotion | 93.79M | 0.19 GB est | https://arxiv.org/abs/2312.15185 (Table 2) |
| emotion2vec+ base / large | speech emotion | about 90M / about 300M (repo wording) | 0.18 / 0.60 GB est | https://github.com/ddlBoJack/emotion2vec |
| WavLM-base+ / WavLM-large | speech | 94.70M / 316.62M | 0.19 / 0.63 GB est | https://arxiv.org/abs/2312.15185 (Table 2) |
| HuBERT-large, data2vec-audio-base | speech | 316.61M, 93.75M | 0.63, 0.19 GB est | https://arxiv.org/abs/2312.15185 (Table 2) |
| Whisper small | ASR (speech-to-text) | 241.7M | about 2 GB doc | https://github.com/openai/whisper |
| Whisper large-v3-turbo | ASR | 808.9M | about 6 GB doc | https://github.com/openai/whisper |
| Whisper large-v3 | ASR | 1,543.5M | about 10 GB doc | https://github.com/openai/whisper |
| BlazeFace short-range | face detector | under 0.12M est (229,746-byte fp16 file / 2 bytes) | negligible | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite |
| HSEmotion enet_b0 | face emotion | about 4M est (16 MB fp32 file / 4 bytes) | 0.01 GB est | https://github.com/av-savchenko/face-emotion-recognition |
| Swin-Tiny | face frames (FacialMMT) | 28.3M | 0.06 GB est | https://huggingface.co/microsoft/swin-tiny-patch4-window7-224 |
| ViT face-expression | face emotion, FER2013 | 85.8M | 0.17 GB est | https://huggingface.co/trpakov/vit-face-expression |
| VideoMAE-base | video clip | 94.2M | 0.19 GB est | https://huggingface.co/MCG-NJU/videomae-base |
| TimeSformer-base | video clip (TelME) | about 121.6M est (486.3 MB fp32 file / 4 bytes) | 0.24 GB est | https://huggingface.co/facebook/timesformer-base-finetuned-k400 |

The ViT face-expression model outputs exactly MELD's seven classes (angry, disgust, fear, happy, neutral, sad, surprise), so it can serve as a zero-training visual baseline. Even the largest encoder set in the paths below (about 1.48B parameters, or 2.96 GB at fp16) would save only about 1.5 GB at int8, so keeping encoders at fp16 is the simple choice.

## Generators and all-in-one models

| Model (release) | Params | Inputs | VRAM | Source |
|---|---|---|---|---|
| Qwen3-0.6B (Apr 2025) | 0.6B card, 0.75B safetensors | text | bf16 1.39 GB, GPTQ-int8 0.99 GB doc | https://huggingface.co/Qwen/Qwen3-0.6B |
| Qwen3-1.7B (Apr 2025) | 1.7B card, 2.03B safetensors | text | bf16 3.41 GB, GPTQ-int8 2.23 GB doc | https://huggingface.co/Qwen/Qwen3-1.7B |
| Qwen3-4B (Apr 2025) | 4.02B | text | bf16 7.97 GB, FP8 5.28 GB, AWQ-int4 2.92 GB doc | https://huggingface.co/Qwen/Qwen3-4B |
| Llama-3.2-1B / 3B-Instruct | 1.24B / 3.21B | text | 2.5 / 6.4 GB est | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| Phi-4-mini-instruct (Feb 2025) | 3.84B | text | 7.7 GB est | https://huggingface.co/microsoft/Phi-4-mini-instruct |
| LFM2.5-1.2B-Instruct (Jan 2026) | 1.17B | text | 2.3 GB est | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |
| LFM2.5-Audio-1.5B (Dec 2025) | 1.47B (1.2B LM + 115M FastConformer encoder) | text, audio | 2.9 GB est | https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B |
| Qwen3.5-2B / 4B (Feb 2026) | 2.27B / 4.66B | text, image | 4.5 / 9.3 GB est | https://huggingface.co/Qwen/Qwen3.5-2B |
| Qwen3-VL-2B / 4B (Oct 2025) | 2.13B / 4.44B | text, image, video | 4.3 / 8.9 GB est | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct |
| SmolVLM2-2.2B (Feb 2025) | 2.25B | text, image, video | 4.5 GB est | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct |
| Phi-4-multimodal (Feb 2025) | 5.57B | text, image, audio | 11.1 GB est | https://huggingface.co/microsoft/Phi-4-multimodal-instruct |
| Qwen2.5-Omni-3B (Apr 2025) | 5.54B with speech "talker" | text, image, audio, video | bf16 18.38 GB doc minimum for a 15 s video, "at least 1.2 times higher" in practice | https://huggingface.co/Qwen/Qwen2.5-Omni-3B |
| Gemma 4 E2B (Mar 2026) | 5.12B (2.3B "effective"; about 150M vision and 300M audio encoders) | text, image, audio | bf16 11.4 GB, SFP8 5.7 GB, Q4_0 2.9 GB doc, weights only | https://huggingface.co/google/gemma-4-E2B-it and https://ai.google.dev/gemma/docs/core |

The Qwen3 figures are measured runtime memory from Qwen's benchmark (Transformers, batch 1, 1-token prompt, 2,048 generated tokens) at https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html. They include the KV cache, so they are good anchors. Qwen3-4B at bf16 rises from 7.97 GB with a 1-token prompt to 12.32 GB with a 30,720-token prompt. The Gemma 4 E2B document figure of 11.4 GB is higher than the 10.2 GB that 5.12B × 2 bytes gives, and the page attributes variation to the inference tool without itemizing the difference. Qwen2.5-Omni-3B's card says `model.disable_talker()` saves about 2 GB when speech output is not needed.

Several models are traps under a total-parameter rule. Gemma 3n E2B has a 5.44B checkpoint, but its card says "the raw parameter count of this model is 6B," so it invites a dispute with the grader. Gemma 4 E4B (8.0B), Gemma 3n E4B (7.85B), MiniCPM-o 2.6 (8.67B), MiniCPM-o 4.5 (9.37B), Qwen2-Audio-7B (8.40B) and Qwen3-Omni-30B-A3B (35.3B) are all over the cap. R1-Omni-0.5B (https://huggingface.co/StarJiaxing/R1-Omni-0.5B) is an emotion-tuned omni model whose checkpoint is 1.37B. Its card also asks for separate SigLIP-base (203M) and Whisper-large-v3 (1.54B) folders, so its true total may be as high as about 3.1B. Count the loaded modules before relying on it.

## Published MELD results (7-class, weighted F1)

| Paper (date) | Model and size | Text | Text+audio | Text+vision | All three | Source |
|---|---|---|---|---|---|---|
| MELD (ACL 2019) | bcLSTM | 56.44 | 59.25 | none | none | https://arxiv.org/abs/1810.02508 |
| MELD (ACL 2019) | DialogueRNN | 57.03 | 60.25 | none | none | same |
| EmoBERTa (2021) | RoBERTa-large, 355M | 65.61 | | | | https://arxiv.org/abs/2108.12009 |
| FacialMMT (ACL 2023) | RoBERTa + Swin-Tiny | 63.98 | 65.55 | 66.20 | 66.58 | https://aclanthology.org/2023.acl-long.861/ |
| TelME (NAACL 2024) | RoBERTa-large + data2vec-audio-base + TimeSformer-base, about 0.57B est | 66.57 | 67.19 | 67.05 | 67.37 | https://arxiv.org/abs/2401.12987 |
| MAMBA-fusion tech report (Nov 2025) | MPNet-base text, face and speaker embeddings | 57.9 | 62.1 | 60.9 | 64.3 | https://arxiv.org/abs/2511.14969 |
| InstructERC (2023) | Llama-2-7B, over cap | 69.15 | | | | https://arxiv.org/abs/2309.11911 |
| SpeechCueLLM (NAACL Findings 2025) | Llama-2-7B LoRA, over cap | 67.44 | 67.60 | | | https://arxiv.org/abs/2407.21315 |
| EmoQ (Sep 2025) | Qwen2.5-7B + HuBERT-large + RoBERTa-base, over cap | | 66.5 | | | https://arxiv.org/abs/2509.15775 |
| OmniVox, zero-shot (Mar 2025) | Phi-4-multimodal, 5.6B | | 43.4 | | | https://arxiv.org/abs/2503.21480 |

The MAMBA-fusion row is a non-peer-reviewed technical report, but it is the only source found that breaks out every modality pair for a small (110M) text encoder. The original MELD baselines used no visual features, because the authors judged video-based speaker localization to be an open problem. EmoQ's widely quoted 71.6 is its 4-class MELD subset, and 66.5 is the 7-class figure. Non-text modalities are weak on their own: audio-only scores 39.08 (bcLSTM), 41.79 (DialogueRNN), 46.60 (TelME student) and 48.70 (frozen emotion2vec with a linear head), and visual-only scores 36.72 (TelME).

The sizing lesson is that fine-tuned encoder stacks under 1B parameters (EmoBERTa, FacialMMT, TelME) land within about two points of 7B LLM pipelines. A 5.6B omni model used zero-shot scores 24 points lower. The cheapest credible design is therefore a fine-tuned encoder classifier plus a small generator for the response text. EmoBERTa fine-tuned RoBERTa-large on a 16 GB GPU with mixed precision at batch sizes of 4 to 16, so training fits the same class of pod as inference.

## Example inference paths

Every path adds a fusion and classification head, estimated at under 1M parameters (for example 2,048 × 256 + 256 × 7 ≈ 0.53M), and counted here as 1M. VRAM is fp16 weights only (est). For runtime, add the CUDA context, the generator's KV cache, and video-frame activations. Qwen's benchmark shows Qwen3-4B growing by about 0.9 GB between a 1-token and a 6,144-token prompt, which is a fair allowance for a short dialogue history.

### Text plus audio

| Path | Components | Total params | fp16 weights | Under 6B? |
|---|---|---|---|---|
| A1 small cascade | Whisper small 241.7M + RoBERTa-large 355.4M + emotion2vec+ base 90M + head 1M + Qwen3-1.7B 1,700M | 2.39B (2.72B with duplicated Qwen3 embedding) | 4.8 GB | Yes, 3.6B spare |
| A2 strong cascade | Whisper large-v3-turbo 808.9M + RoBERTa-large 355.4M + WavLM-large 316.6M + head 1M + Qwen3-4B 4,022.5M | 5.50B | 11.0 GB | Yes, 0.50B spare |
| A3 audio-native LLM | Phi-4-multimodal 5,574.5M + emotion2vec+ base 90M + head 1M | 5.67B | 11.3 GB | Yes, 0.33B spare. Replacing emotion2vec+ base with RoBERTa-large gives 5.93B, but keeping both gives 6.02B and fails the cap |
| A4 edge | LFM2.5-Audio-1.5B 1,470.3M + RoBERTa-large 355.4M + emotion2vec+ base 90M + head 1M | 1.92B | 3.8 GB | Yes, 4.1B spare |

### Text plus vision

| Path | Components | Total params | fp16 weights | Under 6B? |
|---|---|---|---|---|
| V1 typed text, small | RoBERTa-large 355.4M + BlazeFace 0.1M + ViT face-expression 85.8M + head 1M + Qwen3-1.7B 1,700M | 2.14B (2.47B with duplicated embedding) | 4.3 GB | Yes, 3.9B spare |
| V2 spoken input with ASR | Whisper small 241.7M + RoBERTa-large 355.4M + BlazeFace 0.1M + TimeSformer-base 121.6M est + head 1M + Qwen3-4B 4,022.5M | 4.74B | 9.5 GB | Yes, 1.26B spare |
| V3 VLM generator, typed text | Qwen3-VL-4B 4,437.8M + RoBERTa-large 355.4M + ViT face-expression 85.8M + BlazeFace 0.1M + head 1M | 4.88B (5.12B if Whisper small is added for speech) | 9.8 GB | Yes, 1.12B spare |
| V4 single any-to-any model | Gemma 4 E2B 5,123.2M + ViT face-expression 85.8M + BlazeFace 0.1M + head 1M | 5.21B | 10.4 GB by arithmetic, 11.6 GB using Google's 11.4 GB figure | Yes, 0.79B spare; it also takes audio, so no ASR is needed |

Qwen2.5-Omni-3B alone (5.54B, about 0.46B spare) also fits the cap, but its documented bf16 floor of 18.38 GB for a 15 s video makes it the heaviest VRAM choice of all the options here. Every path above holds its weights in under 12 GB at fp16. With a 4-bit generator and fp16 encoders, the large paths drop to roughly 3 to 6 GB. A2 becomes about 5.9 GB (2.96 GB of fp16 encoders plus Qwen3-4B AWQ at 2.92 GB measured), and V4 becomes about 3.1 GB (Gemma 4 E2B Q4_0 at 2.9 GB plus 0.17 GB of face model).
