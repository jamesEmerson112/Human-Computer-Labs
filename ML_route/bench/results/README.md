# Overnight benchmark results

Hardware: one NVIDIA A40 48 GB Runpod pod (Secure Cloud, EU-SE-1), 9 vCPUs. Data: MELD test split, 2,610 lines.

## Speech to text

| Model | Word error rate | Median ms per line | 95th percentile ms | Real-time factor | GPU memory MiB |
|---|---:|---:|---:|---:|---:|
| whisper-turbo-fp16 | 31.9% | 164.0 | 273.0 | 0.0533 | 2700 |
| whisper-turbo-int8 | 31.75% | 167.4 | 272.8 | 0.0546 | 1505 |
| nemotron-1120ms | 33.43% | 56.0 | 120.4 | 0.0214 | 7279 |
| nemotron-160ms | 34.81% | 57.2 | 123.3 | 0.022 | 7279 |
| parakeet-tdt-v3 | 31.93% | 57.1 | 133.9 | 0.0224 | 7135 |

Word error rate is measured against MELD's own text after Whisper's English normaliser. Many clips also hold words from the line before or after, which no model can avoid, so insertions dominate every row and the absolute rates are high; compare the rows with each other. The saved MELD.Whisper test transcripts score 31.80% the same way, which matches the int8 row. Times are for one clip per call with the audio already decoded. GPU memory for the NeMo models includes PyTorch's cached pool, so it overstates what the model needs.

## Frozen encoders with a small head

**Text only.** Weighted F1 on MELD test.

| Features | Logistic regression | MLP |
|---|---:|---:|
| deberta-v3-large/raw | 57.34 | 55.03 |
| deberta-v3-large/completed | 57.53 | 58.01 |
| deberta-v3-large/whisper | 47.76 | 49.08 |
| embeddinggemma-300m/raw | 58.65 | 58.51 |
| embeddinggemma-300m/completed | 58.44 | 60.7 |
| embeddinggemma-300m/whisper | 47.51 | 49.89 |
| emoberta-large/raw | 75.16 | 77.91 |
| emoberta-large/completed | 74.97 | 77.41 |
| emoberta-large/whisper | 58.74 | 61.98 |
| modernbert-large/raw | 55.39 | 59.79 |
| modernbert-large/completed | 55.35 | 59.5 |
| modernbert-large/whisper | 46.88 | 49.23 |
| mpnet-base/raw | 54.08 | 58.12 |
| mpnet-base/completed | 55.1 | 58.2 |
| mpnet-base/whisper | 45.55 | 47.85 |
| roberta-base/raw | 58.31 | 60.98 |
| roberta-base/completed | 57.32 | 61.36 |
| roberta-base/whisper | 47.84 | 49.75 |
| roberta-large/raw | 58.77 | 59.7 |
| roberta-large/completed | 58.46 | 60.0 |
| roberta-large/whisper | 49.01 | 51.1 |

**Voice only.** Weighted F1 on MELD test.

| Features | Logistic regression | MLP |
|---|---:|---:|
| emotion2vec-plus-large | 47.9 | 47.93 |
| hubert-large | 46.81 | 44.08 |
| wav2vec2-superb-er | 44.61 | 44.59 |
| wavlm-base-plus | 46.42 | 47.7 |
| wavlm-large | 47.73 | 47.58 |

**Text plus voice.** Weighted F1 on MELD test.

| Features | Logistic regression | MLP |
|---|---:|---:|
| embeddinggemma-300m+emotion2vec-plus-large | 56.3 | 57.69 |
| embeddinggemma-300m+hubert-large | 57.46 | 61.62 |
| embeddinggemma-300m+wav2vec2-superb-er | 55.51 | 60.52 |
| embeddinggemma-300m+wavlm-base-plus | 56.14 | 61.48 |
| embeddinggemma-300m+wavlm-large | 56.07 | 60.88 |

One line at a time with no dialogue context. Test text for raw and completed is MELD's published test text; for whisper it is Whisper's test transcript. EmoBERTa-large's released weights have likely seen MELD's test lines (77.9 here against 63.46 on its own card), so its scores are not comparable and it is left out of the best-text pick.

## Reply writers

| Model | Parameters loaded | Time to first token, median ms | Tokens per second | Whole reply, median ms |
|---|---:|---:|---:|---:|
| granite-4.1-3b | 3.40B | 28.3 | 39.0 | 333.2 |
| nemotron-3-nano-4b | 3.97B | 55.0 | 48.6 | 857.2 |
| phi-4-mini-instruct | 3.84B | 29.8 | 43.8 | 274.6 |
| smollm3-3b | 3.08B | 25.2 | 47.2 | 356.4 |
| lfm2.5-1.2b-instruct | 1.17B | 14.8 | 85.0 | 122.3 |
| qwen3.5-4b | 4.21B | 53.4 | 26.3 | 619.0 |
| qwen3.5-2b | 1.88B | 41.6 | 34.9 | 365.7 |
| gemma-4-e2b-it | 5.10B | 46.3 | 26.0 | 392.2 |
| llama-3.2-3b-instruct | 3.21B | 21.4 | 56.2 | 235.0 |

Hugging Face transformers generate, bf16, batch size 1, greedy, up to 40 new tokens, 50 MELD test lines with the gold emotion in the prompt

Models built on Mamba, gated delta-rule or short-convolution layers (Nemotron 3 Nano, the Qwen3.5 family, LFM2.5) ran on transformers' slow reference kernels, because causal-conv1d, mamba-ssm and flash-linear-attention were not installed. Their speeds here understate what they reach with those kernels or with vLLM. Parameters loaded counts only the part of the model that writes text, so for Qwen3.5-4B (4.66B in the catalog), Qwen3.5-2B (2.27B) and Gemma 4 E2B it leaves out the vision and audio parts that the 6B limit counts.

## Zero-shot vision-language models

| Model | Clips | Frames only, weighted F1 | Frames and words, weighted F1 | Unparsed answers | Median ms per clip |
|---|---:|---:|---:|---:|---:|
| qwen3.5-4b | 1793 | 36.53 | 54.82 | 0 / 0 | 521.9 / 527.0 |
| oneemo | 1890 | 39.28 | 47.98 | 0 / 0 | 522.7 / 526.5 |
| internvl3.5-4b | 863 | 35.74 | 52.12 | 0 / 0 | 1038.4 / 1038.6 |
| gemma-4-e2b | 1477 | 33.41 | 51.42 | 0 / 0 | 610.0 / 611.7 |
| qwen2.5-omni-3b | 1489 | 37.5 | 53.53 | 0 / 0 | 637.5 / 638.2 |

Each model saw 8 frames and was asked for one word from MELD's seven labels. The prediction is the label whose first token the model scores highest as the start of its answer, from one forward pass, so every answer is a label and the time is that one pass. A first try that let models write up to 8 tokens was thrown away: Qwen3.5-4B opened 1,579 of 1,585 answers with an explanation such as "To determine the emotion", so its 32.9 was the always-neutral score. Always answering neutral scores 31.27 on test. Some labels' first tokens are short fragments, such as a prefix of disgust or the letter f, which can also start other words, so those labels may be picked slightly too often; each result file lists the token ids used. Each model had a 40-minute limit, so a row may cover a random subset of the 2,457 clips whose frames extracted. The slow-kernel caveat above applies to the Qwen3.5-based models (Qwen3.5-4B and OneEmo). OneEmo's paper feeds 16 frames with its own prompt and reports a MELD hit rate, which is not weighted F1, so its lower score here with words is not a direct contradiction of that paper.

## About the run

The pod ran from 08:58 to 16:21 UTC on 2026-10-15, 7.4 hours. At $0.59 an hour that is about $4.35 of GPU time; Runpod's billing showed $3.93 in total, $3.76 GPU and $0.18 disk, when the pod was stopped, and its figures can lag. The pod is stopped with its 150 GB volume kept, which bills about $1 a day.

The volume holds the MELD archive, the decoded audio, the extracted features, the uv package cache and the weights of the VLMs, the reply writers that were rerun, and EmbeddingGemma. The weights of every other finished model were deleted when the volume filled up during the night. The Python environments and the unpacked clips lived on the container disk, which Runpod erases on stop, so they need rebuilding from the cache next time.

Things that went wrong and were fixed: the volume filled up and blocked two VLM downloads; Gemma 4 E2B's weights first failed to download; the free-text VLM answers were unusable (see that section); three reasoning models first wrote their thinking, so they were rerun with thinking turned off; Qwen2.5-Omni-3B first loaded as its speech-output model and was rerun with only its text part, the thinker.

Frames failed to extract for 153 of the 2,610 test clips, so the VLM rows draw from 2,457 clips.

## Vision features with a small head

**Text only.** Weighted F1 on MELD test. The first row repeats last night's best fair text set to check that the run reproduces; the others use Whisper's transcripts, the words the demo reads.

| Features | Test clips | Logistic regression | MLP, mean ± std | Dev MLP, mean |
|---|---:|---:|---:|---:|
| roberta-base/completed | 2610 | 57.32 | 60.98 ± 0.45 | 58.6 |
| embeddinggemma-300m/whisper | 2610 | 47.51 | 50.06 ± 0.22 | 48.7 |
| roberta-base/whisper | 2610 | 47.84 | 50.66 ± 0.86 | 48.91 |

Last night's single MLP run on this set scored 61.36; seed 0 here scored 61.36.

**Vision only.** Weighted F1 on MELD test.

| Features | Test clips | Logistic regression | MLP, mean ± std | Dev MLP, mean |
|---|---:|---:|---:|---:|
| hsemotion-enet-b2-8/face | 2610 | 34.99 | 37.33 ± 0.69 | 35.41 |
| oneemo/reader | 2610 | 37.8 | 38.13 ± 2.19 | 36.24 |
| oneemo/tower | 2610 | 37.05 | 35.36 ± 1.76 | 34.49 |
| qwen3.5-4b/reader | 2610 | 37.5 | 39.72 ± 0.38 | 36.84 |
| qwen3.5-4b/tower | 2610 | 37.17 | 36.42 ± 0.8 | 33.97 |
| siglip2-base/face | 2610 | 36.17 | 35.68 ± 0.6 | 33.25 |
| siglip2-base/frame | 2610 | 34.94 | 34.87 ± 2.25 | 32.82 |
| vit-face-expression/face | 2610 | 34.81 | 36.15 ± 0.29 | 34.43 |
| vjepa2-vitl/frame | 2610 | 35.96 | 34.34 ± 0.78 | 30.81 |

**Text plus vision.** Weighted F1 on MELD test, with the text-only heads trained and scored on exactly the same clips and the gain over them.

| Features | Test clips | Logistic regression | MLP, mean ± std | Text only, same clips, logistic | Text only, same clips, MLP | Gain, logistic | Gain, MLP ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| roberta-base/completed+hsemotion-enet-b2-8/face | 2610 | 54.19 | 59.25 ± 0.3 | 57.29 | 61.36 ± 0.13 | -3.10 | -2.11 ± 0.33 |
| embeddinggemma-300m/whisper+hsemotion-enet-b2-8/face | 2610 | 43.66 | 48.49 ± 0.9 | 47.51 | 50.27 ± 0.34 | -3.85 | -1.78 ± 0.96 |
| roberta-base/whisper+hsemotion-enet-b2-8/face | 2610 | 44.54 | 49.14 ± 0.21 | 47.85 | 50.86 ± 0.3 | -3.31 | -1.72 ± 0.37 |
| roberta-base/completed+oneemo/reader | 2610 | 54.14 | 57.92 ± 1.2 | 57.29 | 61.36 ± 0.13 | -3.15 | -3.44 ± 1.21 |
| embeddinggemma-300m/whisper+oneemo/reader | 2610 | 42.54 | 49.28 ± 0.9 | 47.51 | 50.27 ± 0.34 | -4.97 | -0.99 ± 0.96 |
| roberta-base/whisper+oneemo/reader | 2610 | 45.06 | 50.13 ± 0.23 | 47.85 | 50.86 ± 0.3 | -2.79 | -0.73 ± 0.38 |
| roberta-base/completed+oneemo/tower | 2610 | 54.63 | 57.97 ± 0.49 | 57.29 | 61.36 ± 0.13 | -2.66 | -3.39 ± 0.51 |
| embeddinggemma-300m/whisper+oneemo/tower | 2610 | 44.11 | 48.4 ± 1.23 | 47.51 | 50.27 ± 0.34 | -3.40 | -1.87 ± 1.28 |
| roberta-base/whisper+oneemo/tower | 2610 | 45.1 | 48.34 ± 0.87 | 47.85 | 50.86 ± 0.3 | -2.75 | -2.52 ± 0.92 |
| roberta-base/completed+qwen3.5-4b/reader | 2610 | 55.4 | 59.52 ± 0.7 | 57.29 | 61.36 ± 0.13 | -1.89 | -1.84 ± 0.71 |
| embeddinggemma-300m/whisper+qwen3.5-4b/reader | 2610 | 44.08 | 49.38 ± 0.36 | 47.51 | 50.27 ± 0.34 | -3.43 | -0.89 ± 0.5 |
| roberta-base/whisper+qwen3.5-4b/reader | 2610 | 45.73 | 48.29 ± 1.36 | 47.85 | 50.86 ± 0.3 | -2.12 | -2.57 ± 1.39 |
| roberta-base/completed+qwen3.5-4b/tower | 2610 | 54.6 | 58.37 ± 0.55 | 57.29 | 61.36 ± 0.13 | -2.69 | -2.99 ± 0.57 |
| embeddinggemma-300m/whisper+qwen3.5-4b/tower | 2610 | 43.76 | 48.51 ± 1.15 | 47.51 | 50.27 ± 0.34 | -3.75 | -1.76 ± 1.2 |
| roberta-base/whisper+qwen3.5-4b/tower | 2610 | 45.4 | 48.39 ± 0.75 | 47.85 | 50.86 ± 0.3 | -2.45 | -2.47 ± 0.81 |
| roberta-base/completed+siglip2-base/face | 2610 | 56.83 | 59.86 ± 0.82 | 57.29 | 61.36 ± 0.13 | -0.46 | -1.50 ± 0.83 |
| embeddinggemma-300m/whisper+siglip2-base/face | 2610 | 44.9 | 50.69 ± 0.71 | 47.51 | 50.27 ± 0.34 | -2.61 | +0.42 ± 0.79 |
| roberta-base/whisper+siglip2-base/face | 2610 | 45.92 | 49.89 ± 0.89 | 47.85 | 50.86 ± 0.3 | -1.93 | -0.97 ± 0.94 |
| roberta-base/completed+siglip2-base/frame | 2610 | 54.09 | 59.51 ± 0.36 | 57.29 | 61.36 ± 0.13 | -3.20 | -1.85 ± 0.38 |
| embeddinggemma-300m/whisper+siglip2-base/frame | 2610 | 42.56 | 50.16 ± 0.18 | 47.51 | 50.27 ± 0.34 | -4.95 | -0.11 ± 0.38 |
| roberta-base/whisper+siglip2-base/frame | 2610 | 44.43 | 48.83 ± 1.82 | 47.85 | 50.86 ± 0.3 | -3.42 | -2.03 ± 1.84 |
| roberta-base/completed+vit-face-expression/face | 2610 | 56.34 | 59.93 ± 0.93 | 57.29 | 61.36 ± 0.13 | -0.95 | -1.43 ± 0.94 |
| embeddinggemma-300m/whisper+vit-face-expression/face | 2610 | 44.31 | 46.6 ± 0.97 | 47.51 | 50.27 ± 0.34 | -3.20 | -3.67 ± 1.03 |
| roberta-base/whisper+vit-face-expression/face | 2610 | 46.79 | 47.6 ± 0.82 | 47.85 | 50.86 ± 0.3 | -1.06 | -3.26 ± 0.87 |
| roberta-base/completed+vjepa2-vitl/frame | 2610 | 53.92 | 57.33 ± 0.55 | 57.29 | 61.36 ± 0.13 | -3.37 | -4.03 ± 0.57 |
| embeddinggemma-300m/whisper+vjepa2-vitl/frame | 2610 | 42.67 | 47.88 ± 1.27 | 47.51 | 50.27 ± 0.34 | -4.84 | -2.39 ± 1.31 |
| roberta-base/whisper+vjepa2-vitl/frame | 2610 | 44.46 | 48.81 ± 1.03 | 47.85 | 50.86 ± 0.3 | -3.39 | -2.05 ± 1.07 |

**Text, vision and voice.** Weighted F1 on MELD test, with qwen3.5-4b/reader, the vision set with the best mean dev MLP F1.

| Features | Test clips | Logistic regression | MLP, mean ± std | Text only, same clips, logistic | Text only, same clips, MLP | Gain, logistic | Gain, MLP ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| embeddinggemma-300m/whisper+qwen3.5-4b/reader+emotion2vec-plus-large/audio+hubert-large/audio | 2610 | 46.16 | 49.36 ± 0.88 | 47.51 | 50.27 ± 0.34 | -1.35 | -0.91 ± 0.94 |

One line at a time with no dialogue context. Logistic regression runs once; the MLP runs with seeds 0, 1, 2 and is reported as the mean and sample standard deviation over them; seed 0 is the seed last night's single run used. The whisper rows train, tune and test on Whisper's transcripts of the clips, the words the live demo reads; completed rows use MELD's published words. Vision sets miss clips whose frames failed to extract, so a joined row drops those clips from train, dev and test alike; each joined row is therefore compared with the text-only heads trained and scored on exactly the same clips, and the gain is the difference. The gain's standard deviation is the square root of the sum of both rows' squared seed standard deviations, and a gain smaller than about two of those is not a real difference. Face-view rows end with a has_face column, 1.0 when a face was found in any frame and 0.0 when the row is all zeros. The best vision set for the all-three row is picked by mean dev MLP F1.

## Nemotron transcripts as text

Not run.

## Vision encoder speed

| Encoder / view | Median ms per clip | 95th percentile ms | Clips | Device | Notes |
|---|---:|---:|---:|---|---|
| hsemotion-enet-b2-8/face | 37.68 | 57.22 | 13706 | NVIDIA A40 bfloat16 | batch_clips 16; amortised batch time per clip (preprocessing plus forward), image decoding and cropping excluded; splits train, dev, test |
| oneemo/reader | 591.7 | 692.8 | 13706 | NVIDIA A40 bfloat16 | whole frames-only forward pass, vision plus language model, one clip per call; the first clip of each run includes warm-up |
| oneemo/tower | 196.3 | 289.3 | 13706 | NVIDIA A40 bfloat16 | vision module only, timed by forward hooks with a CUDA sync, one clip of eight frames per call; the first clip of each run includes warm-up |
| qwen3.5-4b/reader | 591.3 | 750.0 | 13706 | NVIDIA A40 bfloat16 | whole frames-only forward pass, vision plus language model, one clip per call; the first clip of each run includes warm-up |
| qwen3.5-4b/tower | 195.9 | 330.0 | 13706 | NVIDIA A40 bfloat16 | vision module only, timed by forward hooks with a CUDA sync, one clip of eight frames per call; the first clip of each run includes warm-up |
| siglip2-base/face | 7.23 | 10.66 | 13706 | NVIDIA A40 bfloat16 | batch_clips 16; amortised batch time per clip (preprocessing plus forward), image decoding and cropping excluded; splits train, dev, test |
| siglip2-base/frame | 18.7 | 34.7 | 13706 | NVIDIA A40 bfloat16 | batch_clips 16; amortised batch time per clip (preprocessing plus forward), image decoding and cropping excluded; splits train, dev, test |
| vit-face-expression/face | 6.68 | 8.6 | 13706 | NVIDIA A40 bfloat16 | batch_clips 16; amortised batch time per clip (preprocessing plus forward), image decoding and cropping excluded; splits train, dev, test |
| vjepa2-vitl/frame | 29.98 | 51.71 | 13706 | NVIDIA A40 bfloat16 | batch_clips 16; amortised batch time per clip (preprocessing plus forward), image decoding and cropping excluded; splits train, dev, test |
| yunet/face-finding | 138.1 | 164.15 | 13706 | cpu | frames_per_clip 8; threads 8; clips_per_s 55.95; per-clip time inside one of the parallel threads |

Each row is milliseconds per clip as recorded by the script that extracted it, and its notes say how it was measured: amortised batch time, one clip per call, or the time inside one of several parallel CPU threads, so rows measured differently are not directly comparable. Face finding has its own row, yunet/face-finding.

## Faces and frames

**Face finding.**

| Split | Clips with frames | Clips without frames | Clips with no face | Frames | Frames with a face | Frames with several faces | Median face score |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 9988 | 1 | 37 | 79904 | 78788 | 49832 | 0.919 |
| dev | 1108 | 1 | 3 | 8864 | 8713 | 5204 | 0.919 |
| test | 2610 | 0 | 13 | 20880 | 20505 | 12611 | 0.919 |

Detector: face_detection_yunet_2023mar.onnx from opencv/face_detection_yunet, score threshold 0.6.

Speaker rule: In each frame the largest YuNet face (by box area, score at least 0.6) is taken as the speaker. The crop is a square centred on that box whose side is 1.25 times the box's longer side, clamped to the frame, so a face at the frame's edge gives a crop that is no longer square. This is a guess at the speaker: in two-shots and over-the-shoulder shots the largest face is often the listener.

A clip with no face in any frame has an all-zero face-view row whose has_face column is 0.0.

**Clips whose frames failed to extract.**

| Split | Failed clips | Clips with repeated frames |
|---|---:|---:|
| train | 1 | 456 |
| dev | 1 | 56 |
| test | 0 | 114 |

A short clip can hold fewer decodable frames than the eight slots; frames.py then fills an empty slot with a copy of the nearest frame that worked rather than dropping the clip.
