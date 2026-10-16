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

