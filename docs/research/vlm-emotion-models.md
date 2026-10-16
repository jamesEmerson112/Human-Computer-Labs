# Open vision-language models that read human emotion, checked against the 6B cap

Written on 2026-10-15 from a multi-agent search. Five Opus agents searched from different angles, and further Opus agents then checked each candidate against its model card and paper. 78 models were checked and 42 kept. Every number below links its source. Our own zero-shot run of five of these models on MELD is in `ML_route/bench/results/README.md`.

## What the scores mean

Most MELD numbers in this research area are **hit rates**, not the weighted F1 the challenge scores. A hit rate counts a prediction as correct when the true label appears anywhere in the model's predicted set, after the free-text answer is grouped onto an emotion wheel (https://arxiv.org/html/2607.12787). That is more lenient than choosing one label. A hit rate therefore cannot be compared with a MELD weighted-F1 leaderboard. The wheel also has no "neutral", and neutral is a MELD label. Mapping a model's open-vocabulary answer onto MELD's seven labels is a step you would have to design yourself.

The 6B cap counts every parameter, vision and audio encoders included. Hugging Face size badges often round in either direction:

- Qwen3.5-4B shows "5B" but is 4.66B.
- Qwen2.5-Omni-3B is 5.54B as shipped. Its text-output part, the thinker, is 4.70B.
- InteractiveOmni-4B is 6.20B, so it is over the cap as shipped.

One model that both reads the emotion and writes the reply is counted once. Two separate models are added together.

## Kept models

Rows marked "(catalog)" are already in your catalog and are listed only for their scores. "A" means the model needs audio, which matters if you pick the text plus vision track. "HR" means hit rate.

| Model | Total params | Fits 6B | Emotion-trained | Inputs | License | Best confirmed emotion result (source) |
|---|---|---|---|---|---|---|
| OneEmo | 4.54B | Yes | Yes | video, image, text | weights CC-BY-NC-4.0 | MELD HR 55.06, MELD used as test only (arxiv.org/html/2608.06013) |
| Nano-EmoX | 2.2B per paper, not confirmed from files | Yes | Yes | video, face, A, text | Apache-2.0 on HF, MIT on GitHub | MELD HR 56.55, zero-shot (arxiv.org/html/2603.02123) |
| Light-MER (student) | 0.85B | Yes | Yes | face video, A, text | Apache-2.0 | MELD HR 57.02 with audio, video and text (arxiv.org/html/2607.12787) |
| InternVL3.5-4B (also 2B at 2.3B, 1B at 1.1B) | 4.73B | Yes | No | image, video, text | Apache-2.0 | MELD HR 52.73, setting and inputs not stated (arxiv.org/html/2608.06013) |
| VidEmo-3B | 3.75B | Yes | Yes | video, image, text | Apache tag on HF conflicts with a non-commercial EULA on the project page | MELD HR 33.75 (arxiv.org/html/2608.06013); MELD is in its training data |
| FaceLLM-1B | 0.94B | Yes | Yes, faces | face image, text | MIT plus Qwen License | FaceXBench Face Analysis 53.00 against 49.88 for its base (arxiv.org/html/2507.10300) |
| FMAE ViT-L (face encoder, not a VLM) | about 0.30B, inferred from file sizes | Yes | Yes | aligned face crops | CC-BY on HF conflicts with CC-BY-NC on GitHub | RAF-DB 93.45%, fine-tuned (arxiv.org/html/2407.11243v2) |
| MiniCPM-V 4.7 1B (replaces 4.6 and 4.5) | 1.55B | Yes | No | image, video, text | Apache-2.0 | None of its own. Predecessor 4.6 scored FACES 0.669 (arxiv.org/html/2610.08162v1) |
| Ovis2.5-2B | 2.57B | Yes | No | image, video frames, text | Apache-2.0 | None found |
| Molmo2-4B | about 4.85B | Yes | No | image, video, text | Apache-2.0, card says research and education use | None found |
| R1-Omni-0.5B (catalog) | 1.37B | Yes | Yes | video, A, text | Apache-2.0 | MELD HR 40.87 (arxiv.org/html/2603.02123v1); MELD accuracy 29.81 (arxiv.org/html/2602.00971) |
| Qwen3.5-4B (catalog) | 4.66B (4.54B without the MTP head) | Yes | No | image, video, text | Apache-2.0 | MELD HR 32.44, zero-shot (arxiv.org/html/2608.06013) |
| Qwen3.5-2B (catalog) | 2.27B | Yes | No | image, video, text | Apache-2.0 | None found |
| Qwen2.5-Omni-3B (catalog) | 5.54B whole, 4.70B thinker | Thinker only | No | video, image, A, text | Qwen Research, non-commercial | MELD accuracy 45.3 direct and 51.1 with CCoT prompting, from audio (arxiv.org/html/2509.25458) |
| Phi-4-multimodal (catalog) | 5.57B | Yes | No | text, image, A | MIT | MELD weighted F1 43.4, text plus audio, zero-shot (arxiv.org/html/2503.21480) |
| InteractiveOmni-4B | 6.20B | No as shipped | No | image, video, A, text | MIT | MELD accuracy 57.16, audio only (arxiv.org/html/2510.13747) |
| AffectGPT | about 8.2 to 8.4B, estimated from components | No | Yes | video, A, text | non-commercial research | MELD HR 57.63 with video and text, 55.65 with all three (arxiv.org/html/2501.16566) |
| Emotion-LLaMAv2 | about 7.7B per paper, 8.4B by component sum | No | Yes | video, A, text | BSD-3 code, Llama 2 base | MELD HR 51.32 for stage 1, 48.61 for stage 2 (arxiv.org/html/2601.16449v1) |
| HumanOmniV2 | 8.93B | No | Yes | video, A, image, text | Apache-2.0 | MELD HR 42.74 (arxiv.org/html/2601.16449v2) |
| VidEmo-7B | 8.29B | No | Yes | video, image, text | Apache-2.0 | MELD HR 32.95 (arxiv.org/html/2608.06013v2) |
| UniFER-7B | 8.29B | No | Yes, faces | face image, text | Apache-2.0 | FERBench 68.84% accuracy (arxiv.org/html/2511.00389) |
| InternVL3.5-8B | 8.5B | No | No | image, video, text | Apache-2.0 | FACES 0.840 verification accuracy (arxiv.org/html/2610.08162v1) |
| Qwen3.5-9B | 9.65B | No | No | image, video, text | Apache-2.0 | MELD HR 41.26 (arxiv.org/html/2608.06013); FACES 0.815 (arxiv.org/html/2610.08162v1) |
| Gemma 4 12B | 11.96B | No | No | text, image, audio | Apache-2.0 | FACES 0.782 verification (arxiv.org/html/2610.08162v1) |
| GLM-4.6V-Flash | 10.29B | No | No | image, video, text | MIT | EmoNet-Face-HQ verification kappa 0.586, best of 11 models (arxiv.org/html/2610.08162v1) |
| Ministral 3 14B | 13.95B | No | No | image, text | Apache-2.0 | FACES 0.684 verification (arxiv.org/html/2610.08162v1) |
| Kimi-VL-A3B-Thinking-2506 | 16.41B | No | No | image, video, text | MIT | Human-MME emotion discrimination 55.3, variant not stated (arxiv.org/html/2509.26165) |
| Qwen2.5-Omni-7B | 10.73B | No | No | video, image, A, text | Apache-2.0 | MELD weighted F1 64.9 after MELD fine-tuning (arxiv.org/html/2508.02429) |
| Qwen3-Omni-30B-A3B | 35.26B | No | No | video, image, A, text | Apache-2.0 | No MELD score; E3mo-Bench evoked perception 60.09 (arxiv.org/html/2608.10796) |
| EmObserver | 8.77B | No | Yes, evoked emotion | image, text | Apache-2.0 | MVEI 86.23 (arxiv.org/html/2607.21061) |

## Which models read human emotion best within 6B

On MELD hit rate, four fitting models are close together:

- Light-MER: 57.02 (https://arxiv.org/html/2607.12787)
- Nano-EmoX: 56.55 (https://arxiv.org/html/2603.02123)
- OneEmo: 55.06 (https://arxiv.org/html/2608.06013)
- InternVL3.5-4B: 52.73 (https://arxiv.org/html/2608.06013)

Light-MER's number comes from a different paper than the other three. Gaps of one or two points may not mean much across papers.

For the text plus vision track, the ranking changes. Light-MER and Nano-EmoX both took audio for these runs, and I found no confirmed audio-free MELD number for either one. OneEmo and InternVL3.5-4B take video and text only, so they suit that track. On that track, OneEmo is the strongest confirmed option and InternVL3.5-4B is next. The OneEmo paper does not say which inputs the comparison models received in its MELD column. So the InternVL3.5-4B figure is less clean than OneEmo's own.

For comparison, Gemini-3.1-Pro scores 62.72 and GPT-5-mini 53.36 in the same table (https://arxiv.org/html/2608.06013). The best fitting open models are therefore near GPT-5-mini and about six to eight points below Gemini.

## Emotion specialists versus strong general VLMs

Emotion training helps a lot when the base model is the same. OneEmo is a fine-tune of the catalog's Qwen3.5-4B. Its MELD hit rate rose from 32.44 to 55.06 (https://arxiv.org/html/2608.06013). Zero-shot, plain Qwen3.5-4B is weak on MELD. It even scores below the removed Qwen3-VL-4B, which got 41.59 in the same table.

Face-only specialists did badly on MELD. VidEmo-3B scored 33.75 and VidEmo-7B 32.95, even though MELD is in their training data (https://arxiv.org/html/2608.06013). InternVL3.5-4B scored 52.73 in the same table. FaceLLM-1B gains only 0.7 points overall over its base on FaceXBench (https://arxiv.org/html/2507.10300), and it reads still images only.

UniFER-7B shows what face-expression tuning can do. It is about 10 points of accuracy above InternVL3.5-8B on FERBench (https://arxiv.org/html/2511.00389), but at 8.29B it does not fit. Its dataset and recipe could be reused to tune a fitting base such as Qwen3.5-4B.

Overall, the specialists that cover the whole scene and conversation are worth it: OneEmo, Nano-EmoX and Light-MER. The face-only specialists are not, for MELD. A strong general VLM such as InternVL3.5-4B is a reasonable baseline without any tuning.

EmObserver and EmoCaliber score evoked emotion, meaning what an image makes a viewer feel. They do not read the on-screen person, so MVEI 86.23 is not evidence of face reading.

## How each could fit into the system

**As the emotion reader whose answer is mapped to the seven MELD labels.** OneEmo, InternVL3.5-4B and Nano-EmoX can be prompted to name the speaker's emotion. You then map the answer to a MELD label. An October 2026 study of 11 open VLMs found that asking the model to verify a given label scored higher than asking it to name the emotion freely (https://arxiv.org/html/2610.08162v1). That suggests constraining the answer to the seven labels, or scoring one yes-or-no question per label. Free-text answers would need the wheel mapping. Expect single-label accuracy to come out below the published hit rates.

**As a frozen encoder feeding a trained head.** You would take features from a model without changing its weights, and train only a small classifier on MELD train to predict the seven labels. That trained classifier is called the head. FMAE ViT-L gives face features at about 0.30B, but it needs a face detector and speaker selection first. A VLM's pooled hidden states can also be the features, for example from OneEmo or InternVL3.5-4B. Two example budgets:

- FMAE plus Qwen3.5-4B comes to about 4.96B.
- Light-MER plus Qwen3.5-4B comes to 5.51B.

**As the reply writer that also sees the face.** OneEmo lists empathetic response generation among its tasks. Nano-EmoX also does recognition and empathetic replies in one model. If one model both classifies and replies, it is counted once against the cap. OneEmo at 4.54B leaves about 1.46B for anything else. InternVL3.5-4B or Qwen3.5-4B could fill the same role untuned.

**Packaging and license notes.**
- Nano-EmoX's Hugging Face repo holds only trained adapter checkpoints. It needs CLIP-L, Chinese HuBERT-Large, FaceXFormer and Qwen2.5-1.5B downloaded separately.
- Light-MER's vision tower is CLIP-B/16. CLIP appears here only as a fixed part of these released models, not as a catalog pick.
- OneEmo's weights are non-commercial, and so are Qwen2.5-Omni-3B's.
- VidEmo's and FMAE's licenses are disputed between sources.

## What is known about vision on MELD

No source gives a fitting model's seven-class weighted F1 on MELD from vision. The weighted-F1 figures that do exist are these:

- Phi-4-multimodal: 43.4 zero-shot from text plus audio, and 36.3 from audio alone (https://arxiv.org/html/2503.21480).
- HumanOmni-7B: 66.6 after fine-tuning on MELD (https://arxiv.org/html/2508.02429). It is over the cap, and HumanOmniV2 has replaced it.
- Qwen2.5-Omni-7B: 64.9 after fine-tuning on MELD, in the same paper.

The input-ablation evidence gives mixed signals about how much video adds:

- AffectGPT scored 57.63 hit rate with video and text, 56.63 with audio and text, and 55.65 with all three (https://arxiv.org/html/2501.16566).
- Its successor AffectGPT-R1 scored 61.09 with video and text against 63.74 with audio and text (https://arxiv.org/html/2508.01318v3).
- A Qwen2.5-Omni plus LoRA baseline scored 55.29 with all three inputs and 54.48 with video and text (https://arxiv.org/html/2605.19950). The paper labels that metric F1, but it is probably a hit rate.

The sweeps also reported other claims, but no source URL was recorded for them and I did not verify them. They are that speaker-face crops add under a point to text-only weighted F1 in TelME, that video-only Qwen2.5-VL-3B features score below always guessing neutral, and that text-only LLM fine-tunes reach about 69 to 70 weighted F1. Treat them as leads, not facts.

## Biggest open questions

- **No usable baseline yet.** No fitting VLM has a published seven-class weighted F1 on MELD from video. You would have to measure one on MELD dev yourself.
- **Inputs behind the comparison scores.** It is unknown which inputs the OneEmo comparison table gave each model. That makes the InternVL3.5-4B figure hard to read.
- **Audio-free scores for the audio models.** Nano-EmoX and Light-MER have no confirmed MELD number without audio.
- **Possible MELD contamination.** It is unconfirmed whether MELD training clips were kept out of Nano-EmoX's and Light-MER's training. VidEmo includes MELD in its training data. OneEmo's Table I lists MELD as test only.
- **Untested sizes.** No emotion result exists for these:
  - Qwen3.5-2B
  - InternVL3.5-1B and InternVL3.5-2B
  - MiniCPM-V 4.7 1B
  - Ovis2.5-2B
  - Molmo2-4B
  - Gemma 4 E2B and E4B
  - the LFM2.5-VL models
- **Unconfirmed details.**
  - Nano-EmoX's 2.2B total is stated in the paper but not confirmed from its files.
  - Which Nano-EmoX checkpoint matches the paper is unknown.
  - Whether VidEmo's DFEW numbers belong to the 3B or the 7B is unknown.
- **Face handling.** MELD scenes often show several small faces. Whether to crop the speaker's face before the VLM sees the frame is an open design choice.

## Dropped models

- **No downloadable weights:** Emotion-Qwen (anonymous repo only), Facial-R1 (also withdrawn from arXiv), Omni-Emotion (repo says "coming soon"), ActFER, TAG, and AffectGPT-RL (Baidu Pan only). AffectGPT-RL and AffectGPT-R1 would replace AffectGPT if their weights reach Hugging Face.
- **Replaced by a newer model from the same maker:**
  - HumanOmni-7B, by HumanOmniV2
  - Emotion-LLaMA, by v2 (it was also on Google Drive only)
  - EmoCaliber, by EmObserver
  - MiniCPM-V 4.5 and 4.6, by 4.7 1B
  - MiMo-VL-7B-RL, by its 2508 update
  - Moondream 3 Preview, by Moondream 3.1
  - Keye-VL-1.5-8B, by Keye-VL 2.0
  - Cosmos-Reason2, by Cosmos3
- **Over the cap with no emotion evidence:** Ministral 3 8B, Eagle 2.5-8B, Nemotron Nano 12B v2 VL, Nemotron 3 Nano Omni, Phi-4-reasoning-vision-15B, and Ovis2.5-9B. Cosmos3-Nano was also dropped as a 16B robotics world model.
- **Over the cap and face-only:** FaceLLM-8B and FaceLLM-38B, where the 8B scores below its own base on face analysis. Face-LLaVA, at 7.5B on an old Vicuna base with no MELD result.
- **Others:**
  - FLIP and Exp-CLIP were dropped because they are built on the removed CLIP and FLIP has no emotion result.
  - PaliGemma 2 is not a chat model, has no emotion result of its own, and is gated.
  - OmniDimen-v2.5-4B-Emotion looks like a text persona fine-tune and publishes no scores.
  - EmoLA is 2024 work, about 7B, and on Google Drive only.
