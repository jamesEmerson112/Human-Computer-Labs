# Reading list for the MELD emotion prototype

Compiled 2026-10-01 for the ML route. Every link below loaded on that date. The list is ordered by group, not by priority. The top three picks are marked at the end.

The plan these readings support is the one discussed so far. Pretrained models stay frozen, which means their numbers are left as their makers trained them. A small head is trained on MELD to pick one of its seven emotions. A pretrained language model writes the short reply from the words and the predicted emotion.

## Background

1. [3Blue1Brown, Neural Networks series](https://www.3blue1brown.com/topics/neural-networks). These are animated videos that show what a neuron, a layer, a gradient and backpropagation are. The later chapters cover transformers and attention. Watch them if the vocabulary in the other readings feels unfamiliar.
2. [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course). Its classification module explains accuracy, precision and recall in plain terms. Those measures are what the write-up will report.
3. [PyTorch, Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). It walks through tensors, datasets, models, automatic gradients and the training loop in PyTorch, the library the head will be written in.
4. [Andrej Karpathy, A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). This is practical advice on training a model without fooling yourself. It says to look at the data first, build the whole pipeline with a simple baseline, and only then make the model more complex.

## Pretrained models and training a head

5. [Hugging Face LLM Course, chapter 1](https://huggingface.co/learn/llm-course/chapter1/1). It explains what a transformer is and what "pretrained" means. It also explains the difference between encoder models such as RoBERTa, which read text and summarize it, and decoder models such as Qwen, which write text.
6. [Hugging Face LLM Course, chapter 3](https://huggingface.co/learn/llm-course/chapter3/1). It fine-tunes a BERT model for text classification with the Transformers library, both with the built-in Trainer and with a hand-written training loop.
7. [PyTorch, Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). It compares retraining a whole pretrained network with freezing it and training only a new final layer. That second option, which the tutorial calls a fixed feature extractor, is our plan. The example uses images, but the idea carries over to text and audio unchanged.
8. [Transformers, text classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification). This is a short, complete recipe for training a text classifier. It is useful as a template for the text side of the head.

## The MELD task

9. [MELD paper (Poria et al., ACL 2019)](https://aclanthology.org/P19-1050/). It describes how the dataset was built from Friends and defines the seven emotion labels. It also shows how unevenly the labels are spread, since neutral dominates and fear and disgust are rare, and it reports the baseline scores to compare against.
10. [MELD GitHub repository](https://github.com/declare-lab/MELD). It covers the download links, the CSV columns and the naming of the video clips. Its issues page lists known problems with a few clips.
11. [Emotion Recognition in Conversation: Research Challenges, Datasets, and Recent Advances (Poria et al., 2019)](https://arxiv.org/abs/1905.02947). This survey explains why emotion in a conversation depends on earlier lines and on who is speaking. It also explains why combining modalities helps.
12. [scikit-learn, classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html). It documents the weighted F1 score, which balances precision and recall for each emotion and weights each emotion by how often it appears. MELD papers report that score. The page also documents the confusion matrix, a table showing which emotions get mistaken for which.

## Text plus audio track

13. [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction). Units 1 to 4 cover how audio is stored and how transformers read it. Unit 4 trains an audio classifier. Unit 5 covers speech recognition.
14. [Transformers, audio classification task guide](https://huggingface.co/docs/transformers/tasks/audio_classification). It is a short recipe that fine-tunes Wav2Vec2 to classify short spoken clips, and it is the closest template for the audio side of the head.
15. [faster-whisper](https://github.com/SYSTRAN/faster-whisper). It is a faster build of OpenAI's Whisper speech recognizer that can run on a CPU at reduced precision, and it turns live speech into the text the model reads.
16. [Silero VAD](https://github.com/snakers4/silero-vad). It is a small voice activity detection model, meaning it tells when someone starts and stops talking. That is how live audio gets cut into separate lines before the model sees them.

## Text plus vision track

17. [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome). It covers how images become numbers a model can use, from convolutional networks to vision transformers.
18. [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker). It finds faces in video frames on a CPU and gives scores for facial movements such as a smile or raised brows. Those scores could feed the head directly.

## Writing the reply and fitting in memory

19. [Transformers, generation with LLMs](https://huggingface.co/docs/transformers/llm_tutorial). It shows how a language model writes text one word piece at a time, and which settings control length and randomness.
20. [Transformers, quantization overview](https://huggingface.co/docs/transformers/quantization/overview). It compares the ways of storing a model's numbers at lower precision so that the model uses less memory. The parameter count, and so the 6B cap, stays the same.

## Top three

The picks are the MELD paper (9), chapters 1 and 3 of the Hugging Face LLM Course (5 and 6), and the PyTorch transfer learning tutorial (7). They hold whichever track is chosen. Once a track is picked, the next reading is unit 4 of the Audio Course (13) or the Computer Vision Course (17).

## Related find, not part of the list

[CesarAugusto00/Multimodal-Emotion-Recognition-with-MELD](https://github.com/CesarAugusto00/Multimodal-Emotion-Recognition-with-MELD) describes itself as a three-day prototype on MELD text plus audio. It feeds a structured emotional state to Llama 3.2 1B through Ollama for a short reply. It is very likely an earlier candidate's public answer to this same brief. Its Future Work section lists microphone input, voice activity detection and streaming as not done. Reading it to see the shape of a finished answer is fine. Copying from it is not, and anything borrowed must be cited, because the brief asks submitters to identify external components and own their results.
