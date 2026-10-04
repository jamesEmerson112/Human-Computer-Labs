"""Runs the sentiment-analysis and text-generation examples from the Hugging Face LLM Course,
chapter 1, section 3 (https://huggingface.co/learn/llm-course/chapter1/3).

Contents:
- classifier: a pretrained sentiment-analysis pipeline.
- Three calls that print the top label for one sentence, the top label for two sentences, and
  every label's score for each of the two sentences.
- generator: a text-generation pipeline running SmolLM2-360M.
- One call that prints two random continuations of a prompt, each at most 30 tokens long.
"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("I've been waiting for a HuggingFace course my whole life."))

print(classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]))

print(classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"],
                 top_k=None))

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

print(generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
))
