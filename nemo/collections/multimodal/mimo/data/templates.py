import json


COMPHREHENSION_PROMPTS = [
    "Summarize the content of this {}.",  # this can be Summarize the content of this image, Summarize the content of this video etc.
    "Provide a brief description of the {} shown.",
    "What is the main subject of the {}?",
    "Describe the {} concisely",
    "What is happening in this {}?",
    "Give a concise overview of the {}.",
    "What does the {} depict?",
    "Highlight the central theme of the {}.",
    "What is the focus of this {}?",
    "Summarize the visual information presented here in {}.",
    "Summarize {}.",
    "Explain in detail {}.",
    "What isd shown here in {}.",
    "What is the message here in this {}.",
]

GENERATION_PROMPTS = ["I want to {} a {} of ", "Please {} a {} of a ", "{} a {} of ", "Can you {} a {} of "]

IMAGE_KEYWORDS = [
    "image",
    "picture",
    "potrait",
    "illustration",
]

GENERATION_KEYWORDS = [
    "generate",
    "create",
    "compose",
    "illustrate",
]

RESPONSES = [
    "Sure, here is the {}.",
    "How about this {}.",
    "Here is an {} you asked.",
    "Of course!",
    "Certainly!",
    "Sure!",
    "Absolutely, here is the {}",
    "Sure thing!",
    "Here is an {}.",
]
