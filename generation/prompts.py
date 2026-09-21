"""
Prompt collections for evaluating text generation.

Provides Shakespeare-specific, challenging, and generic English prompts for
testing the language model's text generation capabilities.
"""


SHAKESPEARE_PROMPTS = [
    "KING HENRY:\nWhat news from the north, good my lord?",
    "Now is the winter of our discontent",
    "The night is dark",
    "To be, or not to be, that is the question:",
    "Fie, fie, upon this quiet life! I want work.",
    "FIRST CITIZEN:\nBefore we proceed any further, hear me speak.",
    "The king shall",
    "Romeo:",
]

CHALLENGING = [
    "The stock market crashed on Monday morning, and",
    "The wizard opened his smartphone and cast a spell that read:",
    "The weather forecast predicts rain for the weekend, so",
    "The president announced a new policy on climate change,",
    "The robot looked at the king and said",
    "The scientists in the lab discovered a new species of",
]

GENERIC_ENGLISH = [
    "Once upon a time, there was a",
    "It was a dark and stomy night",
    "In the beginning, there was",
    "I looked out the window and",
    "In the middle of the forest,",
]