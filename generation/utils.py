"""
Text generation utilities.

Provides functions for generating text completions from a GPT model using
tokenized text prompts and decoding the generated token sequences back into
strings.
"""

import torch

from model.gpt import GPT
from tokenization.tokenizer import Tokenizer

def generate_completions(
        model: GPT,
        prompts: list[str],
        tokenizer: Tokenizer,
        new_tokens: int,
) -> list[str]:
    """
    Generate text completions for a collection of prompts.

    Each prompt is encoded into token IDs, passed to the GPT model for
    autoregressive generation, and decoded back into text.
    """
    encoded = [
        torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        for prompt in prompts
    ]

    completed = [
        model.generate(encoded_prompt, new_tokens)
        for encoded_prompt in encoded
    ]

    decoded = [
        tokenizer.decode(generated.squeeze(0).tolist())
        for generated in completed
    ]

    return decoded

def generate_completions_with_temp(
        model: GPT,
        prompts: list[str],
        tokenizer: Tokenizer,
        new_tokens: int,
        temperature: float,
) -> list[str]:

    encoded = [
        torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        for prompt in prompts
    ]

    completed = [
        model.generate_with_temp(encoded_prompt, new_tokens, temperature)
        for encoded_prompt in encoded
    ]

    decoded = [
        tokenizer.decode(generated.squeeze(0).tolist())
        for generated in completed
    ]

    return decoded