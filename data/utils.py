"""
Utilities for downloading, inspecting, and preprocessing the Tiny Shakespeare dataset.

This module provides reusable functions to:

download the Tiny Shakespeare dataset;
load the dataset from disk;
inspect basic dataset statistics; and
tokenize and save the dataset for reuse during training.

The functions operate on explicitly provided file paths and do not perform
any work when this module is imported.
"""


from pathlib import Path
from urllib.request import urlopen

import torch
from tokenization.tokenizer import CharacterTokenizer


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_shakespeare(path: Path) -> None:
    """
    Download the Tiny Shakespeare dataset used for the NLP project.

    The dataset is downloaded from Andrej Karpathy's char-rnn repository
    and saved to the specified path.

    The function is safe to execute multiple times: if the dataset already
    exists locally, it will not be downloaded again.
    """

    if path.exists():
        print(f"Dataset already exists at {path}")
        return

    print("Downloading Shakespeare dataset...")

    with urlopen(SHAKESPEARE_URL) as response:
        text = response.read().decode("utf-8")

    path.write_text(text, encoding="utf-8")

    print(f"Downloaded {len(text):,} characters.")

def load_data(path: Path) -> str:
    """Load the dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")

    data = path.read_text(encoding="utf-8")
    return data

def inspect_data(path: Path) -> None:
    """
    Inspect the downloaded Shakespeare dataset and print basic statistics.
    """

    data = load_data(path)

    text_length = len(data)
    unique_chars = sorted(set(data))

    print("=" * 40)
    print("Summary:")
    print()
    print(f"Number of characters: {text_length}")
    print(f"Number of unique characters: {len(unique_chars)}")
    print()
    print("Vocabulary:")
    print(unique_chars)

    print()
    print("First 500 characters:")
    print(data[:500])

def save_as_tokens(
        text_path: Path,
        tokens_path: Path,
        tokenizer: CharacterTokenizer,

) -> None:
    """
    Pre-tokenizes the specified dataset and saves the resulting token
    IDs for reuse during training.

    The saved representation includes the tokenizer configuration and
    vocabulary_size used to generate the tokens.
    """
    if not text_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {text_path}."
            "Download Tiny Shakespeare first."
        )

    if tokens_path.exists():
        print(f"Tokens already exists at {tokens_path}")
        return

    data = load_data(text_path)

    tokens = torch.tensor(
        tokenizer.encode(data),
        dtype=torch.long,
    )

    torch.save(
        {
            "tokens": tokens,
            "tokenizer_config": {
                "vocabulary": tokenizer.config.vocabulary,
                "unk_token": tokenizer.config.unk_token,
                "special_tokens": tokenizer.config.special_tokens,
            },
            "vocabulary_size": tokenizer.vocabulary_size,
        },
        tokens_path,
    )

    print(f"Saved {len(tokens):,} tokens to {tokens_path}")