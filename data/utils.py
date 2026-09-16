"""
Utilities for downloading, inspecting, and preprocessing the Tiny Shakespeare dataset.

This module provides reusable functions to:

download the Tiny Shakespeare dataset;
load the dataset from disk;
inspect basic dataset statistics; and
tokenize, split, and save the dataset for reuse during training.

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

def prepare_dataset(
        raw_text_path: Path,
        data_path: Path,
        tokenizer_path: Path,
        tokenizer: CharacterTokenizer,
        train_size: float = 0.9,
        validation_size: float = 0.05,
) -> None:
    """
    Tokenize, split, and save the Shakespeare dataset.

    The tokenized dataset is split sequentially into training, validation,
    and test sets. The resulting splits, split ratios, and vocabulary size
    are saved to tokens_path.

    The tokenizer configuration is saved separately to tokenizer_path
    so that it can be reused independently for encoding and decoding text.
    """
    if not raw_text_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_text_path}. "
            "Download Tiny Shakespeare first."
        )

    if data_path.exists() and tokenizer_path.exists():
        print("Processed dataset and tokenizer already exist.")
        return

    if train_size <= 0 or validation_size <= 0:
        raise ValueError(
            "train_size and validation_size must be greater than 0."
        )

    if train_size + validation_size >= 1:
        raise ValueError(
            "train_size and validation_size must sum to less than 1."
        )

    data = load_data(raw_text_path)

    tokens = torch.tensor(
        tokenizer.encode(data),
        dtype=torch.long,
    )

    train_end = int(len(tokens) * train_size)
    validation_end = int(
        len(tokens) * (train_size + validation_size)
    )

    train_tokens = tokens[:train_end]
    validation_tokens = tokens[train_end:validation_end]
    test_tokens = tokens[validation_end:]

    torch.save(
        {
            "train_split": train_tokens,
            "validation_split": validation_tokens,
            "test_split": test_tokens,
            "split_ratio": {
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": 1.0 - train_size - validation_size,
            },
            "vocabulary_size": tokenizer.vocabulary_size,
        },
        data_path,
    )

    torch.save(
        {
            "vocabulary": tokenizer.config.vocabulary,
            "unk_token": tokenizer.config.unk_token,
            "special_tokens": tokenizer.config.special_tokens,
        }, tokenizer_path
    )

    print(f"Saved dataset to {data_path}")
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(validation_tokens):,}")
    print(f"Test tokens: {len(test_tokens):,}")
    print(f"Saved tokenizer to {tokenizer_path}")