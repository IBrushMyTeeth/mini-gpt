"""
Pre-tokenizes the Tiny Shakespeare dataset and saves the resulting token
IDs for reuse during training.

The saved representation includes the tokenizer configuration used to
generate the tokens.

Run from the project root with:

python -m data.save_data_as_tokens

"""


import torch
from pathlib import Path

from tokenization.config import TOKENIZER_CONFIG
from tokenization.tokenizer import CharacterTokenizer


TEXT_PATH = Path(__file__).parent / "shakespeare.txt"
TOKENS_PATH = Path(__file__).parent / "shakespeare_tokens.pt"

def main() -> None:
    if not TEXT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {TEXT_PATH}. "
            "Download Tiny Shakespeare first."
        )

    if TOKENS_PATH.exists():
        print(f"Tokens already exists at {TOKENS_PATH}")
        return

    text = TEXT_PATH.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer(TOKENIZER_CONFIG)

    tokens = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    torch.save(
        {
            "tokens": tokens,
            "tokenizer_config": TOKENIZER_CONFIG,
        },
        TOKENS_PATH
    )

    print(f"Saved {len(tokens):,} tokens to {TOKENS_PATH}")


if __name__ == "__main__":
    main()
    