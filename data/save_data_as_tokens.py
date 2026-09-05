"""
Pre-tokenizes the Tiny Shakespeare dataset and saves the resulting token
IDs for reuse during training.

The saved representation includes the tokenizer configuration and
vocabulary_size used to generate the tokens.

Run from the project root with:

python -m data.save_data_as_tokens

"""


import torch
from pathlib import Path

from tokenization.config import TokenizerConfig
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

    tokenizer_config = TokenizerConfig()
    tokenizer = CharacterTokenizer(tokenizer_config)

    tokens = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    torch.save(
        {
            "tokens": tokens,
            "tokenizer_config": {
                "vocabulary": tokenizer_config.vocabulary,
                "unk_token": tokenizer_config.unk_token,
                "special_tokens": tokenizer_config.special_tokens,
            },
            "vocabulary_size": tokenizer.vocabulary_size,
        },
        TOKENS_PATH,
    )

    print(f"Saved {len(tokens):,} tokens to {TOKENS_PATH}")


if __name__ == "__main__":
    main()
    