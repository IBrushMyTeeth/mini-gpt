"""
Download and preprocess the Tiny Shakespeare dataset.

This module downloads the Tiny Shakespeare dataset, creates a character
tokenizer, and prepares the tokenized dataset with fixed training,
validation, and test splits.

Run from the project root with:

python -m data.download_and_preprocess
"""


from pathlib import Path
from data.utils import download_shakespeare, prepare_dataset
from tokenization.config import TokenizerConfig
from tokenization.tokenizer import CharacterTokenizer


RAW_TEXT_PATH = Path(__file__).parent / "shakespeare.txt"
DATA_PATH = Path(__file__).parent / "shakespeare_char_tokens.pt"
TOKENIZER_CONFIG_PATH = Path(__file__).parent / "shakespeare_char_tokenizer_config.pt"

def main():
    download_shakespeare(RAW_TEXT_PATH)

    config = TokenizerConfig()
    tokenizer = CharacterTokenizer(config)

    prepare_dataset(
        RAW_TEXT_PATH,
        DATA_PATH,
        TOKENIZER_CONFIG_PATH,
        tokenizer,
    )

if __name__ == "__main__":
    main()