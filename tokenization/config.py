"""
Configuration for the character-level tokenizer.

Defines the tokenizer vocabulary, special tokens, and default configuration
used throughout the project. The vocabulary contains a fixed set of common
characters, while characters outside the vocabulary are represented by the
unknown token.
"""

from dataclasses import dataclass


# Character-level vocabulary used by the tokenizer. Characters outside this
# set are mapped to the unknown token.
CHARACTER_VOCABULARY = (
    # Whitespace
    " ",
    "\t",
    "\n",

    # Lowercase letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",

    # Uppercase letters
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",

    # Numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",

    # Punctuation
    ".", ",", "!", "?", ";", ":", "'", '"',
    "-", "_", "/",

    # Currency / common symbols
    "$", "€", "£",

    # Other common symbols
    "@", "#", "`",

    # Additional punctuation
    "—", "“", "”",
)


@dataclass(frozen=True)
class TokenizerConfig:
    """
    Configuration for a character-level tokenizer.

    The unknown token is always part of the tokenizer vocabulary. Additional
    special tokens, such as BOS or EOS, may optionally be configured.
    """
    vocabulary: tuple[str, ...]
    unk_token: str = "<UNK>"
    special_tokens: tuple[str, ...] = ()


# Default configuration used by the tokenizer.
TOKENIZER_CONFIG = TokenizerConfig(
    vocabulary=CHARACTER_VOCABULARY,
)