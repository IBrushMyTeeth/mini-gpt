"""
Character-level tokenizer implementation.

Provides a tokenizer that maps individual characters to integer token IDs
and can reconstruct text from those token IDs using a fixed vocabulary and
optional special tokens.
"""

from tokenization.config import TokenizerConfig


class CharacterTokenizer:
    """
    Tokenizer that represents text as a sequence of character tokens.

    The tokenizer assigns a unique integer ID to every vocabulary character,
    configured special token, and the required unknown token. Characters not
    present in the vocabulary are mapped to the unknown token during encoding.
    """
    def __init__(
        self,
        cfg: TokenizerConfig,
    ) -> None:

        self.cfg = cfg
        self.tokens = cfg.vocabulary + cfg.special_tokens + (cfg.unk_token,)

        self.token_to_id = {
            token: token_id
            for token_id, token in enumerate(self.tokens)
        }

        self.id_to_token = {
            token_id: token
            for token_id, token in enumerate(self.tokens)
        }

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """Convert text into a sequence of token IDs."""

        unknown_id = self.token_to_id[self.cfg.unk_token]

        return [
            self.token_to_id.get(character, unknown_id)
            for character in text
        ]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """Convert a sequence of token IDs back into text."""

        return "".join(
            self.id_to_token[token_id]
            for token_id in token_ids
        )