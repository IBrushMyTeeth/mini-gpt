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
        config: TokenizerConfig,
    ) -> None:

        self.cfg = config
        vocabulary = (
            config.vocabulary + config.special_tokens + (config.unk_token,))

        self._validate_unique_tokens(vocabulary)
        self._vocabulary = vocabulary


        self._token_to_id = {
            token: token_id
            for token_id, token in enumerate(self._vocabulary)
        }

        self._id_to_token = {
            token_id: token
            for token_id, token in enumerate(self._vocabulary)
        }

    def _validate_unique_tokens(
        self,
        vocabulary: tuple[str, ...],
    ) -> None:
        """Ensure no duplicate tokens are present in the vocabulary."""

        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Vocabulary contains duplicate tokens.")

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocabulary)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """Convert text into a sequence of token IDs."""

        unknown_id = self._token_to_id[self.cfg.unk_token]

        return [
            self._token_to_id.get(character, unknown_id)
            for character in text
        ]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """
        Convert a sequence of token IDs back into text.
        
        Raises:
            KeyError: If a token ID is invalid.
        """

        return "".join(
            self._id_to_token[token_id]
            for token_id in token_ids
        )