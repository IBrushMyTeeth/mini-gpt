"""
Input embeddings for transformer-based language models.

Combines learned token embeddings with learned positional embeddings to
provide the representation used as input to the transformer.
"""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Combines token and positional embeddings.

    Token embeddings represent the identity of each token, while positional
    embeddings provide information about the token's position within the
    sequence. The two representations are combined by element-wise addition.
    """
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_length: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
        )
        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim,
        )

    def forward(
            self,
            token_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape

        positions = torch.arange(sequence_length)

        token_embeddings = self.token_embedding(token_ids)
        positional_embeddings = self.position_embedding(positions)

        return token_embeddings + positional_embeddings