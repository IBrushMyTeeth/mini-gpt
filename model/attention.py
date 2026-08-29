""" 
Self-attention and multi-head attention modules.

Provides causal self-attention for autoregressive transformer models and
a multi-head attention module that combines multiple independent attention
heads by concatenating their outputs.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Causal self-attention mechanism.
    
    Projects the input representations into query, key, and value
    representations and computes scaled dot-product attention. A causal
    mask prevents each position from attending to future positions,
    making the module suitable for autoregressive language modeling.
    """

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        self.q = nn.Linear(embedding_dim, attention_dim, bias=True)
        self.k = nn.Linear(embedding_dim, attention_dim, bias=True)
        self.v = nn.Linear(embedding_dim, attention_dim, bias=True)

        self.attention_dim = attention_dim

        self.causal_mask: torch.Tensor
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    max_sequence_length,
                    max_sequence_length,
                    dtype=torch.bool,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute causal scaled dot-product self-attention.
        """

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        seq_length = x.shape[-2]

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.attention_dim ** 0.5)

        mask = self.causal_mask[:seq_length, :seq_length]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(scores, dim=-1)

        output = attention_weights @ v

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    Combines multiple independent self-attention heads, allowing the
    model to attend to different representation subspaces simultaneously.
    Each head receives the same input but learns its own query, key, and
    value projections. The outputs of all attention heads are concatenated
    along the embedding dimension.

    Note:
        In the standard Transformer architecture, the concatenated
        attention heads are passed through an output projection W_0.
        This implementation omits W_0 because num_heads * attention_dim
        is chosen to equal embedding_dim. The concatenated output can
        therefore be passed directly through the residual connection, while the
        subsequent feed-forward network can naturally mix information across
        the different attention heads.
    """

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        max_sequence_length: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [
                SelfAttention(
                    embedding_dim,
                    attention_dim,
                    max_sequence_length
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        Compute and concatenate the outputs of all attention heads.
        """
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)