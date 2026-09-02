"""
Transformer block module.

Provides a pre-normalization transformer block consisting of causal multi-head
self-attention and a feed-forward MLP. Layer normalization and residual
connections are applied around each sub-block.
"""


import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from mlp import MLP


class TransformerBlock(nn.Module):
    """
    Pre-normalization transformer block.
    
    The block consists of two sub-blocks: a causal multi-head self-attention
    layer followed by a feed-forward MLP. Each sub-block is preceded by layer
    normalization and followed by a residual connection.
    """
    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        hidden_dim: int,
        max_sequence_length: int,
        num_heads: int,

    ) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        self.multi_headed_attention = MultiHeadAttention(
            embedding_dim,
            attention_dim,
            max_sequence_length,
            num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-block
        residual = x

        x = self.layer_norm_1(x)
        x = self.multi_headed_attention(x)
        x = residual + x

        # MLP sub-block
        residual = x

        x = self.layer_norm_2(x)
        x = self.mlp(x)
        x = x + residual

        return x