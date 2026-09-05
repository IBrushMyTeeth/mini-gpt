"""
GPT-style language model.

Provides a decoder-only transformer language model that combines token and
positional embeddings, causal transformer blocks, layer normalization, and a
linear projection to produce vocabulary logits.
"""


from pathlib import Path

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.embeddings import InputEmbedding
from model.transformer import TransformerBlock

class GPT(nn.Module):
    """
    Decoder-only transformer language model.

    The model maps input token IDs to vocabulary logits through an input
    embedding layer, a stack of causal transformer blocks, and a final
    normalization and linear projection layer.

    The model configuration is stored alongside the model weights when saving,
    allowing the model to be reconstructed when loading.
    """
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_embedding = InputEmbedding(
            config.vocabulary_size,
            config.max_sequence_length,
            config.embedding_dim
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embedding_dim,
                    config.attention_dim,
                    config.hidden_dim,
                    config.max_sequence_length,
                    config.num_heads,
                )

                for _ in range(config.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.linear_projection = nn.Linear(
            config.embedding_dim,
            config.vocabulary_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the language model.
        """
        x = self.input_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.linear_projection(x)

        return logits

    def save(self, path: Path) -> None:
        """
        Save the model weights and configuration to a file.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "model_config": {
                    "vocabulary_size": self.config.vocabulary_size,
                    "embedding_dim": self.config.embedding_dim,
                    "attention_dim": self.config.attention_dim,
                    "max_sequence_length": self.config.max_sequence_length,
                    "num_heads": self.config.num_heads,
                    "hidden_dim": self.config.hidden_dim,
                    "num_layers": self.config.num_layers,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "GPT":
        """
        Load a GPT model from a saved file.

        The model configuration is reconstructed from the saved configuration
        before loading the model weights.
        """
        state = torch.load(path, weights_only=True)

        config = ModelConfig(**state["model_config"])

        model = cls(config)
        model.load_state_dict(state["state_dict"])

        return model