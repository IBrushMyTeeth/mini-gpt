import torch
import torch.nn as nn
from model.embeddings import InputEmbedding
from model.transformer import TransformerBlock
from model.config import ModelConfig
from pathlib import Path

class GPT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary_size: int
    ) -> None:
        super().__init__()

        self.config = config
        self.vocabulary_size = vocabulary_size

        self.input_embedding = InputEmbedding(
            vocabulary_size,
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
            vocabulary_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.linear_projection(x)

        return logits

    def save(self, path: Path) -> None:

        torch.save({
            "state_dict" : self.state_dict(),
            "config" : self.config,
            "vocabulary_size" : self.vocabulary_size
        }, path)

    @classmethod
    def load(cls, path: Path) -> "GPT":

        state = torch.load(path)

        state_dict = state["state_dict"]
        config = state["config"]
        vocabulary_size = state["vocabulary_size"]

        model = cls(config, vocabulary_size)
        model.load_state_dict(state_dict)

        return model