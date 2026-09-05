from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
        vocabulary_size: int
        embedding_dim: int = 256
        attention_dim: int = 64
        max_sequence_length: int = 128
        num_heads: int = 4
        hidden_dim: int = 1024
        num_layers: int = 4