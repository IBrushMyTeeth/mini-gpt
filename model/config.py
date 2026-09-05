"""
Configuration for the GPT language model.

Defines the architectural parameters used to construct the model, including
embedding dimensions, attention dimensions, sequence length, and the number
of transformer layers.
"""


from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
        """
        Configuration parameters for the GPT language model.
        """
        vocabulary_size: int
        embedding_dim: int = 256
        attention_dim: int = 64
        max_sequence_length: int = 128
        num_heads: int = 4
        hidden_dim: int = 1024
        num_layers: int = 4