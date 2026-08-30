"""
PyTorch dataset wrapper for next-token language-model training.

The dataset converts a one-dimensional sequence of token IDs into
overlapping input/target pairs. Each input contains context_length
tokens, while the corresponding target is shifted by one token and
is therefore used for next-token prediction.
"""

import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    """
    Dataset for training a character-level language model.
    
    The dataset uses a sliding window over a sequence of token IDs to
    construct training examples for autoregressive next-token prediction.
    """
    def __init__(
        self,
        token_ids: torch.Tensor,
        context_length: int,
    ) -> None:
        super().__init__()

        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor.")

        if context_length <= 0:
            raise ValueError("context_length must be a positive integer.")

        if len(token_ids) <= context_length:
            raise ValueError("token_ids must be longer than context_length.")

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.context_length]
        y = self.token_ids[idx + 1 : idx + self.context_length + 1]

        return x, y