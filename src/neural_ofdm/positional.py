import math

import torch
from torch import nn

POSITIONAL_METHODS = ("circular", "none")


def circular_fourier_features(length: int, dimension: int) -> torch.Tensor:
    if length < 1 or dimension < 2 or dimension % 2:
        raise ValueError("length must be positive and dimension a positive even integer")
    bins = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    harmonics = torch.arange(1, dimension // 2 + 1, dtype=torch.float64).unsqueeze(0)
    phase = 2 * math.pi * bins * harmonics / length
    features = torch.empty(length, dimension, dtype=torch.float64)
    features[:, 0::2] = torch.sin(phase)
    features[:, 1::2] = torch.cos(phase)
    return features.to(torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, method: str, length: int, dimension: int):
        super().__init__()
        if method not in POSITIONAL_METHODS:
            raise ValueError(f"positional method must be one of {POSITIONAL_METHODS}")
        self.method = method
        self.length = length
        table = (
            circular_fourier_features(length, dimension)
            if method == "circular"
            else torch.zeros(length, dimension)
        )
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.length:
            raise ValueError(f"tokens must have shape [B,{self.length},d_model]")
        if self.method == "none":
            return tokens
        return tokens + self.table.to(dtype=tokens.dtype)
