import torch
from torch import nn

from neural_ofdm.neural_models import AffineNormalization, ResidualBlock, activation_layer
from neural_ofdm.positional import PositionalEncoding
from neural_ofdm.transformer import EncoderStack


class ChannelHybrid(nn.Module):
    def __init__(
        self,
        fft_size: int = 64,
        conv_width: int = 32,
        conv_blocks: int = 1,
        kernel_size: int = 5,
        conv_activation: str = "relu",
        padding: str = "circular",
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 128,
        dropout: float = 0.05,
        activation: str = "gelu",
        positional: str = "circular",
        residual: bool = True,
        zero_init_output: bool = True,
        normalization: AffineNormalization | None = None,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.fft_size = fft_size
        self.kernel_size = kernel_size
        self.conv_blocks = conv_blocks
        self.residual = residual
        self.normalization = normalization or AffineNormalization()
        self.input_layer = nn.Conv1d(
            2, conv_width, kernel_size, padding=kernel_size // 2, padding_mode=padding
        )
        self.input_activation = activation_layer(conv_activation)
        self.conv_body = nn.Sequential(
            *[
                ResidualBlock(conv_width, kernel_size, padding, conv_activation)
                for _ in range(conv_blocks)
            ]
        )
        self.token_projection = nn.Linear(conv_width, d_model)
        self.position = PositionalEncoding(positional, fft_size, d_model)
        self.encoder = EncoderStack(d_model, heads, layers, feedforward, dropout, activation)
        self.output_projection = nn.Linear(d_model, 2)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @property
    def receptive_field(self) -> int:
        return self.fft_size

    @property
    def convolution_receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * (1 + 2 * self.conv_blocks)

    def check(self, ls_features: torch.Tensor) -> None:
        if ls_features.ndim != 3 or ls_features.shape[1:] != (2, self.fft_size):
            raise ValueError("input must have shape [B,2,N] with the configured N")

    def tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        features = self.conv_body(self.input_activation(self.input_layer(normalized)))
        return self.position(self.token_projection(features.transpose(1, 2)))

    def finish(self, normalized: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        output = self.output_projection(encoded).transpose(1, 2)
        if self.residual:
            output = normalized + output
        return self.normalization.inverse(output)

    def forward(
        self, ls_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.check(ls_features)
        normalized = self.normalization(ls_features)
        return self.finish(normalized, self.encoder(self.tokens(normalized), attention_mask))

    def forward_with_attention(
        self, ls_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.check(ls_features)
        normalized = self.normalization(ls_features)
        encoded, maps = self.encoder.forward_with_attention(self.tokens(normalized), attention_mask)
        return self.finish(normalized, encoded), maps
