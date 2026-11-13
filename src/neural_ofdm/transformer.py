import torch
from torch import nn

from neural_ofdm.neural_models import AffineNormalization
from neural_ofdm.positional import PositionalEncoding


def validate_attention_mask(mask: torch.Tensor | None, length: int) -> None:
    if mask is None:
        return
    if mask.dtype != torch.bool or mask.shape != (length, length):
        raise ValueError(f"attention_mask must be a boolean [{length},{length}] tensor")
    if bool(torch.any(torch.all(mask, dim=-1))):
        raise ValueError("every query must keep at least one unblocked key")


class EncoderStack(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 128,
        dropout: float = 0.05,
        activation: str = "gelu",
    ):
        super().__init__()
        if d_model < 1 or heads < 1 or d_model % heads:
            raise ValueError("d_model must be a positive multiple of heads")
        if layers < 1 or feedforward < 1 or not 0 <= dropout < 1:
            raise ValueError("layers and feedforward must be positive and dropout in [0,1)")
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be relu or gelu")
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    heads,
                    feedforward,
                    dropout,
                    activation,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_attention_mask(attention_mask, tokens.shape[1])
        for layer in self.layers:
            tokens = layer(tokens, src_mask=attention_mask)
        return self.final_norm(tokens)

    def forward_with_attention(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_attention_mask(attention_mask, tokens.shape[1])
        maps = []
        for layer in self.layers:
            normalized = layer.norm1(tokens)
            attended, weights = layer.self_attn(
                normalized,
                normalized,
                normalized,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            tokens = tokens + layer.dropout1(attended)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(tokens))))
            )
            tokens = tokens + layer.dropout2(feedforward)
            maps.append(weights)
        return self.final_norm(tokens), torch.stack(maps, dim=1)


class ChannelTransformer(nn.Module):
    def __init__(
        self,
        fft_size: int = 64,
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
        self.fft_size = fft_size
        self.residual = residual
        self.normalization = normalization or AffineNormalization()
        self.input_projection = nn.Linear(2, d_model)
        self.position = PositionalEncoding(positional, fft_size, d_model)
        self.encoder = EncoderStack(d_model, heads, layers, feedforward, dropout, activation)
        self.output_projection = nn.Linear(d_model, 2)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @property
    def receptive_field(self) -> int:
        return self.fft_size

    def tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.position(self.input_projection(normalized.transpose(1, 2)))

    def finish(self, normalized: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        output = self.output_projection(encoded).transpose(1, 2)
        if self.residual:
            output = normalized + output
        return self.normalization.inverse(output)

    def check(self, ls_features: torch.Tensor) -> None:
        if ls_features.ndim != 3 or ls_features.shape[1:] != (2, self.fft_size):
            raise ValueError("input must have shape [B,2,N] with the configured N")

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
