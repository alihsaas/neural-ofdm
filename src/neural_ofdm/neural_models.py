import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def activation_layer(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class AffineNormalization(nn.Module):
    def __init__(self, offset: torch.Tensor | None = None, scale: torch.Tensor | None = None):
        super().__init__()
        offset = torch.zeros(1, 2, 1) if offset is None else offset.reshape(1, 2, 1)
        scale = torch.ones(1, 2, 1) if scale is None else scale.reshape(1, 2, 1)
        if torch.any(scale <= 0) or not torch.all(torch.isfinite(scale)):
            raise ValueError("normalization scale must be finite and positive")
        self.register_buffer("offset", offset.to(torch.float32).clone())
        self.register_buffer("scale", scale.to(torch.float32).clone())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.offset) / self.scale

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.offset


class ResidualBlock(nn.Module):
    def __init__(
        self, width: int, kernel_size: int, padding: str, activation: str, dilation: int = 1
    ):
        super().__init__()
        self.first = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            padding_mode=padding,
        )
        self.second = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            padding_mode=padding,
        )
        self.inner_activation = activation_layer(activation)
        self.output_activation = activation_layer(activation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.second(self.inner_activation(self.first(values)))
        return self.output_activation(values + update)


class ChannelCNN(nn.Module):
    def __init__(
        self,
        width: int = 32,
        blocks: int = 3,
        kernel_size: int = 5,
        activation: str = "relu",
        padding: str = "circular",
        residual: bool = True,
        zero_init_output: bool = True,
        normalization: AffineNormalization | None = None,
        dilations: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        dilations = [1] * blocks if dilations is None else list(dilations)
        if len(dilations) != blocks or any(int(value) < 1 for value in dilations):
            raise ValueError("dilations must list one positive integer per residual block")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if padding not in ("circular", "zeros", "reflect", "replicate"):
            raise ValueError("unsupported Conv1d padding mode")
        if width < 1 or blocks < 0:
            raise ValueError("width must be positive and blocks nonnegative")
        self.residual = residual
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.dilations = tuple(int(value) for value in dilations)
        self.normalization = normalization or AffineNormalization()
        self.input_layer = nn.Conv1d(
            2, width, kernel_size, padding=kernel_size // 2, padding_mode=padding
        )
        self.input_activation = activation_layer(activation)
        self.body = nn.Sequential(
            *[
                ResidualBlock(width, kernel_size, padding, activation, dilation)
                for dilation in self.dilations
            ]
        )
        self.output_layer = nn.Conv1d(
            width, 2, kernel_size, padding=kernel_size // 2, padding_mode=padding
        )
        if zero_init_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * (2 + 2 * sum(self.dilations))

    def forward(self, ls_features: torch.Tensor) -> torch.Tensor:
        if ls_features.ndim != 3 or ls_features.shape[1] != 2:
            raise ValueError("input must have shape [B,2,N]")
        normalized = self.normalization(ls_features)
        hidden = self.body(self.input_activation(self.input_layer(normalized)))
        output = self.output_layer(hidden)
        if self.residual:
            output = normalized + output
        return self.normalization.inverse(output)


class ChannelMLP(nn.Module):
    def __init__(
        self,
        fft_size: int = 64,
        hidden: int = 128,
        activation: str = "relu",
        residual: bool = False,
        normalization: AffineNormalization | None = None,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.residual = residual
        self.normalization = normalization or AffineNormalization()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * fft_size, hidden),
            activation_layer(activation),
            nn.Linear(hidden, 2 * fft_size),
        )

    @property
    def receptive_field(self) -> int:
        return self.fft_size

    def forward(self, ls_features: torch.Tensor) -> torch.Tensor:
        if ls_features.ndim != 3 or ls_features.shape[1:] != (2, self.fft_size):
            raise ValueError("input must have shape [B,2,N] with the configured N")
        normalized = self.normalization(ls_features)
        output = self.network(normalized).reshape(-1, 2, self.fft_size)
        if self.residual:
            output = normalized + output
        return self.normalization.inverse(output)


def build_model(
    model_config: dict, fft_size: int, normalization: AffineNormalization | None = None
) -> nn.Module:
    settings = dict(model_config)
    architecture = settings.pop("architecture")
    settings.pop("normalization", None)
    if architecture == "cnn":
        return ChannelCNN(**settings, normalization=normalization)
    if architecture == "mlp":
        return ChannelMLP(fft_size=fft_size, **settings, normalization=normalization)
    if architecture == "transformer":
        from neural_ofdm.transformer import ChannelTransformer

        return ChannelTransformer(fft_size=fft_size, **settings, normalization=normalization)
    if architecture == "hybrid":
        from neural_ofdm.hybrid import ChannelHybrid

        return ChannelHybrid(fft_size=fft_size, **settings, normalization=normalization)
    raise ValueError("architecture must be cnn, mlp, transformer or hybrid")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
