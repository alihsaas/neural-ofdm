from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neural_ofdm._validation import complex_vector, integer
from neural_ofdm.channel import ChannelConfig, reference_noise_variance
from neural_ofdm.modulation import bits_per_symbol


@dataclass(frozen=True)
class OFDMConfig:
    fft_size: int = 64
    cp_length: int = 16
    pilot_spacing: int = 4
    pilot_value: complex = (1 + 1j) / np.sqrt(2)
    modulation_order: int = 4
    snr_db: float = 20.0
    random_seed: int = 2026
    channel_model: Literal["fixed", "rayleigh"] = "fixed"
    channel_taps: tuple[complex, ...] = (1 + 0j, 0.35 + 0.15j, 0.12 - 0.10j)
    rayleigh_powers: tuple[float, ...] = (1.0, 0.4, 0.15, 0.05)
    channel: ChannelConfig | None = None
    noise_mode: Literal["received_snr", "fixed"] = "received_snr"
    reference_esn0_db: float = 20.0
    pilot_values: tuple[complex, ...] | None = None

    def __post_init__(self) -> None:
        integer(self.fft_size, "fft_size", 2)
        integer(self.cp_length, "cp_length")
        integer(self.pilot_spacing, "pilot_spacing", 2)
        integer(self.random_seed, "random_seed")
        if self.cp_length > self.fft_size:
            raise ValueError("cp_length cannot exceed fft_size")
        if self.pilot_spacing >= self.fft_size:
            raise ValueError("at least two pilots are required for interpolation")
        bits_per_symbol(self.modulation_order)
        if not np.isfinite(self.pilot_value) or abs(self.pilot_value) == 0:
            raise ValueError("pilot_value must be finite and nonzero")
        if np.isnan(self.snr_db) or np.isneginf(self.snr_db):
            raise ValueError("snr_db must be finite or +inf")
        if self.channel_model not in ("fixed", "rayleigh"):
            raise ValueError("channel_model must be 'fixed' or 'rayleigh'")
        if not isinstance(self.channel_taps, tuple) or not isinstance(self.rayleigh_powers, tuple):
            raise ValueError("channel_taps and rayleigh_powers must be immutable tuples")
        taps = complex_vector(self.channel_taps, "channel_taps")
        if np.sum(np.abs(taps) ** 2) == 0:
            raise ValueError("channel_taps must have nonzero energy")
        powers = np.asarray(self.rayleigh_powers, dtype=float)
        if powers.ndim != 1 or powers.size == 0 or not np.all(np.isfinite(powers)):
            raise ValueError("rayleigh_powers must be a finite nonempty vector")
        if np.any(powers < 0) or powers.sum() <= 0:
            raise ValueError("rayleigh_powers must be nonnegative with positive total power")
        if self.noise_mode not in ("received_snr", "fixed"):
            raise ValueError("noise_mode must be 'received_snr' or 'fixed'")
        reference_noise_variance(self.reference_esn0_db)
        if self.channel is not None and not isinstance(self.channel, ChannelConfig):
            raise ValueError("channel must be a ChannelConfig")
        if self.pilot_values is not None:
            values = complex_vector(self.pilot_values, "pilot_values")
            if (
                not isinstance(self.pilot_values, tuple)
                or values.size != self.pilot_indices.size
                or np.any(values == 0)
            ):
                raise ValueError("pilot_values must be an immutable tuple of nonzero pilots")
        length = (
            self.channel.delays[-1] + 1
            if self.channel is not None
            else len(self.channel_taps if self.channel_model == "fixed" else self.rayleigh_powers)
        )
        self.validate_channel_length(length)

    def validate_channel_length(self, length: int) -> None:

        integer(length, "channel length", 1)
        if length > self.fft_size or self.cp_length < length - 1:
            raise ValueError("CP must cover channel length - 1 and channel must fit the FFT")

    @property
    def pilot_indices(self) -> NDArray[np.int64]:

        return np.arange(0, self.fft_size, self.pilot_spacing, dtype=np.int64)

    @property
    def data_indices(self) -> NDArray[np.int64]:

        return np.setdiff1d(np.arange(self.fft_size, dtype=np.int64), self.pilot_indices)

    @property
    def bits_per_frame(self) -> int:

        return bits_per_symbol(self.modulation_order) * self.data_indices.size

    @property
    def pilot_symbols(self) -> NDArray[np.complex128]:
        if self.pilot_values is not None:
            return np.asarray(self.pilot_values, dtype=np.complex128)
        return np.full(self.pilot_indices.size, self.pilot_value, dtype=np.complex128)

    @property
    def pilot_overhead(self) -> float:
        return self.pilot_indices.size / self.fft_size

    @property
    def cp_overhead(self) -> float:
        return self.cp_length / (self.fft_size + self.cp_length)

    @property
    def channel_prior(self) -> ChannelConfig:
        if self.channel is not None:
            return self.channel
        if self.channel_model == "fixed":
            return ChannelConfig(
                model="fixed",
                delays=tuple(range(len(self.channel_taps))),
                fixed_taps=self.channel_taps,
            )
        return ChannelConfig(
            delays=tuple(range(len(self.rayleigh_powers))), powers=self.rayleigh_powers
        )
