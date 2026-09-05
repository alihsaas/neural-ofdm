from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_ofdm._validation import complex_vector, integer


def fixed_channel(taps: ArrayLike = (1, 0.35 + 0.15j, 0.12 - 0.10j)) -> NDArray[np.complex128]:

    impulse_response = complex_vector(taps, "taps")
    energy = np.sum(np.abs(impulse_response) ** 2)
    if energy <= 0:
        raise ValueError("channel taps must have nonzero energy")
    return impulse_response / np.sqrt(energy)


def channel_frequency_response(taps: ArrayLike, fft_size: int) -> NDArray[np.complex128]:

    impulse_response = complex_vector(taps, "taps")
    integer(fft_size, "fft_size", 1)
    if impulse_response.size > fft_size:
        raise ValueError("channel cannot be longer than fft_size")
    return np.fft.fft(impulse_response, n=fft_size)


def apply_channel(signal: ArrayLike, taps: ArrayLike) -> NDArray[np.complex128]:

    return np.convolve(complex_vector(signal, "signal"), complex_vector(taps, "taps"))


def rayleigh_channel(powers: ArrayLike, rng: np.random.Generator) -> NDArray[np.complex128]:

    profile = np.asarray(powers, dtype=float)
    if (
        profile.ndim != 1
        or profile.size == 0
        or not np.all(np.isfinite(profile))
        or np.any(profile < 0)
        or profile.sum() <= 0
    ):
        raise ValueError("powers must be finite, nonnegative and have positive total power")
    scale = np.sqrt(profile / profile.sum() / 2)
    return scale * (rng.normal(size=profile.size) + 1j * rng.normal(size=profile.size))


@dataclass(frozen=True)
class AWGNResult:
    signal: NDArray[np.complex128]
    signal_power: float
    noise_variance: float
    measured_noise_power: float


def add_awgn(signal: ArrayLike, snr_db: float, rng: np.random.Generator) -> AWGNResult:

    clean_signal = complex_vector(signal, "signal")
    signal_power = float(np.mean(np.abs(clean_signal) ** 2))
    if signal_power <= 0 or not np.isfinite(signal_power):
        raise ValueError("signal must have finite positive power to define SNR")
    if np.isnan(snr_db) or np.isneginf(snr_db):
        raise ValueError("snr_db must be finite or +inf")
    if np.isposinf(snr_db):
        return AWGNResult(clean_signal.copy(), signal_power, 0.0, 0.0)
    with np.errstate(over="ignore", under="ignore"):
        snr_linear = np.power(10.0, snr_db / 10)
        noise_variance = signal_power / snr_linear if snr_linear > 0 else np.inf
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("snr_db is outside the representable positive noise-variance range")
    scale = np.sqrt(noise_variance / 2)
    noise = scale * (rng.normal(size=clean_signal.size) + 1j * rng.normal(size=clean_signal.size))
    return AWGNResult(
        clean_signal + noise,
        signal_power,
        float(noise_variance),
        float(np.mean(np.abs(noise) ** 2)),
    )


def add_fixed_awgn(signal: ArrayLike, variance: float, rng: np.random.Generator) -> AWGNResult:
    clean = complex_vector(signal, "signal")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    noise = np.zeros(clean.size, dtype=np.complex128)
    if variance:
        noise = np.sqrt(variance / 2) * (
            rng.normal(size=clean.size) + 1j * rng.normal(size=clean.size)
        )
    return AWGNResult(
        clean + noise,
        float(np.mean(np.abs(clean) ** 2)),
        float(variance),
        float(np.mean(np.abs(noise) ** 2)),
    )


def reference_noise_variance(reference_esn0_db: float) -> float:
    if np.isnan(reference_esn0_db) or np.isneginf(reference_esn0_db):
        raise ValueError("reference_esn0_db must be finite or +inf")
    if np.isposinf(reference_esn0_db):
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        variance = float(np.power(10.0, -reference_esn0_db / 10))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("reference_esn0_db is outside the noise-variance range")
    return variance


@dataclass(frozen=True)
class ChannelConfig:
    model: Literal["fixed", "rayleigh", "rician"] = "rayleigh"
    delays: tuple[int, ...] = (0, 1, 2, 3)
    powers: tuple[float, ...] | None = None
    decay_samples: float = 1.5
    k_linear: float = 0.0
    los_path: int = 0
    los_phase_rad: float = 0.0
    fixed_taps: tuple[complex, ...] = (1, 0.35 + 0.15j, 0.12 - 0.10j, 0)

    def __post_init__(self) -> None:
        if self.model not in ("fixed", "rayleigh", "rician"):
            raise ValueError("unknown channel model")
        if not isinstance(self.delays, tuple) or not self.delays:
            raise ValueError("delays must be a nonempty immutable tuple")
        for delay in self.delays:
            integer(delay, "delay")
        if np.any(np.diff(self.delays) <= 0):
            raise ValueError("delays must be sorted and unique")
        if not np.isfinite(self.decay_samples) or self.decay_samples <= 0:
            raise ValueError("decay_samples must be finite and positive")
        if not np.isfinite(self.k_linear) or self.k_linear < 0:
            raise ValueError("k_linear must be finite and nonnegative")
        integer(self.los_path, "los_path")
        if self.los_path >= len(self.delays) or not np.isfinite(self.los_phase_rad):
            raise ValueError("invalid LOS path or phase")
        if self.powers is not None:
            profile = np.asarray(self.powers, dtype=float)
            if (
                not isinstance(self.powers, tuple)
                or profile.shape != (len(self.delays),)
                or not np.all(np.isfinite(profile))
                or np.any(profile < 0)
                or not 0 < profile.sum() < np.inf
            ):
                raise ValueError("powers must match delays and have finite positive total power")
        if self.model == "fixed":
            taps = complex_vector(self.fixed_taps, "fixed_taps")
            if not isinstance(self.fixed_taps, tuple) or taps.size != len(self.delays):
                raise ValueError("fixed_taps must match delays")
            fixed_channel(taps)

    @property
    def diffuse_powers(self) -> NDArray[np.float64]:
        profile = (
            np.exp(-(np.asarray(self.delays) - self.delays[0]) / self.decay_samples)
            if self.powers is None
            else np.asarray(self.powers, dtype=float)
        )
        return profile / profile.sum()

    def tap_statistics(self) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        if self.model == "fixed":
            return fixed_channel(self.fixed_taps), np.zeros(
                (len(self.delays), len(self.delays)), dtype=np.complex128
            )
        k = self.k_linear if self.model == "rician" else 0.0
        mean = np.zeros(len(self.delays), dtype=np.complex128)
        mean[self.los_path] = np.sqrt(k / (k + 1)) * np.exp(1j * self.los_phase_rad)
        return mean, np.diag(self.diffuse_powers / (k + 1)).astype(np.complex128)

    def sample(self, rng: np.random.Generator) -> NDArray[np.complex128]:
        if self.model == "fixed":
            coefficients = fixed_channel(self.fixed_taps)
        else:
            mean, _ = self.tap_statistics()
            k = self.k_linear if self.model == "rician" else 0.0
            coefficients = mean + rayleigh_channel(self.diffuse_powers, rng) / np.sqrt(k + 1)
        impulse = np.zeros(self.delays[-1] + 1, dtype=np.complex128)
        impulse[np.asarray(self.delays)] = coefficients
        return impulse


def frequency_statistics(
    channel: ChannelConfig, fft_size: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    integer(fft_size, "fft_size", 1)
    if channel.delays[-1] >= fft_size:
        raise ValueError("channel delays must fit the FFT")
    fourier = np.exp(-2j * np.pi * np.outer(np.arange(fft_size), channel.delays) / fft_size)
    mean, covariance = channel.tap_statistics()
    return fourier @ mean, fourier @ covariance @ fourier.conj().T
