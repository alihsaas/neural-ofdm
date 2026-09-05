import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_ofdm._validation import complex_vector


@dataclass(frozen=True)
class EqualizationResult:
    symbols: NDArray[np.complex128]
    stabilized_bins: int


def zero_forcing_equalize(
    received_grid: ArrayLike, channel_estimate: ArrayLike, epsilon: float = 1e-10
) -> EqualizationResult:

    received = complex_vector(received_grid, "received_grid")
    estimate = complex_vector(channel_estimate, "channel_estimate")
    if received.shape != estimate.shape:
        raise ValueError("received grid and channel estimate must have identical shapes")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    magnitude = np.abs(estimate)
    small = magnitude < epsilon
    count = int(np.count_nonzero(small))
    denominator = estimate.copy()
    denominator[small] = epsilon * np.exp(1j * np.angle(estimate[small]))
    if count:
        warnings.warn(f"ZF magnitude floor applied on {count} bin(s)", RuntimeWarning, stacklevel=2)
    return EqualizationResult(received / denominator, count)


@dataclass(frozen=True)
class MMSEEqualizationResult:
    symbols: NDArray[np.complex128]
    decision_symbols: NDArray[np.complex128]
    effective_gain: NDArray[np.float64]
    stabilized_bins: int


def mmse_equalize(
    received_grid: ArrayLike,
    channel_estimate: ArrayLike,
    noise_variance: float,
    symbol_energy: float = 1.0,
    epsilon: float = 1e-10,
) -> MMSEEqualizationResult:
    received = complex_vector(received_grid, "received_grid")
    estimate = complex_vector(channel_estimate, "channel_estimate")
    if received.shape != estimate.shape:
        raise ValueError("received grid and channel estimate must have identical shapes")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    if not np.isfinite(symbol_energy) or symbol_energy <= 0:
        raise ValueError("symbol_energy must be finite and positive")
    decisions = zero_forcing_equalize(received, estimate, epsilon)
    magnitude_squared = np.abs(estimate) ** 2
    denominator = magnitude_squared + noise_variance / symbol_energy
    gain = np.divide(
        magnitude_squared, denominator, out=np.zeros_like(magnitude_squared), where=denominator > 0
    )
    raw = np.divide(
        estimate.conj() * received,
        denominator,
        out=np.zeros_like(received),
        where=denominator > 0,
    )
    return MMSEEqualizationResult(raw, decisions.symbols, gain, decisions.stabilized_bins)
