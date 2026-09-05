import numpy as np
from numpy.typing import ArrayLike

from neural_ofdm._validation import bit_vector, complex_vector


def bit_error_rate(reference_bits: ArrayLike, estimated_bits: ArrayLike) -> float:

    reference = bit_vector(reference_bits)
    estimated = bit_vector(estimated_bits)
    if reference.shape != estimated.shape:
        raise ValueError("bit vectors must have identical shapes")
    return float(np.count_nonzero(reference != estimated) / reference.size)


def normalized_mean_squared_error(true_channel: ArrayLike, estimated_channel: ArrayLike) -> float:

    reference = complex_vector(true_channel, "true_channel")
    estimate = complex_vector(estimated_channel, "estimated_channel")
    if reference.shape != estimate.shape:
        raise ValueError("channel vectors must have identical shapes")
    energy = np.sum(np.abs(reference) ** 2)
    if energy == 0:
        raise ValueError("NMSE is undefined for a zero-energy reference")
    return float(np.sum(np.abs(reference - estimate) ** 2) / energy)


def nmse_to_db(nmse: float) -> float:

    if not np.isfinite(nmse) or nmse < 0:
        raise ValueError("nmse must be finite and nonnegative")
    return float(10 * np.log10(nmse)) if nmse > 0 else -np.inf
