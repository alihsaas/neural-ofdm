import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_ofdm._validation import bit_vector, complex_vector, integer


def random_bits(count: int, rng: np.random.Generator) -> NDArray[np.uint8]:

    integer(count, "count", 1)
    return rng.integers(0, 2, size=count, dtype=np.uint8)


def qpsk_modulate(bits: ArrayLike) -> NDArray[np.complex128]:

    array = bit_vector(bits)
    if array.size % 2:
        raise ValueError("QPSK requires an even number of bits")
    pairs = array.reshape(-1, 2).astype(float)
    return ((1 - 2 * pairs[:, 0]) + 1j * (1 - 2 * pairs[:, 1])) / np.sqrt(2)


def qpsk_demodulate(symbols: ArrayLike) -> NDArray[np.uint8]:

    array = complex_vector(symbols, "symbols")
    return np.column_stack((array.real < 0, array.imag < 0)).astype(np.uint8).ravel()


def bits_per_symbol(order: int) -> int:
    integer(order, "modulation_order", 1)
    if order not in (4, 16):
        raise ValueError("modulation_order must be 4 or 16")
    return 2 if order == 4 else 4


def modulate(bits: ArrayLike, order: int = 4) -> NDArray[np.complex128]:
    width = bits_per_symbol(order)
    array = bit_vector(bits)
    if array.size % width:
        raise ValueError("bit length must be divisible by bits per symbol")
    if order == 4:
        return qpsk_modulate(array)
    words = array.reshape(-1, 4).astype(float)
    in_phase = (1 - 2 * words[:, 0]) * (3 - 2 * words[:, 1])
    quadrature = (1 - 2 * words[:, 2]) * (3 - 2 * words[:, 3])
    return (in_phase + 1j * quadrature) / np.sqrt(10)


def demodulate(symbols: ArrayLike, order: int = 4) -> NDArray[np.uint8]:
    bits_per_symbol(order)
    array = complex_vector(symbols, "symbols")
    if order == 4:
        return qpsk_demodulate(array)
    scaled = array * np.sqrt(10)
    return (
        np.column_stack(
            (scaled.real < 0, np.abs(scaled.real) <= 2, scaled.imag < 0, np.abs(scaled.imag) <= 2)
        )
        .astype(np.uint8)
        .ravel()
    )


def constellation(order: int = 4) -> tuple[NDArray[np.complex128], NDArray[np.uint8]]:
    width = bits_per_symbol(order)
    labels = ((np.arange(order)[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)
    return modulate(labels.ravel(), order), labels
