import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_ofdm._validation import complex_vector, integer
from neural_ofdm.config import OFDMConfig


def build_resource_grid(data_symbols: ArrayLike, config: OFDMConfig) -> NDArray[np.complex128]:

    symbols = complex_vector(data_symbols, "data_symbols")
    if symbols.size != config.data_indices.size:
        raise ValueError("data_symbols count must match the data subcarrier allocation")
    frequency_grid = np.empty(config.fft_size, dtype=np.complex128)
    frequency_grid[config.data_indices] = symbols
    frequency_grid[config.pilot_indices] = config.pilot_symbols
    return frequency_grid


def ofdm_modulate(frequency_grid: ArrayLike) -> NDArray[np.complex128]:

    return np.fft.ifft(complex_vector(frequency_grid, "frequency_grid"), norm="ortho")


def ofdm_demodulate(time_signal: ArrayLike) -> NDArray[np.complex128]:

    return np.fft.fft(complex_vector(time_signal, "time_signal"), norm="ortho")


def add_cyclic_prefix(time_signal: ArrayLike, cp_length: int) -> NDArray[np.complex128]:

    signal = complex_vector(time_signal, "time_signal")
    integer(cp_length, "cp_length")
    if cp_length > signal.size:
        raise ValueError("cp_length cannot exceed the symbol length")
    return np.concatenate((signal[-cp_length:], signal)) if cp_length else signal.copy()


def remove_cyclic_prefix(
    received_signal: ArrayLike, cp_length: int, fft_size: int
) -> NDArray[np.complex128]:

    signal = complex_vector(received_signal, "received_signal")
    integer(cp_length, "cp_length")
    integer(fft_size, "fft_size", 1)
    if cp_length > fft_size or signal.size < cp_length + fft_size:
        raise ValueError("invalid prefix or insufficient received samples")
    return signal[cp_length : cp_length + fft_size].copy()
