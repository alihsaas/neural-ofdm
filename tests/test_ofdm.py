import numpy as np
import pytest

from neural_ofdm.config import OFDMConfig
from neural_ofdm.modulation import qpsk_modulate, random_bits
from neural_ofdm.ofdm import (
    add_cyclic_prefix,
    build_resource_grid,
    ofdm_demodulate,
    ofdm_modulate,
    remove_cyclic_prefix,
)


def test_allocation_and_fft_ifft_reconstruction():
    config = OFDMConfig()
    symbols = qpsk_modulate(random_bits(config.bits_per_frame, np.random.default_rng(3)))
    grid = build_resource_grid(symbols, config)
    assert config.bits_per_frame == 96
    assert not np.intersect1d(config.pilot_indices, config.data_indices).size
    assert np.union1d(config.pilot_indices, config.data_indices).size == config.fft_size
    np.testing.assert_array_equal(grid[config.data_indices], symbols)
    np.testing.assert_array_equal(grid[config.pilot_indices], config.pilot_value)
    time_signal = ofdm_modulate(grid)
    np.testing.assert_allclose(ofdm_demodulate(time_signal), grid, atol=1e-14)
    np.testing.assert_allclose(np.sum(np.abs(time_signal) ** 2), np.sum(np.abs(grid) ** 2))


@pytest.mark.parametrize("cp_length", [0, 1, 16, 64])
def test_prefix_copy_and_round_trip(cp_length):
    signal = np.arange(64) + 1j * np.arange(64)[::-1]
    with_cp = add_cyclic_prefix(signal, cp_length)
    assert with_cp.size == 64 + cp_length
    if cp_length:
        np.testing.assert_array_equal(with_cp[:cp_length], signal[-cp_length:])
    np.testing.assert_array_equal(remove_cyclic_prefix(with_cp, cp_length, 64), signal)


@pytest.mark.parametrize(
    "options",
    [
        {"fft_size": 1},
        {"fft_size": 64.0},
        {"cp_length": -1},
        {"cp_length": 65},
        {"cp_length": 1},
        {"pilot_spacing": 1},
        {"pilot_spacing": 64},
        {"pilot_value": 0},
        {"modulation_order": 64},
        {"snr_db": np.nan},
        {"channel_taps": (0j,)},
        {"channel_model": "unknown"},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        OFDMConfig(**options)


def test_bad_grid_and_prefix_lengths():
    with pytest.raises(ValueError):
        build_resource_grid([1j], OFDMConfig())
    with pytest.raises(ValueError):
        add_cyclic_prefix([1, 2], 3)
    with pytest.raises(ValueError):
        remove_cyclic_prefix([1, 2], 1, 2)
