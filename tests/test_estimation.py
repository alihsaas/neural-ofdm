import numpy as np
import pytest

from neural_ofdm.config import OFDMConfig
from neural_ofdm.estimation import ls_channel_estimate
from neural_ofdm.simulation import simulate_frame


def test_pilot_ls_and_periodic_complex_interpolation():
    received = np.zeros(8, dtype=complex)
    pilots = np.array([1j, 2])
    received[[0, 4]] = pilots * np.array([1 + 2j, 3 - 2j])
    estimate = ls_channel_estimate(received, [0, 4], pilots)
    expected = [1 + 2j, 1.5 + 1j, 2, 2.5 - 1j, 3 - 2j, 2.5 - 1j, 2, 1.5 + 1j]
    np.testing.assert_allclose(estimate, expected)
    assert estimate.shape == received.shape
    assert estimate.dtype == np.complex128


def test_noiseless_ls_pilots_exact_but_interpolation_has_bias():
    config = OFDMConfig(snr_db=np.inf)
    frame = simulate_frame(config, np.random.default_rng(1), csi="ls")
    np.testing.assert_allclose(
        frame.estimated_channel[config.pilot_indices],
        frame.channel_frequency_response[config.pilot_indices],
        atol=1e-14,
    )
    assert 0 < frame.nmse < 0.01
    assert frame.ber == 0


@pytest.mark.parametrize("indices", [[0], [4, 0], [0, 0], [-1, 4], [0, 8], [0.0, 4.0]])
def test_invalid_pilot_indices(indices):
    with pytest.raises(ValueError):
        ls_channel_estimate(np.ones(8), indices, 1)


def test_invalid_pilot_values():
    with pytest.raises(ValueError):
        ls_channel_estimate(np.ones(8), [0, 4], 0)
    with pytest.raises(ValueError):
        ls_channel_estimate(np.ones(8), [0, 4], [1, 1, 1])
