import numpy as np
import pytest
from scipy.special import erfc

from neural_ofdm.config import OFDMConfig
from neural_ofdm.simulation import simulate_frame


def test_identity_channel_noiseless_ber_zero():
    config = OFDMConfig(channel_taps=(1 + 0j,), cp_length=0, snr_db=np.inf)
    rng = np.random.default_rng(2026)
    for _ in range(100):
        frame = simulate_frame(config, rng, csi="perfect")
        assert frame.ber == 0
        np.testing.assert_allclose(frame.received_grid, frame.frequency_grid, atol=1e-14)


def test_fixed_multipath_perfect_csi_noiseless_ber_zero():
    config = OFDMConfig(cp_length=2, snr_db=np.inf)
    rng = np.random.default_rng(2026)
    for _ in range(100):
        frame = simulate_frame(config, rng, csi="perfect")
        assert frame.ber == 0
        assert frame.nmse == 0
        np.testing.assert_allclose(
            frame.received_grid, frame.channel_frequency_response * frame.frequency_grid, atol=1e-14
        )
        np.testing.assert_allclose(frame.equalized_grid, frame.frequency_grid, atol=1e-14)


def test_custom_channel_rechecks_cp_length():
    with pytest.raises(ValueError, match="CP"):
        simulate_frame(OFDMConfig(), np.random.default_rng(1), channel_taps=np.ones(18))


def test_perfect_csi_ber_decreases_with_snr():
    rates = []
    for snr in (0, 10, 20):
        config = OFDMConfig(snr_db=snr)
        rng = np.random.default_rng(31)
        rates.append(np.mean([simulate_frame(config, rng, csi="perfect").ber for _ in range(1200)]))
    assert rates[0] > rates[1] > rates[2]
    assert rates[0] > 0.1
    assert rates[2] < 0.001


def test_rayleigh_perfect_csi_noiseless_and_one_realization_per_frame():
    config = OFDMConfig(channel_model="rayleigh", snr_db=np.inf)
    rng = np.random.default_rng(10)
    previous = None
    for _ in range(100):
        frame = simulate_frame(config, rng, csi="perfect")
        assert frame.ber == 0
        if previous is not None:
            assert not np.array_equal(previous, frame.channel_impulse_response)
        previous = frame.channel_impulse_response


def test_identity_awgn_agrees_with_qpsk_theory_at_sample_snr():

    for snr in (0, 5, 10):
        config = OFDMConfig(channel_taps=(1 + 0j,), cp_length=0, snr_db=snr)
        rng = np.random.default_rng(99)
        total_bits = 2000 * config.bits_per_frame
        observed = np.mean([simulate_frame(config, rng, csi="perfect").ber for _ in range(2000)])
        theoretical = float(0.5 * erfc(np.sqrt(10 ** (snr / 10) / 2)))
        standard_error = np.sqrt(theoretical * (1 - theoretical) / total_bits)
        assert abs(observed - theoretical) < 6 * standard_error
