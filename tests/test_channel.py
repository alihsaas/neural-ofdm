import numpy as np
import pytest

from neural_ofdm.channel import (
    add_awgn,
    apply_channel,
    channel_frequency_response,
    fixed_channel,
    rayleigh_channel,
)
from neural_ofdm.config import OFDMConfig
from neural_ofdm.modulation import qpsk_modulate, random_bits
from neural_ofdm.ofdm import (
    add_cyclic_prefix,
    build_resource_grid,
    ofdm_demodulate,
    ofdm_modulate,
    remove_cyclic_prefix,
)


def test_fir_normalization_and_identity_response():
    taps = fixed_channel()
    np.testing.assert_allclose(np.sum(np.abs(taps) ** 2), 1)
    np.testing.assert_allclose(np.mean(np.abs(channel_frequency_response(taps, 64)) ** 2), 1)
    np.testing.assert_array_equal(channel_frequency_response([1], 64), np.ones(64))
    np.testing.assert_array_equal(apply_channel([1, 2, 3], [1, 2j]), [1, 2 + 2j, 3 + 4j, 6j])


def test_continuous_two_symbol_convolution_at_minimum_cp():
    config = OFDMConfig(cp_length=2)
    rng = np.random.default_rng(4)
    grids = [build_resource_grid(qpsk_modulate(random_bits(96, rng)), config) for _ in range(2)]
    waveform = np.concatenate([add_cyclic_prefix(ofdm_modulate(g), 2) for g in grids])
    taps = fixed_channel()
    received = apply_channel(waveform, taps)
    response = channel_frequency_response(taps, 64)
    for index, grid in enumerate(grids):
        window = received[index * 66 : (index + 1) * 66]
        recovered = ofdm_demodulate(remove_cyclic_prefix(window, 2, 64))
        np.testing.assert_allclose(recovered, response * grid, atol=1e-14)


def test_awgn_power_circularity_and_seed():
    signal = np.full(300_000, 1 + 1j)
    result = add_awgn(signal, 12, np.random.default_rng(123))
    noise = result.signal - signal
    measured_snr = 10 * np.log10(result.signal_power / result.measured_noise_power)
    assert abs(measured_snr - 12) < 0.04
    np.testing.assert_allclose(result.noise_variance, 2 / 10**1.2)
    np.testing.assert_allclose(np.var(noise.real), result.noise_variance / 2, rtol=0.01)
    np.testing.assert_allclose(np.var(noise.imag), result.noise_variance / 2, rtol=0.01)
    assert abs(np.mean(noise * noise)) / result.noise_variance < 0.01
    np.testing.assert_array_equal(
        result.signal, add_awgn(signal, 12, np.random.default_rng(123)).signal
    )


def test_awgn_no_noise_and_invalid_power():
    signal = np.array([1, 1j])
    result = add_awgn(signal, np.inf, np.random.default_rng(1))
    np.testing.assert_array_equal(result.signal, signal)
    assert result.noise_variance == result.measured_noise_power == 0
    with pytest.raises(ValueError):
        add_awgn([0, 0], 10, np.random.default_rng(1))
    with pytest.raises(ValueError):
        add_awgn(signal, -np.inf, np.random.default_rng(1))


def test_rayleigh_ensemble_power_profile_and_seed():
    powers = np.array([1, 0.4, 0.15, 0.05])
    rng = np.random.default_rng(17)
    taps = np.array([rayleigh_channel(powers, rng) for _ in range(6000)])
    np.testing.assert_allclose(np.mean(np.abs(taps) ** 2, axis=0), powers / powers.sum(), rtol=0.05)
    assert np.max(np.abs(np.mean(taps, axis=0))) < 0.025
    assert np.std(np.sum(np.abs(taps) ** 2, axis=1)) > 0.3
    np.testing.assert_array_equal(
        rayleigh_channel(powers, np.random.default_rng(9)),
        rayleigh_channel(powers, np.random.default_rng(9)),
    )
