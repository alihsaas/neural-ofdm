from dataclasses import replace

import numpy as np
import pytest

from neural_ofdm.channel import (
    ChannelConfig,
    add_awgn,
    add_fixed_awgn,
    frequency_statistics,
    reference_noise_variance,
)
from neural_ofdm.config import OFDMConfig
from neural_ofdm.equalization import mmse_equalize, zero_forcing_equalize
from neural_ofdm.modulation import constellation, demodulate, modulate, random_bits
from neural_ofdm.simulation import simulate_frame


@pytest.mark.parametrize("order", [4, 16])
def test_complete_constellation_energy_gray_neighbors_and_recovery(order):
    symbols, labels = constellation(order)
    np.testing.assert_array_equal(demodulate(symbols, order).reshape(labels.shape), labels)
    np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 1, atol=1e-15)
    distances = np.abs(symbols[:, None] - symbols[None, :])
    nearest = np.min(distances[distances > 0])
    neighbors = np.isclose(distances, nearest)
    differences = np.sum(labels[:, None, :] != labels[None, :, :], axis=2)
    assert np.all(differences[neighbors] == 1)
    bits = random_bits(20000, np.random.default_rng(17))
    np.testing.assert_array_equal(demodulate(modulate(bits, order), order), bits)


def test_qam_decision_boundaries_and_invalid_inputs():
    eps = 1e-9
    points = np.array([2 - eps, 2, 2 + eps, -2 + eps, -2, -2 - eps]) / np.sqrt(10)
    words = demodulate(points + 1j / np.sqrt(10), 16).reshape(-1, 4)
    np.testing.assert_array_equal(words[:, 1], [1, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(demodulate([0j], 4), [0, 0])
    np.testing.assert_array_equal(demodulate([0j], 16), [0, 1, 0, 1])
    for bits in ([0, 1], [0, 1, 0], [0, 1, 0, 0.5], []):
        with pytest.raises(ValueError):
            modulate(bits, 16)
    with pytest.raises(ValueError):
        modulate([0, 0], 64)


@pytest.mark.parametrize("spacing", [2, 4, 8])
def test_pilot_partitions_payload_and_unequal_pilots(spacing):
    base = OFDMConfig(pilot_spacing=spacing, modulation_order=16)
    pilots = tuple((1 + 1j) * (1 + np.arange(base.pilot_indices.size) / 10))
    config = replace(base, pilot_values=pilots)
    assert np.intersect1d(config.data_indices, config.pilot_indices).size == 0
    assert np.union1d(config.data_indices, config.pilot_indices).size == 64
    assert config.pilot_overhead == 1 / spacing
    assert config.bits_per_frame == 4 * config.data_indices.size
    frame = simulate_frame(replace(config, snr_db=np.inf), np.random.default_rng(1), csi="perfect")
    assert frame.ber == 0
    np.testing.assert_array_equal(frame.frequency_grid[config.pilot_indices], pilots)


@pytest.mark.parametrize("model", ["fixed", "rayleigh", "rician"])
def test_sparse_delays_cp_fourier_relation_and_seed(model):
    channel = ChannelConfig(model=model, delays=(0, 2, 6), fixed_taps=(1, 0.2j, 0.1), k_linear=3)
    with pytest.raises(ValueError, match="CP"):
        OFDMConfig(cp_length=5, channel=channel)
    config = OFDMConfig(channel=channel, cp_length=6, modulation_order=16, snr_db=np.inf)
    first = simulate_frame(config, np.random.default_rng(32), csi="perfect")
    second = simulate_frame(config, np.random.default_rng(32), csi="perfect")
    np.testing.assert_array_equal(first.received_grid, second.received_grid)
    assert first.ber == 0
    np.testing.assert_allclose(
        first.received_grid, first.channel_frequency_response * first.frequency_grid, atol=1e-14
    )


@pytest.mark.parametrize(
    "options",
    [
        {"delays": (0, 1.5)},
        {"delays": (0, True)},
        {"delays": (0, 0)},
        {"delays": (-1, 1)},
        {"powers": (1, 0)},
        {"powers": (0, 0, 0, 0)},
        {"powers": (1, -1, 1, 1)},
        {"powers": (1, np.nan, 1, 1)},
        {"decay_samples": 0},
        {"k_linear": -1},
        {"los_phase_rad": np.inf},
        {"los_path": 4},
        {"model": "unknown"},
    ],
)
def test_channel_validation(options):
    with pytest.raises(ValueError):
        ChannelConfig(**options)


@pytest.mark.parametrize("k", [0, 1, 10, 10000])
def test_empirical_rician_mean_covariance_and_power(k):
    channel = ChannelConfig(
        model="rician", delays=(0, 2, 5), k_linear=k, los_path=1, los_phase_rad=0.7
    )
    rng = np.random.default_rng(110)
    samples = np.array([channel.sample(rng)[list(channel.delays)] for _ in range(18000)])
    mean, covariance = channel.tap_statistics()
    centered = samples - mean
    empirical = centered.T @ centered.conj() / len(samples)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.015 / np.sqrt(k + 1))
    np.testing.assert_allclose(empirical * (k + 1), covariance * (k + 1), atol=0.012)
    expected_powers = np.abs(mean) ** 2 + np.diag(covariance).real
    np.testing.assert_allclose(np.mean(np.abs(samples) ** 2, axis=0), expected_powers, atol=0.012)
    assert abs(np.mean(np.sum(np.abs(samples) ** 2, axis=1)) - 1) < 0.02
    if k == 0:
        rayleigh = replace(channel, model="rayleigh")
        np.testing.assert_array_equal(
            channel.sample(np.random.default_rng(3)), rayleigh.sample(np.random.default_rng(3))
        )
    if k == 10000:
        assert np.mean(np.sum(np.abs(centered) ** 2, axis=1)) < 0.00011


def test_frequency_centered_covariance_and_unscaled_fourier():
    channel = ChannelConfig(model="rician", delays=(0, 2, 5), k_linear=2, los_phase_rad=0.4)
    mean, covariance = frequency_statistics(channel, 16)
    assert covariance.shape == (16, 16)
    np.testing.assert_allclose(covariance, covariance.conj().T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() > -1e-13
    np.testing.assert_allclose(np.diag(covariance), 1 / 3)
    np.testing.assert_allclose(np.abs(mean) ** 2, 2 / 3)
    rng = np.random.default_rng(25)
    responses = np.array([np.fft.fft(channel.sample(rng), n=16) for _ in range(15000)])
    centered = responses - mean
    empirical = centered.T @ centered.conj() / len(responses)
    np.testing.assert_allclose(empirical, covariance, atol=0.012)
    second_moment = responses.T @ responses.conj() / len(responses)
    assert np.max(np.abs(second_moment - covariance)) > 0.5


def test_fixed_noise_independence_fft_variance_and_legacy_amplitude_compensation():
    variance = reference_noise_variance(10)
    signal = np.ones(128000, dtype=np.complex128)
    first = add_fixed_awgn(signal, variance, np.random.default_rng(3))
    second = add_fixed_awgn(7 * signal, variance, np.random.default_rng(3))
    np.testing.assert_allclose(first.signal - signal, second.signal - 7 * signal, atol=1e-15)
    noise = (first.signal - signal).reshape(-1, 64)
    transformed = np.fft.fft(noise, norm="ortho", axis=1)
    assert abs(np.mean(np.abs(transformed) ** 2) / variance - 1) < 0.012
    assert abs(np.var(noise.real) / (variance / 2) - 1) < 0.012
    legacy = add_awgn(7 * signal, 10, np.random.default_rng(3))
    np.testing.assert_allclose(legacy.noise_variance, 49 * variance)
    assert reference_noise_variance(np.inf) == 0
    config = OFDMConfig(noise_mode="fixed", reference_esn0_db=10, modulation_order=16)
    for gain in (0.1, 3):
        frame = simulate_frame(config, np.random.default_rng(5), channel_taps=[gain])
        assert frame.noise_power == variance


def test_mmse_effective_gain_and_qam_decisions():
    symbols, _ = constellation(16)
    channel = np.full(16, 0.2 + 0.3j)
    received = channel * symbols
    mmse = mmse_equalize(received, channel, 0.4)
    expected_gain = 0.13 / 0.53
    np.testing.assert_allclose(mmse.effective_gain, expected_gain)
    np.testing.assert_allclose(mmse.symbols, expected_gain * symbols)
    np.testing.assert_allclose(mmse.decision_symbols, symbols)
    np.testing.assert_array_equal(demodulate(mmse.decision_symbols, 16), demodulate(symbols, 16))
    assert np.any(demodulate(mmse.symbols, 16) != demodulate(symbols, 16))
    np.testing.assert_allclose(
        mmse.decision_symbols, zero_forcing_equalize(received, channel).symbols
    )
    np.testing.assert_allclose(mmse_equalize(received, channel, 0).symbols, symbols)


def test_mmse_true_null_diagnostic():
    with pytest.warns(RuntimeWarning, match="floor"):
        result = mmse_equalize([1, 2], [0, 1], 0.2)
    assert result.stabilized_bins == 1
    assert result.effective_gain[0] == result.symbols[0] == 0
    assert np.all(np.isfinite(result.decision_symbols))
