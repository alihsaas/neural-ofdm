import numpy as np
import pytest

from neural_ofdm.channel import ChannelConfig, frequency_statistics
from neural_ofdm.estimation import ls_channel_estimate, prepare_lmmse


def test_independent_complex_analytic_one_pilot_example():
    mean = np.array([1 + 0.5j, -0.2j])
    covariance = np.array([[2, 0.3 + 0.4j], [0.3 - 0.4j, 1]])
    pilot = 2j
    variance = 0.8
    received = np.array([pilot * (2 - 1j), 1000j])
    estimator = prepare_lmmse(mean, covariance, [0], [pilot], variance)
    innovation = received[0] / pilot - mean[0]
    expected = mean + np.array([2, 0.3 - 0.4j]) * innovation / (2 + 0.8 / 4)
    np.testing.assert_allclose(estimator.estimate(received), expected, atol=1e-14)
    assert estimator.operator.shape == (2, 1)
    assert "Cholesky" in estimator.method


def test_unequal_complex_pilots_and_mean_handling():
    mean = np.array([1j, 2, -0.5j])
    covariance = np.diag([1, 2, 3]).astype(complex)
    pilots = np.array([0.5j, 2 + 1j, -3j])
    z = np.array([2 + 1j, 1j, 3])
    estimator = prepare_lmmse(mean, covariance, [0, 1, 2], pilots, 0.6)
    weights = np.array([1, 2, 3]) / (np.array([1, 2, 3]) + 0.6 / np.abs(pilots) ** 2)
    np.testing.assert_allclose(estimator.estimate(pilots * z), mean + weights * (z - mean))
    zero_cov = prepare_lmmse(mean, np.zeros((3, 3)), [0, 2], pilots[[0, 2]], 0)
    np.testing.assert_array_equal(zero_cov.estimate(pilots * z), mean)
    assert zero_cov.observation_rank == 0


def test_all_pilot_noiseless_rank_deficient_channel_recovery():
    channel = ChannelConfig(model="rician", delays=(0, 2, 5), k_linear=4, los_phase_rad=0.6)
    mean, covariance = frequency_statistics(channel, 16)
    indices = np.arange(16)
    pilots = (1 + 1j) * np.linspace(0.5, 2, 16)
    estimator = prepare_lmmse(mean, covariance, indices, pilots, 0)
    assert estimator.observation_rank == 3
    assert "noiseless" in estimator.method
    response = np.fft.fft(channel.sample(np.random.default_rng(4)), n=16)
    np.testing.assert_allclose(estimator.estimate(pilots * response), response, atol=2e-14)
    assert np.all(np.isfinite(estimator.operator))


def test_matched_ensemble_mse_and_posterior_covariance_prediction():
    channel = ChannelConfig(model="rician", delays=(0, 1, 3), k_linear=2, los_phase_rad=0.7)
    mean, covariance = frequency_statistics(channel, 16)
    indices = np.arange(0, 16, 4)
    pilots = np.array([1, 0.5j, 2 + 1j, -1j])
    variance = 0.3
    estimator = prepare_lmmse(mean, covariance, indices, pilots, variance)
    expected_mse = np.trace(covariance - estimator.operator @ covariance[indices, :]).real / 16
    rng = np.random.default_rng(7)
    mse_lmmse = mse_ls = 0.0
    for _ in range(5000):
        truth = np.fft.fft(channel.sample(rng), n=16)
        grid = np.zeros(16, dtype=complex)
        grid[indices] = pilots * truth[indices] + np.sqrt(variance / 2) * (
            rng.normal(size=4) + 1j * rng.normal(size=4)
        )
        mse_lmmse += np.mean(np.abs(estimator.estimate(grid) - truth) ** 2)
        mse_ls += np.mean(np.abs(ls_channel_estimate(grid, indices, pilots) - truth) ** 2)
    assert mse_lmmse < 0.5 * mse_ls
    np.testing.assert_allclose(mse_lmmse / 5000, expected_mse, rtol=0.05)


@pytest.mark.parametrize(
    "indices,pilots,variance",
    [
        ([1, 0], [1, 1], 1),
        ([0, 0], [1, 1], 1),
        ([0, 2], [1, 1], 1),
        ([0.0], [1], 1),
        ([0], [0], 1),
        ([0], [np.nan], 1),
        ([0], [1], -1),
    ],
)
def test_invalid_lmmse_observations(indices, pilots, variance):
    with pytest.raises(ValueError):
        prepare_lmmse([0, 0], np.eye(2), indices, pilots, variance)


def test_invalid_covariance_rejected_without_jitter():
    for covariance in (np.diag([1, -0.1]), [[1, 1j], [1j, 1]], np.eye(3)):
        with pytest.raises(ValueError, match="covariance"):
            prepare_lmmse([0, 0], covariance, [0], [1], 0.1)
