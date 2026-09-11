import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from neural_ofdm.benchmark import (
    ExperimentConfig,
    benchmark_studies,
    load_benchmark,
    paired_frame_bootstrap,
    run_benchmark,
    save_benchmark,
)
from neural_ofdm.channel import ChannelConfig, frequency_statistics
from neural_ofdm.config import OFDMConfig
from neural_ofdm.estimation import prepare_lmmse
from neural_ofdm.simulation import observe_frame, receive_observation


def test_seeded_benchmark_counts_uncertainty_and_metadata(tmp_path):
    config = ExperimentConfig(
        frames=30,
        reference_values_db=(0, 15),
        bootstrap_replicates=100,
        assumed_channel=ChannelConfig(decay_samples=0.5),
    )
    first = run_benchmark(config)
    second = run_benchmark(config)
    assert first.rows == second.rows
    assert first.comparisons == second.comparisons
    for a, b in zip(first.frame_errors, second.frame_errors, strict=True):
        np.testing.assert_array_equal(a, b)
    for row in first.rows:
        assert row["ber"] == row["errors"] / row["total_bits"]
        assert row["total_bits"] == config.frames * config.ofdm.bits_per_frame
        assert row["noise_variance"] == 10 ** (-row["reference_esn0_db"] / 10)
    save_benchmark(first, tmp_path)
    restored = load_benchmark(tmp_path)
    assert restored.rows == first.rows
    assert restored.comparisons == first.comparisons
    assert restored.config == first.config
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["experiment"]["frames"] == 30
    assert metadata["experiment"]["ofdm"]["noise_mode"] == "fixed"
    assert all(item["jitter"] == 0 for item in metadata["estimator_diagnostics"])
    with (tmp_path / "frame_errors.csv").open() as handle:
        counts = list(csv.DictReader(handle))
    for row in first.rows:
        total = sum(
            int(p[row["receiver"]])
            for p in counts
            if float(p["reference_esn0_db"]) == row["reference_esn0_db"]
        )
        assert total == row["errors"]


def test_zero_errors_have_frame_bound_and_no_bootstrap_coverage_claim():
    config = ExperimentConfig(
        ofdm=OFDMConfig(noise_mode="fixed", channel=ChannelConfig(model="rician", k_linear=1e6)),
        reference_values_db=(80,),
        frames=8,
        bootstrap_replicates=100,
    )
    result = run_benchmark(config)
    for row in result.rows:
        assert row["errors"] == 0
        assert row["ber_frame_bootstrap_low"] is None
        assert row["zero_error_frame_upper95"] > row["zero_error_bit_upper95_independence_approx"]


def test_paired_bootstrap_preserves_receiver_identity():
    counts = np.array([[0, 0], [3, 3], [1, 1], [2, 2]])
    samples = paired_frame_bootstrap(counts, 10, 100, np.random.default_rng(1))
    np.testing.assert_array_equal(samples[:, 0], samples[:, 1])


def test_mismatch_uses_same_observation_and_does_not_read_truth():
    config = OFDMConfig(noise_mode="fixed", channel=ChannelConfig(model="rician", k_linear=4))
    observation = observe_frame(config, np.random.default_rng(22))
    mean, covariance = frequency_statistics(config.channel_prior, 64)
    estimator = prepare_lmmse(mean, covariance, config.pilot_indices, config.pilot_symbols, 0.01)
    first = receive_observation(config, observation, csi="lmmse", estimator=estimator)
    altered = replace(observation, true_channel=np.full(64, 100 + 100j))
    second = receive_observation(config, altered, csi="lmmse", estimator=estimator)
    np.testing.assert_array_equal(first.estimated_channel, second.estimated_channel)
    np.testing.assert_array_equal(first.estimated_bits, second.estimated_bits)
    ls_first = receive_observation(config, observation)
    ls_second = receive_observation(config, altered)
    np.testing.assert_array_equal(ls_first.estimated_channel, ls_second.estimated_channel)
    np.testing.assert_array_equal(ls_first.estimated_bits, ls_second.estimated_bits)
    mean2, cov2 = frequency_statistics(ChannelConfig(decay_samples=0.4), 64)
    wrong = prepare_lmmse(mean2, cov2, config.pilot_indices, config.pilot_symbols, 0.01)
    third = receive_observation(config, observation, csi="lmmse", estimator=wrong)
    assert np.max(np.abs(first.estimated_channel - third.estimated_channel)) > 0.01


def test_matrix_scope_and_common_channel_noise_across_grids():
    studies = benchmark_studies(frames=2)
    first = studies["receiver_comparison"].ofdm
    second = studies["pilot_spacing_8"].ofdm
    observations = []
    for config in (first, second):
        streams = [np.random.default_rng(p) for p in np.random.SeedSequence(2026).spawn(3)]
        observations.append(
            observe_frame(config, streams[0], channel_rng=streams[1], noise_rng=streams[2])
        )
    np.testing.assert_array_equal(observations[0].true_channel, observations[1].true_channel)
    clean = [o.true_channel * o.frequency_grid for o in observations]
    np.testing.assert_allclose(
        observations[0].received_grid - clean[0],
        observations[1].received_grid - clean[1],
        atol=2e-14,
    )


def test_experiment_validation():
    for options in (
        {"frames": 1},
        {"assumed_noise_factor": 0},
        {"reference_values_db": (2, 1)},
        {"ofdm": OFDMConfig()},
        {"equalizer": "bad"},
    ):
        with pytest.raises(ValueError):
            ExperimentConfig(**options)
