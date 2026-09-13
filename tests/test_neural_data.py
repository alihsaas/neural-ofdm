import json
from dataclasses import replace

import numpy as np
import pytest

from neural_ofdm.estimation import ls_channel_estimate
from neural_ofdm.neural_data import (
    DatasetConfig,
    complex_to_features,
    config_from_dict,
    features_to_complex,
    generate_dataset,
    load_dataset,
)
from neural_ofdm.neural_evaluation import mixture_noise_variance
from neural_ofdm.neural_study import evaluation_conditions, load_protocol, training_dataset


def small(split: str = "train", count: int = 24, **changes) -> DatasetConfig:
    return replace(DatasetConfig(), split=split, count=count, **changes)


def test_feature_round_trip_preserves_order_shape_and_float32_precision():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(5, 64)) + 1j * rng.normal(size=(5, 64))
    features = complex_to_features(values)
    assert features.shape == (5, 2, 64) and features.dtype == np.float32
    np.testing.assert_array_equal(features[:, 0, 7], values[:, 7].real.astype(np.float32))
    np.testing.assert_array_equal(features[:, 1, 0], values[:, 0].imag.astype(np.float32))
    restored = features_to_complex(features)
    assert restored.dtype == np.complex128
    np.testing.assert_allclose(restored, values, rtol=2e-7, atol=1e-7)
    assert complex_to_features(values[0]).shape == (2, 64)
    with pytest.raises(ValueError):
        features_to_complex(np.zeros((3, 64)))


def test_sample_generation_is_independent_of_access_order():
    config = small(count=10)
    forward = [config.replay(index)[1] for index in range(10)]
    backward = {index: config.replay(index)[1] for index in reversed(range(10))}
    for index, observation in enumerate(forward):
        np.testing.assert_array_equal(observation.received_grid, backward[index].received_grid)
        np.testing.assert_array_equal(observation.reference_bits, backward[index].reference_bits)


def test_splits_and_conditions_use_independent_channels_noise_and_payloads():
    observations = {
        key: small(split=split, condition_id=condition, count=4).replay(2)[1]
        for key, (split, condition) in {
            "train": ("train", 0),
            "validation": ("validation", 0),
            "test": ("test", 0),
            "shifted": ("train", 1),
        }.items()
    }
    names = list(observations)
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            assert not np.allclose(
                observations[first].true_channel, observations[second].true_channel
            )
            assert not np.array_equal(
                observations[first].reference_bits, observations[second].reference_bits
            )


def test_training_references_are_drawn_inside_declared_range():
    config = small(count=200)
    references = [config.replay(index)[0].reference_esn0_db for index in range(200)]
    assert 0 <= min(references) and max(references) <= 30
    assert np.std(references) > 6


def test_stored_dataset_replays_exactly_and_detects_tampering(tmp_path):
    config = small(split="test", count=12, fixed_reference_db=10.0, store_frames=True)
    dataset = generate_dataset(config, tmp_path / "set")
    assert dataset.verify_replay() == 12
    frame_config, observation = config.replay(5)
    ls = ls_channel_estimate(
        dataset.received_grids[5], frame_config.pilot_indices, frame_config.pilot_symbols
    )
    np.testing.assert_array_equal(dataset.inputs[5], complex_to_features(ls))
    np.testing.assert_array_equal(dataset.reference_bits[5], observation.reference_bits)
    assert np.all(dataset.references_db == 10.0)
    reloaded = load_dataset(tmp_path / "set")
    assert reloaded.config == config
    assert generate_dataset(config, tmp_path / "set").manifest == dataset.manifest
    with pytest.raises(ValueError):
        generate_dataset(replace(config, count=13), tmp_path / "set")
    manifest_path = tmp_path / "set" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["configuration"]["count"] = 11
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "set")


def test_replay_check_rejects_a_modified_stored_frame(tmp_path):
    dataset = generate_dataset(small(count=4), tmp_path / "train")
    dataset.inputs[1, 0, 3] += 1e-3
    with pytest.raises(ValueError):
        dataset.verify_replay([1])


def test_configuration_round_trip_and_validation():
    config = small(fixed_reference_db=-5.0, condition_id=3)
    from dataclasses import asdict

    from neural_ofdm.benchmark import json_safe

    assert config_from_dict(json.loads(json.dumps(json_safe(asdict(config))))) == config
    with pytest.raises(ValueError):
        replace(config, split="holdout")
    with pytest.raises(ValueError):
        replace(config, ofdm=replace(config.ofdm, noise_mode="received_snr"))


def test_protocol_conditions_respect_cp_and_pilot_sampling_limits():
    protocol = load_protocol()
    train = training_dataset(protocol, "train")
    assert train.count == protocol["split_counts"]["train"]
    for condition in evaluation_conditions(protocol):
        for config in condition.datasets:
            ofdm = config.ofdm
            assert ofdm.cp_length >= ofdm.channel.delays[-1]
            assert ofdm.channel.delays[-1] < ofdm.fft_size // ofdm.pilot_spacing
            assert config.split == "test" and config.store_frames
    names = [condition.name for condition in evaluation_conditions(protocol)]
    assert names[0] == "primary" and len(set(names)) == len(names)


def test_mixture_noise_variance_matches_uniform_db_average():
    grid = np.linspace(0, 30, 300001)
    expected = np.trapezoid(10 ** (-grid / 10), grid) / 30
    assert mixture_noise_variance((0.0, 30.0)) == pytest.approx(expected, rel=1e-6)
    assert mixture_noise_variance((10.0, 10.0)) == pytest.approx(0.1)


def test_ls_observation_is_modulation_independent_under_fixed_noise():
    qpsk = small(split="test", count=3, fixed_reference_db=12.0)
    qam = replace(qpsk, ofdm=replace(qpsk.ofdm, modulation_order=16))
    for index in range(3):
        first_config, first = qpsk.replay(index)
        second_config, second = qam.replay(index)
        np.testing.assert_array_equal(first.true_channel, second.true_channel)
        np.testing.assert_allclose(
            ls_channel_estimate(
                first.received_grid, first_config.pilot_indices, first_config.pilot_symbols
            ),
            ls_channel_estimate(
                second.received_grid, second_config.pilot_indices, second_config.pilot_symbols
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        assert not np.array_equal(first.reference_bits.size, second.reference_bits.size)
