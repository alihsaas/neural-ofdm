import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from neural_ofdm.neural_data import (
    DatasetConfig,
    features_to_complex,
    generate_dataset,
)
from neural_ofdm.neural_evaluation import evaluate_paired, linear_estimators
from neural_ofdm.neural_models import (
    AffineNormalization,
    ChannelCNN,
    ChannelMLP,
    build_model,
    parameter_count,
)
from neural_ofdm.neural_training import (
    complex_mse,
    fit_normalization,
    load_checkpoint,
    predict,
    train_channel_estimator,
)

CNN = {
    "architecture": "cnn",
    "width": 8,
    "blocks": 1,
    "kernel_size": 3,
    "activation": "relu",
    "padding": "circular",
    "residual": True,
    "zero_init_output": True,
    "normalization": "none",
}


def random_features(batch: int = 4, size: int = 64, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 2, size, generator=generator)


def test_models_preserve_shape_and_produce_finite_gradients():
    inputs = random_features()
    for model in (
        ChannelCNN(width=16, blocks=2, kernel_size=5, zero_init_output=False),
        ChannelMLP(fft_size=64, hidden=32),
    ):
        output = model(inputs)
        assert output.shape == inputs.shape and output.dtype == torch.float32
        complex_mse(output, inputs.flip(-1)).backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        assert all(
            gradient is not None and torch.all(torch.isfinite(gradient)) for gradient in gradients
        )
    with pytest.raises(ValueError):
        ChannelCNN(kernel_size=4)


def test_zero_initialized_residual_cnn_starts_at_ls_with_and_without_normalization():
    inputs = random_features()
    plain = ChannelCNN(width=8, blocks=2)
    torch.testing.assert_close(plain(inputs), inputs, rtol=0, atol=0)
    scaled = ChannelCNN(
        width=8,
        blocks=2,
        normalization=AffineNormalization(torch.tensor([0.1, -0.2]), torch.tensor([0.7, 1.3])),
    )
    torch.testing.assert_close(scaled(inputs), inputs, rtol=1e-6, atol=1e-6)


def test_receptive_field_and_circular_wrap_match_configuration():
    model = ChannelCNN(width=4, blocks=1, kernel_size=3, zero_init_output=False)
    torch.manual_seed(0)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, 0.1, 0.2)
    assert model.receptive_field == 9
    inputs = torch.zeros(1, 2, 64, requires_grad=True)
    model(inputs)[0, 0, 0].backward()
    sensitive = torch.nonzero(inputs.grad.abs().sum(dim=1)[0]).flatten().tolist()
    assert sorted(sensitive) == sorted([(offset % 64) for offset in range(-4, 5)])


def test_complex_mse_matches_analytic_complex_error():
    target = torch.zeros(2, 2, 4)
    prediction = target.clone()
    prediction[0, 0, 1] = 3.0
    prediction[0, 1, 1] = 4.0
    prediction[1, 1, 2] = 2.0
    expected = (25.0 + 4.0) / (2 * 4)
    assert float(complex_mse(prediction, target)) == pytest.approx(expected)
    assert float(torch.mean((prediction - target) ** 2)) == pytest.approx(expected / 2)


def test_normalization_uses_training_statistics_and_inverts():
    train = np.stack([np.full((2, 8), [[1.0], [-2.0]]) + index for index in range(4)]).astype(
        np.float32
    )
    normalization = fit_normalization(train)
    torch.testing.assert_close(normalization.offset.flatten(), torch.tensor([2.5, -0.5]))
    torch.testing.assert_close(
        normalization.scale.flatten(), torch.tensor([np.std([0, 1, 2, 3])] * 2, dtype=torch.float32)
    )
    values = random_features(size=8)
    torch.testing.assert_close(normalization.inverse(normalization(values)), values)


def tiny_datasets(tmp_path):
    base = DatasetConfig()
    train = generate_dataset(replace(base, split="debug", count=96), tmp_path / "debug")
    validation = generate_dataset(replace(base, split="validation", count=48), tmp_path / "val")
    return train, validation


def test_tiny_training_job_selects_checkpoint_reloads_and_evaluates(tmp_path):
    train, validation = tiny_datasets(tmp_path)
    training = {
        "batch_size": 32,
        "maximum_epochs": 3,
        "patience": 5,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "num_workers": 0,
    }
    torch.manual_seed(5)
    initial = build_model(CNN, 64)
    initial_state = {key: value.clone() for key, value in initial.state_dict().items()}
    summary = train_channel_estimator(
        "tiny",
        5,
        CNN,
        training,
        train,
        validation,
        tmp_path / "run",
        torch.device("cpu"),
        log=lambda message: None,
    )
    with (tmp_path / "run" / "history.csv").open(encoding="ascii") as handle:
        history = list(csv.DictReader(handle))
    losses = [float(row["validation_loss"]) for row in history]
    assert all(np.isfinite(losses)) and len(history) == 3
    assert summary["selected_epoch"] == int(np.argmin(losses)) + 1
    assert summary["selected_validation_mse"] == pytest.approx(min(losses))
    assert summary["reload_prediction_equal"]
    assert summary["parameters"] == parameter_count(initial)
    model, metadata = load_checkpoint(tmp_path / "run" / "best.pt", torch.device("cpu"))
    changed = any(
        not torch.equal(model.state_dict()[key], value) for key, value in initial_state.items()
    )
    assert changed and metadata["run"] == "tiny"
    stored = json.loads((tmp_path / "run" / "best.json").read_text(encoding="ascii"))
    assert stored["checkpoint_sha256"] and stored["epochs_completed"] == 3
    again = predict(model, validation.inputs, torch.device("cpu"), batch_size=7)
    np.testing.assert_array_equal(again, predict(model, validation.inputs, torch.device("cpu")))
    test = generate_dataset(
        replace(
            DatasetConfig(), split="test", count=16, fixed_reference_db=15.0, store_frames=True
        ),
        tmp_path / "test",
    )
    estimates = {
        "tiny/seed_5": features_to_complex(predict(model, test.inputs, torch.device("cpu")))
    }
    evaluation = evaluate_paired(
        test, linear_estimators(test, test.config.ofdm.channel, (0.0, 30.0)), estimates
    )
    assert evaluation.errors.shape == (16, 5)
    assert np.all(np.isfinite(evaluation.nmse))


def test_training_is_repeatable_for_a_seed_on_cpu(tmp_path):
    train, validation = tiny_datasets(tmp_path)
    training = {
        "batch_size": 32,
        "maximum_epochs": 2,
        "patience": 5,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
    }
    first = train_channel_estimator(
        "a",
        9,
        CNN,
        training,
        train,
        validation,
        tmp_path / "a",
        torch.device("cpu"),
        lambda m: None,
    )
    second = train_channel_estimator(
        "a",
        9,
        CNN,
        training,
        train,
        validation,
        tmp_path / "b",
        torch.device("cpu"),
        lambda m: None,
    )
    assert first["selected_validation_mse"] == second["selected_validation_mse"]
