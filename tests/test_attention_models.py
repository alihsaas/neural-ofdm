import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from neural_ofdm.hybrid import ChannelHybrid
from neural_ofdm.neural_data import DatasetConfig, generate_dataset
from neural_ofdm.neural_models import ChannelCNN, build_model
from neural_ofdm.neural_training import (
    load_checkpoint,
    predict,
    save_checkpoint,
    train_channel_estimator,
)
from neural_ofdm.positional import PositionalEncoding, circular_fourier_features
from neural_ofdm.transformer import ChannelTransformer

SMALL = {"d_model": 16, "heads": 4, "layers": 2, "feedforward": 32, "dropout": 0.0}


def features(batch: int = 3, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, 2, 64, generator=torch.Generator().manual_seed(seed))


def randomize(model: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in model.output_projection.parameters():
            parameter.normal_(0, 0.2)
    return model.eval()


def test_circular_features_are_deterministic_periodic_and_movable():
    table = circular_fourier_features(64, 8)
    assert table.shape == (64, 8) and table.dtype == torch.float32
    torch.testing.assert_close(table, circular_fourier_features(64, 8), rtol=0, atol=0)
    assert table[5, 0] == pytest.approx(math.sin(2 * math.pi * 5 / 64), abs=1e-6)
    assert table[5, 3] == pytest.approx(math.cos(2 * math.pi * 2 * 5 / 64), abs=1e-6)
    wrapped = torch.tensor([math.sin(2 * math.pi * r * 64 / 64) for r in (1, 2, 3, 4)])
    torch.testing.assert_close(table[0, 0::2], wrapped, atol=1e-6, rtol=0)
    encoding = PositionalEncoding("circular", 64, 8).to(torch.float64)
    assert encoding.table.dtype == torch.float64
    tokens = torch.zeros(2, 64, 8, dtype=torch.float64)
    assert encoding(tokens).dtype == torch.float64
    torch.testing.assert_close(encoding(tokens)[1], table.double())
    with pytest.raises(ValueError):
        encoding(torch.zeros(2, 32, 8, dtype=torch.float64))
    with pytest.raises(ValueError):
        circular_fourier_features(64, 7)
    assert "table" not in encoding.state_dict()


def test_configuration_validation():
    with pytest.raises(ValueError):
        ChannelTransformer(d_model=30, heads=4)
    with pytest.raises(ValueError):
        ChannelTransformer(positional="learned")
    with pytest.raises(ValueError):
        ChannelHybrid(kernel_size=4)
    with pytest.raises(ValueError):
        ChannelCNN(blocks=2, dilations=[1])


def test_token_shapes_and_finite_gradients_for_batches():
    inputs = features(batch=5)
    transformer = ChannelTransformer(**SMALL, zero_init_output=False)
    hybrid = ChannelHybrid(conv_width=8, **SMALL, zero_init_output=False)
    assert transformer.tokens(inputs).shape == (5, 64, 16)
    assert hybrid.tokens(inputs).shape == (5, 64, 16)
    for model in (transformer, hybrid):
        output = model(inputs)
        assert output.shape == (5, 2, 64)
        output.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
            for parameter in model.parameters()
        )


def test_projection_keeps_bin_order():
    model = ChannelTransformer(**SMALL, positional="none")
    inputs = torch.zeros(1, 2, 64)
    inputs[0, 0, 7] = 1.0
    tokens = model.tokens(inputs)
    bias = model.input_projection.bias
    differs = torch.nonzero(~torch.isclose(tokens[0], bias.expand(64, -1)).all(dim=1)).flatten()
    assert differs.tolist() == [7]


def test_zero_initialized_attention_models_start_at_ls():
    inputs = features()
    for model in (ChannelTransformer(**SMALL), ChannelHybrid(conv_width=8, **SMALL)):
        model.eval()
        torch.testing.assert_close(model(inputs), inputs, rtol=0, atol=0)


def test_encoder_layers_are_initialized_independently():
    torch.manual_seed(0)
    model = ChannelTransformer(**SMALL)
    first, second = model.encoder.layers
    assert not torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight)
    assert not torch.equal(first.linear1.weight, second.linear1.weight)


def test_no_cross_sample_mixing_in_evaluation():
    inputs = features(batch=6)
    for model in (
        randomize(ChannelTransformer(**SMALL)),
        randomize(ChannelHybrid(conv_width=8, **SMALL)),
    ):
        with torch.inference_mode():
            batch = model(inputs)
            alone = model(inputs[3:4])
        torch.testing.assert_close(alone[0], batch[3], rtol=1e-5, atol=1e-6)


def test_pure_transformer_without_position_is_permutation_equivariant():
    model = randomize(ChannelTransformer(**SMALL, positional="none"))
    inputs = features()
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        torch.testing.assert_close(
            model(inputs[:, :, permutation]),
            model(inputs)[:, :, permutation],
            rtol=1e-5,
            atol=1e-5,
        )
    positioned = randomize(ChannelTransformer(**SMALL, positional="circular"))
    with torch.no_grad():
        positioned.load_state_dict(model.state_dict())
        difference = positioned(inputs[:, :, permutation]) - positioned(inputs)[:, :, permutation]
    assert float(difference.abs().max()) > 1e-4


def test_hybrid_without_position_is_circular_shift_equivariant_only():
    model = randomize(ChannelHybrid(conv_width=8, **SMALL, positional="none"))
    inputs = features()
    with torch.inference_mode():
        torch.testing.assert_close(
            model(torch.roll(inputs, 5, dims=-1)),
            torch.roll(model(inputs), 5, dims=-1),
            rtol=1e-5,
            atol=1e-5,
        )
        permutation = torch.randperm(64, generator=torch.Generator().manual_seed(2))
        difference = model(inputs[:, :, permutation]) - model(inputs)[:, :, permutation]
    assert float(difference.abs().max()) > 1e-4


def test_mask_polarity_blocks_true_positions_and_rejects_empty_rows():
    model = randomize(ChannelTransformer(**SMALL, positional="none"))
    self_only = ~torch.eye(64, dtype=torch.bool)
    inputs = features(batch=1)
    changed = inputs.clone()
    changed[:, :, 20:] += 3.0
    with torch.inference_mode():
        isolated = model(inputs, self_only)[..., :20]
        isolated_changed = model(changed, self_only)[..., :20]
        full = model(inputs)[..., :20]
        full_changed = model(changed)[..., :20]
    torch.testing.assert_close(isolated, isolated_changed, rtol=1e-5, atol=1e-5)
    assert float((full - full_changed).abs().max()) > 1e-4
    blocked = torch.zeros(64, 64, dtype=torch.bool)
    blocked[3] = True
    with pytest.raises(ValueError):
        model(inputs, blocked)
    with pytest.raises(ValueError):
        model(inputs, torch.zeros(64, 64))


def test_attention_diagnostics_match_predictions_and_are_normalized():
    inputs = features(batch=2)
    for model in (
        randomize(ChannelTransformer(**SMALL)),
        randomize(ChannelHybrid(conv_width=8, **SMALL)),
    ):
        with torch.inference_mode():
            ordinary = model(inputs)
            diagnostic, maps = model.forward_with_attention(inputs)
        torch.testing.assert_close(diagnostic, ordinary, rtol=1e-5, atol=1e-5)
        assert maps.shape == (2, 2, 4, 64, 64)
        torch.testing.assert_close(maps.sum(-1), torch.ones(2, 2, 4, 64), rtol=0, atol=1e-5)
        assert float(maps.min()) >= 0


def test_dilated_cnn_receptive_field_matches_gradient_support():
    model = ChannelCNN(width=4, blocks=2, kernel_size=3, dilations=[1, 2], zero_init_output=False)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, 0.1, 0.2)
    assert model.receptive_field == 1 + 2 * (2 + 2 * 3)
    inputs = torch.zeros(1, 2, 64, requires_grad=True)
    model(inputs)[0, 0, 0].backward()
    support = torch.nonzero(inputs.grad.abs().sum(dim=1)[0]).flatten().tolist()
    assert sorted(support) == sorted(offset % 64 for offset in range(-8, 9))


@pytest.mark.parametrize(
    "config",
    [
        {"architecture": "transformer", **SMALL, "positional": "circular"},
        {"architecture": "hybrid", "conv_width": 8, "conv_blocks": 1, **SMALL},
        {"architecture": "cnn", "width": 8, "blocks": 2, "dilations": [1, 4]},
    ],
)
def test_checkpoint_reload_preserves_predictions_for_each_type(tmp_path, config):
    model = build_model(config, 64)
    if hasattr(model, "output_projection"):
        randomize(model)
    model.eval()
    save_checkpoint(tmp_path / "model.pt", model, {"model": config, "fft_size": 64})
    reloaded, metadata = load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    assert metadata["model"] == config and type(reloaded) is type(model)
    inputs = features().numpy()
    np.testing.assert_array_equal(
        predict(model, inputs, torch.device("cpu")), predict(reloaded, inputs, torch.device("cpu"))
    )


def test_shared_trainer_runs_attention_models(tmp_path):
    base = DatasetConfig()
    train = generate_dataset(replace(base, split="debug", count=64), tmp_path / "debug")
    validation = generate_dataset(replace(base, split="validation", count=32), tmp_path / "val")
    training = {
        "batch_size": 16,
        "maximum_epochs": 2,
        "patience": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    for name, config in (
        ("transformer", {"architecture": "transformer", **SMALL}),
        ("hybrid", {"architecture": "hybrid", "conv_width": 8, **SMALL}),
    ):
        summary = train_channel_estimator(
            name,
            3,
            config,
            training,
            train,
            validation,
            tmp_path / name,
            torch.device("cpu"),
            lambda message: None,
        )
        assert summary["reload_prediction_equal"]
        assert np.isfinite(summary["selected_validation_mse"])
        assert summary["selected_validation_mse"] != summary["initial_validation_mse"]
