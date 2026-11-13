import time
from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from neural_ofdm.config import OFDMConfig
from neural_ofdm.estimation import ls_channel_estimate
from neural_ofdm.neural_data import complex_to_features, features_to_complex
from neural_ofdm.neural_evaluation import lmmse_for
from neural_ofdm.simulation import Observation, receive_with_estimate


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    function: Callable[[], object], device: torch.device, warmup: int, repeats: int
) -> dict:
    for _ in range(warmup):
        function()
    synchronize(device)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        durations.append(time.perf_counter() - start)
    values = np.asarray(durations)
    return {
        "median_seconds": float(np.median(values)),
        "p10_seconds": float(np.quantile(values, 0.1)),
        "p90_seconds": float(np.quantile(values, 0.9)),
        "repeats": repeats,
        "warmup": warmup,
    }


def forward_resident(model: torch.nn.Module, values: torch.Tensor) -> None:
    with torch.inference_mode():
        model(values)


def forward_with_transfer(model: torch.nn.Module, values: np.ndarray, device: torch.device) -> None:
    with torch.inference_mode():
        model(torch.as_tensor(values, device=device)).cpu().numpy()


def forward_flops(model: torch.nn.Module, features: np.ndarray) -> dict:
    from torch.utils.flop_counter import FlopCounterMode

    batch = torch.as_tensor(features[:1], device=next(model.parameters()).device)
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter:
        model(batch)
    return {
        "flops_per_frame": int(counter.get_total_flops()),
        "flop_convention": (
            "torch.utils.flop_counter.FlopCounterMode, batch 1: 2 FLOPs per multiply-add in"
            " matmul/linear/conv/attention kernels; elementwise ops, norms and softmax excluded;"
            " counted with autograd enabled so the unfused Transformer path is measured"
        ),
    }


def gpu_memory(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(torch.as_tensor(features, device=device))
    torch.cuda.synchronize(device)
    return {
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def estimator_timings(
    model: torch.nn.Module,
    name: str,
    features: np.ndarray,
    device: torch.device,
    batch_sizes: tuple[int, ...] = (1, 256),
    warmup: int = 20,
    repeats: int = 200,
) -> list[dict]:
    rows = []
    model.eval()
    for batch_size in batch_sizes:
        batch = features[:batch_size]
        resident = torch.as_tensor(batch, device=device)

        for scope, function in (
            (
                "model forward only, input already on device",
                partial(forward_resident, model, resident),
            ),
            (
                "host float32 -> device -> forward -> host",
                partial(forward_with_transfer, model, batch, device),
            ),
        ):
            result = time_call(function, device, warmup, repeats)
            rows.append(
                {
                    "estimator": name,
                    "device": str(device),
                    "threads": torch.get_num_threads(),
                    "batch_size": batch_size,
                    "scope": scope,
                    **result,
                    "median_seconds_per_frame": result["median_seconds"] / batch_size,
                    **gpu_memory(model, batch, device),
                }
            )
    return rows


def classical_and_pipeline_timings(
    config: OFDMConfig,
    observation: Observation,
    model: torch.nn.Module,
    device: torch.device,
    model_name: str = "cnn_residual",
    warmup: int = 20,
    repeats: int = 200,
) -> list[dict]:
    grid = observation.received_grid
    estimator = lmmse_for(config, config.channel, observation.noise_variance)

    def ls_only() -> None:
        ls_channel_estimate(grid, config.pilot_indices, config.pilot_symbols)

    def lmmse_prepare() -> None:
        lmmse_for(config, config.channel, observation.noise_variance)

    def lmmse_apply() -> None:
        estimator.estimate(grid)

    def cnn_pipeline() -> None:
        ls = ls_channel_estimate(grid, config.pilot_indices, config.pilot_symbols)
        with torch.inference_mode():
            output = model(torch.as_tensor(complex_to_features(ls)[None], device=device))
        estimate = features_to_complex(output.cpu().numpy())[0]
        receive_with_estimate(config, observation, estimate, csi="external", equalizer="zf")

    def lmmse_pipeline() -> None:
        receive_with_estimate(
            config, observation, estimator.estimate(grid), csi="external", equalizer="zf"
        )

    def ls_pipeline() -> None:
        estimate = ls_channel_estimate(grid, config.pilot_indices, config.pilot_symbols)
        receive_with_estimate(config, observation, estimate, csi="external", equalizer="zf")

    cases = (
        ("ls", "LS pilot division and periodic interpolation, NumPy complex128", ls_only),
        ("lmmse", "offline: prior statistics + Cholesky operator preparation", lmmse_prepare),
        ("lmmse", "online: precomputed (N,P) operator applied to pilot innovations", lmmse_apply),
        ("ls", "complete frame: LS + ZF + demodulation + metrics", ls_pipeline),
        ("lmmse", "complete frame: LMMSE apply + ZF + demodulation + metrics", lmmse_pipeline),
        (
            model_name,
            "complete frame: LS + float32 features + transfer + network + ZF + demodulation"
            " + metrics",
            cnn_pipeline,
        ),
    )
    rows = []
    for name, scope, function in cases:
        result = time_call(function, device, warmup, repeats)
        rows.append(
            {
                "estimator": name,
                "device": str(device) if name not in ("ls", "lmmse") else "cpu",
                "threads": torch.get_num_threads(),
                "batch_size": 1,
                "scope": scope,
                **result,
                "median_seconds_per_frame": result["median_seconds"],
            }
        )
    return rows
