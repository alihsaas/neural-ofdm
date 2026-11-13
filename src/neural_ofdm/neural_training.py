import copy
import csv
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from neural_ofdm.neural_data import StoredDataset, file_digest, source_digest
from neural_ofdm.neural_models import AffineNormalization, build_model, parameter_count

CHECKPOINT_SCHEMA_VERSION = 1


def complex_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must share shape [B,2,N]")
    return torch.sum((prediction - target) ** 2, dim=1).mean()


def per_frame_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.sum((prediction - target) ** 2, dim=(1, 2))
    energy = torch.sum(target**2, dim=(1, 2))
    return error / energy


def fit_normalization(train_inputs: NDArray[np.float32]) -> AffineNormalization:
    values = np.asarray(train_inputs, dtype=np.float64)
    offset = values.mean(axis=(0, 2))
    scale = values.std(axis=(0, 2))
    return AffineNormalization(
        torch.as_tensor(offset, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def hardware_description(device: torch.device) -> dict:
    description = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
    }
    if device.type == "cuda":
        description["gpu_name"] = torch.cuda.get_device_name(device)
    return description


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("neural-ofdm", "numpy", "scipy", "torch")},
    }


def predict(
    model: torch.nn.Module,
    features: NDArray[np.float32],
    device: torch.device,
    batch_size: int = 1024,
) -> NDArray[np.float32]:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], device=device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def evaluate_split(
    model: torch.nn.Module, dataset: StoredDataset, device: torch.device, batch_size: int
) -> tuple[float, float]:
    prediction = torch.as_tensor(predict(model, dataset.inputs, device, batch_size))
    target = torch.as_tensor(dataset.targets)
    return (
        float(complex_mse(prediction.double(), target.double())),
        float(per_frame_nmse(prediction.double(), target.double()).mean()),
    )


def checkpoint_metadata(
    run: str,
    seed: int,
    model_config: dict,
    training: dict,
    train: StoredDataset,
    validation: StoredDataset,
    fft_size: int,
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run": run,
        "seed": seed,
        "model": model_config,
        "fft_size": fft_size,
        "training": training,
        "input_features": "periodic pilot LS real/imag, shape [B,2,N], float32",
        "output": "refined H real/imag, shape [B,2,N], physical units",
        "train_dataset": {
            "path": train.path.as_posix(),
            "arrays_sha256": train.manifest["arrays_sha256"],
            "configuration_sha256": train.manifest["configuration_sha256"],
        },
        "validation_dataset": {
            "path": validation.path.as_posix(),
            "arrays_sha256": validation.manifest["arrays_sha256"],
            "configuration_sha256": validation.manifest["configuration_sha256"],
        },
        "source_sha256": source_digest(),
        "runtime": runtime_versions(),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": state,
            "metadata_json": json.dumps(metadata, sort_keys=True),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    metadata = json.loads(payload["metadata_json"])
    state = payload["model_state"]
    normalization = AffineNormalization(
        state["normalization.offset"].cpu(), state["normalization.scale"].cpu()
    )
    model = build_model(metadata["model"], metadata["fft_size"], normalization)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, metadata


def write_history(directory: Path, history: list[dict]) -> None:
    with (directory / "history.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_channel_estimator(
    run: str,
    seed: int,
    model_config: dict,
    training: dict,
    train: StoredDataset,
    validation: StoredDataset,
    output_directory: str | Path,
    device: torch.device,
    log=print,
) -> dict:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    fft_size = train.config.ofdm.fft_size
    normalization = (
        fit_normalization(train.inputs) if model_config.get("normalization") == "affine" else None
    )
    model = build_model(model_config, fft_size, normalization).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train.inputs), torch.as_tensor(train.targets)),
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training.get("num_workers", 0),
        generator=torch.Generator().manual_seed(seed),
    )
    metadata = checkpoint_metadata(run, seed, model_config, training, train, validation, fft_size)
    metadata["parameters"] = parameter_count(model)
    metadata["receptive_field_bins"] = model.receptive_field
    metadata["hardware"] = hardware_description(device)
    metadata["determinism"] = (
        "seeded Python/NumPy/torch and DataLoader generator; deterministic algorithms"
        " requested with warn_only; bitwise equality across devices or releases not guaranteed"
    )
    initial_loss, initial_nmse = evaluate_split(model, validation, device, 4096)
    log(f"{run} seed {seed}: initial validation MSE {initial_loss:.6g}, NMSE {initial_nmse:.6g}")
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, training["maximum_epochs"] + 1):
        model.train()
        squared_error_sum = 0.0
        sample_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = complex_mse(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(bool(torch.all(torch.isfinite(gradient))) for gradient in gradients):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            squared_error_sum += float(loss.detach()) * len(inputs)
            sample_count += len(inputs)
        train_loss = squared_error_sum / sample_count
        validation_loss, validation_nmse = evaluate_split(model, validation, device, 4096)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            metadata.update(
                {
                    "selected_epoch": epoch,
                    "selected_validation_mse": validation_loss,
                    "selected_validation_nmse_mean": validation_nmse,
                }
            )
            save_checkpoint(directory / "best.pt", model, metadata)
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_nmse_mean": validation_nmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - start,
                "selected_so_far": best_epoch,
            }
        )
        write_history(directory, history)
        log(
            f"{run} seed {seed} epoch {epoch}: train {train_loss:.6g}, validation"
            f" {validation_loss:.6g}, NMSE {validation_nmse:.6g}, best epoch {best_epoch}"
        )
        if stale >= training["patience"]:
            break
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    reloaded, _ = load_checkpoint(directory / "best.pt", device)
    sample = torch.as_tensor(validation.inputs[:256], device=device)
    with torch.inference_mode():
        model.eval()
        reload_equal = bool(torch.equal(model(sample), reloaded(sample)))
    summary = {
        **metadata,
        "initial_validation_mse": initial_loss,
        "initial_validation_nmse_mean": initial_nmse,
        "epochs_completed": len(history),
        "examples_seen": len(history) * train.config.count,
        "train_samples": train.config.count,
        "stopped_early": len(history) < training["maximum_epochs"],
        "training_seconds": elapsed,
        "checkpoint_bytes": (directory / "best.pt").stat().st_size,
        "checkpoint_sha256": file_digest(directory / "best.pt"),
        "reload_prediction_equal": reload_equal,
        "resume_supported": False,
    }
    (directory / "best.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="ascii")
    log(
        f"{run} seed {seed}: selected epoch {best_epoch} with validation MSE {best_loss:.6g};"
        f" reload equal {reload_equal}"
    )
    return summary
