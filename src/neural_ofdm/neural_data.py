import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_ofdm._validation import integer
from neural_ofdm.benchmark import json_safe
from neural_ofdm.channel import ChannelConfig
from neural_ofdm.config import OFDMConfig
from neural_ofdm.estimation import ls_channel_estimate
from neural_ofdm.simulation import Observation, observe_frame

DATASET_SCHEMA_VERSION = 2
SPLITS = ("train", "validation", "test", "debug")
RNG_COMPONENTS = ("reference", "payload", "channel", "noise")


def source_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_digest(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("ascii")).hexdigest()


def complex_to_features(values: ArrayLike) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim not in (1, 2) or array.shape[-1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError("complex values must be finite [N] or [B,N]")
    features = np.stack((array.real, array.imag), axis=-2).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("complex values exceed the float32 feature range")
    return features


def features_to_complex(values: ArrayLike) -> NDArray[np.complex128]:
    array = np.asarray(values)
    if array.ndim not in (2, 3) or array.shape[-2] != 2 or not np.all(np.isfinite(array)):
        raise ValueError("features must be finite [2,N] or [B,2,N]")
    return array[..., 0, :].astype(np.float64) + 1j * array[..., 1, :].astype(np.float64)


def ls_estimate_for(config: OFDMConfig, observation: Observation) -> NDArray[np.complex128]:
    return ls_channel_estimate(
        observation.received_grid, config.pilot_indices, config.pilot_symbols
    )


@dataclass(frozen=True)
class DatasetConfig:
    ofdm: OFDMConfig = OFDMConfig(
        noise_mode="fixed", channel=ChannelConfig(delays=(0, 1, 3, 6), decay_samples=2.5)
    )
    count: int = 30000
    root_seed: int = 7301
    split: str = "train"
    condition_id: int = 0
    reference_range_db: tuple[float, float] = (0.0, 30.0)
    fixed_reference_db: float | None = None
    store_frames: bool = False

    def __post_init__(self) -> None:
        integer(self.count, "count", 1)
        integer(self.root_seed, "root_seed")
        integer(self.condition_id, "condition_id")
        if self.split not in SPLITS:
            raise ValueError("split must be train, validation, test or debug")
        if self.ofdm.noise_mode != "fixed":
            raise ValueError("learning data requires fixed independent noise")
        if self.ofdm.channel is None:
            raise ValueError("learning data requires an explicit ChannelConfig")
        if (
            not isinstance(self.reference_range_db, tuple)
            or len(self.reference_range_db) != 2
            or not np.all(np.isfinite(self.reference_range_db))
            or self.reference_range_db[0] > self.reference_range_db[1]
        ):
            raise ValueError("reference_range_db must be a finite increasing pair")
        if self.fixed_reference_db is not None and not np.isfinite(self.fixed_reference_db):
            raise ValueError("fixed_reference_db must be finite")
        if not isinstance(self.store_frames, bool):
            raise ValueError("store_frames must be boolean")

    @property
    def split_id(self) -> int:
        return SPLITS.index(self.split)

    def sample_streams(self, index: int) -> list[np.random.Generator]:
        integer(index, "sample index")
        if index >= self.count:
            raise ValueError("sample index is outside the split")
        identifiers = [self.root_seed, self.split_id, self.condition_id, index]
        return [
            np.random.default_rng(np.random.SeedSequence([*identifiers, component]))
            for component in range(len(RNG_COMPONENTS))
        ]

    def replay(self, index: int) -> tuple[OFDMConfig, Observation]:
        reference_rng, payload_rng, channel_rng, noise_rng = self.sample_streams(index)
        reference = self.fixed_reference_db
        if reference is None:
            reference = float(reference_rng.uniform(*self.reference_range_db))
        frame_config = replace(self.ofdm, reference_esn0_db=float(reference))
        observation = observe_frame(
            frame_config, payload_rng, channel_rng=channel_rng, noise_rng=noise_rng
        )
        return frame_config, observation


@dataclass(frozen=True)
class StoredDataset:
    config: DatasetConfig
    inputs: NDArray[np.float32]
    targets: NDArray[np.float32]
    references_db: NDArray[np.float64]
    received_grids: NDArray[np.complex128] | None
    reference_bits: NDArray[np.uint8] | None
    manifest: dict
    path: Path

    def verify_replay(self, indices: ArrayLike | None = None) -> int:
        selected = range(self.config.count) if indices is None else np.asarray(indices).tolist()
        checked = 0
        for index in selected:
            frame_config, observation = self.config.replay(int(index))
            ls = ls_estimate_for(frame_config, observation)
            equal = (
                np.array_equal(self.inputs[index], complex_to_features(ls))
                and np.array_equal(
                    self.targets[index], complex_to_features(observation.true_channel)
                )
                and self.references_db[index] == frame_config.reference_esn0_db
            )
            if self.received_grids is not None and self.reference_bits is not None:
                equal = (
                    equal
                    and np.array_equal(self.received_grids[index], observation.received_grid)
                    and np.array_equal(self.reference_bits[index], observation.reference_bits)
                )
            if not equal:
                raise ValueError(f"replayed sample {index} differs from the stored dataset")
            checked += 1
        return checked


def channel_from_dict(values: dict | None) -> ChannelConfig | None:
    if values is None:
        return None
    return ChannelConfig(
        **{
            key: value
            for key, value in values.items()
            if key not in ("delays", "powers", "fixed_taps")
        },
        delays=tuple(values["delays"]),
        powers=None if values.get("powers") is None else tuple(values["powers"]),
        fixed_taps=tuple(
            complex(*tap) if isinstance(tap, list) else complex(tap)
            for tap in values.get("fixed_taps", ChannelConfig().fixed_taps)
        ),
    )


def ofdm_from_dict(values: dict) -> OFDMConfig:
    settings = dict(values)

    def as_complex(value) -> complex:
        return complex(*value) if isinstance(value, list) else complex(value)

    channel = channel_from_dict(settings.pop("channel", None))
    if "pilot_value" in settings:
        settings["pilot_value"] = as_complex(settings["pilot_value"])
    if "channel_taps" in settings:
        settings["channel_taps"] = tuple(as_complex(tap) for tap in settings["channel_taps"])
    if "rayleigh_powers" in settings:
        settings["rayleigh_powers"] = tuple(settings["rayleigh_powers"])
    if settings.get("pilot_values") is not None:
        settings["pilot_values"] = tuple(as_complex(tap) for tap in settings["pilot_values"])
    return OFDMConfig(**settings, channel=channel)


def config_from_dict(values: dict) -> DatasetConfig:
    remaining = {
        key: value for key, value in values.items() if key not in ("ofdm", "reference_range_db")
    }
    return DatasetConfig(
        **remaining,
        ofdm=ofdm_from_dict(values["ofdm"]),
        reference_range_db=tuple(values["reference_range_db"]),
    )


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("neural-ofdm", "numpy", "scipy")},
    }


def build_manifest(config: DatasetConfig, path: Path, elapsed: float) -> dict:
    configuration = json_safe(asdict(config))
    references = (
        "fixed at fixed_reference_db"
        if config.fixed_reference_db is not None
        else "continuous uniform reference_esn0_db over reference_range_db"
    )
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "configuration": configuration,
        "configuration_sha256": text_digest(configuration),
        "source_sha256": source_digest(),
        "arrays_sha256": file_digest(path),
        "rng": (
            "PCG64 per sample: SeedSequence([root_seed, split_id, condition_id, sample_index,"
            " component]); components 0 reference, 1 payload, 2 channel, 3 noise"
        ),
        "split_ids": {name: index for index, name in enumerate(SPLITS)},
        "split": config.split,
        "split_id": config.split_id,
        "condition_id": config.condition_id,
        "groups": (
            "one independent channel per sample; group key (root_seed, split_id,"
            " condition_id, sample_index); no augmentation or shared channels"
        ),
        "count": config.count,
        "channel": json_safe(asdict(config.ofdm.channel)),
        "pilot_indices": config.ofdm.pilot_indices.tolist(),
        "pilot_symbols": json_safe([complex(value) for value in config.ofdm.pilot_symbols]),
        "modulation_order": config.ofdm.modulation_order,
        "noise_mode": "fixed",
        "noise_variance": "sigma^2=10^(-reference_esn0_db/10) with Es=1, independent of channel",
        "reference_sampling": references,
        "feature_information": (
            "periodic pilot LS real/imag only; no noise level, truth, payload,"
            " realized power or pilot mask is passed to model.forward"
        ),
        "target": "true H, ordinary unscaled FIR FFT, unshifted FFT order including DC",
        "arrays": {
            "inputs": {"shape": [config.count, 2, config.ofdm.fft_size], "dtype": "float32"},
            "targets": {"shape": [config.count, 2, config.ofdm.fft_size], "dtype": "float32"},
            "reference_esn0_db": {"shape": [config.count], "dtype": "float64"},
            "sample_ids": {"shape": [config.count], "dtype": "int64"},
            **(
                {
                    "received_grids": {
                        "shape": [config.count, config.ofdm.fft_size],
                        "dtype": "complex128",
                    },
                    "reference_bits": {
                        "shape": [config.count, config.ofdm.bits_per_frame],
                        "dtype": "uint8",
                    },
                }
                if config.store_frames
                else {}
            ),
        },
        "classical_dtype": "complex128",
        "generation_seconds": elapsed,
        "storage_bytes": path.stat().st_size,
        "evaluation_replay": (
            "replay identifiers plus this configuration and source digest; evaluation"
            " verifies LS, target, reference, received grid and bits before comparing"
        ),
        "runtime": runtime_versions(),
    }


def generate_dataset(config: DatasetConfig, directory: str | Path) -> StoredDataset:
    destination = Path(directory)
    if (destination / "manifest.json").exists():
        existing = load_dataset(destination)
        if existing.config != config:
            raise ValueError("existing dataset configuration differs; choose a new directory")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    size = config.ofdm.fft_size
    inputs = np.empty((config.count, 2, size), dtype=np.float32)
    targets = np.empty_like(inputs)
    references = np.empty(config.count, dtype=np.float64)
    grids = np.empty((config.count, size), dtype=np.complex128)
    bits = np.empty((config.count, config.ofdm.bits_per_frame), dtype=np.uint8)
    for index in range(config.count):
        frame_config, observation = config.replay(index)
        inputs[index] = complex_to_features(ls_estimate_for(frame_config, observation))
        targets[index] = complex_to_features(observation.true_channel)
        references[index] = frame_config.reference_esn0_db
        grids[index] = observation.received_grid
        bits[index] = observation.reference_bits
    elapsed = time.perf_counter() - start
    arrays = {
        "inputs": inputs,
        "targets": targets,
        "reference_esn0_db": references,
        "sample_ids": np.arange(config.count, dtype=np.int64),
    }
    if config.store_frames:
        arrays |= {"received_grids": grids, "reference_bits": bits}
    path = destination / "tensors.npz"
    np.savez(path, **arrays)
    manifest = build_manifest(config, path, elapsed)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    return load_dataset(destination)


def load_dataset(directory: str | Path) -> StoredDataset:
    path = Path(directory)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run: uv run python scripts/generate_dataset.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema")
    configuration = manifest["configuration"]
    if text_digest(configuration) != manifest["configuration_sha256"]:
        raise ValueError("dataset configuration failed integrity validation")
    if file_digest(path / "tensors.npz") != manifest["arrays_sha256"]:
        raise ValueError("dataset arrays failed integrity validation")
    config = config_from_dict(configuration)
    with np.load(path / "tensors.npz", allow_pickle=False) as arrays:
        inputs, targets = arrays["inputs"], arrays["targets"]
        references, identifiers = arrays["reference_esn0_db"], arrays["sample_ids"]
        grids = arrays["received_grids"] if config.store_frames else None
        bits = arrays["reference_bits"] if config.store_frames else None
    expected = (config.count, 2, config.ofdm.fft_size)
    if (
        inputs.shape != expected
        or targets.shape != expected
        or inputs.dtype != np.float32
        or targets.dtype != np.float32
    ):
        raise ValueError("stored tensor shape or dtype does not match manifest")
    if references.shape != (config.count,) or references.dtype != np.float64:
        raise ValueError("stored reference values are invalid")
    if not np.array_equal(identifiers, np.arange(config.count)):
        raise ValueError("stored sample identifiers are invalid")
    if not all(np.all(np.isfinite(array)) for array in (inputs, targets, references)):
        raise ValueError("stored values must be finite")
    if config.store_frames and (
        grids.shape != (config.count, config.ofdm.fft_size)
        or bits.shape != (config.count, config.ofdm.bits_per_frame)
    ):
        raise ValueError("stored frame arrays do not match the configuration")
    return StoredDataset(config, inputs, targets, references, grids, bits, manifest, path)
