import json
from dataclasses import dataclass
from pathlib import Path

from neural_ofdm.neural_data import DatasetConfig, ofdm_from_dict

PROTOCOL_PATH = Path("configs/neural_channel_estimation/protocol.json")


@dataclass(frozen=True)
class TestCondition:
    name: str
    condition_id: int
    kind: str
    references_db: tuple[float, ...]
    datasets: tuple[DatasetConfig, ...]


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="ascii"))
    if protocol.get("schema_version") != 2:
        raise ValueError("unsupported neural protocol schema")
    return protocol


def protocol_path(protocol: dict, key: str) -> Path:
    return Path(protocol["paths"][key])


def run_training_settings(protocol: dict, run: str) -> dict:
    settings = dict(protocol["training"])
    settings.update(protocol["runs"][run].get("training", {}))
    return settings


def checkpoint_entries(protocol: dict) -> list[tuple[str, int, Path]]:
    entries = [
        (run, seed, Path(item["checkpoints"]))
        for run, item in protocol.get("external_runs", {}).items()
        for seed in item["seeds"]
    ]
    entries += [
        (run, seed, protocol_path(protocol, "checkpoints"))
        for run, item in protocol["runs"].items()
        for seed in item["seeds"]
    ]
    return entries


def merged_ofdm(protocol: dict, overrides: dict | None = None):
    values = dict(protocol["ofdm"])
    values.update(overrides or {})
    return ofdm_from_dict(values)


def reference_label(reference_db: float) -> str:
    sign = "m" if reference_db < 0 else ""
    return f"ref_{sign}{abs(reference_db):04.1f}db".replace(".", "p")


def training_dataset(protocol: dict, split: str) -> DatasetConfig:
    return DatasetConfig(
        ofdm=merged_ofdm(protocol),
        count=protocol["split_counts"][split],
        root_seed=protocol["dataset_root_seed"],
        split=split,
        condition_id=0,
        reference_range_db=tuple(float(value) for value in protocol["training_reference_range_db"]),
    )


def evaluation_conditions(protocol: dict) -> tuple[TestCondition, ...]:
    conditions = []
    for item in protocol["test_conditions"]:
        references = list(protocol["primary_references_db"])
        if item["references"] == "primary_and_extrapolation":
            references = sorted(references + list(protocol["extrapolation_references_db"]))
        ofdm = merged_ofdm(protocol, item["ofdm"])
        datasets = tuple(
            DatasetConfig(
                ofdm=ofdm,
                count=protocol["test_frames_per_point"],
                root_seed=protocol["dataset_root_seed"],
                split="test",
                condition_id=item["condition_id"],
                reference_range_db=tuple(
                    float(value) for value in protocol["training_reference_range_db"]
                ),
                fixed_reference_db=float(reference),
                store_frames=True,
            )
            for reference in references
        )
        conditions.append(
            TestCondition(
                item["name"],
                item["condition_id"],
                item["kind"],
                tuple(float(value) for value in references),
                datasets,
            )
        )
    identifiers = [condition.condition_id for condition in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("test condition identifiers must be unique")
    return tuple(conditions)


def dataset_directory(root: Path, config: DatasetConfig, condition: str | None = None) -> Path:
    if config.split != "test":
        return root / config.split
    return root / "test" / condition / reference_label(config.fixed_reference_db)


def run_directory(root: Path, run: str, seed: int) -> Path:
    return root / run / f"seed_{seed}"


def file_logger(path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with destination.open("a", encoding="ascii", errors="replace") as handle:
            handle.write(message + "\n")

    return log
