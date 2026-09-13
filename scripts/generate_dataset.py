import argparse
import json
from pathlib import Path

import numpy as np

from neural_ofdm.neural_data import generate_dataset
from neural_ofdm.neural_study import (
    PROTOCOL_PATH,
    dataset_directory,
    evaluation_conditions,
    file_logger,
    load_protocol,
    protocol_path,
    training_dataset,
)


def describe(dataset, name: str) -> dict:
    references = dataset.references_db
    histogram, edges = np.histogram(references, bins=6)
    return {
        "name": name,
        "path": dataset.path.as_posix(),
        "split": dataset.config.split,
        "condition_id": dataset.config.condition_id,
        "count": dataset.config.count,
        "reference_min_db": float(references.min()),
        "reference_max_db": float(references.max()),
        "reference_mean_db": float(references.mean()),
        "reference_histogram_counts": histogram.tolist(),
        "reference_histogram_edges_db": edges.tolist(),
        "generation_seconds": dataset.manifest["generation_seconds"],
        "storage_bytes": dataset.manifest["storage_bytes"],
        "arrays_sha256": dataset.manifest["arrays_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible neural estimation datasets")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["debug", "train", "validation", "test"],
        choices=["debug", "train", "validation", "test"],
    )
    parser.add_argument("--verify", type=int, default=64, help="replayed samples checked per set")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    results = protocol_path(protocol, "results")
    log = file_logger(results / "logs" / "generate_dataset.log")
    jobs = [
        (split, training_dataset(protocol, split), protocol_path(protocol, "training_datasets"))
        for split in args.splits
        if split != "test"
    ]
    if "test" in args.splits:
        jobs += [
            (condition.name, config, protocol_path(protocol, "test_datasets"))
            for condition in evaluation_conditions(protocol)
            for config in condition.datasets
        ]
    summaries = []
    for name, config, root in jobs:
        directory = dataset_directory(root, config, name)
        dataset = generate_dataset(config, directory)
        rng = np.random.default_rng(np.random.SeedSequence([protocol["dataset_root_seed"], 99]))
        indices = rng.choice(config.count, size=min(args.verify, config.count), replace=False)
        summary = describe(dataset, name)
        summary["replay_checked"] = dataset.verify_replay(np.sort(indices))
        summaries.append(summary)
        log(
            f"{directory.as_posix()}: {config.count} samples, references"
            f" {summary['reference_min_db']:.2f}..{summary['reference_max_db']:.2f} dB,"
            f" {summary['storage_bytes'] / 1e6:.2f} MB, {summary['generation_seconds']:.1f} s,"
            f" replay verified {summary['replay_checked']}"
        )
    output = results / "datasets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output.read_text(encoding="ascii")) if output.exists() else {}
    existing.update({item["path"]: item for item in summaries})
    output.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="ascii")
    log(f"Dataset summary written to {output.as_posix()}")


if __name__ == "__main__":
    main()
