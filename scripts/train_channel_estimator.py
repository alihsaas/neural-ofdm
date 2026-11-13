import argparse
import json
import os
from pathlib import Path

from neural_ofdm.neural_data import load_dataset
from neural_ofdm.neural_study import (
    PROTOCOL_PATH,
    file_logger,
    load_protocol,
    protocol_path,
    run_directory,
    run_training_settings,
)


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from neural_ofdm.neural_training import select_device, train_channel_estimator

    parser = argparse.ArgumentParser(description="Train learned OFDM channel estimators")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--run", nargs="+", default=["all"], help="protocol run names, or all")
    parser.add_argument("--seed", type=int, default=None, help="default: every protocol seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--train-split", default="train", choices=["train", "debug"])
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke tests only")
    parser.add_argument("--output", type=Path, default=None, help="default: protocol checkpoints")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    runs = list(protocol["runs"]) if args.run == ["all"] else args.run
    unknown = [run for run in runs if run not in protocol["runs"]]
    if unknown:
        parser.error(f"unknown runs: {unknown}")
    output = args.output or protocol_path(protocol, "checkpoints")
    device = select_device(args.device)
    log = file_logger(protocol_path(protocol, "results") / "logs" / "train.log")
    datasets = protocol_path(protocol, "training_datasets")
    train = load_dataset(datasets / args.train_split)
    validation = load_dataset(datasets / "validation")
    log(f"Device {device}; train {train.path.as_posix()} ({train.config.count} samples)")
    for run in runs:
        training = run_training_settings(protocol, run)
        if args.epochs is not None:
            training["maximum_epochs"] = args.epochs
            log(f"Epoch budget overridden to {args.epochs}; smoke run, not a study result")
        seeds = protocol["runs"][run]["seeds"] if args.seed is None else [args.seed]
        for seed in seeds:
            directory = run_directory(output, run, seed)
            if args.skip_existing and (directory / "best.json").exists():
                log(f"{run} seed {seed}: existing completed run kept")
                continue
            summary = train_channel_estimator(
                run,
                seed,
                protocol["runs"][run]["model"],
                training,
                train,
                validation,
                directory,
                device,
                log,
            )
            keys = (
                "run",
                "seed",
                "parameters",
                "receptive_field_bins",
                "selected_epoch",
                "selected_validation_mse",
                "epochs_completed",
                "training_seconds",
                "checkpoint_bytes",
            )
            log(json.dumps({key: summary[key] for key in keys}))


if __name__ == "__main__":
    main()
