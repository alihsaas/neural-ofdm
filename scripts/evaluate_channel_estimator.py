import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

import matplotlib
import numpy as np

from neural_ofdm.benchmark import json_safe, write_csv
from neural_ofdm.neural_data import (
    features_to_complex,
    file_digest,
    load_dataset,
    ls_estimate_for,
    source_digest,
)
from neural_ofdm.neural_evaluation import (
    evaluate_paired,
    linear_estimators,
    mixture_noise_variance,
    replay_observations,
    save_condition,
    summarize,
)
from neural_ofdm.neural_study import (
    PROTOCOL_PATH,
    checkpoint_entries,
    dataset_directory,
    evaluation_conditions,
    file_logger,
    load_protocol,
    merged_ofdm,
    protocol_path,
    run_directory,
)
from neural_ofdm.simulation import receive_with_estimate


def load_models(protocol: dict, device, load_checkpoint) -> dict:
    models = {}
    for run, seed, root in checkpoint_entries(protocol):
        path = run_directory(root, run, seed) / "best.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run: uv run python scripts/train_channel_estimator.py"
                f" --protocol <protocol defining {run}> --run {run} --seed {seed}"
            )
        model, metadata = load_checkpoint(path, device)
        models[f"{run}/seed_{seed}"] = (model, metadata, path)
    return models


def collect_training(models: dict, results: Path) -> list:
    summary = []
    for name, (_, _, path) in models.items():
        run, seed_text = name.split("/seed_")
        destination = results / "training" / run / f"seed_{seed_text}"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("history.csv", "best.json"):
            shutil.copyfile(path.parent / filename, destination / filename)
        details = json.loads((path.parent / "best.json").read_text(encoding="ascii"))
        summary.append(
            {
                "run": run,
                "seed": int(seed_text),
                "architecture": details["model"]["architecture"],
                "parameters": details["parameters"],
                "receptive_field_bins": details["receptive_field_bins"],
                "learning_rate": details["training"]["learning_rate"],
                "epochs_completed": details["epochs_completed"],
                "examples_seen": details["examples_seen"],
                "stopped_early": details["stopped_early"],
                "selected_epoch": details["selected_epoch"],
                "selected_validation_mse": details["selected_validation_mse"],
                "selected_validation_nmse_mean": details["selected_validation_nmse_mean"],
                "initial_validation_mse": details["initial_validation_mse"],
                "training_seconds": details["training_seconds"],
                "seconds_per_epoch": details["training_seconds"] / details["epochs_completed"],
                "checkpoint_bytes": details["checkpoint_bytes"],
                "checkpoint_sha256": file_digest(path),
                "checkpoint_path": path.as_posix(),
                "reload_prediction_equal": details["reload_prediction_equal"],
                "device": details["hardware"]["device"],
                "gpu_name": details["hardware"].get("gpu_name", ""),
            }
        )
    (results / "models.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    write_csv(results / "models.csv", summary)
    return summary


def example_models(protocol: dict) -> list[str]:
    seed = protocol["representative_seed"]
    return [f"{run}/seed_{seed}" for run in protocol["main_runs"]]


def save_example(protocol, dataset, observations, estimators, neural, results: Path) -> None:
    frame = protocol["evaluation"]["example_frame"]
    names = example_models(protocol)
    frame_config, observation = observations[frame]
    estimates = {
        "true": observation.true_channel,
        "ls": ls_estimate_for(frame_config, observation),
        "lmmse_matched": estimators["lmmse_matched"].estimate(observation.received_grid),
    }
    estimates |= {name.split("/")[0]: neural[name][frame] for name in names}
    nmse = {
        key: float(
            np.sum(np.abs(values - observation.true_channel) ** 2)
            / np.sum(np.abs(observation.true_channel) ** 2)
        )
        for key, values in estimates.items()
        if key != "true"
    }
    count = 50
    equalized, ber = {}, {}
    for key in ("perfect", "ls", "lmmse_matched", *[name.split("/")[0] for name in names]):
        symbols, errors = [], 0
        for index in range(count):
            config, item = observations[index]
            if key == "perfect":
                estimate = item.true_channel
            elif key == "ls":
                estimate = ls_estimate_for(config, item)
            elif key == "lmmse_matched":
                estimate = estimators["lmmse_matched"].estimate(item.received_grid)
            else:
                estimate = neural[f"{key}/seed_{protocol['representative_seed']}"][index]
            result = receive_with_estimate(config, item, estimate, csi="external", equalizer="zf")
            symbols.append(result.decision_grid[config.data_indices])
            errors += int(np.count_nonzero(result.reference_bits != result.estimated_bits))
        equalized[f"equalized_{key}"] = np.concatenate(symbols)
        ber[key] = errors / (count * frame_config.bits_per_frame)
    np.savez(
        results / "example.npz",
        pilot_indices=frame_config.pilot_indices,
        **estimates,
        **equalized,
    )
    details = {
        "condition": "primary",
        "frame": frame,
        "reference_db": dataset.config.fixed_reference_db,
        "selection": "frame index and reference predeclared in the protocol",
        "nmse": nmse,
        "constellation_frames": count,
        "constellation_ber": ber,
        "models": names,
    }
    (results / "example.json").write_text(json.dumps(details, indent=2) + "\n", encoding="ascii")


def save_attention_maps(protocol, models, datasets_root: Path, results: Path, device, torch):
    seed = protocol["representative_seed"]
    names = [
        f"{run}/seed_{seed}"
        for run in protocol["main_runs"]
        if hasattr(models.get(f"{run}/seed_{seed}", (None,))[0], "forward_with_attention")
    ]
    if not names:
        return []
    conditions = {condition.name: condition for condition in evaluation_conditions(protocol)}
    cases = [
        ("primary", reference)
        for reference in protocol["evaluation"]["attention_example_references_db"]
    ]
    cases.append(("pilot_spacing_8", protocol["evaluation"]["attention_example_references_db"][-1]))
    arrays, records = {}, []
    frame = protocol["evaluation"]["example_frame"]
    for condition_name, reference in cases:
        condition = conditions[condition_name]
        config = next(item for item in condition.datasets if item.fixed_reference_db == reference)
        dataset = load_dataset(dataset_directory(datasets_root, config, condition_name))
        sample = torch.as_tensor(dataset.inputs[frame : frame + 1], device=device)
        for name in names:
            model = models[name][0]
            model.eval()
            with torch.inference_mode():
                ordinary = model(sample)
                diagnostic, maps = model.forward_with_attention(sample)
            difference = float((ordinary - diagnostic).abs().max())
            row_sums = maps.sum(dim=-1)
            key = f"{name.split('/')[0]}__{condition_name}__{reference:g}db".replace("-", "m")
            arrays[key] = maps[0].float().cpu().numpy()
            records.append(
                {
                    "key": key,
                    "model": name,
                    "condition": condition_name,
                    "reference_db": reference,
                    "frame": frame,
                    "pilot_indices": config.ofdm.pilot_indices.tolist(),
                    "shape": "[layers, heads, query_bin, key_bin]",
                    "map_dims": list(maps.shape[1:]),
                    "row_sum_max_error": float((row_sums - 1).abs().max()),
                    "prediction_max_abs_difference": difference,
                    "source": "post-softmax MultiheadAttention weights, eval mode, per head",
                }
            )
    if arrays:
        np.savez(results / "attention_maps.npz", **arrays)
        (results / "attention_maps.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="ascii"
        )
    return records


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    matplotlib.use("Agg")
    from neural_ofdm.study_figures import save_study_figures

    parser = argparse.ArgumentParser(description="Paired receiver evaluation of learned estimators")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="rerun timing, FLOPs, metadata and figures; keep existing evaluation outputs",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    results = protocol_path(protocol, "results")
    datasets_root = protocol_path(protocol, "test_datasets")
    log = file_logger(results / "logs" / "evaluate.log")
    if args.plots_only:
        for path in save_study_figures(results, protocol):
            log(f"Saved {path.as_posix()}")
        return
    import torch

    from neural_ofdm.neural_models import parameter_count
    from neural_ofdm.neural_timing import (
        classical_and_pipeline_timings,
        estimator_timings,
        forward_flops,
    )
    from neural_ofdm.neural_training import (
        hardware_description,
        load_checkpoint,
        predict,
        runtime_versions,
        select_device,
    )

    device = select_device(args.device)
    start = time.perf_counter()
    models = load_models(protocol, device, load_checkpoint)
    summary = collect_training(models, results)
    log(f"Loaded {len(models)} checkpoints on {device}")
    training_prior = merged_ofdm(protocol).channel
    training_range = tuple(float(value) for value in protocol["training_reference_range_db"])
    settings = protocol["evaluation"]
    dataset_digests = {}
    representative = f"{protocol['representative_run']}/seed_{protocol['representative_seed']}"
    selected = [
        condition
        for condition in evaluation_conditions(protocol)
        if args.conditions is None or condition.name in args.conditions
    ]
    for condition in selected:
        evaluations, rows, comparisons = [], [], []
        for config in condition.datasets:
            dataset = load_dataset(dataset_directory(datasets_root, config, condition.name))
            if dataset.config != config:
                raise ValueError(f"{dataset.path} does not match the protocol configuration")
            dataset_digests[dataset.path.as_posix()] = dataset.manifest["arrays_sha256"]
            if args.timing_only:
                continue
            observations = replay_observations(dataset)
            estimators = linear_estimators(dataset, training_prior, training_range)
            neural = {
                name: features_to_complex(predict(model, dataset.inputs, device))
                for name, (model, _, _) in models.items()
            }
            evaluation = evaluate_paired(
                dataset, estimators, neural, settings["equalizer"], observations
            )
            condition_rows, condition_comparisons = summarize(
                evaluation,
                settings["bootstrap_replicates"],
                settings["bootstrap_seed"],
                condition.name,
            )
            evaluations.append(evaluation)
            rows += condition_rows
            comparisons += condition_comparisons
            shown = ("ls", "lmmse_matched", representative)
            log(
                f"{condition.name} {config.fixed_reference_db:g} dB: replay verified"
                f" {config.count} frames; "
                + ", ".join(
                    f"{row['receiver']} BER {row['ber']:.3g} NMSE {row['nmse_mean_per_frame']:.3g}"
                    for row in condition_rows
                    if row["receiver"] in shown
                )
            )
            if (
                condition.name == "primary"
                and config.fixed_reference_db == settings["example_reference_db"]
            ):
                save_example(protocol, dataset, observations, estimators, neural, results)
        if not args.timing_only:
            save_condition(results / condition.name, evaluations, rows, comparisons)
    if not args.timing_only:
        attention_records = save_attention_maps(
            protocol, models, datasets_root, results, device, torch
        )
        log(f"Attention diagnostics saved: {len(attention_records)}")
    if not args.skip_timing:
        reference_config = evaluation_conditions(protocol)[0].datasets[3]
        reference = load_dataset(dataset_directory(datasets_root, reference_config, "primary"))
        frame_config, observation = reference.config.replay(0)
        timing_names = [
            f"{run}/seed_{protocol['representative_seed']}"
            for run in ("mlp", *protocol["main_runs"])
        ]
        timing_names = list(dict.fromkeys(name for name in timing_names if name in models))
        devices = [torch.device("cpu")] + ([device] if device.type != "cpu" else [])
        timing_rows, cost_rows = [], []
        for timing_device in devices:
            for name in timing_names:
                model, _ = load_checkpoint(models[name][2], timing_device)
                timing_rows += estimator_timings(model, name, reference.inputs[:256], timing_device)
                if timing_device.type == "cpu":
                    cost_rows.append(
                        {
                            "model": name,
                            "parameters": parameter_count(model),
                            **forward_flops(model, reference.inputs),
                        }
                    )
            model, _ = load_checkpoint(models[representative][2], timing_device)
            timing_rows += classical_and_pipeline_timings(
                frame_config, observation, model, timing_device, representative
            )
        write_csv(results / "timing.csv", timing_rows)
        write_csv(results / "flops.csv", cost_rows)
        log(f"Timing rows written: {len(timing_rows)}")
    metadata = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": file_digest(args.protocol),
        "source_sha256": source_digest(),
        "datasets_arrays_sha256": dataset_digests,
        "checkpoints": {
            f"{item['run']}/seed_{item['seed']}": item["checkpoint_sha256"] for item in summary
        },
        "runtime": runtime_versions(),
        "hardware": hardware_description(device),
        "timing_hardware": {"cpu": platform.processor(), "cpu_threads": torch.get_num_threads()},
        "mixture_noise_variance": mixture_noise_variance(training_range),
        "receiver_information": {
            "perfect": "true realized H (oracle; only receiver given truth at inference)",
            "ls": "known pilots only",
            "lmmse_matched": "known pilots, true test prior mean/covariance, configured sigma^2",
            "lmmse_training_prior": "known pilots, Rayleigh training prior, configured sigma^2",
            "lmmse_snr_unaware": (
                "known pilots, Rayleigh training prior, sigma^2 replaced by its mean over the"
                " training uniform-dB mixture; best linear estimator without SNR knowledge"
            ),
            "neural": "LS real/imag features only; no sigma^2, pilot mask or truth",
        },
        "equalizer": settings["equalizer"],
        "uncertainty": (
            "95% percentile bootstrap over independent frames with shared indices across"
            " receivers; zero-error frames report a conservative frame-event upper95 instead"
        ),
        "timing_definitions": (
            "median of repeated perf_counter wall times after warmup; CUDA synchronized;"
            " attention-map collection disabled; scopes listed per row"
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (results / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2) + "\n", encoding="ascii"
    )
    if args.conditions is None:
        for path in save_study_figures(results, protocol):
            log(f"Saved {path.as_posix()}")
    log(f"Evaluation finished in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
