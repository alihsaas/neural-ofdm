import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from neural_ofdm.neural_plotting import (
    AXIS_LABEL,
    STYLES,
    load_condition,
    plot_ber,
    plot_metric,
    plot_training_curves,
    save_neural_figures,
    series,
    shade_extrapolation,
)
from neural_ofdm.plotting import _format_ascii_ticks


def style(name: str) -> tuple[str, str]:
    return STYLES.get(name, ("tab:pink", name))


def load_example(results: Path) -> tuple[dict, dict]:
    with np.load(results / "example.npz", allow_pickle=False) as arrays:
        example = {key: arrays[key] for key in arrays.files}
    return example, json.loads((results / "example.json").read_text(encoding="ascii"))


def plot_attention_example(example: dict, details: dict, runs: list[str]) -> Figure:
    bins = np.arange(len(example["true"]))
    figure, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, layout="constrained")
    curves = [("true", "black", "-", "True H"), ("ls", "tab:gray", ":", "LS")]
    curves.append(("lmmse_matched", "tab:blue", "--", "LMMSE matched"))
    curves += [(run, style(run)[0], "-", style(run)[1]) for run in runs]
    for key, color, line, label in curves:
        values = example[key]
        axes[0].plot(bins, values.real, line, color=color, label=label, lw=1.2)
        axes[1].plot(bins, values.imag, line, color=color, label=label, lw=1.2)
        axes[2].plot(bins, 20 * np.log10(np.abs(values)), line, color=color, label=label, lw=1.2)
    pilots = example["pilot_indices"]
    axes[0].plot(pilots, example["ls"][pilots].real, "k^", ms=4, label="pilot LS samples")
    axes[0].set_ylabel("Real part")
    axes[1].set_ylabel("Imaginary part")
    axes[2].set_ylabel("Magnitude (dB)")
    axes[2].set_xlabel("FFT bin k (unshifted order, DC at 0)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=7)
    summary = ", ".join(f"{key} {value:.4f}" for key, value in details["nmse"].items())
    figure.suptitle(
        f"Held-out frame {details['frame']} at {details['reference_db']:g} dB; NMSE: {summary}",
        fontsize=9,
    )
    return _format_ascii_ticks(figure)


def plot_attention_constellations(example: dict, details: dict, runs: list[str]) -> Figure:
    names = ["perfect", "ls", "lmmse_matched", *runs]
    figure, axes = plt.subplots(
        1, len(names), figsize=(3.3 * len(names), 3.8), layout="constrained"
    )
    for axis, name in zip(axes, names, strict=True):
        symbols = example[f"equalized_{name}"]
        axis.scatter(symbols.real, symbols.imag, s=2, alpha=0.3, color=style(name)[0])
        axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal", xlabel="In-phase")
        axis.set_title(
            f"{style(name)[1]}\nBER {details['constellation_ber'][name]:.4f}", fontsize=8
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Quadrature")
    figure.suptitle(
        f"ZF-equalized QPSK payload, same {details['constellation_frames']} frames at"
        f" {details['reference_db']:g} dB; points beyond +-2.5 not shown"
    )
    return _format_ascii_ticks(figure)


def condition_panels(results: Path, conditions: list[str], runs: list[str], title: str) -> Figure:
    figure, axes = plt.subplots(
        2, len(conditions), figsize=(4.4 * len(conditions), 8), layout="constrained", squeeze=False
    )
    for column, condition in enumerate(conditions):
        rows = load_condition(results, condition)
        receivers = ["ls", "lmmse_matched"]
        if any(row["receiver"] == "lmmse_training_prior" for row in rows):
            receivers.append("lmmse_training_prior")
        receivers += runs
        plot_metric(axes[0, column], rows, "nmse_mean_per_frame", receivers)
        plot_ber(axes[1, column], rows, ["perfect", *receivers])
        axes[0, column].set_title(condition)
        for axis in axes[:, column]:
            axis.grid(which="both", alpha=0.25)
            axis.set_xlabel("Reference Es/N0 (dB), fixed noise")
    axes[0, 0].set_ylabel("Mean per-frame NMSE (dB)")
    axes[1, 0].set_ylabel("Payload BER, ZF")
    axes[0, 0].legend(fontsize=6)
    figure.suptitle(title)
    return _format_ascii_ticks(figure)


def plot_attention_primary(rows: list[dict], protocol: dict, runs: list[str]) -> Figure:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    receivers = ["ls", "lmmse_matched", "lmmse_snr_unaware", *runs]
    plot_metric(axes[0], rows, "nmse_mean_per_frame", receivers)
    _, lmmse = series(rows, "lmmse_matched", "nmse_mean_per_frame")
    for run in runs:
        names = sorted({row["receiver"] for row in rows if row["run"] == run})
        ratios = []
        for name in names:
            levels, values = series(rows, name, "nmse_mean_per_frame")
            ratios.append(10 * np.log10(values / lmmse))
        ratios = np.array(ratios)
        color, label = style(run)
        axes[1].fill_between(levels, ratios.min(0), ratios.max(0), color=color, alpha=0.2)
        axes[1].plot(
            levels, ratios.mean(0), "o-", color=color, label=f"{label} ({len(names)} seeds)"
        )
    axes[1].axhline(0, color="tab:blue", label="matched LMMSE")
    plot_ber(axes[2], rows, ["perfect", "ls", "lmmse_matched", *runs])
    axes[0].set_ylabel("Mean per-frame NMSE (dB of linear mean)")
    axes[1].set_ylabel("NMSE / matched-LMMSE NMSE (dB); > 0 is worse")
    for axis in axes:
        shade_extrapolation(axis, protocol)
        axis.set_xlabel(AXIS_LABEL, fontsize=8)
        axis.grid(which="both", alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=6)
    figure.suptitle(
        "Fresh holdout, training prior. Gray band = training range; bands = min-max over seeds;"
        " BER bars = 95% paired frame bootstrap (seed 11 shown)"
    )
    return _format_ascii_ticks(figure)


def plot_relative_ablation(rows: list[dict], protocol: dict, groups: dict) -> Figure:
    figure, axes = plt.subplots(
        1, len(groups), figsize=(6 * len(groups), 4.6), layout="constrained"
    )
    _, lmmse = series(rows, "lmmse_matched", "nmse_mean_per_frame")
    seed = protocol["representative_seed"]
    for axis, (title, runs) in zip(np.atleast_1d(axes), groups.items(), strict=True):
        for run in runs:
            levels, values = series(rows, f"{run}/seed_{seed}", "nmse_mean_per_frame")
            if values.size == 0:
                continue
            color, label = style(run)
            axis.plot(levels, 10 * np.log10(values / lmmse), "o-", color=color, label=label)
        axis.axhline(0, color="tab:blue", lw=1)
        shade_extrapolation(axis, protocol)
        axis.set(title=f"{title} (seed {seed})", xlabel="Reference Es/N0 (dB)")
        axis.set_ylabel("NMSE / matched-LMMSE NMSE (dB)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Ablations on the primary holdout; identical frames, data, loss and budget")
    return _format_ascii_ticks(figure)


def plot_cost(results: Path, rows: list[dict], protocol: dict, runs: list[str]) -> Figure:
    timing = list(csv.DictReader((results / "timing.csv").open(encoding="ascii")))
    _, lmmse = series(rows, "lmmse_matched", "nmse_mean_per_frame")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    models = json.loads((results / "models.json").read_text(encoding="ascii"))
    for run in runs:
        name = f"{run}/seed_{protocol['representative_seed']}"
        levels, values = series(rows, name, "nmse_mean_per_frame")
        inside = (levels >= 0) & (levels <= 30)
        relative = float(np.mean(10 * np.log10(values[inside] / lmmse[inside])))
        latency = [
            float(row["median_seconds"]) * 1e3
            for row in timing
            if row["estimator"] == name
            and row["device"] == "cpu"
            and row["batch_size"] == "1"
            and row["scope"].startswith("model forward")
        ]
        parameters = next(item["parameters"] for item in models if item["run"] == run)
        color, label = style(run)
        axes[0].scatter(parameters, relative, color=color, s=60, label=label)
        if latency:
            axes[1].scatter(latency[0], relative, color=color, s=60, label=label)
    axes[0].set(xlabel="Trainable parameters", xscale="log")
    axes[1].set(xlabel="CPU forward latency, batch 1 (ms, median)")
    for axis in axes:
        axis.axhline(0, color="tab:blue", lw=1)
        axis.set_ylabel("Mean over 0..30 dB of NMSE/LMMSE (dB)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Accuracy versus cost, seed 11 checkpoints, primary holdout")
    return _format_ascii_ticks(figure)


def plot_attention_maps(results: Path) -> Figure | None:
    path = results / "attention_maps.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="ascii"))
    with np.load(results / "attention_maps.npz", allow_pickle=False) as arrays:
        maps = {key: arrays[key] for key in arrays.files}
    chosen = [record for record in records if record["condition"] == "primary"][-2:]
    layers, heads = maps[chosen[0]["key"]].shape[:2]
    figure, axes = plt.subplots(
        len(chosen) * layers,
        heads,
        figsize=(3 * heads, 3 * layers * len(chosen)),
        layout="constrained",
        squeeze=False,
    )
    for block, record in enumerate(chosen):
        weights = maps[record["key"]]
        for layer in range(layers):
            for head in range(heads):
                axis = axes[block * layers + layer, head]
                image = axis.imshow(weights[layer, head], origin="lower", cmap="viridis")
                axis.set_title(
                    f"{record['model'].split('/')[0][:14]} L{layer + 1} H{head + 1}", fontsize=7
                )
                axis.set_xlabel("key bin", fontsize=7)
                axis.set_ylabel("query bin", fontsize=7)
        figure.colorbar(image, ax=axes[block * layers + layers - 1, heads - 1], shrink=0.6)
    figure.suptitle(
        f"Post-softmax attention, eval mode, frame {chosen[0]['frame']},"
        f" {chosen[0]['condition']} {chosen[0]['reference_db']:g} dB, pilot spacing"
        f" {chosen[0]['pilot_indices'][1] - chosen[0]['pilot_indices'][0]}; diagnostic only"
    )
    return _format_ascii_ticks(figure)


def save_attention_figures(results: Path, protocol: dict) -> list[Path]:
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = protocol["main_runs"]
    primary = load_condition(results, "primary")
    example, details = load_example(results)
    outputs = {
        "training_curves.png": plot_training_curves(results),
        "channel_example.png": plot_attention_example(example, details, runs),
        "constellations.png": plot_attention_constellations(example, details, runs),
        "primary_nmse_ber.png": plot_attention_primary(primary, protocol, runs),
        "pilot_density.png": condition_panels(
            results,
            ["pilot_spacing_2", "primary", "pilot_spacing_8"],
            runs,
            "Pilot pattern shift with frozen checkpoints (primary = spacing 4)",
        ),
        "delay_spread.png": condition_panels(
            results,
            ["delay_short", "primary", "delay_long", "pdp_flat"],
            runs,
            "Delay spread and PDP shifts with frozen checkpoints (primary = delays 0,1,3,6)",
        ),
        "robustness.png": condition_panels(
            results,
            ["rician_k1", "rician_k10", "qam16"],
            runs,
            "Rician channel shift and 16-QAM payload with frozen checkpoints",
        ),
        "ablations.png": plot_relative_ablation(
            primary,
            protocol,
            {
                "Positional information": [
                    "transformer_d64_h4_l2",
                    "transformer_no_position",
                    "hybrid_d64_h4_l2",
                    "hybrid_no_position",
                ],
                "Context and capacity": [
                    "cnn_residual",
                    "dilated_cnn_w40",
                    "transformer_d64_h4_l2",
                    "hybrid_d64_h4_l2",
                ],
                "Learning rate 1e-3 vs 3e-4": [
                    "cnn_residual",
                    "cnn_residual_lr3e-4",
                    "transformer_d64_h4_l2",
                    "transformer_d64_h4_l2_lr3e-4",
                    "hybrid_d64_h4_l2",
                    "hybrid_d64_h4_l2_lr3e-4",
                ],
            },
        ),
        "accuracy_cost.png": plot_cost(results, primary, protocol, runs),
    }
    attention = plot_attention_maps(results)
    if attention is not None:
        outputs["attention_maps.png"] = attention
    paths = []
    for name, figure in outputs.items():
        path = figures / name
        figure.savefig(path, dpi=120)
        plt.close(figure)
        paths.append(path)
    return paths


def save_study_figures(results: str | Path, protocol: dict) -> list[Path]:
    if protocol["name"] == "attention_channel_estimation":
        return save_attention_figures(Path(results), protocol)
    return save_neural_figures(results, protocol)
