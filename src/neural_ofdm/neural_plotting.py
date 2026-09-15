import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from neural_ofdm.neural_evaluation import read_rows
from neural_ofdm.plotting import _format_ascii_ticks

AXIS_LABEL = "Reference Es/N0 (dB), Es=1, fixed noise sigma^2=10^(-ref/10)"
STYLES = {
    "perfect": ("black", "Perfect CSI"),
    "ls": ("tab:gray", "LS (periodic interp.)"),
    "lmmse_matched": ("tab:blue", "LMMSE matched prior + sigma^2"),
    "lmmse_training_prior": ("tab:cyan", "LMMSE frozen Rayleigh training prior"),
    "lmmse_snr_unaware": ("tab:purple", "LMMSE training prior, mixture sigma^2"),
    "mlp": ("tab:olive", "MLP baseline"),
    "cnn_residual": ("tab:red", "Residual CNN"),
    "cnn_direct": ("tab:orange", "CNN direct output (ablation)"),
    "cnn_normalized": ("tab:green", "Residual CNN + affine norm (ablation)"),
    "dilated_cnn_w40": ("tab:brown", "Dilated CNN w40 (context control)"),
    "transformer_d64_h4_l2": ("tab:green", "Transformer d64 h4 l2"),
    "hybrid_d64_h4_l2": ("tab:orange", "Hybrid CNN/Transformer"),
    "transformer_no_position": ("lime", "Transformer, no position"),
    "hybrid_no_position": ("gold", "Hybrid, no explicit position"),
    "cnn_residual_lr3e-4": ("salmon", "Residual CNN, lr 3e-4"),
    "transformer_d64_h4_l2_lr3e-4": ("darkgreen", "Transformer, lr 3e-4"),
    "hybrid_d64_h4_l2_lr3e-4": ("darkorange", "Hybrid, lr 3e-4"),
}


def load_condition(results: Path, condition: str) -> list[dict]:
    path = results / condition / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run: uv run python scripts/evaluate_channel_estimator.py"
        )
    return read_rows(path)


def series(rows: list[dict], receiver: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["receiver"] == receiver]
    return (
        np.array([row["reference_esn0_db"] for row in selected]),
        np.array([row[key] for row in selected], dtype=float),
    )


def seeds_for(rows: list[dict], run: str) -> list[str]:
    return sorted({row["receiver"] for row in rows if row["run"] == run})


def shade_extrapolation(axis, protocol: dict) -> None:
    low, high = protocol["training_reference_range_db"]
    axis.axvspan(low, high, color="0.92", zorder=0)


def plot_metric(
    axis, rows: list[dict], key: str, receivers: list[str], decibels: bool = True
) -> None:
    for receiver in receivers:
        color, label = STYLES[receiver]
        names = (
            [receiver]
            if receiver in ("perfect", "ls") or receiver.startswith("lmmse")
            else seeds_for(rows, receiver)
        )
        if not names:
            continue
        curves = []
        for name in names:
            levels, values = series(rows, name, key)
            curves.append(values)
        values = np.array(curves)
        transform = (lambda array: 10 * np.log10(array)) if decibels else (lambda array: array)
        if len(names) > 1:
            axis.fill_between(
                levels,
                transform(values.min(axis=0)),
                transform(values.max(axis=0)),
                color=color,
                alpha=0.2,
            )
            label = f"{label} ({len(names)} seeds, mean and min-max)"
        axis.plot(levels, transform(values.mean(axis=0)), "o-", color=color, label=label, ms=4)


def plot_training_curves(results: Path) -> Figure:
    summary = json.loads((results / "models.json").read_text(encoding="ascii"))
    runs = list(dict.fromkeys(item["run"] for item in summary))
    columns = min(4, len(runs))
    rows = -(-len(runs) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.6 * rows), layout="constrained", squeeze=False
    )
    for axis in axes.flat[len(runs) :]:
        axis.set_visible(False)
    for axis, run in zip(axes.flat, runs, strict=False):
        for item in [entry for entry in summary if entry["run"] == run]:
            history = list(
                csv.DictReader(
                    (results / "training" / run / f"seed_{item['seed']}" / "history.csv").open(
                        encoding="ascii"
                    )
                )
            )
            epochs = [int(row["epoch"]) for row in history]
            (line,) = axis.semilogy(
                epochs,
                [float(row["train_loss"]) for row in history],
                "--",
                alpha=0.7,
                label=f"seed {item['seed']} train",
            )
            axis.semilogy(
                epochs,
                [float(row["validation_loss"]) for row in history],
                "-",
                color=line.get_color(),
                label=f"seed {item['seed']} validation",
            )
            axis.axvline(item["selected_epoch"], color=line.get_color(), alpha=0.3)
        axis.set(xlabel="Epoch", ylabel="Complex MSE per bin")
        axis.set_title(run, fontsize=9)
        axis.grid(which="both", alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Training and validation loss; vertical lines mark validation-selected epochs")
    return _format_ascii_ticks(figure)


def plot_channel_example(example: dict) -> Figure:
    bins = np.arange(len(example["true"]))
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    curves = {
        "True H": (example["true"], "black", "-"),
        "LS": (example["ls"], "tab:gray", ":"),
        "LMMSE matched": (example["lmmse_matched"], "tab:blue", "--"),
        "Residual CNN": (example["cnn_residual"], "tab:red", "-"),
    }
    for label, (values, color, style) in curves.items():
        axes[0].plot(bins, values.real, style, color=color, label=label)
        axes[1].plot(bins, values.imag, style, color=color, label=label)
        axes[2].plot(bins, 20 * np.log10(np.abs(values)), style, color=color, label=label)
    pilots = example["pilot_indices"]
    axes[0].plot(pilots, example["ls"][pilots].real, "k^", ms=4, label="pilot LS samples")
    axes[0].set_ylabel("Real part")
    axes[1].set_ylabel("Imaginary part")
    axes[2].set_ylabel("Magnitude (dB)")
    axes[2].set_xlabel("FFT bin k (unshifted order, DC at 0)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=5, fontsize=8)
    figure.suptitle(
        f"Held-out test frame {example['frame']} at {example['reference_db']:g} dB reference;"
        f" NMSE LS {example['nmse']['ls']:.4f}, LMMSE {example['nmse']['lmmse_matched']:.4f},"
        f" CNN {example['nmse']['cnn_residual']:.4f}"
    )
    return _format_ascii_ticks(figure)


def plot_constellations(example: dict) -> Figure:
    names = ("perfect", "ls", "lmmse_matched", "cnn_residual")
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    for axis, name in zip(axes, names, strict=True):
        symbols = example["equalized"][name]
        axis.scatter(symbols.real, symbols.imag, s=3, alpha=0.35, color=STYLES[name][0])
        axis.set(
            title=f"{STYLES[name][1]}\nBER {example['ber'][name]:.4f}",
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            aspect="equal",
            xlabel="In-phase",
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Quadrature")
    figure.suptitle(
        f"ZF-equalized payload symbols, same {example['constellation_frames']} test frames at"
        f" {example['reference_db']:g} dB; points beyond +-2.5 clipped from view"
    )
    return _format_ascii_ticks(figure)


def plot_primary(rows: list[dict], protocol: dict) -> Figure:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    receivers = ["ls", "lmmse_matched", "lmmse_snr_unaware", "mlp", "cnn_residual"]
    plot_metric(axes[0], rows, "nmse_mean_per_frame", receivers)
    plot_metric(axes[1], rows, "ensemble_mse_per_bin", receivers)
    plot_ber(axes[2], rows, ["perfect", *receivers])
    axes[0].set(ylabel="Mean per-frame NMSE (dB of linear mean)")
    axes[1].set(ylabel="Ensemble MSE per bin (dB)")
    for axis in axes:
        shade_extrapolation(axis, protocol)
        axis.set_xlabel(AXIS_LABEL, fontsize=8)
        axis.grid(which="both", alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.suptitle(
        "Rayleigh test frames (unseen channels, same prior). Gray band: training range;"
        " outside it is SNR extrapolation"
    )
    return _format_ascii_ticks(figure)


def plot_ber(axis, rows: list[dict], receivers: list[str]) -> None:
    for receiver in receivers:
        color, label = STYLES[receiver]
        names = (
            [receiver]
            if receiver in ("perfect", "ls") or receiver.startswith("lmmse")
            else seeds_for(rows, receiver)[:1]
        )
        for name in names:
            selected = [row for row in rows if row["receiver"] == name]
            levels = np.array([row["reference_esn0_db"] for row in selected])
            ber = np.array([row["ber"] for row in selected])
            suffix = f" ({name.split('/')[-1]})" if "/" in name else ""
            axis.semilogy(
                levels,
                np.where(ber > 0, ber, np.nan),
                "o-",
                color=color,
                ms=4,
                label=label + suffix,
            )
            for row in selected:
                if row["errors"]:
                    axis.vlines(
                        row["reference_esn0_db"],
                        max(row["ber_frame_bootstrap_low"], 1 / row["total_bits"]),
                        row["ber_frame_bootstrap_high"],
                        color=color,
                        alpha=0.6,
                    )
                else:
                    axis.scatter(
                        row["reference_esn0_db"],
                        row["zero_error_frame_upper95"],
                        marker="v",
                        color=color,
                    )
    axis.set_ylabel("Payload BER, ZF; bars: 95% paired frame bootstrap")


def plot_robustness(results: Path, protocol: dict) -> Figure:
    conditions = [item["name"] for item in protocol["test_conditions"] if item["name"] != "primary"]
    figure, axes = plt.subplots(
        2, len(conditions), figsize=(4.3 * len(conditions), 8), layout="constrained"
    )
    for column, condition in enumerate(conditions):
        rows = load_condition(results, condition)
        receivers = ["ls", "lmmse_matched"]
        if any(row["receiver"] == "lmmse_training_prior" for row in rows):
            receivers.append("lmmse_training_prior")
        receivers += ["lmmse_snr_unaware", "mlp", "cnn_residual"]
        plot_metric(axes[0, column], rows, "nmse_mean_per_frame", receivers)
        plot_ber(axes[1, column], rows, ["perfect", *receivers])
        axes[0, column].set_title(condition)
        for axis in axes[:, column]:
            axis.grid(which="both", alpha=0.25)
            axis.set_xlabel("Reference Es/N0 (dB)")
    axes[0, 0].set_ylabel("Mean per-frame NMSE (dB)")
    axes[1, 0].set_ylabel("Payload BER, ZF")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    figure.suptitle(
        "Frozen Rayleigh-trained networks under changed priors and pilot layouts;"
        " matched LMMSE knows the shifted statistics"
    )
    return _format_ascii_ticks(figure)


def plot_ablation(rows: list[dict], protocol: dict) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    receivers = ["ls", "lmmse_matched", "cnn_residual", "cnn_direct", "cnn_normalized"]
    plot_metric(axes[0], rows, "nmse_mean_per_frame", receivers)
    lmmse_levels, lmmse = series(rows, "lmmse_matched", "nmse_mean_per_frame")
    for run in ("cnn_residual", "cnn_direct", "cnn_normalized"):
        name = f"{run}/seed_{protocol['representative_seed']}"
        levels, values = series(rows, name, "nmse_mean_per_frame")
        axes[1].plot(
            levels,
            10 * np.log10(values / lmmse),
            "o-",
            color=STYLES[run][0],
            label=f"{STYLES[run][1]} seed {protocol['representative_seed']}",
        )
    axes[1].axhline(0, color=STYLES["lmmse_matched"][0], label="matched LMMSE")
    axes[0].set_ylabel("Mean per-frame NMSE (dB)")
    axes[1].set_ylabel("NMSE relative to matched LMMSE (dB); > 0 is worse")
    for axis in axes:
        shade_extrapolation(axis, protocol)
        axis.set_xlabel(AXIS_LABEL, fontsize=8)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(
        "Ablations with identical data, budget and validation selection (one seed each)"
    )
    return _format_ascii_ticks(figure)


def save_neural_figures(results: str | Path, protocol: dict) -> list[Path]:
    root = Path(results)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = load_condition(root, "primary")
    with np.load(root / "example.npz", allow_pickle=False) as arrays:
        example = {key: arrays[key] for key in arrays.files}
    details = json.loads((root / "example.json").read_text(encoding="ascii"))
    example |= {
        "frame": details["frame"],
        "reference_db": details["reference_db"],
        "nmse": details["nmse"],
        "ber": details["constellation_ber"],
        "constellation_frames": details["constellation_frames"],
        "equalized": {
            name: example[f"equalized_{name}"]
            for name in ("perfect", "ls", "lmmse_matched", "cnn_residual")
        },
    }
    outputs = {
        "training_curves.png": plot_training_curves(root),
        "channel_example.png": plot_channel_example(example),
        "constellations.png": plot_constellations(example),
        "primary_nmse_ber.png": plot_primary(primary, protocol),
        "robustness.png": plot_robustness(root, protocol),
        "ablations.png": plot_ablation(primary, protocol),
    }
    paths = []
    for name, figure in outputs.items():
        path = figures / name
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(path)
    return paths
