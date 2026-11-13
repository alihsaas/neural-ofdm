from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator

from neural_ofdm.benchmark import BenchmarkResult, ExperimentConfig
from neural_ofdm.channel import frequency_statistics
from neural_ofdm.estimation import prepare_lmmse
from neural_ofdm.modulation import constellation
from neural_ofdm.plotting import _format_ascii_ticks
from neural_ofdm.simulation import observe_frame, receive_observation


def plot_benchmark_curves(result: BenchmarkResult) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for name in result.receiver_names:
        rows = [row for row in result.rows if row["receiver"] == name]
        levels = np.array([row["reference_esn0_db"] for row in rows])
        ber = np.array([row["ber"] for row in rows])
        measured = ber > 0
        (line,) = axes[0].semilogy(levels, np.where(measured, ber, np.nan), "o-", label=name)
        color = line.get_color()
        for row in rows:
            level = row["reference_esn0_db"]
            if row["errors"] > 0:
                axes[0].vlines(
                    level,
                    max(row["ber_frame_bootstrap_low"], 1 / row["total_bits"]),
                    row["ber_frame_bootstrap_high"],
                    color=color,
                    alpha=0.5,
                )
            else:
                axes[0].scatter(level, row["zero_error_frame_upper95"], marker="v", color=color)
        nmse = np.array([row["nmse_mean_per_frame"] for row in rows])
        if np.any(nmse > 1e-25):
            axes[1].semilogy(
                levels, np.where(nmse > 1e-25, nmse, np.nan), "o-", color=color, label=name
            )
    axes[0].set(ylabel="Payload BER", title="Paired receivers; 95% frame bootstrap")
    axes[1].set(
        ylabel="Mean per-frame NMSE (linear)", title="All FFT bins; centered prior covariance"
    )
    for axis in axes:
        axis.set_xlabel("Reference Es/N0 (dB), Es=1; fixed noise")
        axis.grid(which="both", alpha=0.25)
        axis.legend()
    figure.suptitle(
        f"{result.config.frames} independent frames/level; {result.config.ofdm.bits_per_frame * result.config.frames} bits/receiver\n"
        "Triangles: zero-error conservative frame upper95, separate from measured curves"
    )
    axes[0].text(
        0.02,
        0.02,
        "CI lower ends at zero clipped to 1/bit budget",
        transform=axes[0].transAxes,
        fontsize=7,
    )
    axes[1].text(
        0.02,
        0.02,
        "Numerical zero (<1e-25) omitted; raw values in CSV",
        transform=axes[1].transAxes,
        fontsize=7,
    )
    return _format_ascii_ticks(figure)


def plot_channel_statistics(config: ExperimentConfig) -> Figure:
    channel = config.ofdm.channel_prior
    mean_tap, covariance_tap = channel.tap_statistics()
    mean, covariance = frequency_statistics(channel, config.ofdm.fft_size)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(channel.delays, channel.diffuse_powers, "o--", label="Diffuse PDP (unit sum)")
    axes[0].plot(
        channel.delays,
        np.abs(mean_tap) ** 2 + np.diag(covariance_tap).real,
        "s-",
        label="Combined expected tap power",
    )
    axes[0].set(
        xlabel="Delay (samples)",
        ylabel="Average tap power",
        title=f"{channel.model}; global K={channel.k_linear:g}",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    heatmap = axes[1].imshow(np.abs(covariance), origin="lower", aspect="equal")
    axes[1].set(xlabel="FFT bin", ylabel="FFT bin", title="|R_H|, centered covariance")
    figure.colorbar(heatmap, ax=axes[1], label="Channel covariance magnitude")
    assert mean.shape == (config.ofdm.fft_size,)
    return _format_ascii_ticks(figure)


def plot_channel_examples(config: ExperimentConfig) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    base = config.ofdm.channel_prior
    for column, k in enumerate((0.0, 10.0)):
        channel = replace(base, model="rayleigh" if k == 0 else "rician", k_linear=k)
        impulse = channel.sample(np.random.default_rng(config.seed))
        response = np.fft.fft(impulse, n=config.ofdm.fft_size)
        axes[0, column].stem(np.arange(impulse.size), np.abs(impulse))
        axes[0, column].set(
            xlabel="Delay (samples)", ylabel="|h|", title=f"{channel.model}, global K={k:g}"
        )
        axes[1, column].plot(np.abs(response), label="Realization |H|")
        mean, covariance = frequency_statistics(channel, config.ofdm.fft_size)
        axes[1, column].plot(
            np.sqrt(np.abs(mean) ** 2 + np.diag(covariance).real), "--", label="sqrt(E[|H|^2])"
        )
        axes[1, column].set(xlabel="FFT bin k", ylabel="Frequency-response magnitude")
        axes[1, column].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    return _format_ascii_ticks(figure)


def plot_estimation_example(config: ExperimentConfig, reference: float = 15) -> Figure:
    ofdm = replace(config.ofdm, reference_esn0_db=reference)
    observation = observe_frame(ofdm, np.random.default_rng(config.seed))
    mean, covariance = frequency_statistics(ofdm.channel_prior, ofdm.fft_size)
    estimator = prepare_lmmse(
        mean, covariance, ofdm.pilot_indices, ofdm.pilot_symbols, observation.noise_variance
    )
    ls = receive_observation(ofdm, observation)
    lmmse = receive_observation(ofdm, observation, csi="lmmse", estimator=estimator)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    for axis, component in zip(axes, ("real", "imag"), strict=True):
        for name, values in (
            ("True H", observation.true_channel),
            ("LS", ls.estimated_channel),
            ("LMMSE", lmmse.estimated_channel),
        ):
            axis.plot(getattr(values, component), label=name)
        axis.scatter(
            ofdm.pilot_indices,
            getattr(ls.estimated_channel, component)[ofdm.pilot_indices],
            marker="x",
            label="Pilot LS",
        )
        axis.set(xlabel="FFT bin k", ylabel=f"H {component}")
        axis.grid(alpha=0.25)
        axis.legend(ncols=4)
    figure.suptitle(
        f"Identical observation at reference Es/N0={reference:g} dB; example, not an ensemble ranking"
    )
    return _format_ascii_ticks(figure)


def plot_constellation_examples(config: ExperimentConfig) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for axis, order in zip(axes, (4, 16), strict=True):
        ofdm = replace(config.ofdm, modulation_order=order, reference_esn0_db=20)
        mean, covariance = frequency_statistics(ofdm.channel_prior, ofdm.fft_size)
        estimator = prepare_lmmse(mean, covariance, ofdm.pilot_indices, ofdm.pilot_symbols, 0.01)
        values = []
        rng = np.random.default_rng(config.seed)
        for _ in range(12):
            observation = observe_frame(ofdm, rng)
            frame = receive_observation(ofdm, observation, csi="lmmse", estimator=estimator)
            values.extend(frame.decision_grid[ofdm.data_indices])
        values = np.asarray(values)
        ideal, _ = constellation(order)
        axis.scatter(values.real, values.imag, s=8, alpha=0.3, label="LMMSE + ZF decisions")
        axis.scatter(
            ideal.real,
            ideal.imag,
            marker="x",
            c="black",
            s=70,
            label="Ideal unit-mean-energy points",
        )
        outside = int(np.count_nonzero((np.abs(values.real) > 2) | (np.abs(values.imag) > 2)))
        axis.set(
            xlabel="In-phase amplitude",
            ylabel="Quadrature amplitude",
            xlim=(-2, 2),
            ylim=(-2, 2),
            aspect="equal",
            title=f"{'QPSK' if order == 4 else '16-QAM'}, reference 20 dB\n{outside} symbols outside displayed axes",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left", fontsize=7)
    return _format_ascii_ticks(figure)


def plot_pilot_comparison(results: dict[str, BenchmarkResult], reference: float = 15) -> Figure:
    selected = [
        results[name] for name in ("pilot_spacing_2", "receiver_comparison", "pilot_spacing_8")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for receiver in ("perfect", "ls", "lmmse"):
        rows = [
            next(
                row
                for row in result.rows
                if row["receiver"] == receiver and row["reference_esn0_db"] == reference
            )
            for result in selected
        ]
        x = [row["pilot_overhead"] for row in rows]
        axes[0].semilogy(
            x, [row["ber"] if row["ber"] > 0 else np.nan for row in rows], "o-", label=receiver
        )
        if receiver != "perfect":
            axes[1].semilogy(x, [row["nmse_mean_per_frame"] for row in rows], "o-", label=receiver)
    for axis in axes:
        axis.set_xlabel("Pilot overhead (pilots / 64 used bins)")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.grid(which="both", alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Payload BER")
    axes[1].set_ylabel("Mean per-frame NMSE (linear)")
    figure.suptitle(
        f"Reference Es/N0={reference:g} dB, fixed noise; spacings 8, 4, 2\nQPSK payload: 112, 96, 64 bits/frame; common channels and standardized time noise"
    )
    return _format_ascii_ticks(figure)


def plot_study_comparison(
    results: dict[str, BenchmarkResult], studies: tuple[str, ...], title: str
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for study in studies:
        result = results[study]
        for receiver in ("ls", "lmmse"):
            rows = [row for row in result.rows if row["receiver"] == receiver]
            levels = [row["reference_esn0_db"] for row in rows]
            axes[0].semilogy(
                levels,
                [row["ber"] if row["ber"] else np.nan for row in rows],
                "o-",
                label=f"{study}: {receiver}",
            )
            axes[1].semilogy(
                levels,
                [row["nmse_mean_per_frame"] for row in rows],
                "o-",
                label=f"{study}: {receiver}",
            )
            for row in rows:
                if row["errors"] == 0:
                    axes[0].scatter(
                        row["reference_esn0_db"], row["zero_error_frame_upper95"], marker="v"
                    )
    for axis in axes:
        axis.set_xlabel("Reference Es/N0 (dB), Es=1; fixed noise")
        axis.grid(which="both", alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Payload BER; triangles are zero-error frame bounds")
    axes[1].set_ylabel("Mean per-frame NMSE (linear)")
    figure.suptitle(title)
    return _format_ascii_ticks(figure)


def save_benchmark_figures(results: dict[str, BenchmarkResult], output: str | Path) -> None:
    directory = Path(output)
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for study, result in results.items():
        figure = plot_benchmark_curves(result)
        figure.savefig(directory / study / "receiver_curves.png", dpi=160)
        plt.close(figure)
    base = results["receiver_comparison"].config
    plots = {
        "pdp_covariance_rayleigh": plot_channel_statistics(base),
        "pdp_covariance_rician": plot_channel_statistics(results["rician_k10"].config),
        "channel_examples": plot_channel_examples(base),
        "estimation_example": plot_estimation_example(base),
        "constellation_examples": plot_constellation_examples(base),
        "pilot_overhead": plot_pilot_comparison(results),
        "modulation_comparison": plot_study_comparison(
            results,
            ("receiver_comparison", "modulation_16qam"),
            "Common Es=1, PDP and fixed noise; payload bits differ",
        ),
        "fading_comparison": plot_study_comparison(
            results,
            ("receiver_comparison", "rician_k1", "rician_k10"),
            "Global K=0, 1, 10; combined PDP changes with K",
        ),
    }
    for name, figure in plots.items():
        figure.savefig(figures / f"{name}.png", dpi=160)
        plt.close(figure)
