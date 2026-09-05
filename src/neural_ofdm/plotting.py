from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogFormatter, NullFormatter, ScalarFormatter
from numpy.typing import ArrayLike

from neural_ofdm.channel import channel_frequency_response, fixed_channel
from neural_ofdm.config import OFDMConfig
from neural_ofdm.equalization import zero_forcing_equalize
from neural_ofdm.estimation import ls_channel_estimate
from neural_ofdm.experiments import SweepResult
from neural_ofdm.metrics import normalized_mean_squared_error
from neural_ofdm.simulation import simulate_frame


def _format_ascii_ticks(figure: Figure) -> Figure:
    for axis in figure.axes:
        for dimension in (axis.xaxis, axis.yaxis):
            if isinstance(dimension.get_major_formatter(), (ScalarFormatter, LogFormatter)):
                dimension.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
                dimension.set_minor_formatter(NullFormatter())
    return figure


def plot_constellation(symbols: ArrayLike, title: str) -> Figure:

    values = np.asarray(symbols)
    figure, axis = plt.subplots(figsize=(5, 5), layout="constrained")
    axis.scatter(values.real, values.imag, s=16, alpha=0.6)
    ideal = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    axis.scatter(ideal.real, ideal.imag, marker="x", c="black", s=65, label="Ideal QPSK")
    axis.axhline(0, color="gray", linewidth=0.7)
    axis.axvline(0, color="gray", linewidth=0.7)
    axis.set(xlabel="In-phase amplitude", ylabel="Quadrature amplitude", title=title)
    limit = max(1.3, float(np.max(np.abs(np.concatenate((values.real, values.imag))))) * 1.1)
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    axis.grid(alpha=0.25)
    axis.legend()
    return _format_ascii_ticks(figure)


def plot_resource_grid(config: OFDMConfig) -> Figure:

    figure, axis = plt.subplots(figsize=(9, 2.5), layout="constrained")
    axis.scatter(config.data_indices, np.zeros(config.data_indices.size), label="Data", marker="s")
    axis.scatter(
        config.pilot_indices, np.ones(config.pilot_indices.size), label="Known pilots", marker="s"
    )
    axis.set(
        xlabel="FFT bin k (unshifted)",
        yticks=[0, 1],
        yticklabels=["Data", "Pilot"],
        ylim=(-0.6, 1.6),
        title="OFDM allocation: all FFT bins active",
    )
    axis.legend(loc="upper right")
    return _format_ascii_ticks(figure)


def plot_time_frequency(frequency_grid: ArrayLike, time_signal: ArrayLike) -> Figure:

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    axes[0].stem(np.abs(frequency_grid))
    axes[0].set(xlabel="FFT bin k", ylabel="|X[k]|", title="Frequency-domain QPSK and pilots")
    time = np.asarray(time_signal)
    axes[1].plot(time.real, label="Real")
    axes[1].plot(time.imag, label="Imaginary", alpha=0.8)
    axes[1].set(
        xlabel="Sample n", ylabel="Amplitude", title="Unitary IFFT: sum of orthogonal subcarriers"
    )
    axes[1].legend()
    return _format_ascii_ticks(figure)


def plot_cyclic_prefix(
    time_signal: ArrayLike, transmitted_signal: ArrayLike, cp_length: int
) -> Figure:

    time = np.asarray(time_signal)
    transmitted = np.asarray(transmitted_signal)
    figure, axes = plt.subplots(2, 1, figsize=(9, 5), layout="constrained")
    axes[0].plot(time.real, label="Useful symbol (real part)")
    if cp_length:
        axes[0].axvspan(
            time.size - cp_length - 0.5,
            time.size - 0.5,
            alpha=0.2,
            color="orange",
            label="Samples copied into CP",
        )
    axes[1].plot(
        np.arange(transmitted.size) - cp_length, transmitted.real, label="Transmitted (real)"
    )
    if cp_length:
        axes[1].axvspan(-cp_length - 0.5, -0.5, alpha=0.2, color="orange", label="Cyclic prefix")
    for axis in axes:
        axis.set(xlabel="Sample relative to useful-symbol start", ylabel="Real amplitude")
        axis.legend()
    axes[0].set_title("Before cyclic prefix")
    axes[1].set_title("Transmitted symbol with cyclic prefix")
    return _format_ascii_ticks(figure)


def plot_channel(impulse_response: ArrayLike, frequency_response: ArrayLike) -> Figure:

    figure, axes = plt.subplots(3, 1, figsize=(9, 8), layout="constrained")
    axes[0].stem(np.abs(impulse_response))
    axes[0].set(xlabel="Delay (samples)", ylabel="|h[l]|", title="Multipath FIR impulse response")
    axes[1].plot(np.abs(frequency_response))
    axes[1].set(
        xlabel="FFT bin k (unshifted)",
        ylabel="|H[k]|",
        title="Channel frequency-response magnitude",
    )
    axes[2].plot(np.angle(frequency_response))
    axes[2].set(
        xlabel="FFT bin k (unshifted)",
        ylabel="Phase (radians)",
        title="Channel frequency-response phase",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    return _format_ascii_ticks(figure)


def plot_channel_estimate(
    true_channel: ArrayLike, estimate: ArrayLike, pilot_indices: ArrayLike
) -> Figure:

    true = np.asarray(true_channel)
    estimated = np.asarray(estimate)
    pilots = np.asarray(pilot_indices)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    for axis, transform, label in zip(
        axes, (np.abs, np.angle), ("Magnitude", "Phase (radians)"), strict=True
    ):
        axis.plot(transform(true), label="True channel")
        axis.plot(transform(estimated), label="LS + periodic linear interpolation", linestyle="--")
        axis.scatter(pilots, transform(estimated)[pilots], c="black", s=22, label="Pilot LS")
        axis.set(xlabel="FFT bin k (unshifted)", ylabel=label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title("Channel estimation: noise at pilots and interpolation between them")
    return _format_ascii_ticks(figure)


def plot_noise_examples(config: OFDMConfig, seed: int = 2026) -> Figure:

    figure, axes = plt.subplots(3, 2, figsize=(11, 9), layout="constrained")
    for row, snr in enumerate((0, 10, 20)):
        frame_config = replace(config, channel_model="fixed", channel_taps=(1 + 0j,), snr_db=snr)
        frame = simulate_frame(frame_config, np.random.default_rng(seed), csi="perfect")
        axes[row, 0].plot(frame.transmitted_signal.real, label="Clean", alpha=0.8)
        axes[row, 0].plot(frame.received_signal.real, label="Noisy", alpha=0.7)
        axes[row, 0].set(
            xlabel="Sample n (including CP)",
            ylabel="Real amplitude",
            title=f"Identity channel, {snr} dB sample SNR",
        )
        axes[row, 0].legend()
        symbols = frame.received_grid[config.data_indices]
        axes[row, 1].scatter(symbols.real, symbols.imag, s=15)
        axes[row, 1].set(
            xlabel="In-phase amplitude",
            ylabel="Quadrature amplitude",
            xlim=(-2.8, 2.8),
            ylim=(-2.8, 2.8),
            aspect="equal",
            title="Received QPSK data",
        )
        axes[row, 1].axhline(0, color="gray", linewidth=0.5)
        axes[row, 1].axvline(0, color="gray", linewidth=0.5)
    return _format_ascii_ticks(figure)


def plot_ber_sweep(result: SweepResult) -> Figure:

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    snrs = np.array([p.snr_db for p in result.points])
    bounds = np.array([p.zero_error_upper_bound for p in result.points])
    for attribute, label, color, marker in (
        ("ber_perfect", "Perfect CSI + ZF", "tab:blue", "o"),
        ("ber_ls", "Pilot LS + ZF", "tab:orange", "s"),
    ):
        rates = np.array([getattr(p, attribute) for p in result.points])
        shown = np.where(rates > 0, rates, bounds)

        axis.semilogy(snrs, np.where(rates > 0, rates, np.nan), color=color, label=label)
        positive = rates > 0
        axis.scatter(snrs[positive], shown[positive], color=color, marker=marker)
        if np.any(~positive):
            axis.scatter(
                snrs[~positive],
                shown[~positive],
                color=color,
                marker="v",
                s=70,
                label=f"{label}: zero errors (approx. 95% upper bound)",
            )
    axis.set(
        xlabel="Received sample SNR (dB)",
        ylabel="Payload BER",
        title=f"{result.config.channel_model.capitalize()} multipath: {result.points[0].total_bits:,} bits per SNR",
    )
    axis.grid(which="both", alpha=0.3)
    axis.legend(fontsize=8)
    return _format_ascii_ticks(figure)


def plot_nmse_sweep(result: SweepResult) -> Figure:

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    axis.semilogy(
        [p.snr_db for p in result.points],
        [p.nmse_ls for p in result.points],
        "o-",
        label="LS mean NMSE",
    )
    if result.config.channel_model == "fixed":
        config = result.config
        true = channel_frequency_response(fixed_channel(config.channel_taps), config.fft_size)
        estimate = ls_channel_estimate(
            true * config.pilot_value, config.pilot_indices, config.pilot_value
        )
        floor = normalized_mean_squared_error(true, estimate)
        if floor > 1e-25:
            axis.axhline(
                floor,
                linestyle="--",
                color="gray",
                label=f"Noiseless interpolation floor: {floor:.3g}",
            )
    axis.set(
        xlabel="Received sample SNR (dB)",
        ylabel="NMSE (linear ratio, all FFT bins)",
        title=f"{result.config.channel_model.capitalize()} multipath: mean over {result.num_frames:,} frames per SNR",
    )
    axis.grid(which="both", alpha=0.3)
    axis.legend()
    return _format_ascii_ticks(figure)


def save_sweep_figures(result: SweepResult, output_dir: str | Path) -> None:

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, function in (("ber_vs_snr", plot_ber_sweep), ("nmse_vs_snr", plot_nmse_sweep)):
        figure = function(result)
        figure.savefig(directory / f"{name}.png", dpi=160)
        plt.close(figure)


def save_baseline_figures(config: OFDMConfig, output_dir: str | Path, seed: int = 2026) -> None:

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frame = simulate_frame(replace(config, snr_db=15), np.random.default_rng(seed))
    perfect = zero_forcing_equalize(frame.received_grid, frame.channel_frequency_response)
    data = config.data_indices
    figures = {
        "qpsk_clean": plot_constellation(frame.frequency_grid[data], "Unit-energy Gray-coded QPSK"),
        "noise_examples": plot_noise_examples(config, seed),
        "before_equalization": plot_constellation(
            frame.received_grid[data], "Before equalization: fixed multipath, 15 dB"
        ),
        "after_equalization": plot_constellation(
            frame.equalized_grid[data], "After LS + ZF equalization, 15 dB"
        ),
        "perfect_equalization": plot_constellation(
            perfect.symbols[data], "After perfect-CSI ZF equalization, 15 dB"
        ),
        "channel": plot_channel(frame.channel_impulse_response, frame.channel_frequency_response),
        "channel_estimate": plot_channel_estimate(
            frame.channel_frequency_response, frame.estimated_channel, config.pilot_indices
        ),
        "resource_grid": plot_resource_grid(config),
        "time_frequency": plot_time_frequency(frame.frequency_grid, frame.time_signal),
        "cyclic_prefix": plot_cyclic_prefix(
            frame.time_signal, frame.transmitted_signal, config.cp_length
        ),
    }
    for name, figure in figures.items():
        figure.savefig(directory / f"{name}.png", dpi=160)
        plt.close(figure)
