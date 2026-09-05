import csv
import json
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from neural_ofdm._validation import integer
from neural_ofdm.config import OFDMConfig
from neural_ofdm.equalization import zero_forcing_equalize
from neural_ofdm.modulation import demodulate
from neural_ofdm.simulation import simulate_frame


@dataclass(frozen=True)
class SweepPoint:
    snr_db: float
    total_bits: int
    errors_perfect: int
    errors_ls: int
    ber_perfect: float
    ber_ls: float
    nmse_ls: float
    stabilized_bins_perfect: int
    stabilized_bins_ls: int

    @property
    def zero_error_upper_bound(self) -> float:

        return float(-np.expm1(np.log(0.05) / self.total_bits))


@dataclass(frozen=True)
class SweepResult:
    config: OFDMConfig
    num_frames: int
    seed: int
    points: tuple[SweepPoint, ...]


def run_ber_sweep(
    config: OFDMConfig, snr_values: ArrayLike, num_frames: int = 5000, seed: int | None = None
) -> SweepResult:
    if config.noise_mode != "received_snr":
        raise ValueError(
            "run_ber_sweep uses received sample SNR; use run_benchmark for fixed noise"
        )
    integer(num_frames, "num_frames", 1)
    seed = config.random_seed if seed is None else seed
    integer(seed, "seed")
    snrs = np.asarray(snr_values, dtype=float)
    if snrs.ndim != 1 or snrs.size == 0 or not np.all(np.isfinite(snrs)):
        raise ValueError("snr_values must be a finite, nonempty one-dimensional vector")
    points = []
    for snr_db in snrs:
        frame_config = replace(config, snr_db=float(snr_db))
        rng = np.random.default_rng(seed)
        errors_perfect = errors_ls = stabilized_perfect = stabilized_ls = 0
        nmse_sum = 0.0
        for _ in range(num_frames):
            frame = simulate_frame(frame_config, rng, csi="ls")
            perfect = zero_forcing_equalize(frame.received_grid, frame.channel_frequency_response)
            perfect_bits = demodulate(perfect.symbols[config.data_indices], config.modulation_order)
            errors_perfect += int(np.count_nonzero(frame.reference_bits != perfect_bits))
            errors_ls += int(np.count_nonzero(frame.reference_bits != frame.estimated_bits))
            nmse_sum += frame.nmse
            stabilized_perfect += perfect.stabilized_bins
            stabilized_ls += frame.stabilized_bins
        total_bits = num_frames * config.bits_per_frame
        points.append(
            SweepPoint(
                float(snr_db),
                total_bits,
                errors_perfect,
                errors_ls,
                errors_perfect / total_bits,
                errors_ls / total_bits,
                nmse_sum / num_frames,
                stabilized_perfect,
                stabilized_ls,
            )
        )
    return SweepResult(config, num_frames, seed, tuple(points))


def save_sweep(result: SweepResult, output_dir: str | Path) -> None:

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in result.points]
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    configuration = asdict(result.config)
    configuration["pilot_value"] = [result.config.pilot_value.real, result.config.pilot_value.imag]
    configuration["channel_taps"] = [[tap.real, tap.imag] for tap in result.config.channel_taps]
    if result.config.pilot_values is not None:
        configuration["pilot_values"] = [[tap.real, tap.imag] for tap in result.config.pilot_values]
    if result.config.channel is not None:
        configuration["channel"]["fixed_taps"] = [
            [tap.real, tap.imag] for tap in result.config.channel.fixed_taps
        ]

    configuration.pop("snr_db")
    metadata = {
        "config": configuration,
        "seed": result.seed,
        "rng": "numpy.random.default_rng / PCG64; reset to seed at each SNR",
        "snr_values_db": [point.snr_db for point in result.points],
        "num_frames_per_snr": result.num_frames,
        "bits_per_snr": result.points[0].total_bits,
        "snr_definition": "received power averaged over retained N+CP samples / complex noise variance",
        "fft_convention": "ortho for signals; backward (ordinary FFT) for FIR channel response",
        "channel_policy": "fixed normalized taps, or independent Rayleigh realization per one-symbol frame",
        "versions": {
            name: version(name) for name in ("neural-ofdm", "numpy", "scipy", "matplotlib")
        },
        "python_version": platform.python_version(),
        "paired_receivers": True,
        "nmse_aggregation": "arithmetic mean of per-frame linear NMSE over all FFT bins",
        "zero_error_plot": "approximate 95% binomial upper bound; independent-bit assumption",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def format_sweep_table(result: SweepResult) -> str:

    lines = ["SNR dB | BER perfect | BER LS      | NMSE LS     | errors perfect/LS | bits"]
    lines.extend(
        f"{p.snr_db:6.1f} | {p.ber_perfect:11.4g} | {p.ber_ls:11.4g} | {p.nmse_ls:11.4g} | "
        f"{p.errors_perfect:7d}/{p.errors_ls:<7d} | {p.total_bits}"
        for p in result.points
    )
    return "\n".join(lines)
