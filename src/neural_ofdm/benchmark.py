import csv
import json
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neural_ofdm._validation import integer
from neural_ofdm.channel import ChannelConfig, frequency_statistics, reference_noise_variance
from neural_ofdm.config import OFDMConfig
from neural_ofdm.estimation import prepare_lmmse
from neural_ofdm.simulation import observe_frame, receive_observation


@dataclass(frozen=True)
class ExperimentConfig:
    ofdm: OFDMConfig = OFDMConfig(noise_mode="fixed", channel=ChannelConfig())
    reference_values_db: tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)
    frames: int = 3000
    seed: int = 2026
    equalizer: Literal["zf", "mmse"] = "zf"
    assumed_channel: ChannelConfig | None = None
    assumed_noise_factor: float = 1.0
    bootstrap_replicates: int = 1000

    def __post_init__(self) -> None:
        integer(self.frames, "frames", 2)
        integer(self.seed, "seed")
        integer(self.bootstrap_replicates, "bootstrap_replicates", 100)
        if self.ofdm.noise_mode != "fixed":
            raise ValueError("statistical benchmark requires independent fixed noise")
        if self.equalizer not in ("zf", "mmse"):
            raise ValueError("equalizer must be zf or mmse")
        values = np.asarray(self.reference_values_db, dtype=float)
        if (
            not isinstance(self.reference_values_db, tuple)
            or values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError("reference_values_db must be a finite increasing tuple")
        for value in values:
            reference_noise_variance(value)
        if not np.isfinite(self.assumed_noise_factor) or self.assumed_noise_factor <= 0:
            raise ValueError("assumed_noise_factor must be finite and positive")
        if self.assumed_channel is not None:
            frequency_statistics(self.assumed_channel, self.ofdm.fft_size)


@dataclass(frozen=True)
class BenchmarkResult:
    config: ExperimentConfig
    rows: tuple[dict, ...]
    comparisons: tuple[dict, ...]
    frame_errors: tuple[NDArray[np.int64], ...]
    receiver_names: tuple[str, ...]
    diagnostics: tuple[dict, ...]


def paired_frame_bootstrap(
    errors: NDArray[np.int64], bits_per_frame: int, replicates: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    samples = np.empty((replicates, errors.shape[1]))
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        indices = rng.integers(0, errors.shape[0], size=(count, errors.shape[0]))
        samples[start : start + count] = errors[indices].mean(axis=1) / bits_per_frame
    return samples


def run_benchmark(config: ExperimentConfig) -> BenchmarkResult:
    ofdm = config.ofdm
    mean, covariance = frequency_statistics(ofdm.channel_prior, ofdm.fft_size)
    mismatch = config.assumed_channel is not None or config.assumed_noise_factor != 1
    assumed = config.assumed_channel or ofdm.channel_prior
    assumed_mean, assumed_covariance = frequency_statistics(assumed, ofdm.fft_size)
    names = ("perfect", "ls", "lmmse") + (("mismatched",) if mismatch else ())
    rows, comparisons, errors_by_level, diagnostics = [], [], [], []
    for level_index, reference in enumerate(config.reference_values_db):
        variance = reference_noise_variance(reference)
        frame_config = replace(ofdm, reference_esn0_db=float(reference))
        estimators = {
            "lmmse": prepare_lmmse(
                mean, covariance, ofdm.pilot_indices, ofdm.pilot_symbols, variance
            )
        }
        if mismatch:
            estimators["mismatched"] = prepare_lmmse(
                assumed_mean,
                assumed_covariance,
                ofdm.pilot_indices,
                ofdm.pilot_symbols,
                variance * config.assumed_noise_factor,
            )
        for name, estimator in estimators.items():
            diagnostics.append(
                {
                    "reference_esn0_db": reference,
                    "receiver": name,
                    "method": estimator.method,
                    "observation_rank": estimator.observation_rank,
                    "condition_number": estimator.condition_number,
                    "relative_tolerance": estimator.relative_tolerance,
                    "eigenvalue_cutoff": estimator.eigenvalue_cutoff,
                    "assumed_noise_variance": estimator.noise_variance,
                    "jitter": 0.0,
                }
            )
        payload_rng, channel_rng, noise_rng = [
            np.random.default_rng(stream) for stream in np.random.SeedSequence(config.seed).spawn(3)
        ]
        errors = np.zeros((config.frames, len(names)), dtype=np.int64)
        nmse_sum = np.zeros(len(names))
        squared_error_sum = np.zeros(len(names))
        evm_squared_sum = np.zeros(len(names))
        evm_raw_squared_sum = np.zeros(len(names))
        stabilized = np.zeros(len(names), dtype=np.int64)
        truth_energy_sum = clean_power_sum = measured_noise_sum = 0.0
        for frame_index in range(config.frames):
            observation = observe_frame(
                frame_config, payload_rng, channel_rng=channel_rng, noise_rng=noise_rng
            )
            truth_energy_sum += float(np.sum(np.abs(observation.true_channel) ** 2))
            clean_power_sum += observation.clean_received_power
            measured_noise_sum += observation.measured_noise_power
            for receiver_index, name in enumerate(names):
                frame = receive_observation(
                    frame_config,
                    observation,
                    csi=name if name in ("perfect", "ls") else "lmmse",
                    estimator=estimators.get(name),
                    equalizer=config.equalizer,
                )
                errors[frame_index, receiver_index] = np.count_nonzero(
                    frame.reference_bits != frame.estimated_bits
                )
                nmse_sum[receiver_index] += frame.nmse
                squared_error_sum[receiver_index] += np.sum(
                    np.abs(frame.estimated_channel - observation.true_channel) ** 2
                )
                evm_squared_sum[receiver_index] += frame.evm_decision**2
                evm_raw_squared_sum[receiver_index] += frame.evm_raw**2
                stabilized[receiver_index] += frame.stabilized_bins
        total_bits = config.frames * ofdm.bits_per_frame
        bootstrap_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4, level_index]))
        bootstrap = paired_frame_bootstrap(
            errors, ofdm.bits_per_frame, config.bootstrap_replicates, bootstrap_rng
        )
        total_errors = errors.sum(axis=0)
        for index, name in enumerate(names):
            lower, upper = np.quantile(bootstrap[:, index], [0.025, 0.975])
            nonzero = int(np.count_nonzero(errors[:, index]))
            zero_frame_bound = float(-np.expm1(np.log(0.05) / config.frames))
            rows.append(
                {
                    "reference_esn0_db": reference,
                    "noise_mode": "fixed",
                    "noise_variance": variance,
                    "receiver": name,
                    "equalizer": config.equalizer,
                    "frames": config.frames,
                    "total_bits": total_bits,
                    "payload_bits_per_frame": ofdm.bits_per_frame,
                    "pilot_count": ofdm.pilot_indices.size,
                    "pilot_overhead": ofdm.pilot_overhead,
                    "cp_overhead": ofdm.cp_overhead,
                    "modulation_order": ofdm.modulation_order,
                    "errors": int(total_errors[index]),
                    "error_frames": nonzero,
                    "ber": float(total_errors[index] / total_bits),
                    "ber_frame_bootstrap_low": float(lower) if nonzero else None,
                    "ber_frame_bootstrap_high": float(upper) if nonzero else None,
                    "zero_error_frame_upper95": zero_frame_bound if not nonzero else None,
                    "zero_error_bit_upper95_independence_approx": (
                        float(-np.expm1(np.log(0.05) / total_bits)) if not nonzero else None
                    ),
                    "nmse_mean_per_frame": float(nmse_sum[index] / config.frames),
                    "ensemble_mse_per_bin": float(
                        squared_error_sum[index] / config.frames / ofdm.fft_size
                    ),
                    "ensemble_nmse": float(squared_error_sum[index] / truth_energy_sum),
                    "evm_decision_frame_rms": float(
                        np.sqrt(evm_squared_sum[index] / config.frames)
                    ),
                    "evm_raw_frame_rms": float(np.sqrt(evm_raw_squared_sum[index] / config.frames)),
                    "stabilized_bins": int(stabilized[index]),
                    "clean_received_power_mean": clean_power_sum / config.frames,
                    "measured_noise_power_mean": measured_noise_sum / config.frames,
                }
            )
        for name in names:
            if name == "lmmse":
                continue
            index = names.index(name)
            differences = bootstrap[:, index] - bootstrap[:, names.index("lmmse")]
            lower, upper = np.quantile(differences, [0.025, 0.975])
            comparisons.append(
                {
                    "reference_esn0_db": reference,
                    "receiver_minus_lmmse": name,
                    "ber_difference": float((total_errors[index] - total_errors[2]) / total_bits),
                    "paired_bootstrap_low": float(lower),
                    "paired_bootstrap_high": float(upper),
                    "informative": bool(np.any(errors[:, index] != errors[:, 2])),
                }
            )
        errors_by_level.append(errors)
    return BenchmarkResult(
        config, tuple(rows), tuple(comparisons), tuple(errors_by_level), names, tuple(diagnostics)
    )


def json_safe(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: tuple[dict, ...] | list[dict]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def save_benchmark(result: BenchmarkResult, directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "metrics.csv", result.rows)
    write_csv(destination / "paired_comparisons.csv", result.comparisons)
    with (destination / "frame_errors.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference_esn0_db", "frame", *result.receiver_names])
        for reference, counts in zip(
            result.config.reference_values_db, result.frame_errors, strict=True
        ):
            for index, row in enumerate(counts):
                writer.writerow([reference, index, *row])
    configuration = asdict(result.config)
    configuration["ofdm"].pop("snr_db")
    configuration["ofdm"].pop("reference_esn0_db")
    metadata = {
        "experiment": configuration,
        "schema_version": 1,
        "versions": {
            name: version(name) for name in ("neural-ofdm", "numpy", "scipy", "matplotlib")
        },
        "python_version": platform.python_version(),
        "noise_definition": "Es=1; sigma^2=10^(-reference_esn0_db/10), independent of channel and payload",
        "noise_variance_units": "total complex sample variance; unitary FFT preserves it",
        "fft_convention": "unitary signal FFT/IFFT; unscaled channel Fourier matrix",
        "channel_policy": "independent channel per one-symbol frame; held constant within symbol",
        "rician_definition": "global linear K; fixed-phase single unit-energy LOS path plus diffuse PDP",
        "rng_policy": "PCG64, SeedSequence(seed).spawn(3): payload, channel, noise; reset each level and study",
        "pairing": "all receivers including mismatch use identical received observations",
        "cross_grid_policy": "same channel and standardized time noise across densities/modulations; payload length and waveform differ",
        "uncertainty": "paired percentile bootstrap of independent frames, shared indices for all receivers; 95%",
        "sparse_error_limit": "bootstrap can be unreliable with few error frames; raw counts always retained",
        "zero_error_limit": "no bootstrap interval; frame event upper95 bounds BER conservatively because BER <= probability(any frame error)",
        "bit_bound": "independent-bit approximation only; shared fading violates bit independence",
        "nmse": "arithmetic mean of per-frame ||error||^2/||H||^2, all bins, linear units",
        "ensemble_nmse": "sum channel squared error / sum true channel energy, all frames and bins",
        "evm": "RMS over per-frame payload EVM squared; decision output gain corrected; raw output separately labeled",
        "estimator_diagnostics": result.diagnostics,
    }
    (destination / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, allow_nan=False) + "\n", encoding="ascii"
    )


def benchmark_studies(frames: int = 3000, seed: int = 2026) -> dict[str, ExperimentConfig]:
    channel = ChannelConfig(delays=(0, 1, 3, 6), decay_samples=2.5)
    ofdm = OFDMConfig(noise_mode="fixed", channel=channel)
    base = ExperimentConfig(ofdm=ofdm, frames=frames, seed=seed)
    return {
        "receiver_comparison": base,
        "pilot_spacing_2": replace(base, ofdm=replace(ofdm, pilot_spacing=2)),
        "pilot_spacing_8": replace(base, ofdm=replace(ofdm, pilot_spacing=8)),
        "modulation_16qam": replace(base, ofdm=replace(ofdm, modulation_order=16)),
        "rician_k1": replace(
            base, ofdm=replace(ofdm, channel=replace(channel, model="rician", k_linear=1))
        ),
        "rician_k10": replace(
            base, ofdm=replace(ofdm, channel=replace(channel, model="rician", k_linear=10))
        ),
        "prior_mismatch": replace(
            base,
            assumed_channel=ChannelConfig(delays=(0, 1, 2, 3), decay_samples=0.8),
            assumed_noise_factor=2,
        ),
        "rician_mean_mismatch": replace(
            base,
            ofdm=replace(ofdm, channel=replace(channel, model="rician", k_linear=10)),
            assumed_channel=channel,
        ),
        "fixed_reference": replace(
            base,
            ofdm=replace(
                ofdm,
                channel=ChannelConfig(
                    model="fixed", delays=(0, 1, 2), fixed_taps=ofdm.channel_taps
                ),
            ),
        ),
    }


def load_benchmark(directory: str | Path) -> BenchmarkResult:
    source = Path(directory)
    metadata = json.loads((source / "metadata.json").read_text(encoding="ascii"))
    if metadata["schema_version"] != 1:
        raise ValueError("unsupported benchmark schema version")
    configuration = metadata["experiment"]

    def decode_complex(value) -> complex:
        return complex(*value) if isinstance(value, list) else complex(value)

    def decode_channel(values: dict | None) -> ChannelConfig | None:
        if values is None:
            return None
        return ChannelConfig(
            **{
                key: value
                for key, value in values.items()
                if key not in ("delays", "powers", "fixed_taps")
            },
            delays=tuple(values["delays"]),
            powers=None if values["powers"] is None else tuple(values["powers"]),
            fixed_taps=tuple(decode_complex(value) for value in values["fixed_taps"]),
        )

    values = configuration["ofdm"]
    ofdm = OFDMConfig(
        **{
            key: value
            for key, value in values.items()
            if key
            not in ("channel", "pilot_value", "channel_taps", "rayleigh_powers", "pilot_values")
        },
        channel=decode_channel(values["channel"]),
        pilot_value=decode_complex(values["pilot_value"]),
        channel_taps=tuple(decode_complex(value) for value in values["channel_taps"]),
        rayleigh_powers=tuple(values["rayleigh_powers"]),
        pilot_values=None
        if values["pilot_values"] is None
        else tuple(decode_complex(value) for value in values["pilot_values"]),
    )
    config = ExperimentConfig(
        **{
            key: value
            for key, value in configuration.items()
            if key not in ("ofdm", "assumed_channel", "reference_values_db")
        },
        ofdm=ofdm,
        assumed_channel=decode_channel(configuration["assumed_channel"]),
        reference_values_db=tuple(configuration["reference_values_db"]),
    )
    integers = {
        "frames",
        "total_bits",
        "payload_bits_per_frame",
        "pilot_count",
        "modulation_order",
        "errors",
        "error_frames",
        "stabilized_bins",
    }

    def read_rows(filename: str) -> tuple[dict, ...]:
        with (source / filename).open(encoding="ascii") as handle:
            output = []
            for row in csv.DictReader(handle):
                parsed = {}
                for key, value in row.items():
                    if key in ("receiver", "equalizer", "noise_mode", "receiver_minus_lmmse"):
                        parsed[key] = value
                    elif key == "informative":
                        parsed[key] = value == "True"
                    elif value == "":
                        parsed[key] = None
                    else:
                        parsed[key] = int(value) if key in integers else float(value)
                output.append(parsed)
        return tuple(output)

    rows = read_rows("metrics.csv")
    names = tuple(
        row["receiver"] for row in rows if row["reference_esn0_db"] == config.reference_values_db[0]
    )
    counts = np.loadtxt(source / "frame_errors.csv", delimiter=",", skiprows=1, ndmin=2)
    errors = tuple(
        counts[counts[:, 0] == level, 2:].astype(np.int64) for level in config.reference_values_db
    )
    if any(array.shape != (config.frames, len(names)) for array in errors):
        raise ValueError("saved frame counts do not match the configuration")
    return BenchmarkResult(
        config,
        rows,
        read_rows("paired_comparisons.csv"),
        errors,
        names,
        tuple(metadata["estimator_diagnostics"]),
    )
