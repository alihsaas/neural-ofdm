import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neural_ofdm.benchmark import write_csv
from neural_ofdm.channel import ChannelConfig, frequency_statistics, reference_noise_variance
from neural_ofdm.config import OFDMConfig
from neural_ofdm.estimation import LMMSEEstimator, prepare_lmmse
from neural_ofdm.neural_data import StoredDataset, ls_estimate_for
from neural_ofdm.simulation import Observation, receive_with_estimate

CLASSICAL_RECEIVERS = ("perfect", "ls")


def mixture_noise_variance(reference_range_db: tuple[float, float]) -> float:
    low, high = reference_range_db
    if high == low:
        return reference_noise_variance(low)
    scale = np.log(10) / 10
    return float((np.exp(-scale * low) - np.exp(-scale * high)) / (scale * (high - low)))


def lmmse_for(ofdm: OFDMConfig, prior: ChannelConfig, noise_variance: float) -> LMMSEEstimator:
    mean, covariance = frequency_statistics(prior, ofdm.fft_size)
    return prepare_lmmse(mean, covariance, ofdm.pilot_indices, ofdm.pilot_symbols, noise_variance)


def linear_estimators(
    dataset: StoredDataset, training_prior: ChannelConfig, training_range_db: tuple[float, float]
) -> dict[str, LMMSEEstimator]:
    ofdm = dataset.config.ofdm
    variance = reference_noise_variance(dataset.config.fixed_reference_db)
    estimators = {"lmmse_matched": lmmse_for(ofdm, ofdm.channel, variance)}
    if ofdm.channel != training_prior:
        estimators["lmmse_training_prior"] = lmmse_for(ofdm, training_prior, variance)
    estimators["lmmse_snr_unaware"] = lmmse_for(
        ofdm, training_prior, mixture_noise_variance(training_range_db)
    )
    return estimators


@dataclass(frozen=True)
class PairedEvaluation:
    receivers: tuple[str, ...]
    reference_db: float
    errors: NDArray[np.int64]
    nmse: NDArray[np.float64]
    squared_error: NDArray[np.float64]
    stabilized: NDArray[np.int64]
    channel_energy: NDArray[np.float64]
    bits_per_frame: int
    fft_size: int


def replay_observations(dataset: StoredDataset) -> list[tuple[OFDMConfig, Observation]]:
    dataset.verify_replay()
    return [dataset.config.replay(index) for index in range(dataset.config.count)]


def evaluate_paired(
    dataset: StoredDataset,
    estimators: dict[str, LMMSEEstimator],
    neural_estimates: dict[str, NDArray[np.complex128]],
    equalizer: str = "zf",
    observations: list[tuple[OFDMConfig, Observation]] | None = None,
) -> PairedEvaluation:
    frames = observations if observations is not None else replay_observations(dataset)
    if len(frames) != dataset.config.count:
        raise ValueError("observations must cover every stored frame")
    for name, estimates in neural_estimates.items():
        if estimates.shape != (dataset.config.count, dataset.config.ofdm.fft_size):
            raise ValueError(f"neural estimates for {name} have the wrong shape")
    receivers = (*CLASSICAL_RECEIVERS, *estimators, *neural_estimates)
    shape = (len(frames), len(receivers))
    errors = np.zeros(shape, dtype=np.int64)
    nmse = np.zeros(shape)
    squared = np.zeros(shape)
    stabilized = np.zeros(shape, dtype=np.int64)
    energy = np.zeros(len(frames))
    for index, (frame_config, observation) in enumerate(frames):
        estimates = {
            "perfect": observation.true_channel,
            "ls": ls_estimate_for(frame_config, observation),
        }
        estimates |= {
            name: estimator.estimate(observation.received_grid)
            for name, estimator in estimators.items()
        }
        estimates |= {name: values[index] for name, values in neural_estimates.items()}
        energy[index] = np.sum(np.abs(observation.true_channel) ** 2)
        for column, name in enumerate(receivers):
            result = receive_with_estimate(
                frame_config, observation, estimates[name], csi="external", equalizer=equalizer
            )
            errors[index, column] = np.count_nonzero(result.reference_bits != result.estimated_bits)
            nmse[index, column] = result.nmse
            squared[index, column] = np.sum(
                np.abs(result.estimated_channel - observation.true_channel) ** 2
            )
            stabilized[index, column] = result.stabilized_bins
    return PairedEvaluation(
        receivers,
        float(dataset.config.fixed_reference_db),
        errors,
        nmse,
        squared,
        stabilized,
        energy,
        dataset.config.ofdm.bits_per_frame,
        dataset.config.ofdm.fft_size,
    )


def bootstrap_means(
    values: NDArray[np.float64], replicates: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    samples = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        indices = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        samples[start : start + count] = values[indices].mean(axis=1)
    return samples


def summarize(
    evaluation: PairedEvaluation, replicates: int, seed: int, condition: str
) -> tuple[list[dict], list[dict]]:
    frames = evaluation.errors.shape[0]
    bits = evaluation.bits_per_frame
    total_bits = frames * bits
    combined = np.concatenate((evaluation.errors / bits, evaluation.nmse), axis=1)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, int(round(evaluation.reference_db * 10)) + 1000])
    )
    samples = bootstrap_means(combined, replicates, rng)
    receivers = evaluation.receivers
    count = len(receivers)
    ber_samples, nmse_samples = samples[:, :count], samples[:, count:]
    rows = []
    zero_frame_bound = float(-np.expm1(np.log(0.05) / frames))
    for column, name in enumerate(receivers):
        errors = int(evaluation.errors[:, column].sum())
        error_frames = int(np.count_nonzero(evaluation.errors[:, column]))
        ber_low, ber_high = np.quantile(ber_samples[:, column], [0.025, 0.975])
        nmse_low, nmse_high = np.quantile(nmse_samples[:, column], [0.025, 0.975])
        run, _, seed_text = name.partition("/seed_")
        rows.append(
            {
                "condition": condition,
                "reference_esn0_db": evaluation.reference_db,
                "noise_variance": reference_noise_variance(evaluation.reference_db),
                "receiver": name,
                "run": run,
                "seed": int(seed_text) if seed_text else "",
                "frames": frames,
                "total_bits": total_bits,
                "errors": errors,
                "error_frames": error_frames,
                "ber": errors / total_bits,
                "ber_frame_bootstrap_low": float(ber_low) if error_frames else "",
                "ber_frame_bootstrap_high": float(ber_high) if error_frames else "",
                "zero_error_frame_upper95": "" if error_frames else zero_frame_bound,
                "nmse_mean_per_frame": float(evaluation.nmse[:, column].mean()),
                "nmse_frame_bootstrap_low": float(nmse_low),
                "nmse_frame_bootstrap_high": float(nmse_high),
                "ensemble_mse_per_bin": float(
                    evaluation.squared_error[:, column].sum() / (frames * evaluation.fft_size)
                ),
                "ensemble_nmse": float(
                    evaluation.squared_error[:, column].sum() / evaluation.channel_energy.sum()
                ),
                "stabilized_bins": int(evaluation.stabilized[:, column].sum()),
            }
        )
    comparisons = []
    for baseline in ("lmmse_matched", "ls"):
        if baseline not in receivers:
            continue
        base = receivers.index(baseline)
        for column, name in enumerate(receivers):
            if name in ("perfect", baseline):
                continue
            ber_difference = ber_samples[:, column] - ber_samples[:, base]
            nmse_difference = nmse_samples[:, column] - nmse_samples[:, base]
            comparisons.append(
                {
                    "condition": condition,
                    "reference_esn0_db": evaluation.reference_db,
                    "receiver": name,
                    "baseline": baseline,
                    "ber_difference": float(
                        (evaluation.errors[:, column].sum() - evaluation.errors[:, base].sum())
                        / total_bits
                    ),
                    "ber_difference_low": float(np.quantile(ber_difference, 0.025)),
                    "ber_difference_high": float(np.quantile(ber_difference, 0.975)),
                    "ber_informative": bool(
                        np.any(evaluation.errors[:, column] != evaluation.errors[:, base])
                    ),
                    "nmse_mean_difference": float(
                        evaluation.nmse[:, column].mean() - evaluation.nmse[:, base].mean()
                    ),
                    "nmse_difference_low": float(np.quantile(nmse_difference, 0.025)),
                    "nmse_difference_high": float(np.quantile(nmse_difference, 0.975)),
                    "fraction_frames_lower_nmse": float(
                        np.mean(evaluation.nmse[:, column] < evaluation.nmse[:, base])
                    ),
                }
            )
    return rows, comparisons


def save_condition(
    directory: str | Path,
    evaluations: list[PairedEvaluation],
    rows: list[dict],
    comparisons: list[dict],
) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "metrics.csv", rows)
    write_csv(destination / "paired_comparisons.csv", comparisons)
    with (destination / "frame_errors.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference_esn0_db", "frame", *evaluations[0].receivers])
        for evaluation in evaluations:
            for index, counts in enumerate(evaluation.errors):
                writer.writerow([evaluation.reference_db, index, *counts.tolist()])
    np.savez(
        destination / "frame_metrics.npz",
        receivers=np.asarray(evaluations[0].receivers),
        reference_esn0_db=np.asarray([item.reference_db for item in evaluations]),
        errors=np.stack([item.errors for item in evaluations]),
        nmse=np.stack([item.nmse for item in evaluations]),
        squared_error=np.stack([item.squared_error for item in evaluations]),
        channel_energy=np.stack([item.channel_energy for item in evaluations]),
    )


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    numeric = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            if key in ("condition", "receiver", "run", "baseline"):
                parsed[key] = value
            elif value in ("True", "False"):
                parsed[key] = value == "True"
            elif value == "":
                parsed[key] = None
            else:
                number = float(value)
                parsed[key] = (
                    int(number)
                    if number.is_integer()
                    and key
                    in (
                        "seed",
                        "frames",
                        "total_bits",
                        "errors",
                        "error_frames",
                        "stabilized_bins",
                    )
                    else number
                )
        numeric.append(parsed)
    return numeric
