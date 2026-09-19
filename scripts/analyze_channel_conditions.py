import argparse
import json
from pathlib import Path

import numpy as np

from neural_ofdm.channel import frequency_statistics
from neural_ofdm.neural_study import (
    PROTOCOL_PATH,
    evaluation_conditions,
    load_protocol,
    protocol_path,
)


def delay_profile(channel) -> dict:
    mean, covariance = channel.tap_statistics()
    power = np.abs(mean) ** 2 + np.real(np.diag(covariance))
    delays = np.asarray(channel.delays, dtype=float)
    average = float(np.sum(power * delays) / power.sum())
    rms = float(np.sqrt(np.sum(power * (delays - average) ** 2) / power.sum()))
    return {
        "delays_samples": list(channel.delays),
        "tap_count": len(channel.delays),
        "max_delay_samples": int(channel.delays[-1]),
        "expected_tap_power": power.tolist(),
        "expected_total_power": float(power.sum()),
        "mean_delay_samples": average,
        "rms_delay_spread_samples": rms,
    }


def pilot_observability(ofdm) -> dict:
    channel = ofdm.channel
    pilots = ofdm.pilot_indices
    delays = np.asarray(channel.delays)
    matrix = np.exp(-2j * np.pi * np.outer(pilots, delays) / ofdm.fft_size)
    singular = np.linalg.svd(matrix, compute_uv=False)
    full_support = np.arange(ofdm.cp_length + 1)
    full = np.exp(-2j * np.pi * np.outer(pilots, full_support) / ofdm.fft_size)
    return {
        "pilot_count": int(pilots.size),
        "pilot_spacing": ofdm.pilot_spacing,
        "pilot_overhead": ofdm.pilot_overhead,
        "payload_bits_per_frame": ofdm.bits_per_frame,
        "tap_support_rank": int(np.linalg.matrix_rank(matrix)),
        "tap_support_condition_number": float(singular[0] / singular[-1]),
        "aliased_delay_classes_mod_pilot_count": sorted({int(d % pilots.size) for d in delays}),
        "cp_support_taps": int(full_support.size),
        "cp_support_rank": int(np.linalg.matrix_rank(full)),
        "unaliased_delay_limit_samples": int(pilots.size - 1),
    }


def frequency_correlation(ofdm) -> dict:
    mean, covariance = frequency_statistics(ofdm.channel, ofdm.fft_size)
    second = covariance + np.outer(mean, mean.conj())
    diagonal = np.real(np.diag(second))
    lags = [1, 2, 4, 8, 16, 32]
    return {
        f"abs_correlation_lag_{lag}": float(
            np.mean(np.abs(np.diag(second, lag)) / np.sqrt(diagonal[:-lag] * diagonal[lag:]))
        )
        for lag in lags
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Delay profile and pilot observability table")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    rows = []
    for condition in evaluation_conditions(protocol):
        ofdm = condition.datasets[0].ofdm
        rows.append(
            {
                "condition": condition.name,
                "kind": condition.kind,
                "channel_model": ofdm.channel.model,
                "k_linear": ofdm.channel.k_linear,
                **delay_profile(ofdm.channel),
                **pilot_observability(ofdm),
                **frequency_correlation(ofdm),
            }
        )
    output = protocol_path(protocol, "results") / "channel_conditions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="ascii")
    for row in rows:
        print(
            f"{row['condition']:>16}: taps {row['tap_count']}, max delay"
            f" {row['max_delay_samples']}, rms {row['rms_delay_spread_samples']:.2f},"
            f" pilots {row['pilot_count']}, rank {row['tap_support_rank']}/{row['tap_count']},"
            f" cond {row['tap_support_condition_number']:.3g},"
            f" |rho(4)| {row['abs_correlation_lag_4']:.3f}"
        )
    print(f"Saved {output.as_posix()}")


if __name__ == "__main__":
    main()
