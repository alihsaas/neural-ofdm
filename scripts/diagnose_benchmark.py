from dataclasses import replace
from pathlib import Path

import numpy as np

from neural_ofdm.benchmark import benchmark_studies, write_csv
from neural_ofdm.channel import frequency_statistics, reference_noise_variance
from neural_ofdm.estimation import prepare_lmmse
from neural_ofdm.simulation import observe_frame, receive_observation


def main() -> None:
    configuration = benchmark_studies()["receiver_comparison"]
    ofdm = configuration.ofdm
    mean, covariance = frequency_statistics(ofdm.channel_prior, ofdm.fft_size)
    rows = []
    for reference in (20, 25):
        frame_config = replace(ofdm, reference_esn0_db=reference)
        estimator = prepare_lmmse(
            mean,
            covariance,
            ofdm.pilot_indices,
            ofdm.pilot_symbols,
            reference_noise_variance(reference),
        )
        payload_rng, channel_rng, noise_rng = [
            np.random.default_rng(stream)
            for stream in np.random.SeedSequence(configuration.seed).spawn(3)
        ]
        details = []
        for index in range(configuration.frames):
            observation = observe_frame(
                frame_config, payload_rng, channel_rng=channel_rng, noise_rng=noise_rng
            )
            frame = receive_observation(frame_config, observation, csi="lmmse", estimator=estimator)
            data = ofdm.data_indices
            worst_bin = data[
                np.argmax(np.abs(frame.decision_grid[data] - frame.frequency_grid[data]))
            ]
            details.append(
                {
                    "frame": index,
                    "evm_squared": frame.evm_decision**2,
                    "min_estimated_payload_gain": float(
                        np.min(np.abs(frame.estimated_channel[data]))
                    ),
                    "worst_bin": int(worst_bin),
                    "estimated_gain_at_worst_bin": float(
                        np.abs(frame.estimated_channel[worst_bin])
                    ),
                    "true_gain_at_worst_bin": float(np.abs(observation.true_channel[worst_bin])),
                    "stabilized_bins": frame.stabilized_bins,
                }
            )
        values = np.array([item["evm_squared"] for item in details])
        largest = details[int(np.argmax(values))]
        rows.append(
            {
                "reference_esn0_db": reference,
                "seed": configuration.seed,
                "frames": configuration.frames,
                "receiver": "lmmse",
                "equalizer": "zf",
                "frame_rms_evm": float(np.sqrt(values.mean())),
                "median_frame_evm": float(np.sqrt(np.median(values))),
                "largest_frame_fraction_of_evm_squared_sum": float(values.max() / values.sum()),
                "largest_five_frames_fraction": float(np.sort(values)[-5:].sum() / values.sum()),
                **{f"largest_{key}": value for key, value in largest.items()},
            }
        )
        print(rows[-1], flush=True)
    destination = (
        Path(__file__).resolve().parents[1] / "results" / "classical_benchmark" / "diagnostics"
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "zf_tail.csv", rows)


if __name__ == "__main__":
    main()
