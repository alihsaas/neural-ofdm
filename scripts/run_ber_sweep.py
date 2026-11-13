import argparse
from pathlib import Path

import matplotlib
import numpy as np

from neural_ofdm.config import OFDMConfig
from neural_ofdm.experiments import format_sweep_table, run_ber_sweep, save_sweep


def main() -> None:
    matplotlib.use("Agg")
    from neural_ofdm.plotting import save_baseline_figures, save_sweep_figures

    parser = argparse.ArgumentParser(description="Compare perfect-CSI and pilot-LS OFDM receivers")
    parser.add_argument("--frames", type=int, default=5000, help="one-symbol frames per SNR")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-start", type=float, default=0)
    parser.add_argument("--snr-stop", type=float, default=30)
    parser.add_argument("--snr-step", type=float, default=5)
    parser.add_argument("--channel", choices=("fixed", "rayleigh"), default="fixed")
    parser.add_argument("--output", type=Path, default=Path("results/ber_sweep"))
    parser.add_argument(
        "--demo-figures", action="store_true", help="also save fixed-channel teaching figures"
    )
    args = parser.parse_args()
    if args.frames < 1 or args.seed < 0:
        parser.error("frames must be positive and seed nonnegative")
    if not np.all(np.isfinite([args.snr_start, args.snr_stop, args.snr_step])):
        parser.error("SNR values must be finite")
    if args.snr_step <= 0 or args.snr_stop < args.snr_start:
        parser.error("SNR step must be positive and stop >= start")
    count = int(np.floor((args.snr_stop - args.snr_start) / args.snr_step + 1e-10)) + 1
    snrs = args.snr_start + args.snr_step * np.arange(count)
    config = OFDMConfig(channel_model=args.channel, random_seed=args.seed)
    result = run_ber_sweep(config, snrs, args.frames)
    save_sweep(result, args.output)
    save_sweep_figures(result, args.output)
    if args.demo_figures:
        save_baseline_figures(OFDMConfig(random_seed=args.seed), args.output / "demo", args.seed)
    print(format_sweep_table(result))
    print(f"Saved CSV, metadata and figures to {args.output.resolve()}")
    print("Zero observed errors do not prove zero BER; see counts and plotted upper bounds.")


if __name__ == "__main__":
    main()
