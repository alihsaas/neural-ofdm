import argparse
from pathlib import Path

import matplotlib

from neural_ofdm.benchmark import benchmark_studies, load_benchmark, run_benchmark, save_benchmark


def main() -> None:
    matplotlib.use("Agg")
    from neural_ofdm.benchmark_plotting import save_benchmark_figures

    parser = argparse.ArgumentParser(description="Run controlled classical OFDM estimation studies")
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/classical_benchmark"))
    parser.add_argument(
        "--plots-only", action="store_true", help="rebuild figures from saved CSV/JSON"
    )
    args = parser.parse_args()
    if args.frames < 2 or args.seed < 0:
        parser.error("frames must be >=2 and seed nonnegative")
    results = {}
    for name, configuration in benchmark_studies(args.frames, args.seed).items():
        if args.plots_only:
            results[name] = load_benchmark(args.output / name)
            continue
        print(f"Running {name}: {args.frames} independent frames per reference level", flush=True)
        result = run_benchmark(configuration)
        save_benchmark(result, args.output / name)
        results[name] = result
        row = next(
            row
            for row in result.rows
            if row["receiver"] == "lmmse" and row["reference_esn0_db"] == 15
        )
        print(
            f"  15 dB: LMMSE BER={row['ber']:.6g}, NMSE={row['nmse_mean_per_frame']:.6g}, errors={row['errors']}/{row['total_bits']}",
            flush=True,
        )
    save_benchmark_figures(results, args.output)
    print(f"Saved CSV, metadata, frame counts and figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
