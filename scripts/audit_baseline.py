import csv
import json
from pathlib import Path

import numpy as np

from neural_ofdm.config import OFDMConfig
from neural_ofdm.experiments import run_ber_sweep


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = {}
    for model in ("fixed", "rayleigh"):
        directory = root / "results" / f"baseline_{model}"
        metadata = json.loads((directory / "metadata.json").read_text())
        with (directory / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        reference = next(row for row in rows if float(row["snr_db"]) == 10)
        result = run_ber_sweep(
            OFDMConfig(channel_model=model), [10], metadata["num_frames_per_snr"], metadata["seed"]
        ).points[0]
        for name in ("total_bits", "errors_perfect", "errors_ls"):
            assert getattr(result, name) == int(reference[name])
        np.testing.assert_allclose(result.nmse_ls, float(reference["nmse_ls"]), rtol=1e-13)
        evidence[model] = {
            "snr_db": 10,
            "frames": metadata["num_frames_per_snr"],
            "errors_perfect": result.errors_perfect,
            "errors_ls": result.errors_ls,
            "nmse_ls": result.nmse_ls,
            "reference_counts_match": True,
        }
        print(f"{model}: seeded 10 dB reference counts and NMSE match", flush=True)
    destination = root / "results" / "baseline_audit" / "reference_check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
