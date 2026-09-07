import json

import numpy as np

from neural_ofdm.config import OFDMConfig
from neural_ofdm.experiments import run_ber_sweep, save_sweep


def test_paired_ls_performance_and_nmse_trend():
    result = run_ber_sweep(OFDMConfig(), [0, 10, 20, 30], num_frames=1500, seed=2026)
    perfect = np.array([p.ber_perfect for p in result.points])
    ls = np.array([p.ber_ls for p in result.points])
    nmse = np.array([p.nmse_ls for p in result.points])
    assert np.all(np.diff(perfect) <= 0)
    assert np.all(np.diff(ls) <= 0)
    assert np.all(np.diff(nmse) < 0)
    assert perfect[0] < ls[0]
    assert perfect[1] < ls[1]
    assert perfect[2] <= ls[2]
    assert nmse[-1] < 0.005
    assert result.points[0].total_bits == 144_000


def test_sweep_seed_counts_and_metadata(tmp_path):
    config = OFDMConfig()
    first = run_ber_sweep(config, [0, 15], num_frames=12)
    assert first == run_ber_sweep(config, [0, 15], num_frames=12)
    for point in first.points:
        assert point.ber_ls == point.errors_ls / point.total_bits
        assert point.ber_perfect == point.errors_perfect / point.total_bits
    save_sweep(first, tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["seed"] == 2026
    assert metadata["num_frames_per_snr"] == 12
    assert metadata["snr_values_db"] == [0, 15]
    assert metadata["paired_receivers"]
    assert (tmp_path / "metrics.csv").read_text().startswith("snr_db,total_bits,errors_perfect")
