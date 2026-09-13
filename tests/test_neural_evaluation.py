from dataclasses import replace

import numpy as np

from neural_ofdm.neural_data import DatasetConfig, features_to_complex, generate_dataset
from neural_ofdm.neural_evaluation import (
    evaluate_paired,
    linear_estimators,
    read_rows,
    save_condition,
    summarize,
)


def test_paired_branches_share_frames_and_counting(tmp_path):
    config = replace(
        DatasetConfig(), split="test", count=30, fixed_reference_db=5.0, store_frames=True
    )
    dataset = generate_dataset(config, tmp_path / "test")
    estimators = linear_estimators(dataset, config.ofdm.channel, (0.0, 30.0))
    assert list(estimators) == ["lmmse_matched", "lmmse_snr_unaware"]
    rician = replace(config.ofdm.channel, model="rician", k_linear=10.0)
    shifted = generate_dataset(
        replace(config, ofdm=replace(config.ofdm, channel=rician), condition_id=2),
        tmp_path / "shifted",
    )
    assert "lmmse_training_prior" in linear_estimators(shifted, config.ofdm.channel, (0.0, 30.0))
    copies = {
        "copy_of_ls/seed_1": features_to_complex(dataset.inputs),
        "oracle_copy/seed_2": features_to_complex(dataset.targets),
    }
    evaluation = evaluate_paired(dataset, estimators, copies)
    columns = {name: index for index, name in enumerate(evaluation.receivers)}
    np.testing.assert_array_equal(
        evaluation.errors[:, columns["ls"]], evaluation.errors[:, columns["copy_of_ls/seed_1"]]
    )
    np.testing.assert_array_equal(
        evaluation.errors[:, columns["perfect"]],
        evaluation.errors[:, columns["oracle_copy/seed_2"]],
    )
    assert np.all(evaluation.nmse[:, columns["perfect"]] == 0)
    assert np.all(evaluation.nmse[:, columns["oracle_copy/seed_2"]] < 1e-12)
    assert (
        evaluation.errors.sum(axis=0)[columns["perfect"]]
        <= evaluation.errors.sum(axis=0)[columns["ls"]]
    )
    rows, comparisons = summarize(evaluation, 200, 5, "primary")
    by_name = {row["receiver"]: row for row in rows}
    assert by_name["copy_of_ls/seed_1"]["run"] == "copy_of_ls"
    assert by_name["copy_of_ls/seed_1"]["seed"] == 1
    assert by_name["ls"]["total_bits"] == 30 * config.ofdm.bits_per_frame
    expected = evaluation.squared_error[:, columns["ls"]].sum() / (30 * 64)
    assert by_name["ls"]["ensemble_mse_per_bin"] == expected
    same = [
        item
        for item in comparisons
        if item["receiver"] == "copy_of_ls/seed_1" and item["baseline"] == "ls"
    ][0]
    assert same["ber_difference"] == 0 and not same["ber_informative"]
    save_condition(tmp_path / "out", [evaluation], rows, comparisons)
    restored = read_rows(tmp_path / "out" / "metrics.csv")
    assert restored[1]["errors"] == rows[1]["errors"]


def test_zero_error_rows_report_frame_bound_without_bootstrap(tmp_path):
    config = replace(
        DatasetConfig(), split="test", count=20, fixed_reference_db=60.0, store_frames=True
    )
    dataset = generate_dataset(config, tmp_path / "quiet")
    evaluation = evaluate_paired(dataset, {}, {})
    rows, _ = summarize(evaluation, 100, 1, "quiet")
    perfect = rows[0]
    assert perfect["errors"] == 0
    assert perfect["ber_frame_bootstrap_low"] == ""
    assert perfect["zero_error_frame_upper95"] == float(-np.expm1(np.log(0.05) / 20))
