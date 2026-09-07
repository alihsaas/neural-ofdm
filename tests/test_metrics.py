import numpy as np
import pytest

from neural_ofdm.metrics import bit_error_rate, nmse_to_db, normalized_mean_squared_error


def test_ber_and_nmse_values():
    assert bit_error_rate([0, 1, 0, 1], [1, 1, 0, 0]) == 0.5
    assert normalized_mean_squared_error([1, 1j], [0, 0]) == 1
    assert normalized_mean_squared_error([1, 1j], [1, 1j]) == 0
    assert normalized_mean_squared_error([1, 1j], [0.5, 0.5j]) == 0.25
    assert nmse_to_db(0.01) == -20
    assert nmse_to_db(0) == -np.inf


def test_invalid_metric_inputs():
    with pytest.raises(ValueError):
        bit_error_rate([0, 1], [1])
    with pytest.raises(ValueError):
        bit_error_rate([], [])
    with pytest.raises(ValueError):
        normalized_mean_squared_error([0, 0], [1, 1])
    with pytest.raises(ValueError):
        normalized_mean_squared_error([1, 1], [1])
