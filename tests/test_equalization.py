import numpy as np
import pytest

from neural_ofdm.equalization import zero_forcing_equalize


def test_zf_inverts_nonzero_complex_channel():
    symbols = np.array([1 + 1j, -1 + 1j])
    channel = np.array([0.5j, 2 - 1j])
    result = zero_forcing_equalize(symbols * channel, channel)
    np.testing.assert_allclose(result.symbols, symbols)
    assert result.stabilized_bins == 0


def test_zf_null_is_reported_and_finite():
    with pytest.warns(RuntimeWarning, match="2 bin"):
        result = zero_forcing_equalize([1, 1, 1], [0, 1e-12j, 1])
    assert result.stabilized_bins == 2
    assert np.all(np.isfinite(result.symbols))
    np.testing.assert_allclose(result.symbols, [1e10, -1e10j, 1], atol=1e-5)
