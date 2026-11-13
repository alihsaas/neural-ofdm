import numpy as np
import pytest

from neural_ofdm.modulation import qpsk_demodulate, qpsk_modulate, random_bits


def test_expected_gray_constellation_and_energy():
    bits = [0, 0, 0, 1, 1, 0, 1, 1]
    expected = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    symbols = qpsk_modulate(bits)
    np.testing.assert_allclose(symbols, expected)
    np.testing.assert_allclose(np.abs(symbols) ** 2, 1, atol=1e-15)
    np.testing.assert_array_equal(qpsk_demodulate(symbols), bits)


def test_seeded_round_trip():
    bits = random_bits(20_000, np.random.default_rng(7))
    np.testing.assert_array_equal(bits, random_bits(20_000, np.random.default_rng(7)))
    np.testing.assert_array_equal(qpsk_demodulate(qpsk_modulate(bits)), bits)


@pytest.mark.parametrize("bits", [[0, 1, 0], [0, 2], [0, 0.5], [], [[0, 1]]])
def test_invalid_bits(bits):
    with pytest.raises(ValueError):
        qpsk_modulate(bits)
