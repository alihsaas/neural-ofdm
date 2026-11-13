from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def integer(value: int, name: str, minimum: int = 0) -> None:

    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def complex_vector(values: ArrayLike, name: str) -> NDArray[np.complex128]:

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite, nonempty one-dimensional vector")
    return array


def bit_vector(bits: ArrayLike) -> NDArray[np.uint8]:

    array = np.asarray(bits)
    if array.ndim != 1 or array.size == 0 or not np.all((array == 0) | (array == 1)):
        raise ValueError("bits must be a nonempty one-dimensional vector of zeros and ones")
    return array.astype(np.uint8)
