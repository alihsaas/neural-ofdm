from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve

from neural_ofdm._validation import complex_vector


def ls_channel_estimate(
    received_grid: ArrayLike, pilot_indices: ArrayLike, pilot_values: ArrayLike
) -> NDArray[np.complex128]:

    received = complex_vector(received_grid, "received_grid")
    indices = np.asarray(pilot_indices)
    if (
        indices.ndim != 1
        or indices.size < 2
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= received.size)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("pilot_indices must be >=2 sorted, unique integer bins within the grid")
    pilots = np.asarray(pilot_values, dtype=np.complex128)
    if pilots.ndim == 0:
        pilots = np.full(indices.size, pilots.item(), dtype=np.complex128)
    if pilots.shape != indices.shape or not np.all(np.isfinite(pilots)) or np.any(pilots == 0):
        raise ValueError("pilot_values must be a finite nonzero scalar or one value per pilot")
    pilot_estimates = received[indices] / pilots
    extended_indices = np.concatenate(
        ([indices[-1] - received.size], indices, [indices[0] + received.size])
    )
    extended_estimates = np.concatenate(
        (pilot_estimates[-1:], pilot_estimates, pilot_estimates[:1])
    )
    bins = np.arange(received.size)
    return np.interp(bins, extended_indices, extended_estimates.real) + 1j * np.interp(
        bins, extended_indices, extended_estimates.imag
    )


@dataclass(frozen=True)
class LMMSEEstimator:
    mean: NDArray[np.complex128]
    pilot_indices: NDArray[np.int64]
    pilot_values: NDArray[np.complex128]
    operator: NDArray[np.complex128]
    noise_variance: float
    method: str
    observation_rank: int
    condition_number: float
    relative_tolerance: float
    eigenvalue_cutoff: float

    def estimate(self, received_grid: ArrayLike) -> NDArray[np.complex128]:
        received = complex_vector(received_grid, "received_grid")
        if received.shape != self.mean.shape:
            raise ValueError("received grid must match the prior dimensions")
        innovation = (
            received[self.pilot_indices] / self.pilot_values - self.mean[self.pilot_indices]
        )
        return self.mean + self.operator @ innovation


def prepare_lmmse(
    mean: ArrayLike,
    covariance: ArrayLike,
    pilot_indices: ArrayLike,
    pilot_values: ArrayLike,
    noise_variance: float,
    relative_tolerance: float = 1e-12,
) -> LMMSEEstimator:
    prior_mean = complex_vector(mean, "mean").copy()
    prior_covariance = np.asarray(covariance, dtype=np.complex128)
    n = prior_mean.size
    if prior_covariance.shape != (n, n) or not np.all(np.isfinite(prior_covariance)):
        raise ValueError("covariance must be a finite N by N matrix")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be in (0,1)")
    scale = float(np.linalg.norm(prior_covariance, ord=2))
    tolerance = relative_tolerance * max(scale, np.finfo(float).tiny)
    if not np.allclose(prior_covariance, prior_covariance.conj().T, rtol=0, atol=tolerance):
        raise ValueError("covariance must be Hermitian")
    if np.linalg.eigvalsh(prior_covariance).min() < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    indices = np.asarray(pilot_indices)
    if (
        indices.ndim != 1
        or indices.size == 0
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= n)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("pilot_indices must be sorted unique integer bins within the grid")
    indices = indices.astype(np.int64, copy=True)
    pilots = np.asarray(pilot_values, dtype=np.complex128)
    if pilots.ndim == 0:
        pilots = np.full(indices.size, pilots.item(), dtype=np.complex128)
    if pilots.shape != indices.shape or not np.all(np.isfinite(pilots)) or np.any(pilots == 0):
        raise ValueError("pilot_values must be finite nonzero and match pilot_indices")
    pilots = pilots.copy()
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    cross_covariance = prior_covariance[:, indices]
    observation_covariance = prior_covariance[np.ix_(indices, indices)] + np.diag(
        noise_variance / np.abs(pilots) ** 2
    )
    eigenvalues = np.linalg.eigvalsh(observation_covariance)
    cutoff = relative_tolerance * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    if noise_variance > 0:
        factor = cho_factor(observation_covariance, lower=True, check_finite=False)
        operator = cho_solve(factor, cross_covariance.conj().T).conj().T
        method = "Hermitian Cholesky solve"
        condition = float(eigenvalues[-1] / eigenvalues[0])
    else:
        values, vectors = np.linalg.eigh(observation_covariance)
        retained = values > cutoff
        basis = vectors[:, retained]
        operator = ((cross_covariance @ basis) / values[retained]) @ basis.conj().T
        method = "rank-aware noiseless eigensolve"
        condition = float(values[-1] / values[retained][0]) if rank else float("inf")
    for array in (prior_mean, indices, pilots, operator):
        array.setflags(write=False)
    return LMMSEEstimator(
        prior_mean,
        indices,
        pilots,
        operator,
        float(noise_variance),
        method,
        rank,
        condition,
        relative_tolerance,
        cutoff,
    )
