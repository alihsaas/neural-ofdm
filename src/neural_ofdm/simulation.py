from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_ofdm._validation import complex_vector
from neural_ofdm.channel import (
    add_awgn,
    add_fixed_awgn,
    apply_channel,
    channel_frequency_response,
    fixed_channel,
    frequency_statistics,
    rayleigh_channel,
    reference_noise_variance,
)
from neural_ofdm.config import OFDMConfig
from neural_ofdm.equalization import mmse_equalize, zero_forcing_equalize
from neural_ofdm.estimation import LMMSEEstimator, ls_channel_estimate, prepare_lmmse
from neural_ofdm.metrics import bit_error_rate, normalized_mean_squared_error
from neural_ofdm.modulation import demodulate, modulate, random_bits
from neural_ofdm.ofdm import (
    add_cyclic_prefix,
    build_resource_grid,
    ofdm_demodulate,
    ofdm_modulate,
    remove_cyclic_prefix,
)


@dataclass(frozen=True)
class FrameResult:
    reference_bits: NDArray[np.uint8]
    estimated_bits: NDArray[np.uint8]
    frequency_grid: NDArray[np.complex128]
    time_signal: NDArray[np.complex128]
    transmitted_signal: NDArray[np.complex128]
    channel_impulse_response: NDArray[np.complex128]
    channel_frequency_response: NDArray[np.complex128]
    received_signal: NDArray[np.complex128]
    received_grid: NDArray[np.complex128]
    estimated_channel: NDArray[np.complex128]
    equalized_grid: NDArray[np.complex128]
    ber: float
    nmse: float
    noise_power: float
    measured_noise_power: float
    stabilized_bins: int
    csi: Literal["perfect", "ls", "lmmse", "external"]
    decision_grid: NDArray[np.complex128]
    evm_decision: float
    evm_raw: float


@dataclass(frozen=True)
class Observation:
    reference_bits: NDArray[np.uint8]
    frequency_grid: NDArray[np.complex128]
    time_signal: NDArray[np.complex128]
    transmitted_signal: NDArray[np.complex128]
    channel_impulse_response: NDArray[np.complex128]
    true_channel: NDArray[np.complex128]
    received_signal: NDArray[np.complex128]
    received_grid: NDArray[np.complex128]
    noise_variance: float
    measured_noise_power: float
    clean_received_power: float


def simulate_frame(
    config: OFDMConfig,
    rng: np.random.Generator,
    *,
    csi: Literal["perfect", "ls", "lmmse"] = "ls",
    channel_taps: ArrayLike | None = None,
    estimator: LMMSEEstimator | None = None,
    equalizer: Literal["zf", "mmse"] = "zf",
) -> FrameResult:
    observation = observe_frame(config, rng, channel_taps=channel_taps)
    if csi == "lmmse" and estimator is None:
        if config.noise_mode != "fixed":
            raise ValueError("automatic matched LMMSE requires independent fixed noise")
        if channel_taps is not None:
            raise ValueError("custom taps require an explicitly assumed LMMSE estimator")
        mean, covariance = frequency_statistics(config.channel_prior, config.fft_size)
        estimator = prepare_lmmse(
            mean, covariance, config.pilot_indices, config.pilot_symbols, observation.noise_variance
        )
    return receive_observation(
        config, observation, csi=csi, estimator=estimator, equalizer=equalizer
    )


def observe_frame(
    config: OFDMConfig,
    rng: np.random.Generator,
    *,
    channel_taps: ArrayLike | None = None,
    channel_rng: np.random.Generator | None = None,
    noise_rng: np.random.Generator | None = None,
) -> Observation:
    channel_rng = rng if channel_rng is None else channel_rng
    noise_rng = rng if noise_rng is None else noise_rng
    if channel_taps is not None:
        impulse_response = complex_vector(channel_taps, "channel_taps").copy()
    elif config.channel is not None:
        impulse_response = config.channel.sample(channel_rng)
    elif config.channel_model == "fixed":
        impulse_response = fixed_channel(config.channel_taps)
    else:
        impulse_response = rayleigh_channel(config.rayleigh_powers, channel_rng)
    config.validate_channel_length(impulse_response.size)
    reference_bits = random_bits(config.bits_per_frame, rng)
    frequency_grid = build_resource_grid(modulate(reference_bits, config.modulation_order), config)
    time_signal = ofdm_modulate(frequency_grid)
    transmitted_signal = add_cyclic_prefix(time_signal, config.cp_length)
    received_signal = apply_channel(transmitted_signal, impulse_response)[: transmitted_signal.size]
    noisy = (
        add_awgn(received_signal, config.snr_db, noise_rng)
        if config.noise_mode == "received_snr"
        else add_fixed_awgn(
            received_signal, reference_noise_variance(config.reference_esn0_db), noise_rng
        )
    )
    received_signal = noisy.signal
    received_grid = ofdm_demodulate(
        remove_cyclic_prefix(received_signal, config.cp_length, config.fft_size)
    )
    true_channel = channel_frequency_response(impulse_response, config.fft_size)
    return Observation(
        reference_bits,
        frequency_grid,
        time_signal,
        transmitted_signal,
        impulse_response,
        true_channel,
        received_signal,
        received_grid,
        noisy.noise_variance,
        noisy.measured_noise_power,
        noisy.signal_power,
    )


def receive_observation(
    config: OFDMConfig,
    observation: Observation,
    *,
    csi: Literal["perfect", "ls", "lmmse"] = "ls",
    estimator: LMMSEEstimator | None = None,
    equalizer: Literal["zf", "mmse"] = "zf",
) -> FrameResult:
    if csi == "perfect":
        estimate = observation.true_channel.copy()
    elif csi == "ls":
        estimate = ls_channel_estimate(
            observation.received_grid, config.pilot_indices, config.pilot_symbols
        )
    elif csi == "lmmse" and estimator is not None:
        estimate = estimator.estimate(observation.received_grid)
    else:
        raise ValueError("csi must be perfect, ls, or lmmse with a prepared estimator")
    return receive_with_estimate(config, observation, estimate, csi=csi, equalizer=equalizer)


def receive_with_estimate(
    config: OFDMConfig,
    observation: Observation,
    channel_estimate: ArrayLike,
    *,
    csi: Literal["perfect", "ls", "lmmse", "external"] = "external",
    equalizer: Literal["zf", "mmse"] = "zf",
) -> FrameResult:
    estimate = complex_vector(channel_estimate, "channel_estimate")
    if estimate.size != config.fft_size:
        raise ValueError("channel estimate must match configured FFT size")
    if equalizer == "zf":
        equalized = zero_forcing_equalize(observation.received_grid, estimate)
        decisions = equalized.symbols
        raw = decisions
        stabilized = equalized.stabilized_bins
    elif equalizer == "mmse":
        mmse = mmse_equalize(observation.received_grid, estimate, observation.noise_variance)
        decisions, raw, stabilized = mmse.decision_symbols, mmse.symbols, mmse.stabilized_bins
    else:
        raise ValueError("equalizer must be zf or mmse")
    data = config.data_indices
    estimated_bits = demodulate(decisions[data], config.modulation_order)
    reference = observation.frequency_grid[data]
    energy = np.sum(np.abs(reference) ** 2)
    return FrameResult(
        reference_bits=observation.reference_bits,
        estimated_bits=estimated_bits,
        frequency_grid=observation.frequency_grid,
        time_signal=observation.time_signal,
        transmitted_signal=observation.transmitted_signal,
        channel_impulse_response=observation.channel_impulse_response,
        channel_frequency_response=observation.true_channel,
        received_signal=observation.received_signal,
        received_grid=observation.received_grid,
        estimated_channel=estimate,
        equalized_grid=raw,
        ber=bit_error_rate(observation.reference_bits, estimated_bits),
        nmse=normalized_mean_squared_error(observation.true_channel, estimate),
        noise_power=observation.noise_variance,
        measured_noise_power=observation.measured_noise_power,
        stabilized_bins=stabilized,
        csi=csi,
        decision_grid=decisions,
        evm_decision=float(np.sqrt(np.sum(np.abs(decisions[data] - reference) ** 2) / energy)),
        evm_raw=float(np.sqrt(np.sum(np.abs(raw[data] - reference) ** 2) / energy)),
    )
