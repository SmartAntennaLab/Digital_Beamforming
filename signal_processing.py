"""Wideband, near-field, channel, adaptive beamforming, and MUSIC models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from array_math import steering_vector
from element_pattern_data import evaluate_element_pattern
from pattern_metrics import array_factor
from physical_effects import near_field_response

if TYPE_CHECKING:
    from simulation import SimulationState

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
MAX_ADAPTIVE_ELEMENTS = 256


@dataclass(frozen=True)
class WidebandPoint:
    frequency_ghz: float
    peak_azimuth_deg: float
    peak_elevation_deg: float
    squint_angle_deg: float
    target_loss_db: float


@dataclass(frozen=True)
class WidebandAnalysis:
    bandwidth_percent: float
    points: tuple[WidebandPoint, ...]
    maximum_squint_deg: float
    edge_target_loss_db: float


@dataclass(frozen=True)
class NearFieldAnalysis:
    focus_range_m: float
    rayleigh_distance_m: float
    focus_inside_rayleigh_distance: bool
    focus_coherence_db: float
    focus_response_magnitude: float


@dataclass(frozen=True)
class ChannelAnalysis:
    snapshots: int
    analysis_elements: int
    multipath_count: int
    input_snr_db: float
    conventional_sinr_db: float
    desired_output_power: float
    interference_output_power: float
    noise_output_power: float


@dataclass(frozen=True)
class AdaptiveBeamformingAnalysis:
    method: str
    diagonal_loading: float
    output_sinr_db: float
    improvement_db: float
    constraint_error: float
    condition_number: float


@dataclass(frozen=True)
class DoaPeak:
    azimuth_deg: float
    spectrum_db: float


@dataclass(frozen=True)
class DoaAnalysis:
    method: str
    source_count: int
    elevation_deg: float
    peaks: tuple[DoaPeak, ...]


@dataclass(frozen=True)
class AdvancedAnalysis:
    wideband: WidebandAnalysis | None = None
    near_field: NearFieldAnalysis | None = None
    channel: ChannelAnalysis | None = None
    adaptive: AdaptiveBeamformingAnalysis | None = None
    doa: DoaAnalysis | None = None


def _safe_db_power(value: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(float).tiny)))


def _angular_distance_deg(
    azimuth_a_deg: float,
    elevation_a_deg: float,
    azimuth_b_deg: float,
    elevation_b_deg: float,
) -> float:
    azimuth_a, azimuth_b, elevation_a, elevation_b = np.radians(
        (azimuth_a_deg, azimuth_b_deg, elevation_a_deg, elevation_b_deg)
    )
    cosine = np.sin(elevation_a) * np.sin(elevation_b) + np.cos(elevation_a) * np.cos(
        elevation_b
    ) * np.cos(azimuth_a - azimuth_b)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def calculate_wideband_squint(
    state: SimulationState,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> WidebandAnalysis | None:
    """Evaluate fixed phase-shifter beam pointing across configured bandwidth."""

    bandwidth_percent = float(state.config.wideband_bandwidth_percent)
    if bandwidth_percent <= 0.0:
        return None
    sample_count = int(state.config.wideband_frequency_samples)
    if not 3 <= sample_count <= 33:
        raise ValueError("Wideband frequency samples must be between 3 and 33.")
    center_frequency_hz = state.config.frequency_ghz * 1.0e9
    fractional = bandwidth_percent / 100.0
    frequencies_hz = np.linspace(
        center_frequency_hz * (1.0 - fractional / 2.0),
        center_frequency_hz * (1.0 + fractional / 2.0),
        sample_count,
    )
    azimuth_axis = np.radians(np.linspace(-90.0, 90.0, 361))
    elevation_axis = np.radians(np.linspace(-90.0, 90.0, 361))
    target_azimuth = np.radians(state.current_azimuth_deg)
    target_elevation = np.radians(state.current_elevation_deg)
    points = []
    for frequency_hz in frequencies_hz:
        if cancel_check is not None:
            cancel_check()
        wavelength = 299_792_458.0 / frequency_hz
        azimuth_pattern = array_factor(
            state.coordinates.y,
            state.coordinates.z,
            state.complex_weights,
            wavelength,
            azimuth_axis,
            target_elevation,
            cancel_check=cancel_check,
        )
        azimuth_pattern *= evaluate_element_pattern(
            state.config.element_option,
            azimuth_axis,
            target_elevation,
            pattern_grid=state.config.element_pattern_grid,
            polarization_angle_deg=state.config.polarization_angle_deg,
        )
        elevation_pattern = array_factor(
            state.coordinates.y,
            state.coordinates.z,
            state.complex_weights,
            wavelength,
            target_azimuth,
            elevation_axis,
            cancel_check=cancel_check,
        )
        elevation_pattern *= evaluate_element_pattern(
            state.config.element_option,
            target_azimuth,
            elevation_axis,
            pattern_grid=state.config.element_pattern_grid,
            polarization_angle_deg=state.config.polarization_angle_deg,
        )
        peak_azimuth = float(
            np.degrees(azimuth_axis[np.argmax(np.abs(azimuth_pattern))])
        )
        peak_elevation = float(
            np.degrees(elevation_axis[np.argmax(np.abs(elevation_pattern))])
        )
        target_response = array_factor(
            state.coordinates.y,
            state.coordinates.z,
            state.complex_weights,
            wavelength,
            target_azimuth,
            target_elevation,
            cancel_check=cancel_check,
        ).item()
        target_response *= evaluate_element_pattern(
            state.config.element_option,
            target_azimuth,
            target_elevation,
            pattern_grid=state.config.element_pattern_grid,
            polarization_angle_deg=state.config.polarization_angle_deg,
        ).item()
        peak_magnitude = max(
            float(np.max(np.abs(azimuth_pattern))),
            float(np.max(np.abs(elevation_pattern))),
            np.finfo(float).tiny,
        )
        target_loss = 20.0 * np.log10(
            max(abs(target_response), np.finfo(float).tiny) / peak_magnitude
        )
        points.append(
            WidebandPoint(
                frequency_ghz=float(frequency_hz / 1.0e9),
                peak_azimuth_deg=peak_azimuth,
                peak_elevation_deg=peak_elevation,
                squint_angle_deg=_angular_distance_deg(
                    state.current_azimuth_deg,
                    state.current_elevation_deg,
                    peak_azimuth,
                    peak_elevation,
                ),
                target_loss_db=float(target_loss),
            )
        )
    return WidebandAnalysis(
        bandwidth_percent=bandwidth_percent,
        points=tuple(points),
        maximum_squint_deg=max(point.squint_angle_deg for point in points),
        edge_target_loss_db=min(points[0].target_loss_db, points[-1].target_loss_db),
    )


def calculate_near_field_analysis(state: SimulationState) -> NearFieldAnalysis | None:
    range_m = state.config.near_field_focus_range_m
    if range_m is None:
        return None
    physical = state.coordinates.element_mask
    response = near_field_response(
        state.coordinates.y[physical],
        state.coordinates.z[physical],
        state.complex_weights[physical],
        state.wavelength_m,
        np.radians(state.current_azimuth_deg),
        np.radians(state.current_elevation_deg),
        range_m,
    )
    aperture = float(
        np.hypot(
            np.ptp(state.coordinates.y[physical]), np.ptp(state.coordinates.z[physical])
        )
    )
    rayleigh = 2.0 * aperture**2 / state.wavelength_m
    ideal_magnitude = float(np.sum(np.abs(state.complex_weights[physical])))
    coherence_db = 20.0 * np.log10(
        max(abs(response), np.finfo(float).tiny)
        / max(ideal_magnitude, np.finfo(float).tiny)
    )
    return NearFieldAnalysis(
        focus_range_m=float(range_m),
        rayleigh_distance_m=rayleigh,
        focus_inside_rayleigh_distance=bool(range_m <= rayleigh),
        focus_coherence_db=float(coherence_db),
        focus_response_magnitude=float(abs(response)),
    )


def _analysis_subarray(state: SimulationState):
    physical_indices = np.flatnonzero(state.active_mask.ravel())
    if physical_indices.size > MAX_ADAPTIVE_ELEMENTS:
        positions = np.linspace(
            0,
            physical_indices.size - 1,
            MAX_ADAPTIVE_ELEMENTS,
            dtype=int,
        )
        physical_indices = physical_indices[positions]
    y = state.coordinates.y.ravel()[physical_indices]
    z = state.coordinates.z.ravel()[physical_indices]
    weights = state.complex_weights.ravel()[physical_indices]
    return y, z, weights


def _source_response(
    state: SimulationState,
    y: FloatArray,
    z: FloatArray,
    azimuth_deg: float,
    elevation_deg: float,
) -> ComplexArray:
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    response = steering_vector(y, z, state.wavelength_m, azimuth, elevation).ravel()
    factor = evaluate_element_pattern(
        state.config.element_option,
        azimuth,
        elevation,
        pattern_grid=state.config.element_pattern_grid,
        polarization_angle_deg=state.config.polarization_angle_deg,
    ).item()
    return np.asarray(response * factor, dtype=complex)


def _output_components(
    weights_plot: ComplexArray,
    desired: ComplexArray,
    interferers: tuple[ComplexArray, ...],
    signal_power: float,
    interference_power: float,
    noise_power: float,
) -> tuple[float, float, float, float]:
    desired_power = signal_power * abs(np.sum(weights_plot * desired)) ** 2
    interferer_power = interference_power * sum(
        abs(np.sum(weights_plot * response)) ** 2 for response in interferers
    )
    output_noise = noise_power * float(np.sum(np.abs(weights_plot) ** 2))
    sinr = desired_power / max(interferer_power + output_noise, np.finfo(float).tiny)
    return (
        float(desired_power),
        float(interferer_power),
        float(output_noise),
        _safe_db_power(float(sinr)),
    )


def _local_maxima(values: FloatArray, count: int) -> tuple[int, ...]:
    candidates = [
        index
        for index in range(1, values.size - 1)
        if values[index] >= values[index - 1] and values[index] >= values[index + 1]
    ]
    candidates.sort(key=lambda index: values[index], reverse=True)
    selected: list[int] = []
    for candidate in candidates:
        if all(abs(candidate - existing) >= 3 for existing in selected):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return tuple(selected)


def calculate_channel_and_adaptive(
    state: SimulationState,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[
    ChannelAnalysis | None,
    AdaptiveBeamformingAnalysis | None,
    DoaAnalysis | None,
]:
    """Simulate deterministic snapshots and optional MVDR/LCMV/MUSIC analysis."""

    if not state.config.enable_channel_analysis:
        return None, None, None
    snapshots = int(state.config.channel_snapshots)
    if not 16 <= snapshots <= 4096:
        raise ValueError("Channel snapshots must be between 16 and 4096.")
    y, z, conventional_weights = _analysis_subarray(state)
    element_count = y.size
    if element_count < 2:
        raise ValueError("Channel analysis requires at least two active elements.")
    rng = np.random.default_rng(np.random.SeedSequence([state.config.random_seed, 303]))
    desired = _source_response(
        state,
        y,
        z,
        state.current_azimuth_deg,
        state.current_elevation_deg,
    )
    for path_index in range(int(state.config.multipath_count)):
        azimuth_offset = float(rng.uniform(-25.0, 25.0))
        elevation_offset = float(rng.uniform(-10.0, 10.0))
        attenuation = 10.0 ** (-(6.0 + 3.0 * path_index) / 20.0)
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi))
        desired += (
            attenuation
            * phase
            * _source_response(
                state,
                y,
                z,
                float(np.clip(state.current_azimuth_deg + azimuth_offset, -90.0, 90.0)),
                float(
                    np.clip(state.current_elevation_deg + elevation_offset, -90.0, 90.0)
                ),
            )
        )

    interferer_directions = tuple(
        (float(np.degrees(azimuth)), float(np.degrees(elevation)))
        for azimuth, elevation in state.weight_result.null_directions_rad
    )
    if not interferer_directions:
        interferer_directions = (
            (
                float(np.clip(state.current_azimuth_deg + 30.0, -90.0, 90.0)),
                state.current_elevation_deg,
            ),
        )
    interferers = tuple(
        _source_response(state, y, z, azimuth, elevation)
        for azimuth, elevation in interferer_directions
    )
    signal_power = 10.0 ** (state.config.signal_power_dbm / 10.0)
    interference_power = 10.0 ** (state.config.interference_power_dbm / 10.0)
    noise_power = 10.0 ** (state.config.noise_power_dbm / 10.0)
    desired_samples = (
        rng.normal(size=snapshots) + 1j * rng.normal(size=snapshots)
    ) / np.sqrt(2.0)
    snapshot_matrix = desired[:, None] * np.sqrt(signal_power) * desired_samples
    interference_covariance = noise_power * np.eye(element_count, dtype=complex)
    for response in interferers:
        samples = (
            rng.normal(size=snapshots) + 1j * rng.normal(size=snapshots)
        ) / np.sqrt(2.0)
        snapshot_matrix += response[:, None] * np.sqrt(interference_power) * samples
        interference_covariance += interference_power * np.outer(
            response, response.conj()
        )
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=(element_count, snapshots))
        + 1j * rng.normal(size=(element_count, snapshots))
    )
    snapshot_matrix += noise
    sample_covariance = snapshot_matrix @ snapshot_matrix.conj().T / snapshots
    desired_output, interference_output, noise_output, conventional_sinr = (
        _output_components(
            conventional_weights,
            desired,
            interferers,
            signal_power,
            interference_power,
            noise_power,
        )
    )
    channel = ChannelAnalysis(
        snapshots=snapshots,
        analysis_elements=element_count,
        multipath_count=int(state.config.multipath_count),
        input_snr_db=float(
            state.config.signal_power_dbm - state.config.noise_power_dbm
        ),
        conventional_sinr_db=conventional_sinr,
        desired_output_power=desired_output,
        interference_output_power=interference_output,
        noise_output_power=noise_output,
    )

    adaptive = None
    method = state.config.adaptive_beamforming_method
    if method in {"mvdr", "lcmv"}:
        loading = float(state.config.diagonal_loading)
        scale = max(float(np.trace(sample_covariance).real / element_count), 1e-12)
        loaded = interference_covariance + loading * scale * np.eye(element_count)
        condition = float(np.linalg.cond(loaded))
        if method == "mvdr":
            solved = np.linalg.solve(loaded, desired)
            denominator = np.vdot(desired, solved)
            receive_weights = solved / denominator
            constraint_error = float(abs(np.vdot(receive_weights, desired) - 1.0))
        else:
            constraints = np.column_stack((desired, *interferers))
            requested = np.zeros(constraints.shape[1], dtype=complex)
            requested[0] = 1.0
            solved = np.linalg.solve(loaded, constraints)
            gram = constraints.conj().T @ solved
            receive_weights = solved @ (np.linalg.pinv(gram) @ requested)
            constraint_error = float(
                np.linalg.norm(constraints.conj().T @ receive_weights - requested)
            )
        adaptive_plot_weights = np.conj(receive_weights)
        _, _, _, adaptive_sinr = _output_components(
            adaptive_plot_weights,
            desired,
            interferers,
            signal_power,
            interference_power,
            noise_power,
        )
        adaptive = AdaptiveBeamformingAnalysis(
            method=method,
            diagonal_loading=loading,
            output_sinr_db=adaptive_sinr,
            improvement_db=float(adaptive_sinr - conventional_sinr),
            constraint_error=constraint_error,
            condition_number=condition,
        )

    doa = None
    if state.config.enable_doa_estimation:
        if cancel_check is not None:
            cancel_check()
        source_count = min(1 + len(interferers), element_count - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)
        order = np.argsort(eigenvalues)
        noise_subspace = eigenvectors[:, order[: element_count - source_count]]
        azimuth_axis_deg = np.linspace(-90.0, 90.0, 361)
        steering = steering_vector(
            y,
            z,
            state.wavelength_m,
            np.radians(azimuth_axis_deg),
            np.radians(state.current_elevation_deg),
        ).reshape(azimuth_axis_deg.size, element_count)
        projection = steering.conj() @ noise_subspace
        denominator = np.sum(np.abs(projection) ** 2, axis=1)
        spectrum = 1.0 / np.maximum(denominator, np.finfo(float).tiny)
        spectrum_db = 10.0 * np.log10(spectrum / np.max(spectrum))
        peak_indices = _local_maxima(spectrum_db, source_count)
        doa = DoaAnalysis(
            method="music",
            source_count=source_count,
            elevation_deg=float(state.current_elevation_deg),
            peaks=tuple(
                DoaPeak(
                    azimuth_deg=float(azimuth_axis_deg[index]),
                    spectrum_db=float(spectrum_db[index]),
                )
                for index in peak_indices
            ),
        )
    return channel, adaptive, doa


def calculate_advanced_analysis(
    state: SimulationState,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> AdvancedAnalysis:
    channel, adaptive, doa = calculate_channel_and_adaptive(
        state,
        cancel_check=cancel_check,
    )
    return AdvancedAnalysis(
        wideband=calculate_wideband_squint(state, cancel_check=cancel_check),
        near_field=calculate_near_field_analysis(state),
        channel=channel,
        adaptive=adaptive,
        doa=doa,
    )


__all__ = [
    "AdaptiveBeamformingAnalysis",
    "AdvancedAnalysis",
    "ChannelAnalysis",
    "DoaAnalysis",
    "DoaPeak",
    "NearFieldAnalysis",
    "WidebandAnalysis",
    "WidebandPoint",
    "calculate_advanced_analysis",
]
