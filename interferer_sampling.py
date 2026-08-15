"""Exact Null comparisons and target-to-interferer great-circle sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from beamforming import array_factor
from element_pattern_data import evaluate_element_pattern
from pattern_sampling import pattern_cut_local_sample_count

if TYPE_CHECKING:
    from simulation import SimulationState

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
INTERFERER_RESPONSE_SCHEMA_VERSION = 1
INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION = 1
INTERFERER_GREAT_CIRCLE_BASE_SAMPLE_COUNT = 360


def _element_pattern(state: SimulationState, azimuth_rad, elevation_rad):
    return evaluate_element_pattern(
        state.config.element_option,
        azimuth_rad,
        elevation_rad,
        pattern_grid=state.config.element_pattern_grid,
        polarization_angle_deg=state.config.polarization_angle_deg,
    )


@dataclass(frozen=True)
class InterfererResponseComparison:
    """Exact array response before and after Null steering at one direction."""

    interferer_index: int
    azimuth_rad: float
    elevation_rad: float
    angular_distance_rad: float
    before_relative_db: float
    after_relative_db: float
    additional_suppression_db: float


@dataclass(frozen=True)
class InterfererGreatCircleCut:
    """Target-to-interferer great-circle comparison with the exact Null included."""

    schema_version: int
    comparison: InterfererResponseComparison
    base_sample_count: int
    local_sample_count: int
    refinement_half_width_deg: float
    offsets_rad: FloatArray
    azimuth_rad: FloatArray
    elevation_rad: FloatArray
    before_pattern_db: FloatArray
    after_pattern_db: FloatArray


def _direction_vector(azimuth_rad: float, elevation_rad: float) -> FloatArray:
    cosine_elevation = np.cos(elevation_rad)
    return np.asarray(
        (
            cosine_elevation * np.cos(azimuth_rad),
            cosine_elevation * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ),
        dtype=float,
    )


def _target_interferer_plane(
    target_azimuth_rad: float,
    target_elevation_rad: float,
    interferer_azimuth_rad: float,
    interferer_elevation_rad: float,
) -> tuple[FloatArray, FloatArray, float]:
    target = _direction_vector(target_azimuth_rad, target_elevation_rad)
    interferer = _direction_vector(interferer_azimuth_rad, interferer_elevation_rad)
    cosine_distance = float(np.clip(np.dot(target, interferer), -1.0, 1.0))
    distance = float(np.arccos(cosine_distance))
    tangent_component = interferer - cosine_distance * target
    tangent_norm = float(np.linalg.norm(tangent_component))
    if tangent_norm > 1e-12:
        tangent = tangent_component / tangent_norm
    else:
        tangent = np.asarray(
            (-np.sin(target_azimuth_rad), np.cos(target_azimuth_rad), 0.0),
            dtype=float,
        )
    return target, np.asarray(tangent, dtype=float), distance


def _directions_on_plane(
    target: FloatArray,
    tangent: FloatArray,
    offsets_rad: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    directions = (
        np.cos(offsets_rad)[..., None] * target
        + np.sin(offsets_rad)[..., None] * tangent
    )
    return (
        np.asarray(np.arctan2(directions[..., 1], directions[..., 0]), dtype=float),
        np.asarray(
            np.arcsin(np.clip(directions[..., 2], -1.0, 1.0)),
            dtype=float,
        ),
    )


def _relative_array_pattern_db(
    pattern: ComplexArray,
    target_response: complex,
) -> FloatArray:
    target_magnitude = abs(target_response)
    if target_magnitude <= np.finfo(float).tiny:
        return np.full(np.asarray(pattern).shape, -300.0, dtype=float)
    relative = np.abs(pattern) / target_magnitude
    return np.asarray(20.0 * np.log10(np.maximum(relative, 1e-15)), dtype=float)


def _exact_array_response(
    state: SimulationState,
    azimuth_rad: float,
    elevation_rad: float,
    *,
    max_chunk_entries: int,
    cancel_check: Callable[[], None] | None,
) -> complex:
    response = array_factor(
            state.coordinates.y,
            state.coordinates.z,
            state.complex_weights,
            state.wavelength_m,
            azimuth_rad,
            elevation_rad,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
    response *= _element_pattern(state, azimuth_rad, elevation_rad)
    return complex(response.item())


def _local_angular_half_width(aperture_m: float, wavelength_m: float) -> float:
    if aperture_m <= np.finfo(float).eps:
        return float(np.radians(12.0))
    return float(
        np.clip(
            4.0 * wavelength_m / aperture_m,
            np.radians(0.5),
            np.radians(12.0),
        )
    )


def _refined_multi_angular_axis(
    base_count: int,
    targets_rad: tuple[float, ...],
    local_half_width_rad: float,
    local_sample_count: int,
) -> FloatArray:
    axes = [np.linspace(-np.pi, np.pi, base_count)]
    for target_rad in targets_rad:
        target = float(np.clip(target_rad, -np.pi, np.pi))
        axes.append(
            np.linspace(
                max(-np.pi, target - local_half_width_rad),
                min(np.pi, target + local_half_width_rad),
                local_sample_count,
            )
        )
        axes.append(np.asarray([target], dtype=float))
    return np.asarray(np.unique(np.concatenate(axes)), dtype=float)


def calculate_interferer_response_comparisons(
    state: SimulationState,
    baseline_state: SimulationState,
    *,
    max_chunk_entries: int = 1_000_000,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[InterfererResponseComparison, ...]:
    """Compare exact relative array responses with Null steering off and on."""

    if not state.config.enable_null_steering:
        return ()
    if baseline_state.config.enable_null_steering:
        raise ValueError("Baseline state must have Null steering disabled.")

    target_azimuth = np.radians(state.current_azimuth_deg)
    target_elevation = np.radians(state.current_elevation_deg)
    before_target = _exact_array_response(
        baseline_state,
        target_azimuth,
        target_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    after_target = _exact_array_response(
        state,
        target_azimuth,
        target_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    comparisons = []
    for interferer_index, (azimuth, elevation) in enumerate(
        state.weight_result.null_directions_rad,
        start=1,
    ):
        _, _, angular_distance = _target_interferer_plane(
            target_azimuth,
            target_elevation,
            azimuth,
            elevation,
        )
        before_response = _exact_array_response(
            baseline_state,
            azimuth,
            elevation,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        after_response = _exact_array_response(
            state,
            azimuth,
            elevation,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        before_db = float(
            _relative_array_pattern_db(
                np.asarray([before_response], dtype=complex), before_target
            )[0]
        )
        after_db = float(
            _relative_array_pattern_db(
                np.asarray([after_response], dtype=complex), after_target
            )[0]
        )
        comparisons.append(
            InterfererResponseComparison(
                interferer_index=interferer_index,
                azimuth_rad=float(azimuth),
                elevation_rad=float(elevation),
                angular_distance_rad=angular_distance,
                before_relative_db=before_db,
                after_relative_db=after_db,
                additional_suppression_db=before_db - after_db,
            )
        )
    return tuple(comparisons)


def calculate_interferer_great_circle_cuts(
    state: SimulationState,
    baseline_state: SimulationState,
    *,
    comparisons: tuple[InterfererResponseComparison, ...] | None = None,
    sample_count: int = INTERFERER_GREAT_CIRCLE_BASE_SAMPLE_COUNT,
    local_sample_count: int | None = None,
    max_chunk_entries: int = 1_000_000,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[InterfererGreatCircleCut, ...]:
    """Calculate one target-to-interferer great-circle cut per Null direction."""

    if sample_count < 5:
        raise ValueError("Interferer great-circle cuts require at least five samples.")
    local_count = (
        pattern_cut_local_sample_count(state.coordinates.element_count)
        if local_sample_count is None
        else local_sample_count
    )
    if local_count < 3:
        raise ValueError("Local interferer refinement requires three samples.")
    response_comparisons = (
        calculate_interferer_response_comparisons(
            state,
            baseline_state,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        if comparisons is None
        else comparisons
    )
    if len(response_comparisons) != len(state.weight_result.null_directions_rad):
        raise ValueError("Each Null direction requires one response comparison.")

    target_azimuth = np.radians(state.current_azimuth_deg)
    target_elevation = np.radians(state.current_elevation_deg)
    before_target = _exact_array_response(
        baseline_state,
        target_azimuth,
        target_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    after_target = _exact_array_response(
        state,
        target_azimuth,
        target_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    physical_mask = state.coordinates.element_mask
    physical_y = state.coordinates.y[physical_mask]
    physical_z = state.coordinates.z[physical_mask]
    cuts = []
    for comparison in response_comparisons:
        target, tangent, angular_distance = _target_interferer_plane(
            target_azimuth,
            target_elevation,
            comparison.azimuth_rad,
            comparison.elevation_rad,
        )
        projected_positions = physical_y * tangent[1] + physical_z * tangent[2]
        half_width = _local_angular_half_width(
            float(np.ptp(projected_positions)), state.wavelength_m
        )
        offsets = _refined_multi_angular_axis(
            sample_count,
            (0.0, angular_distance),
            half_width,
            local_count,
        )
        azimuth, elevation = _directions_on_plane(target, tangent, offsets)
        before_pattern = array_factor(
            baseline_state.coordinates.y,
            baseline_state.coordinates.z,
            baseline_state.complex_weights,
            baseline_state.wavelength_m,
            azimuth,
            elevation,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        after_pattern = array_factor(
            state.coordinates.y,
            state.coordinates.z,
            state.complex_weights,
            state.wavelength_m,
            azimuth,
            elevation,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        before_pattern *= _element_pattern(baseline_state, azimuth, elevation)
        after_pattern *= _element_pattern(state, azimuth, elevation)
        cuts.append(
            InterfererGreatCircleCut(
                schema_version=INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION,
                comparison=comparison,
                base_sample_count=sample_count,
                local_sample_count=local_count,
                refinement_half_width_deg=float(np.degrees(half_width)),
                offsets_rad=np.asarray(offsets, dtype=float),
                azimuth_rad=np.asarray(azimuth, dtype=float),
                elevation_rad=np.asarray(elevation, dtype=float),
                before_pattern_db=_relative_array_pattern_db(
                    np.asarray(before_pattern, dtype=complex), before_target
                ),
                after_pattern_db=_relative_array_pattern_db(
                    np.asarray(after_pattern, dtype=complex), after_target
                ),
            )
        )
    return tuple(cuts)


__all__ = [
    "INTERFERER_GREAT_CIRCLE_BASE_SAMPLE_COUNT",
    "INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION",
    "INTERFERER_RESPONSE_SCHEMA_VERSION",
    "InterfererGreatCircleCut",
    "InterfererResponseComparison",
    "calculate_interferer_great_circle_cuts",
    "calculate_interferer_response_comparisons",
]
