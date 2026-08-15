"""Validation and hardware realization helpers for simulation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from array_geometry import ArrayCoordinates
from array_math import steering_vector
from element_pattern_data import evaluate_element_pattern
from null_solver import BeamformingWeights
from physical_effects import (
    HardwareEffectDiagnostics,
    apply_position_errors,
    apply_weight_errors_and_coupling,
    near_field_focusing_weights,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from simulation import SimulationConfig


def validate_advanced_config(config: SimulationConfig) -> None:
    if isinstance(config.random_seed, bool) or not 0 <= config.random_seed <= 2**32 - 1:
        raise ValueError("Random seed must be between 0 and 4294967295.")
    if config.near_field_focus_range_m is not None and config.enable_null_steering:
        raise ValueError(
            "Near-field focusing cannot be combined with far-field null steering."
        )
    if config.adaptive_beamforming_method not in {"none", "mvdr", "lcmv"}:
        raise ValueError("Unsupported adaptive beamforming method.")
    if config.enable_doa_estimation and not config.enable_channel_analysis:
        raise ValueError("DOA estimation requires channel analysis.")
    if (
        config.adaptive_beamforming_method != "none"
        and not config.enable_channel_analysis
    ):
        raise ValueError("Adaptive beamforming requires channel analysis.")


def realize_hardware(
    config: SimulationConfig,
    nominal_coordinates: ArrayCoordinates,
    weight_result: BeamformingWeights,
    base_amplitudes: NDArray,
    active_mask: NDArray,
    *,
    wavelength_m: float,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    cancel_check: Callable[[], None] | None,
) -> tuple[ArrayCoordinates, NDArray, HardwareEffectDiagnostics]:
    """Apply focusing, position error, calibration error, and coupling."""

    ideal_weights = weight_result.weights
    if config.near_field_focus_range_m is not None:
        ideal_weights = near_field_focusing_weights(
            nominal_coordinates.y,
            nominal_coordinates.z,
            wavelength_m,
            azimuth_rad,
            elevation_rad,
            config.near_field_focus_range_m,
            base_amplitudes,
        )
    coordinates, realized_position_error = apply_position_errors(
        nominal_coordinates,
        wavelength_m,
        config.position_error_rms_wavelength,
        random_seed=config.random_seed,
    )
    weights, diagnostics = apply_weight_errors_and_coupling(
        coordinates,
        ideal_weights,
        active_mask,
        amplitude_error_rms_db=config.amplitude_error_rms_db,
        phase_error_rms_deg=config.phase_error_rms_deg,
        mutual_coupling_db=config.mutual_coupling_db,
        mutual_coupling_phase_deg=config.mutual_coupling_phase_deg,
        reference_spacing_m=min(horizontal_spacing_m, vertical_spacing_m),
        random_seed=config.random_seed,
        position_error_rms_wavelength=config.position_error_rms_wavelength,
        realized_position_error_rms_wavelength=realized_position_error,
        cancel_check=cancel_check,
    )
    return coordinates, weights, diagnostics


def realized_null_depths(
    config: SimulationConfig,
    coordinates: ArrayCoordinates,
    weights: NDArray,
    wavelength_m: float,
    target_direction_rad: tuple[float, float],
    null_directions_rad: tuple[tuple[float, float], ...],
) -> tuple[float | None, ...]:
    """Measure post-impairment Null depths including the element pattern."""

    def response(direction: tuple[float, float]) -> complex:
        value = (
            steering_vector(
                coordinates.y,
                coordinates.z,
                wavelength_m,
                direction[0],
                direction[1],
            ).ravel()
            @ weights.ravel()
        )
        factor = evaluate_element_pattern(
            config.element_option,
            direction[0],
            direction[1],
            pattern_grid=config.element_pattern_grid,
            polarization_angle_deg=config.polarization_angle_deg,
        ).item()
        return complex(value * factor)

    target_magnitude = abs(response(target_direction_rad))
    if target_magnitude <= 0.0:
        return tuple(None for _ in null_directions_rad)
    return tuple(
        float(-20.0 * np.log10(max(abs(response(direction)) / target_magnitude, 1e-15)))
        for direction in null_directions_rad
    )


__all__ = ["realize_hardware", "realized_null_depths", "validate_advanced_config"]
