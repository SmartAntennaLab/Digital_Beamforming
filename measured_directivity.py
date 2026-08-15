"""Numerical full-sphere directivity for uploaded complex element patterns."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from directivity import DIRECTIVITY_SCHEMA_VERSION, DirectivityResult
from element_pattern_data import evaluate_element_pattern
from pattern_metrics import array_factor

if TYPE_CHECKING:
    from simulation import SimulationState

MEASURED_DIRECTIVITY_AZIMUTH_SAMPLES = 96
MEASURED_DIRECTIVITY_ELEVATION_SAMPLES = 48


def calculate_measured_pattern_directivity(
    state: SimulationState,
    *,
    max_chunk_entries: int = 1_000_000,
    cancel_check: Callable[[], None] | None = None,
) -> DirectivityResult:
    """Integrate a measured element pattern with Gauss-Legendre quadrature."""

    if state.config.element_pattern_grid is None:
        raise ValueError("Measured-pattern directivity requires an uploaded pattern.")
    mu, quadrature_weights = np.polynomial.legendre.leggauss(
        MEASURED_DIRECTIVITY_ELEVATION_SAMPLES
    )
    azimuth = np.linspace(
        -np.pi,
        np.pi,
        MEASURED_DIRECTIVITY_AZIMUTH_SAMPLES,
        endpoint=False,
    )
    azimuth_grid, mu_grid = np.meshgrid(azimuth, mu)
    elevation_grid = np.arcsin(mu_grid)
    pattern = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        azimuth_grid,
        elevation_grid,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    pattern *= evaluate_element_pattern(
        state.config.element_option,
        azimuth_grid,
        elevation_grid,
        pattern_grid=state.config.element_pattern_grid,
        polarization_angle_deg=state.config.polarization_angle_deg,
    )
    azimuth_step = 2.0 * np.pi / MEASURED_DIRECTIVITY_AZIMUTH_SAMPLES
    power_integral = float(
        azimuth_step * np.sum(np.abs(pattern) ** 2 * quadrature_weights[:, None])
    )
    target_azimuth = np.radians(state.current_azimuth_deg)
    target_elevation = np.radians(state.current_elevation_deg)
    target = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        target_azimuth,
        target_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    target *= evaluate_element_pattern(
        state.config.element_option,
        target_azimuth,
        target_elevation,
        pattern_grid=state.config.element_pattern_grid,
        polarization_angle_deg=state.config.polarization_angle_deg,
    )
    target_intensity = float(abs(target.item()) ** 2)
    directivity_linear = (
        4.0 * np.pi * target_intensity / power_integral
        if power_integral > np.finfo(float).tiny
        else None
    )
    return DirectivityResult(
        schema_version=DIRECTIVITY_SCHEMA_VERSION,
        directivity_linear=directivity_linear,
        directivity_dbi=(
            float(10.0 * np.log10(directivity_linear))
            if directivity_linear is not None and directivity_linear > 0.0
            else None
        ),
        target_radiation_intensity=target_intensity,
        radiated_power_integral=power_integral,
        integration_method="measured-pattern Gauss-Legendre full-sphere quadrature",
        azimuth_samples=MEASURED_DIRECTIVITY_AZIMUTH_SAMPLES,
        elevation_samples=MEASURED_DIRECTIVITY_ELEVATION_SAMPLES,
        requested_mode=state.config.directivity_mode,
        effective_mode="fast",
        is_approximate=True,
        element_count=state.coordinates.element_count,
        warning_message=(
            "Uploaded element patterns use bounded numerical full-sphere integration."
        ),
    )


__all__ = ["calculate_measured_pattern_directivity"]
