"""Pure simulation orchestration for the Streamlit beamforming application.

The numerical primitives live in :mod:`beamforming`.  This module combines
them into one immutable simulation frame so the UI can render only the active
view and unit tests can exercise the workflow without starting Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from beamforming import (
    ArrayCoordinates,
    ArrayGainMetrics,
    BeamformingWeights,
    GratingLobeAssessment,
    PatternMetrics,
    array_factor,
    assess_grating_lobes,
    calculate_array_gain_metrics,
    calculate_pattern_metrics,
    compute_beamforming_weights,
    create_array_coordinates,
    create_array_taper,
    create_failure_mask,
    element_pattern_factor,
    normalize_pattern_db,
    normalize_pattern_linear,
)


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
LIGHT_SPEED_M_S = 299_792_458.0
SURFACE_PATTERN_SCHEMA_VERSION = 5
PATTERN_CUT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SimulationConfig:
    """All expensive numerical inputs for a simulation frame."""

    frequency_ghz: float = 28.0
    vertical_count: int = 4
    horizontal_count: int = 4
    horizontal_spacing_wavelength: float = 0.5
    vertical_spacing_wavelength: float = 0.5
    geometry: str = "UPA"
    taper_option: str = "uniform"
    element_option: str = "isotropic"
    phase_bits: int | None = None
    failure_rate_percent: float = 0.0
    target_azimuth_deg: float = 0.0
    target_elevation_deg: float = 0.0
    enable_null_steering: bool = False
    null_azimuth_deg: float = 30.0
    null_elevation_deg: float = 0.0


@dataclass(frozen=True)
class SimulationState:
    """Geometry, weights, and diagnostics shared by all result views."""

    config: SimulationConfig
    current_azimuth_deg: float
    current_elevation_deg: float
    wavelength_m: float
    horizontal_spacing_m: float
    vertical_spacing_m: float
    coordinates: ArrayCoordinates
    base_amplitudes: FloatArray
    active_mask: BoolArray
    weight_result: BeamformingWeights
    complex_weights: ComplexArray
    actual_amplitudes: FloatArray
    gain_metrics: ArrayGainMetrics
    grating_assessment: GratingLobeAssessment


@dataclass(frozen=True)
class ArrayLayoutSummary:
    """Physical spacing, aperture, and element counts for UI/reporting."""

    horizontal_spacing_wavelength: float
    horizontal_spacing_cm: float
    vertical_spacing_wavelength: float | None
    vertical_spacing_cm: float | None
    horizontal_extent_wavelength: float
    horizontal_extent_cm: float
    vertical_extent_wavelength: float
    vertical_extent_cm: float
    total_elements: int
    active_elements: int
    failed_elements: int
    requested_failure_rate_percent: float
    actual_failure_rate_percent: float


@dataclass(frozen=True)
class PatternCuts:
    """Azimuth/elevation cuts and their derived performance metrics."""

    base_sample_count: int
    local_sample_count: int
    azimuth_refinement_half_width_deg: float
    elevation_refinement_half_width_deg: float
    azimuth_angles_rad: FloatArray
    elevation_angles_rad: FloatArray
    azimuth_pattern: ComplexArray
    elevation_pattern: ComplexArray
    azimuth_pattern_db: FloatArray
    elevation_pattern_db: FloatArray
    azimuth_metrics: PatternMetrics
    elevation_metrics: PatternMetrics


@dataclass(frozen=True)
class SurfacePattern:
    """A chunk-computed spherical pattern ready for 3D rendering."""

    schema_version: int
    base_resolution: int
    polar_angle_rad: FloatArray
    azimuth_angle_rad: FloatArray
    pattern: ComplexArray
    pattern_linear: FloatArray
    pattern_db: FloatArray
    sampled_peak_magnitude: float
    target_response_magnitude: float


def build_simulation_state(
    config: SimulationConfig,
    *,
    current_azimuth_deg: float | None = None,
    current_elevation_deg: float | None = None,
) -> SimulationState:
    """Build one deterministic beamforming state from a UI-independent config."""

    if config.frequency_ghz <= 0.0 or not np.isfinite(config.frequency_ghz):
        raise ValueError("Frequency must be a finite positive value.")
    if config.horizontal_spacing_wavelength <= 0.0 or not np.isfinite(
        config.horizontal_spacing_wavelength
    ):
        raise ValueError("Horizontal spacing must be finite and positive.")
    if config.vertical_spacing_wavelength <= 0.0 or not np.isfinite(
        config.vertical_spacing_wavelength
    ):
        raise ValueError("Vertical spacing must be finite and positive.")

    azimuth_deg = (
        config.target_azimuth_deg
        if current_azimuth_deg is None
        else float(current_azimuth_deg)
    )
    elevation_deg = (
        config.target_elevation_deg
        if current_elevation_deg is None
        else float(current_elevation_deg)
    )
    if not np.isfinite(azimuth_deg) or not np.isfinite(elevation_deg):
        raise ValueError("Steering angles must be finite.")

    wavelength_m = LIGHT_SPEED_M_S / (config.frequency_ghz * 1.0e9)
    horizontal_spacing_m = config.horizontal_spacing_wavelength * wavelength_m
    vertical_spacing_m = config.vertical_spacing_wavelength * wavelength_m
    coordinates = create_array_coordinates(
        config.vertical_count,
        config.horizontal_count,
        horizontal_spacing_m,
        config.geometry,
        vertical_spacing_m=vertical_spacing_m,
    )
    if coordinates.geometry == "UHA":
        vertical_spacing_m = horizontal_spacing_m * np.sin(np.pi / 3.0)

    active_mask = create_failure_mask(
        coordinates.rows,
        coordinates.columns,
        config.failure_rate_percent,
        seed=42,
        element_mask=coordinates.element_mask,
    )
    base_amplitudes = create_array_taper(coordinates, config.taper_option) * active_mask

    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    null_direction = (
        (
            np.radians(config.null_azimuth_deg),
            np.radians(config.null_elevation_deg),
        )
        if config.enable_null_steering
        else None
    )
    weight_result = compute_beamforming_weights(
        coordinates.y,
        coordinates.z,
        wavelength_m,
        azimuth_rad,
        elevation_rad,
        base_amplitudes,
        phase_bits=config.phase_bits,
        null_direction_rad=null_direction,
    )
    complex_weights = weight_result.weights
    gain_metrics = calculate_array_gain_metrics(
        coordinates.y,
        coordinates.z,
        complex_weights,
        active_mask,
        wavelength_m,
        azimuth_rad,
        elevation_rad,
        element_mask=coordinates.element_mask,
    )
    grating_assessment = assess_grating_lobes(
        config.geometry,
        config.horizontal_spacing_wavelength,
        azimuth_rad,
        elevation_rad,
        vertical_count=coordinates.rows,
        horizontal_count=coordinates.columns,
        vertical_spacing_over_wavelength=(
            config.horizontal_spacing_wavelength * np.sin(np.pi / 3.0)
            if coordinates.geometry == "UHA"
            else config.vertical_spacing_wavelength
        ),
    )

    return SimulationState(
        config=config,
        current_azimuth_deg=azimuth_deg,
        current_elevation_deg=elevation_deg,
        wavelength_m=wavelength_m,
        horizontal_spacing_m=horizontal_spacing_m,
        vertical_spacing_m=vertical_spacing_m,
        coordinates=coordinates,
        base_amplitudes=np.asarray(base_amplitudes, dtype=float),
        active_mask=np.asarray(active_mask, dtype=bool),
        weight_result=weight_result,
        complex_weights=np.asarray(complex_weights, dtype=complex),
        actual_amplitudes=np.asarray(np.abs(complex_weights), dtype=float),
        gain_metrics=gain_metrics,
        grating_assessment=grating_assessment,
    )


def summarize_array_layout(state: SimulationState) -> ArrayLayoutSummary:
    """Summarize center-to-center array dimensions in wavelengths and cm."""

    physical_mask = state.coordinates.element_mask
    if physical_mask.shape != state.active_mask.shape or not np.any(physical_mask):
        raise ValueError("Array layout requires at least one physical element.")

    physical_y = state.coordinates.y[physical_mask]
    physical_z = state.coordinates.z[physical_mask]
    horizontal_extent_m = float(np.ptp(physical_y))
    vertical_extent_m = float(np.ptp(physical_z))
    uses_vertical_spacing = state.coordinates.geometry in {"UPA", "UHA"}
    total_elements = state.coordinates.element_count
    active_elements = int(np.count_nonzero(state.active_mask & physical_mask))

    return ArrayLayoutSummary(
        horizontal_spacing_wavelength=float(
            state.horizontal_spacing_m / state.wavelength_m
        ),
        horizontal_spacing_cm=float(state.horizontal_spacing_m * 100.0),
        vertical_spacing_wavelength=(
            float(state.vertical_spacing_m / state.wavelength_m)
            if uses_vertical_spacing
            else None
        ),
        vertical_spacing_cm=(
            float(state.vertical_spacing_m * 100.0)
            if uses_vertical_spacing
            else None
        ),
        horizontal_extent_wavelength=float(horizontal_extent_m / state.wavelength_m),
        horizontal_extent_cm=float(horizontal_extent_m * 100.0),
        vertical_extent_wavelength=float(vertical_extent_m / state.wavelength_m),
        vertical_extent_cm=float(vertical_extent_m * 100.0),
        total_elements=total_elements,
        active_elements=active_elements,
        failed_elements=total_elements - active_elements,
        requested_failure_rate_percent=float(
            state.config.failure_rate_percent
        ),
        actual_failure_rate_percent=float(
            100.0 * (total_elements - active_elements) / total_elements
        ),
    )


def pattern_cut_local_sample_count(element_count: int) -> int:
    """Select target-region cut detail while bounding array-factor work."""

    if element_count < 1:
        raise ValueError("Element count must be positive.")
    return 65 if element_count <= 64 else 129


def _projected_cut_apertures(
    state: SimulationState,
    target_azimuth_rad: float,
    target_elevation_rad: float,
) -> tuple[float, float]:
    """Return first-order phase apertures for the two principal cuts."""

    physical_mask = state.coordinates.element_mask
    physical_y = state.coordinates.y[physical_mask]
    physical_z = state.coordinates.z[physical_mask]
    azimuth_phase_positions = (
        physical_y
        * np.cos(target_elevation_rad)
        * np.cos(target_azimuth_rad)
    )
    elevation_phase_positions = (
        -physical_y
        * np.sin(target_elevation_rad)
        * np.sin(target_azimuth_rad)
        + physical_z * np.cos(target_elevation_rad)
    )
    return (
        float(np.ptp(azimuth_phase_positions)),
        float(np.ptp(elevation_phase_positions)),
    )


def calculate_pattern_cuts(
    state: SimulationState,
    *,
    sample_count: int = 360,
    local_sample_count: int | None = None,
    max_chunk_entries: int = 1_000_000,
) -> PatternCuts:
    """Calculate target-refined cuts using bounded array-factor workspaces."""

    if sample_count < 3:
        raise ValueError("Pattern cuts require at least three angle samples.")
    local_count = (
        pattern_cut_local_sample_count(state.coordinates.element_count)
        if local_sample_count is None
        else local_sample_count
    )
    if local_count < 3:
        raise ValueError("Local cut refinement requires at least three samples.")

    target_azimuth = np.radians(state.current_azimuth_deg)
    target_elevation = np.radians(state.current_elevation_deg)
    azimuth_aperture, elevation_aperture = _projected_cut_apertures(
        state,
        target_azimuth,
        target_elevation,
    )
    azimuth_half_width = _local_angular_half_width(
        azimuth_aperture,
        state.wavelength_m,
    )
    elevation_half_width = _local_angular_half_width(
        elevation_aperture,
        state.wavelength_m,
    )
    azimuth_angles = _refined_angular_axis(
        -np.pi / 2.0,
        np.pi / 2.0,
        sample_count,
        target_azimuth,
        azimuth_half_width,
        local_count,
    )
    elevation_angles = _refined_angular_axis(
        -np.pi / 2.0,
        np.pi / 2.0,
        sample_count,
        target_elevation,
        elevation_half_width,
        local_count,
    )

    azimuth_pattern = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        azimuth_angles,
        np.full_like(azimuth_angles, target_elevation),
        max_chunk_entries=max_chunk_entries,
    )
    azimuth_pattern *= element_pattern_factor(
        state.config.element_option,
        azimuth_angles,
        np.full_like(azimuth_angles, target_elevation),
    )

    elevation_pattern = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        np.full_like(elevation_angles, target_azimuth),
        elevation_angles,
        max_chunk_entries=max_chunk_entries,
    )
    elevation_pattern *= element_pattern_factor(
        state.config.element_option,
        np.full_like(elevation_angles, target_azimuth),
        elevation_angles,
    )

    return PatternCuts(
        base_sample_count=sample_count,
        local_sample_count=local_count,
        azimuth_refinement_half_width_deg=float(np.degrees(azimuth_half_width)),
        elevation_refinement_half_width_deg=float(
            np.degrees(elevation_half_width)
        ),
        azimuth_angles_rad=np.asarray(azimuth_angles, dtype=float),
        elevation_angles_rad=np.asarray(elevation_angles, dtype=float),
        azimuth_pattern=np.asarray(azimuth_pattern, dtype=complex),
        elevation_pattern=np.asarray(elevation_pattern, dtype=complex),
        azimuth_pattern_db=normalize_pattern_db(azimuth_pattern),
        elevation_pattern_db=normalize_pattern_db(elevation_pattern),
        azimuth_metrics=calculate_pattern_metrics(azimuth_pattern, azimuth_angles),
        elevation_metrics=calculate_pattern_metrics(
            elevation_pattern, elevation_angles
        ),
    )


def surface_resolution(element_count: int) -> int:
    """Select a bounded 3D angular grid for the current array size."""

    if element_count < 1:
        raise ValueError("Element count must be positive.")
    if element_count <= 256:
        return 50
    if element_count <= 1024:
        return 40
    if element_count <= 4096:
        return 30
    return 20


def surface_local_sample_count(element_count: int) -> int:
    """Select target-region detail while bounding large-array work."""

    if element_count < 1:
        raise ValueError("Element count must be positive.")
    if element_count <= 64:
        return 33
    if element_count <= 1024:
        return 65
    if element_count <= 4096:
        return 49
    return 33


def _local_angular_half_width(
    aperture_m: float,
    wavelength_m: float,
) -> float:
    """Estimate a target-centered refinement span from one array aperture."""

    minimum_width = np.radians(0.5)
    maximum_width = np.radians(12.0)
    if aperture_m <= np.finfo(float).eps:
        return maximum_width
    estimated_width = 4.0 * wavelength_m / aperture_m
    return float(np.clip(estimated_width, minimum_width, maximum_width))


def _refined_angular_axis(
    start_rad: float,
    stop_rad: float,
    base_count: int,
    target_rad: float,
    local_half_width_rad: float,
    local_sample_count: int,
) -> FloatArray:
    """Merge a global axis with dense samples around the exact target angle."""

    target = float(np.clip(target_rad, start_rad, stop_rad))
    local_start = max(start_rad, target - local_half_width_rad)
    local_stop = min(stop_rad, target + local_half_width_rad)
    base_axis = np.linspace(start_rad, stop_rad, base_count)
    local_axis = np.linspace(local_start, local_stop, local_sample_count)
    return np.asarray(
        np.unique(np.concatenate((base_axis, local_axis, [target]))),
        dtype=float,
    )


def calculate_surface_pattern(
    state: SimulationState,
    *,
    resolution: int | None = None,
    local_sample_count: int | None = None,
    max_chunk_entries: int = 1_000_000,
) -> SurfacePattern:
    """Calculate a target-refined spherical surface with bounded workspaces.

    A coarse global grid keeps the full sphere inexpensive.  A nonuniform
    local grid is merged around the exact steering direction so a narrow main
    lobe cannot fall between samples as the array aperture grows.
    """

    element_count = state.coordinates.element_count
    grid_size = resolution or surface_resolution(element_count)
    local_count = (
        surface_local_sample_count(element_count)
        if local_sample_count is None
        else local_sample_count
    )
    if grid_size < 3:
        raise ValueError("Surface resolution must be at least three.")
    if local_count < 3:
        raise ValueError("Local surface refinement requires at least three samples.")

    target_azimuth = np.radians(state.current_azimuth_deg)
    target_polar = np.pi / 2.0 - np.radians(state.current_elevation_deg)
    horizontal_aperture = float(np.ptp(state.coordinates.y))
    vertical_aperture = float(np.ptp(state.coordinates.z))
    azimuth = _refined_angular_axis(
        -np.pi,
        np.pi,
        grid_size,
        target_azimuth,
        _local_angular_half_width(horizontal_aperture, state.wavelength_m),
        local_count,
    )
    polar = _refined_angular_axis(
        0.0,
        np.pi,
        grid_size,
        target_polar,
        _local_angular_half_width(vertical_aperture, state.wavelength_m),
        local_count,
    )
    polar_grid, azimuth_grid = np.meshgrid(polar, azimuth)
    elevation_grid = np.pi / 2.0 - polar_grid
    pattern = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        azimuth_grid,
        elevation_grid,
        max_chunk_entries=max_chunk_entries,
    )
    pattern *= element_pattern_factor(
        state.config.element_option,
        azimuth_grid,
        elevation_grid,
    )
    target_response = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        target_azimuth,
        np.radians(state.current_elevation_deg),
        max_chunk_entries=max_chunk_entries,
    )
    target_response *= element_pattern_factor(
        state.config.element_option,
        target_azimuth,
        np.radians(state.current_elevation_deg),
    )
    return SurfacePattern(
        schema_version=SURFACE_PATTERN_SCHEMA_VERSION,
        base_resolution=grid_size,
        polar_angle_rad=np.asarray(polar_grid, dtype=float),
        azimuth_angle_rad=np.asarray(azimuth_grid, dtype=float),
        pattern=np.asarray(pattern, dtype=complex),
        pattern_linear=normalize_pattern_linear(pattern),
        pattern_db=normalize_pattern_db(pattern),
        sampled_peak_magnitude=float(np.max(np.abs(pattern))),
        target_response_magnitude=float(np.abs(target_response.item())),
    )


def scan_direction(
    index: int,
    azimuth_range_deg: tuple[float, float],
    elevation_range_deg: tuple[float, float],
    azimuth_steps: int,
    elevation_steps: int,
) -> tuple[float, float, int]:
    """Resolve one raster-scan index into azimuth/elevation and total count."""

    if azimuth_steps < 1 or elevation_steps < 1:
        raise ValueError("Scan step counts must be positive.")
    total_steps = azimuth_steps * elevation_steps
    if not 0 <= index < total_steps:
        raise ValueError("Scan index is outside the configured raster.")

    elevation_index, azimuth_index = divmod(index, azimuth_steps)
    azimuth_fraction = (
        azimuth_index / (azimuth_steps - 1) if azimuth_steps > 1 else 0.0
    )
    elevation_fraction = (
        elevation_index / (elevation_steps - 1)
        if elevation_steps > 1
        else 0.0
    )
    azimuth_deg = azimuth_range_deg[0] + azimuth_fraction * (
        azimuth_range_deg[1] - azimuth_range_deg[0]
    )
    elevation_deg = elevation_range_deg[0] + elevation_fraction * (
        elevation_range_deg[1] - elevation_range_deg[0]
    )
    return float(azimuth_deg), float(elevation_deg), total_steps
