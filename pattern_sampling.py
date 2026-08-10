"""Adaptive 2D cuts, great-circle cuts, and 3D surface sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from beamforming import (
    PatternMetrics,
    array_factor,
    calculate_pattern_metrics,
    element_pattern_factor,
    great_circle_directions,
    normalize_pattern_db,
    normalize_pattern_linear,
)
from model_options import SCAN_MODE_OPTIONS

if TYPE_CHECKING:
    from simulation import SimulationState

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SURFACE_PATTERN_SCHEMA_VERSION = 5
PATTERN_CUT_SCHEMA_VERSION = 2
GREAT_CIRCLE_CUT_SCHEMA_VERSION = 1
PATTERN_CUT_BASE_SAMPLE_COUNT = 360
GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT = 720
PREVIEW_SURFACE_RESOLUTION = 16
PREVIEW_SURFACE_LOCAL_SAMPLE_COUNT = 17


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
class GreatCircleCuts:
    """Target-centered principal-plane cuts parameterized by angular distance."""

    schema_version: int
    base_sample_count: int
    local_sample_count: int
    horizontal_refinement_half_width_deg: float
    vertical_refinement_half_width_deg: float
    horizontal_offsets_rad: FloatArray
    vertical_offsets_rad: FloatArray
    horizontal_azimuth_rad: FloatArray
    horizontal_elevation_rad: FloatArray
    vertical_azimuth_rad: FloatArray
    vertical_elevation_rad: FloatArray
    horizontal_pattern: ComplexArray
    vertical_pattern: ComplexArray
    horizontal_pattern_db: FloatArray
    vertical_pattern_db: FloatArray
    horizontal_metrics: PatternMetrics
    vertical_metrics: PatternMetrics


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


@dataclass(frozen=True)
class SurfaceSamplingPlan:
    """3D sampling selected for one scan frame."""

    render_3d: bool
    resolution: int | None
    local_sample_count: int | None
    quality: str


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
        physical_y * np.cos(target_elevation_rad) * np.cos(target_azimuth_rad)
    )
    elevation_phase_positions = -physical_y * np.sin(target_elevation_rad) * np.sin(
        target_azimuth_rad
    ) + physical_z * np.cos(target_elevation_rad)
    return (
        float(np.ptp(azimuth_phase_positions)),
        float(np.ptp(elevation_phase_positions)),
    )


def _projected_great_circle_apertures(
    state: SimulationState,
    target_azimuth_rad: float,
    target_elevation_rad: float,
) -> tuple[float, float]:
    """Return phase apertures along unit-speed spherical principal planes."""

    physical_mask = state.coordinates.element_mask
    physical_y = state.coordinates.y[physical_mask]
    physical_z = state.coordinates.z[physical_mask]
    horizontal_phase_positions = physical_y * np.cos(target_azimuth_rad)
    vertical_phase_positions = -physical_y * np.sin(target_elevation_rad) * np.sin(
        target_azimuth_rad
    ) + physical_z * np.cos(target_elevation_rad)
    return (
        float(np.ptp(horizontal_phase_positions)),
        float(np.ptp(vertical_phase_positions)),
    )


def calculate_pattern_cuts(
    state: SimulationState,
    *,
    sample_count: int = PATTERN_CUT_BASE_SAMPLE_COUNT,
    local_sample_count: int | None = None,
    max_chunk_entries: int = 1_000_000,
    cancel_check: Callable[[], None] | None = None,
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
        cancel_check=cancel_check,
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
        cancel_check=cancel_check,
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
        elevation_refinement_half_width_deg=float(np.degrees(elevation_half_width)),
        azimuth_angles_rad=np.asarray(azimuth_angles, dtype=float),
        elevation_angles_rad=np.asarray(elevation_angles, dtype=float),
        azimuth_pattern=np.asarray(azimuth_pattern, dtype=complex),
        elevation_pattern=np.asarray(elevation_pattern, dtype=complex),
        azimuth_pattern_db=normalize_pattern_db(azimuth_pattern),
        elevation_pattern_db=normalize_pattern_db(elevation_pattern),
        azimuth_metrics=calculate_pattern_metrics(
            azimuth_pattern,
            azimuth_angles,
            peak_index=int(np.argmin(np.abs(azimuth_angles - target_azimuth))),
        ),
        elevation_metrics=calculate_pattern_metrics(
            elevation_pattern,
            elevation_angles,
            peak_index=int(np.argmin(np.abs(elevation_angles - target_elevation))),
        ),
    )


def calculate_great_circle_cuts(
    state: SimulationState,
    *,
    sample_count: int = GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT,
    local_sample_count: int | None = None,
    max_chunk_entries: int = 1_000_000,
    cancel_check: Callable[[], None] | None = None,
) -> GreatCircleCuts:
    """Calculate physical-angular-distance cuts through the target direction."""

    if sample_count < 5:
        raise ValueError("Great-circle cuts require at least five samples.")
    local_count = (
        pattern_cut_local_sample_count(state.coordinates.element_count)
        if local_sample_count is None
        else local_sample_count
    )
    if local_count < 3:
        raise ValueError("Local great-circle refinement requires three samples.")

    target_azimuth = np.radians(state.current_azimuth_deg)
    target_elevation = np.radians(state.current_elevation_deg)
    horizontal_aperture, vertical_aperture = _projected_great_circle_apertures(
        state,
        target_azimuth,
        target_elevation,
    )
    horizontal_half_width = _local_angular_half_width(
        horizontal_aperture,
        state.wavelength_m,
    )
    vertical_half_width = _local_angular_half_width(
        vertical_aperture,
        state.wavelength_m,
    )
    horizontal_offsets = _refined_angular_axis(
        -np.pi,
        np.pi,
        sample_count,
        0.0,
        horizontal_half_width,
        local_count,
    )
    vertical_offsets = _refined_angular_axis(
        -np.pi,
        np.pi,
        sample_count,
        0.0,
        vertical_half_width,
        local_count,
    )
    horizontal_azimuth, horizontal_elevation = great_circle_directions(
        target_azimuth,
        target_elevation,
        horizontal_offsets,
        plane="horizontal",
    )
    vertical_azimuth, vertical_elevation = great_circle_directions(
        target_azimuth,
        target_elevation,
        vertical_offsets,
        plane="vertical",
    )

    horizontal_pattern = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        horizontal_azimuth,
        horizontal_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    horizontal_pattern *= element_pattern_factor(
        state.config.element_option,
        horizontal_azimuth,
        horizontal_elevation,
    )
    vertical_pattern = array_factor(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        vertical_azimuth,
        vertical_elevation,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    vertical_pattern *= element_pattern_factor(
        state.config.element_option,
        vertical_azimuth,
        vertical_elevation,
    )
    return GreatCircleCuts(
        schema_version=GREAT_CIRCLE_CUT_SCHEMA_VERSION,
        base_sample_count=sample_count,
        local_sample_count=local_count,
        horizontal_refinement_half_width_deg=float(np.degrees(horizontal_half_width)),
        vertical_refinement_half_width_deg=float(np.degrees(vertical_half_width)),
        horizontal_offsets_rad=np.asarray(horizontal_offsets, dtype=float),
        vertical_offsets_rad=np.asarray(vertical_offsets, dtype=float),
        horizontal_azimuth_rad=np.asarray(horizontal_azimuth, dtype=float),
        horizontal_elevation_rad=np.asarray(horizontal_elevation, dtype=float),
        vertical_azimuth_rad=np.asarray(vertical_azimuth, dtype=float),
        vertical_elevation_rad=np.asarray(vertical_elevation, dtype=float),
        horizontal_pattern=np.asarray(horizontal_pattern, dtype=complex),
        vertical_pattern=np.asarray(vertical_pattern, dtype=complex),
        horizontal_pattern_db=normalize_pattern_db(horizontal_pattern),
        vertical_pattern_db=normalize_pattern_db(vertical_pattern),
        horizontal_metrics=calculate_pattern_metrics(
            horizontal_pattern,
            horizontal_offsets,
            peak_index=int(np.argmin(np.abs(horizontal_offsets))),
        ),
        vertical_metrics=calculate_pattern_metrics(
            vertical_pattern,
            vertical_offsets,
            peak_index=int(np.argmin(np.abs(vertical_offsets))),
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


def scan_surface_sampling(
    element_count: int,
    scan_mode: str,
    *,
    scanning: bool,
) -> SurfaceSamplingPlan:
    """Choose 3D work for a running scan or a stationary final frame.

    Stationary frames always use the normal adaptive resolution. During a
    scan, the 2D mode skips the surface and preview mode uses a deliberately
    small global/local mesh. Full-quality mode keeps the normal resolution.
    """

    if element_count < 1:
        raise ValueError("Element count must be positive.")
    if scan_mode not in SCAN_MODE_OPTIONS:
        raise ValueError("Unsupported scan mode.")

    full_resolution = surface_resolution(element_count)
    full_local_count = surface_local_sample_count(element_count)
    if not scanning or scan_mode == "full_3d":
        return SurfaceSamplingPlan(
            render_3d=True,
            resolution=full_resolution,
            local_sample_count=full_local_count,
            quality="full",
        )
    if scan_mode == "2d":
        return SurfaceSamplingPlan(
            render_3d=False,
            resolution=None,
            local_sample_count=None,
            quality="2d",
        )
    return SurfaceSamplingPlan(
        render_3d=True,
        resolution=min(full_resolution, PREVIEW_SURFACE_RESOLUTION),
        local_sample_count=min(
            full_local_count,
            PREVIEW_SURFACE_LOCAL_SAMPLE_COUNT,
        ),
        quality="preview",
    )


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
    cancel_check: Callable[[], None] | None = None,
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
        cancel_check=cancel_check,
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
        cancel_check=cancel_check,
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


__all__ = [
    "GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT",
    "GREAT_CIRCLE_CUT_SCHEMA_VERSION",
    "GreatCircleCuts",
    "PATTERN_CUT_BASE_SAMPLE_COUNT",
    "PATTERN_CUT_SCHEMA_VERSION",
    "PatternCuts",
    "SURFACE_PATTERN_SCHEMA_VERSION",
    "SurfacePattern",
    "SurfaceSamplingPlan",
    "calculate_great_circle_cuts",
    "calculate_pattern_cuts",
    "calculate_surface_pattern",
    "pattern_cut_local_sample_count",
    "scan_surface_sampling",
    "surface_local_sample_count",
    "surface_resolution",
]
