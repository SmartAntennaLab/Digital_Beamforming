"""Pure simulation orchestration for the digital-beamforming application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from beamforming import (
    ArrayCoordinates,
    ArrayGainMetrics,
    BeamformingWeights,
    GratingLobeAssessment,
    assess_grating_lobes,
    calculate_array_gain_metrics,
    compute_beamforming_weights,
    create_array_coordinates,
    create_array_taper,
    create_failure_mask,
)
from directivity import DirectivityResult, calculate_directivity
from interferer_sampling import (
    INTERFERER_GREAT_CIRCLE_BASE_SAMPLE_COUNT,
    INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION,
    INTERFERER_RESPONSE_SCHEMA_VERSION,
    InterfererGreatCircleCut,
    InterfererResponseComparison,
    calculate_interferer_great_circle_cuts,
    calculate_interferer_response_comparisons,
)
from model_options import SCAN_MODE_OPTIONS
from pattern_sampling import (
    GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT,
    GREAT_CIRCLE_CUT_SCHEMA_VERSION,
    PATTERN_CUT_BASE_SAMPLE_COUNT,
    PATTERN_CUT_SCHEMA_VERSION,
    PREVIEW_SURFACE_LOCAL_SAMPLE_COUNT,
    PREVIEW_SURFACE_RESOLUTION,
    SURFACE_PATTERN_SCHEMA_VERSION,
    GreatCircleCuts,
    PatternCuts,
    SurfacePattern,
    SurfaceSamplingPlan,
    calculate_great_circle_cuts,
    calculate_pattern_cuts,
    calculate_surface_pattern,
    pattern_cut_local_sample_count,
    scan_surface_sampling,
    surface_local_sample_count,
    surface_resolution,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
LIGHT_SPEED_M_S = 299_792_458.0
SCAN_REFERENCE_ELEMENT_COUNT = 4096
SCAN_REFERENCE_FULL_FRAME_SECONDS = 0.85


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
    directivity_mode: str = "auto"
    directivity_warning_elements: int = 1_024
    directivity_exact_max_elements: int = 4_096
    phase_bits: int | None = None
    failure_rate_percent: float = 0.0
    target_azimuth_deg: float = 0.0
    target_elevation_deg: float = 0.0
    enable_null_steering: bool = False
    null_azimuth_deg: float = 30.0
    null_elevation_deg: float = 0.0
    null_constraints_deg: tuple[tuple[float, float, float], ...] = ()
    null_optimization_mode: str = "amplitude_phase"
    maximum_element_amplitude: float | None = None
    null_optimizer_tolerance: float = 1e-8
    null_optimizer_max_iterations: int = 400
    null_optimizer_restart_count: int = 4


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
class ScanTimingEstimate:
    """Empirical scan time estimate calibrated at a 64×64 array."""

    frame_seconds: float
    effective_interval_seconds: float
    finalization_seconds: float
    total_seconds: float
    frame_count: int


def build_simulation_state(
    config: SimulationConfig,
    *,
    current_azimuth_deg: float | None = None,
    current_elevation_deg: float | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> SimulationState:
    """Build one deterministic beamforming state from a UI-independent config."""

    if cancel_check is not None:
        cancel_check()
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
    null_specs: tuple[tuple[float, float, float], ...] = ()
    if config.enable_null_steering:
        null_specs = config.null_constraints_deg or (
            (config.null_azimuth_deg, config.null_elevation_deg, 40.0),
        )
    null_directions: tuple[tuple[float, float], ...] = tuple(
        (float(np.radians(azimuth)), float(np.radians(elevation)))
        for azimuth, elevation, _ in null_specs
    )
    required_suppression: tuple[float, ...] = tuple(
        float(suppression) for _, _, suppression in null_specs
    )
    weight_result = compute_beamforming_weights(
        coordinates.y,
        coordinates.z,
        wavelength_m,
        azimuth_rad,
        elevation_rad,
        base_amplitudes,
        phase_bits=config.phase_bits,
        null_directions_rad=null_directions,
        null_required_suppression_db=required_suppression,
        maximum_element_amplitude=(
            config.maximum_element_amplitude if config.enable_null_steering else None
        ),
        optimization_mode=config.null_optimization_mode,
        optimizer_tolerance=config.null_optimizer_tolerance,
        optimizer_max_iterations=config.null_optimizer_max_iterations,
        optimizer_restart_count=config.null_optimizer_restart_count,
        cancel_check=cancel_check,
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
            float(state.vertical_spacing_m * 100.0) if uses_vertical_spacing else None
        ),
        horizontal_extent_wavelength=float(horizontal_extent_m / state.wavelength_m),
        horizontal_extent_cm=float(horizontal_extent_m * 100.0),
        vertical_extent_wavelength=float(vertical_extent_m / state.wavelength_m),
        vertical_extent_cm=float(vertical_extent_m * 100.0),
        total_elements=total_elements,
        active_elements=active_elements,
        failed_elements=total_elements - active_elements,
        requested_failure_rate_percent=float(state.config.failure_rate_percent),
        actual_failure_rate_percent=float(
            100.0 * (total_elements - active_elements) / total_elements
        ),
    )


def calculate_state_directivity(
    state: SimulationState,
    *,
    max_chunk_entries: int = 1_000_000,
    cancel_check: Callable[[], None] | None = None,
) -> DirectivityResult:
    """Calculate target-direction directivity for one completed state."""

    return calculate_directivity(
        state.coordinates.y,
        state.coordinates.z,
        state.complex_weights,
        state.wavelength_m,
        np.radians(state.current_azimuth_deg),
        np.radians(state.current_elevation_deg),
        state.config.element_option,
        element_mask=state.coordinates.element_mask,
        max_chunk_entries=max_chunk_entries,
        directivity_mode=state.config.directivity_mode,
        warning_element_count=state.config.directivity_warning_elements,
        exact_max_elements=state.config.directivity_exact_max_elements,
        cancel_check=cancel_check,
    )


def _scan_render_work_units(element_count: int, scan_mode: str) -> int:
    """Approximate array-factor entries rendered by the pattern tab."""

    sampling = scan_surface_sampling(element_count, scan_mode, scanning=True)
    coordinate_cut_points = 2 * (
        PATTERN_CUT_BASE_SAMPLE_COUNT + pattern_cut_local_sample_count(element_count)
    )
    great_circle_cut_points = 2 * (
        GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT
        + pattern_cut_local_sample_count(element_count)
    )
    surface_points = 0
    if sampling.render_3d:
        surface_points = int(sampling.resolution or 0) + int(
            sampling.local_sample_count or 0
        )
        surface_points **= 2
    return element_count * (
        coordinate_cut_points + great_circle_cut_points + surface_points
    )


def estimate_scan_timing(
    element_count: int,
    frame_count: int,
    scan_mode: str,
    frame_interval_seconds: float,
    *,
    session_calculations_per_minute: int | None = None,
    session_burst: int = 1,
) -> ScanTimingEstimate:
    """Estimate scan duration from relative array-factor work.

    The model is intentionally presented as an estimate: it anchors a full
    64×64 pattern-tab frame at 0.85 seconds and scales by element/sample work.
    The requested fragment interval cannot make a slower calculation faster.
    """

    if element_count < 1:
        raise ValueError("Element count must be positive.")
    if frame_count < 1:
        raise ValueError("Frame count must be positive.")
    if not np.isfinite(frame_interval_seconds) or frame_interval_seconds < 0.0:
        raise ValueError("Frame interval must be finite and non-negative.")
    if scan_mode not in SCAN_MODE_OPTIONS:
        raise ValueError("Unsupported scan mode.")
    if (
        session_calculations_per_minute is not None
        and session_calculations_per_minute < 1
    ):
        raise ValueError("Session calculation rate must be positive.")
    if session_burst < 1:
        raise ValueError("Session burst must be positive.")

    reference_work = _scan_render_work_units(
        SCAN_REFERENCE_ELEMENT_COUNT,
        "full_3d",
    )
    mode_work = _scan_render_work_units(element_count, scan_mode)
    frame_seconds = max(
        0.02,
        SCAN_REFERENCE_FULL_FRAME_SECONDS * mode_work / reference_work,
    )
    effective_interval = max(frame_seconds, frame_interval_seconds)

    finalization_seconds = 0.0
    if scan_mode != "full_3d":
        full_work = _scan_render_work_units(element_count, "full_3d")
        finalization_seconds = max(
            0.02,
            SCAN_REFERENCE_FULL_FRAME_SECONDS * full_work / reference_work,
        )

    total_seconds = frame_count * effective_interval
    if session_calculations_per_minute is not None:
        refill_rate = session_calculations_per_minute / 60.0
        capacity = float(min(session_burst, session_calculations_per_minute))
        tokens = capacity
        elapsed_seconds = 0.0
        updated_at = 0.0
        for _ in range(frame_count):
            refill_elapsed = elapsed_seconds - updated_at
            tokens = min(capacity, tokens + refill_elapsed * refill_rate)
            updated_at = elapsed_seconds
            if tokens < 1.0:
                wait_seconds = (1.0 - tokens) / refill_rate
                elapsed_seconds += wait_seconds
                tokens = 1.0
                updated_at = elapsed_seconds
            tokens -= 1.0
            elapsed_seconds += effective_interval
        total_seconds = elapsed_seconds

    return ScanTimingEstimate(
        frame_seconds=float(frame_seconds),
        effective_interval_seconds=float(effective_interval),
        finalization_seconds=float(finalization_seconds),
        total_seconds=float(total_seconds + finalization_seconds),
        frame_count=frame_count,
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
    azimuth_fraction = azimuth_index / (azimuth_steps - 1) if azimuth_steps > 1 else 0.0
    elevation_fraction = (
        elevation_index / (elevation_steps - 1) if elevation_steps > 1 else 0.0
    )
    azimuth_deg = azimuth_range_deg[0] + azimuth_fraction * (
        azimuth_range_deg[1] - azimuth_range_deg[0]
    )
    elevation_deg = elevation_range_deg[0] + elevation_fraction * (
        elevation_range_deg[1] - elevation_range_deg[0]
    )
    return float(azimuth_deg), float(elevation_deg), total_steps
