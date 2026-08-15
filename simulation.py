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
from element_pattern_data import ElementPatternGrid
from golden_validation import GoldenDataset
from interferer_sampling import (
    INTERFERER_GREAT_CIRCLE_BASE_SAMPLE_COUNT,
    INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION,
    INTERFERER_RESPONSE_SCHEMA_VERSION,
    InterfererGreatCircleCut,
    InterfererResponseComparison,
    calculate_interferer_great_circle_cuts,
    calculate_interferer_response_comparisons,
)
from measured_directivity import calculate_measured_pattern_directivity
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
from physical_effects import HardwareEffectDiagnostics
from scan_estimation import ScanTimingEstimate, estimate_scan_timing, scan_direction
from simulation_effects import (
    realize_hardware,
    realized_null_depths,
    validate_advanced_config,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
LIGHT_SPEED_M_S = 299_792_458.0


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
    random_seed: int = 42
    position_error_rms_wavelength: float = 0.0
    amplitude_error_rms_db: float = 0.0
    phase_error_rms_deg: float = 0.0
    mutual_coupling_db: float | None = None
    mutual_coupling_phase_deg: float = 0.0
    polarization_angle_deg: float = 0.0
    element_pattern_grid: ElementPatternGrid | None = None
    wideband_bandwidth_percent: float = 0.0
    wideband_frequency_samples: int = 7
    near_field_focus_range_m: float | None = None
    enable_channel_analysis: bool = False
    channel_snapshots: int = 128
    multipath_count: int = 0
    signal_power_dbm: float = 0.0
    interference_power_dbm: float = -10.0
    noise_power_dbm: float = -30.0
    adaptive_beamforming_method: str = "none"
    diagonal_loading: float = 1e-3
    enable_doa_estimation: bool = False
    golden_dataset: GoldenDataset | None = None


@dataclass(frozen=True)
class SimulationState:
    """Geometry, weights, and diagnostics shared by all result views."""

    config: SimulationConfig
    current_azimuth_deg: float
    current_elevation_deg: float
    wavelength_m: float
    horizontal_spacing_m: float
    vertical_spacing_m: float
    nominal_coordinates: ArrayCoordinates
    coordinates: ArrayCoordinates
    base_amplitudes: FloatArray
    active_mask: BoolArray
    weight_result: BeamformingWeights
    complex_weights: ComplexArray
    actual_amplitudes: FloatArray
    hardware_diagnostics: HardwareEffectDiagnostics
    realized_null_depths_db: tuple[float | None, ...]
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
    validate_advanced_config(config)

    wavelength_m = LIGHT_SPEED_M_S / (config.frequency_ghz * 1.0e9)
    horizontal_spacing_m = config.horizontal_spacing_wavelength * wavelength_m
    vertical_spacing_m = config.vertical_spacing_wavelength * wavelength_m
    nominal_coordinates = create_array_coordinates(
        config.vertical_count,
        config.horizontal_count,
        horizontal_spacing_m,
        config.geometry,
        vertical_spacing_m=vertical_spacing_m,
    )
    if nominal_coordinates.geometry == "UHA":
        vertical_spacing_m = horizontal_spacing_m * np.sin(np.pi / 3.0)

    active_mask = create_failure_mask(
        nominal_coordinates.rows,
        nominal_coordinates.columns,
        config.failure_rate_percent,
        seed=config.random_seed,
        element_mask=nominal_coordinates.element_mask,
    )
    base_amplitudes = (
        create_array_taper(nominal_coordinates, config.taper_option) * active_mask
    )
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
        nominal_coordinates.y,
        nominal_coordinates.z,
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
    coordinates, complex_weights, hardware_diagnostics = realize_hardware(
        config,
        nominal_coordinates,
        weight_result,
        base_amplitudes,
        active_mask,
        wavelength_m=wavelength_m,
        horizontal_spacing_m=horizontal_spacing_m,
        vertical_spacing_m=vertical_spacing_m,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        cancel_check=cancel_check,
    )
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
    post_impairment_null_depths = realized_null_depths(
        config,
        coordinates,
        complex_weights,
        wavelength_m,
        (azimuth_rad, elevation_rad),
        weight_result.null_directions_rad,
    )
    grating_assessment = assess_grating_lobes(
        config.geometry,
        config.horizontal_spacing_wavelength,
        azimuth_rad,
        elevation_rad,
        vertical_count=nominal_coordinates.rows,
        horizontal_count=nominal_coordinates.columns,
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
        nominal_coordinates=nominal_coordinates,
        coordinates=coordinates,
        base_amplitudes=np.asarray(base_amplitudes, dtype=float),
        active_mask=np.asarray(active_mask, dtype=bool),
        weight_result=weight_result,
        complex_weights=np.asarray(complex_weights, dtype=complex),
        actual_amplitudes=np.asarray(np.abs(complex_weights), dtype=float),
        hardware_diagnostics=hardware_diagnostics,
        realized_null_depths_db=post_impairment_null_depths,
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

    if state.config.element_pattern_grid is not None:
        return calculate_measured_pattern_directivity(
            state,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
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
