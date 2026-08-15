"""Serializable calculation bundles suitable for inline or worker execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from directivity import DirectivityResult
from golden_validation import GoldenValidationResult, validate_golden_dataset
from interferer_sampling import (
    InterfererGreatCircleCut,
    InterfererResponseComparison,
    calculate_interferer_great_circle_cuts,
    calculate_interferer_response_comparisons,
)
from pattern_sampling import (
    GreatCircleCuts,
    PatternCuts,
    SurfacePattern,
    SurfaceSamplingPlan,
    calculate_great_circle_cuts,
    calculate_pattern_cuts,
    calculate_surface_pattern,
    scan_surface_sampling,
)
from signal_processing import AdvancedAnalysis, calculate_advanced_analysis
from simulation import (
    SimulationConfig,
    SimulationState,
    build_simulation_state,
    calculate_state_directivity,
)

COMPUTE_VIEW_SCHEMA_VERSION = 2
ViewName = Literal["pattern", "metrics", "elements"]


@dataclass(frozen=True)
class ViewComputeRequest:
    """One complete result-view calculation request."""

    config: SimulationConfig
    current_azimuth_deg: float
    current_elevation_deg: float
    view_name: ViewName
    scan_mode: str = "preview_3d"
    scanning: bool = False
    schema_version: int = COMPUTE_VIEW_SCHEMA_VERSION


@dataclass(frozen=True)
class ViewComputeResult:
    """All numerical data needed to render one active result tab."""

    state: SimulationState
    cuts: PatternCuts | None = None
    great_circle_cuts: GreatCircleCuts | None = None
    directivity: DirectivityResult | None = None
    interferer_comparisons: tuple[InterfererResponseComparison, ...] = ()
    interferer_great_circle_cuts: tuple[InterfererGreatCircleCut, ...] = ()
    surface_sampling: SurfaceSamplingPlan | None = None
    surface: SurfacePattern | None = None
    advanced_analysis: AdvancedAnalysis | None = None
    golden_validation: GoldenValidationResult | None = None


def _without_null_steering(config: SimulationConfig) -> SimulationConfig:
    return replace(
        config,
        enable_null_steering=False,
        maximum_element_amplitude=None,
    )


def calculate_view(
    request: ViewComputeRequest,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> ViewComputeResult:
    """Calculate an active tab as one serializable worker task."""

    if request.schema_version != COMPUTE_VIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported view-compute schema version.")
    if request.view_name not in {"pattern", "metrics", "elements"}:
        raise ValueError("Unsupported result view.")
    if cancel_check is not None:
        cancel_check()

    state = build_simulation_state(
        request.config,
        current_azimuth_deg=request.current_azimuth_deg,
        current_elevation_deg=request.current_elevation_deg,
        cancel_check=cancel_check,
    )
    cuts = None
    great_circle_cuts = None
    directivity = None
    comparisons: tuple[InterfererResponseComparison, ...] = ()
    interferer_cuts: tuple[InterfererGreatCircleCut, ...] = ()
    surface_sampling = None
    surface = None
    advanced_analysis = None
    golden_validation = None

    if request.view_name in {"pattern", "metrics"}:
        cuts = calculate_pattern_cuts(state, cancel_check=cancel_check)
        great_circle_cuts = calculate_great_circle_cuts(
            state,
            cancel_check=cancel_check,
        )
    if request.view_name == "metrics":
        directivity = calculate_state_directivity(
            state,
            cancel_check=cancel_check,
        )
        advanced_analysis = calculate_advanced_analysis(
            state,
            cancel_check=cancel_check,
        )
        if request.config.golden_dataset is not None:
            golden_validation = validate_golden_dataset(
                state,
                request.config.golden_dataset,
                cancel_check=cancel_check,
            )

    baseline_state = None
    if request.config.enable_null_steering and request.view_name in {
        "pattern",
        "metrics",
    }:
        baseline_state = build_simulation_state(
            _without_null_steering(request.config),
            current_azimuth_deg=request.current_azimuth_deg,
            current_elevation_deg=request.current_elevation_deg,
            cancel_check=cancel_check,
        )
        comparisons = calculate_interferer_response_comparisons(
            state,
            baseline_state,
            cancel_check=cancel_check,
        )
    if request.config.enable_null_steering and request.view_name == "pattern":
        if baseline_state is None:
            raise RuntimeError("Null pattern calculation requires a baseline state.")
        interferer_cuts = calculate_interferer_great_circle_cuts(
            state,
            baseline_state,
            comparisons=comparisons,
            cancel_check=cancel_check,
        )

    if request.view_name == "pattern":
        surface_sampling = scan_surface_sampling(
            state.coordinates.element_count,
            request.scan_mode,
            scanning=request.scanning,
        )
        if surface_sampling.render_3d:
            surface = calculate_surface_pattern(
                state,
                resolution=surface_sampling.resolution,
                local_sample_count=surface_sampling.local_sample_count,
                cancel_check=cancel_check,
            )

    if cancel_check is not None:
        cancel_check()
    return ViewComputeResult(
        state=state,
        cuts=cuts,
        great_circle_cuts=great_circle_cuts,
        directivity=directivity,
        interferer_comparisons=comparisons,
        interferer_great_circle_cuts=interferer_cuts,
        surface_sampling=surface_sampling,
        surface=surface,
        advanced_analysis=advanced_analysis,
        golden_validation=golden_validation,
    )


__all__ = [
    "COMPUTE_VIEW_SCHEMA_VERSION",
    "ViewComputeRequest",
    "ViewComputeResult",
    "ViewName",
    "calculate_view",
]
