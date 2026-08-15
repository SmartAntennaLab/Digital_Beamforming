"""Bounded Streamlit caches around pure simulation orchestration."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from compute_executor import ComputeExecutor
from compute_governor import check_current_computation
from compute_tasks import ViewComputeRequest, ViewComputeResult
from directivity import DIRECTIVITY_SCHEMA_VERSION, DirectivityResult
from interferer_sampling import (
    INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION,
    INTERFERER_RESPONSE_SCHEMA_VERSION,
    InterfererGreatCircleCut,
    InterfererResponseComparison,
    calculate_interferer_great_circle_cuts,
    calculate_interferer_response_comparisons,
)
from pattern_sampling import (
    GREAT_CIRCLE_CUT_SCHEMA_VERSION,
    PATTERN_CUT_SCHEMA_VERSION,
    SURFACE_PATTERN_SCHEMA_VERSION,
    GreatCircleCuts,
    PatternCuts,
    SurfacePattern,
    calculate_great_circle_cuts,
    calculate_pattern_cuts,
    calculate_surface_pattern,
)
from simulation import (
    SimulationConfig,
    SimulationState,
    build_simulation_state,
    calculate_state_directivity,
)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_view_result(
    request: ViewComputeRequest,
    *,
    _executor: ComputeExecutor,
    _session_id: str,
    _timeout_seconds: float,
    _cancel_check,
) -> ViewComputeResult:
    """Cache one complete view while allowing inline or Process Pool execution."""

    check_current_computation()
    result = _executor.execute(
        request,
        session_id=_session_id,
        timeout_seconds=_timeout_seconds,
        cancel_check=_cancel_check,
    )
    check_current_computation()
    return result


@st.cache_data(max_entries=32, show_spinner=False)
def cached_state(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
) -> SimulationState:
    check_current_computation()
    state = build_simulation_state(
        config,
        current_azimuth_deg=azimuth_deg,
        current_elevation_deg=elevation_deg,
        cancel_check=check_current_computation,
    )
    check_current_computation()
    return state


@st.cache_data(max_entries=24, show_spinner=False)
def cached_pattern_cuts(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> PatternCuts:
    if schema_version != PATTERN_CUT_SCHEMA_VERSION:
        raise ValueError("Unsupported cached pattern-cut schema version.")
    return calculate_pattern_cuts(
        cached_state(config, azimuth_deg, elevation_deg),
        cancel_check=check_current_computation,
    )


@st.cache_data(max_entries=24, show_spinner=False)
def cached_great_circle_cuts(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> GreatCircleCuts:
    if schema_version != GREAT_CIRCLE_CUT_SCHEMA_VERSION:
        raise ValueError("Unsupported cached great-circle schema version.")
    return calculate_great_circle_cuts(
        cached_state(config, azimuth_deg, elevation_deg),
        cancel_check=check_current_computation,
    )


def _without_null_steering(config: SimulationConfig) -> SimulationConfig:
    return replace(
        config,
        enable_null_steering=False,
        maximum_element_amplitude=None,
    )


@st.cache_data(max_entries=24, show_spinner=False)
def cached_interferer_response_comparisons(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> tuple[InterfererResponseComparison, ...]:
    if schema_version != INTERFERER_RESPONSE_SCHEMA_VERSION:
        raise ValueError("Unsupported cached interferer-response schema version.")
    if not config.enable_null_steering:
        return ()
    return calculate_interferer_response_comparisons(
        cached_state(config, azimuth_deg, elevation_deg),
        cached_state(_without_null_steering(config), azimuth_deg, elevation_deg),
        cancel_check=check_current_computation,
    )


@st.cache_data(max_entries=16, show_spinner=False)
def cached_interferer_great_circle_cuts(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> tuple[InterfererGreatCircleCut, ...]:
    if schema_version != INTERFERER_GREAT_CIRCLE_SCHEMA_VERSION:
        raise ValueError("Unsupported cached interferer great-circle schema version.")
    if not config.enable_null_steering:
        return ()
    state = cached_state(config, azimuth_deg, elevation_deg)
    baseline_state = cached_state(
        _without_null_steering(config),
        azimuth_deg,
        elevation_deg,
    )
    comparisons = cached_interferer_response_comparisons(
        config,
        azimuth_deg,
        elevation_deg,
        INTERFERER_RESPONSE_SCHEMA_VERSION,
    )
    return calculate_interferer_great_circle_cuts(
        state,
        baseline_state,
        comparisons=comparisons,
        cancel_check=check_current_computation,
    )


@st.cache_data(max_entries=16, show_spinner=False)
def cached_directivity(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> DirectivityResult:
    if schema_version != DIRECTIVITY_SCHEMA_VERSION:
        raise ValueError("Unsupported cached directivity schema version.")
    return calculate_state_directivity(
        cached_state(config, azimuth_deg, elevation_deg),
        cancel_check=check_current_computation,
    )


@st.cache_data(max_entries=8, show_spinner=False)
def cached_surface_pattern(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    resolution: int,
    local_sample_count: int,
    schema_version: int,
) -> SurfacePattern:
    if schema_version != SURFACE_PATTERN_SCHEMA_VERSION:
        raise ValueError("Unsupported cached surface-pattern schema version.")
    return calculate_surface_pattern(
        cached_state(config, azimuth_deg, elevation_deg),
        resolution=resolution,
        local_sample_count=local_sample_count,
        cancel_check=check_current_computation,
    )
