"""Bounded Streamlit caches around pure simulation orchestration."""

from __future__ import annotations

import streamlit as st

from compute_governor import check_current_computation
from directivity import DIRECTIVITY_SCHEMA_VERSION, DirectivityResult
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
