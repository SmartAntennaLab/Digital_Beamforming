"""Bounded Streamlit caches around pure simulation orchestration."""

from __future__ import annotations

import streamlit as st

from simulation import (
    PATTERN_CUT_SCHEMA_VERSION,
    SURFACE_PATTERN_SCHEMA_VERSION,
    PatternCuts,
    SimulationConfig,
    SimulationState,
    SurfacePattern,
    build_simulation_state,
    calculate_pattern_cuts,
    calculate_surface_pattern,
)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_state(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
) -> SimulationState:
    return build_simulation_state(
        config,
        current_azimuth_deg=azimuth_deg,
        current_elevation_deg=elevation_deg,
    )


@st.cache_data(max_entries=24, show_spinner=False)
def cached_pattern_cuts(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> PatternCuts:
    if schema_version != PATTERN_CUT_SCHEMA_VERSION:
        raise ValueError("Unsupported cached pattern-cut schema version.")
    return calculate_pattern_cuts(cached_state(config, azimuth_deg, elevation_deg))


@st.cache_data(max_entries=8, show_spinner=False)
def cached_surface_pattern(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> SurfacePattern:
    if schema_version != SURFACE_PATTERN_SCHEMA_VERSION:
        raise ValueError("Unsupported cached surface-pattern schema version.")
    return calculate_surface_pattern(
        cached_state(config, azimuth_deg, elevation_deg)
    )
