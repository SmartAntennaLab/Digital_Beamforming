"""Streamlit UI for the digital beamforming simulator."""

from __future__ import annotations

import streamlit as st

from resource_policy import ResourcePolicy
from settings_panel import render_settings_panel
from simulation import PATTERN_CUT_SCHEMA_VERSION, scan_direction
from simulation_cache import cached_pattern_cuts, cached_state
from ui_renderers import (
    render_diagnostics,
    render_elements_tab,
    render_metrics_tab,
    render_pattern_tab,
)


RESOURCE_POLICY = ResourcePolicy.from_environment()


st.set_page_config(page_title="Digital Beamforming Simulator", layout="wide")



panel = render_settings_panel(RESOURCE_POLICY)
config = panel.config
scale_option = panel.scale_option
coordinate_option = panel.coordinate_option
show_3db = panel.show_3db
show_3db_value = panel.show_3db_value
azimuth_range = panel.azimuth_range
azimuth_steps = panel.azimuth_steps
elevation_range = panel.elevation_range
elevation_steps = panel.elevation_steps
scan_delay = panel.scan_delay
resource_error = panel.resource_error


tab_labels = ["📊 빔 패턴 (2D/3D)", "🔍 성능 지표", "🔴 안테나 배치 및 위상"]
pattern_tab, metrics_tab, elements_tab = st.tabs(
    tab_labels,
    key="active_result_tab",
    on_change="rerun",
)

fragment_interval = scan_delay if st.session_state.is_scanning else None


@st.fragment(run_every=fragment_interval)
def render_active_result(view_name: str) -> None:
    if resource_error is not None:
        st.error(
            "계산 요청이 리소스 정책을 초과했습니다. 배열 크기 또는 스캔 "
            f"스텝을 줄이세요. {resource_error}"
        )
        return
    scanning = bool(st.session_state.is_scanning)
    scan_index = int(st.session_state.scan_idx)
    if scanning:
        total_steps = azimuth_steps * elevation_steps
        scan_index = min(scan_index, total_steps - 1)
        current_azimuth, current_elevation, total_steps = scan_direction(
            scan_index,
            azimuth_range,
            elevation_range,
            azimuth_steps,
            elevation_steps,
        )
        st.info(
            f"🔄 자동 스캔: Az {current_azimuth:.1f}°, El {current_elevation:.1f}° "
            f"({scan_index + 1}/{total_steps})"
        )
        st.progress((scan_index + 1) / total_steps)
    else:
        current_azimuth = config.target_azimuth_deg
        current_elevation = config.target_elevation_deg
        total_steps = 0

    with st.spinner("활성 탭 계산 중…", show_time=True):
        state = cached_state(config, current_azimuth, current_elevation)
        render_diagnostics(state)
        if view_name == "pattern":
            cuts = cached_pattern_cuts(
                config,
                current_azimuth,
                current_elevation,
                PATTERN_CUT_SCHEMA_VERSION,
            )
            render_pattern_tab(
                state,
                cuts,
                coordinate_option=coordinate_option,
                scale_option=scale_option,
                show_band=show_3db,
                show_band_value=show_3db_value,
            )
        elif view_name == "metrics":
            cuts = cached_pattern_cuts(
                config,
                current_azimuth,
                current_elevation,
                PATTERN_CUT_SCHEMA_VERSION,
            )
            render_metrics_tab(state, cuts)
        else:
            render_elements_tab(state)

    if scanning:
        if scan_index < total_steps - 1:
            st.session_state.scan_idx = scan_index + 1
        else:
            st.session_state.is_scanning = False
            st.session_state.scan_completed = True
            # One app rerun releases the fragment timer after the final frame.
            st.rerun(scope="app")


# Streamlit 1.60 dynamic tabs expose `.open`; only the visible branch executes.
if pattern_tab.open:
    with pattern_tab:
        render_active_result("pattern")
elif metrics_tab.open:
    with metrics_tab:
        render_active_result("metrics")
elif elements_tab.open:
    with elements_tab:
        render_active_result("elements")
