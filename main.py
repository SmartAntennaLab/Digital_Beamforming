"""Streamlit UI for the digital beamforming simulator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from beamforming import get_steering_limits, normalize_pattern_linear
from device_settings import (
    COORDINATE_OPTIONS,
    DEFAULT_DEVICE_SETTINGS,
    DEVICE_SETTING_KEYS,
    ELEMENT_OPTIONS,
    GEOMETRY_OPTIONS,
    PHASE_BIT_OPTIONS,
    SCALE_OPTIONS,
    TAPER_OPTIONS,
    collect_device_settings,
    decode_share_token,
    encode_share_token,
    sanitize_device_settings,
    settings_envelope,
)
from device_storage import mount_device_storage
from simulation import (
    PatternCuts,
    SimulationConfig,
    SimulationState,
    SurfacePattern,
    build_simulation_state,
    calculate_pattern_cuts,
    calculate_surface_pattern,
    scan_direction,
    summarize_array_layout,
)


SURFACE_CACHE_SCHEMA_VERSION = 5


st.set_page_config(page_title="Digital Beamforming Simulator", layout="wide")


@st.cache_data(max_entries=32, show_spinner=False)
def cached_state(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
) -> SimulationState:
    """Cache immutable frames while bounding scan-history memory."""

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
) -> PatternCuts:
    state = cached_state(config, azimuth_deg, elevation_deg)
    return calculate_pattern_cuts(state)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_surface_pattern(
    config: SimulationConfig,
    azimuth_deg: float,
    elevation_deg: float,
    schema_version: int,
) -> SurfacePattern:
    if schema_version != SURFACE_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported cached surface-pattern schema version.")
    state = cached_state(config, azimuth_deg, elevation_deg)
    return calculate_surface_pattern(state)


def format_depth(depth_db: float | None) -> str:
    if depth_db is None:
        return "N/A"
    if depth_db >= 299.95:
        return "≥ 300 dB"
    displayed = 0.0 if abs(depth_db) < 0.005 else depth_db
    return f"{displayed:.2f} dB"


def steering_axes_text(state: SimulationState) -> str:
    limits = get_steering_limits(
        state.coordinates.rows,
        state.coordinates.columns,
        state.coordinates.geometry,
    )
    if limits.azimuth_controllable and limits.elevation_controllable:
        return "Azimuth 및 Elevation"
    if limits.azimuth_controllable:
        return "Azimuth (Elevation 0° 고정)"
    if limits.elevation_controllable:
        return "Elevation (Azimuth 0° 고정)"
    return "전자 조향 불가 (Azimuth/Elevation 0° 고정)"


def spacing_text(state: SimulationState) -> str:
    """Describe only the spacings used by the effective geometry."""

    horizontal = state.config.horizontal_spacing_wavelength
    vertical = state.config.vertical_spacing_wavelength
    if state.coordinates.geometry == "UCA":
        return f"인접 chord dy={horizontal:.2f} λ"
    if state.coordinates.geometry == "ULA":
        return f"수평 dy={horizontal:.2f} λ"
    if state.coordinates.geometry == "UHA":
        row_spacing = horizontal * np.sin(np.pi / 3.0)
        return f"최근접 dy={horizontal:.2f} λ / 행 간격 dz={row_spacing:.3f} λ"
    return f"수평 dy={horizontal:.2f} λ / 수직 dz={vertical:.2f} λ"


def array_size_text(state: SimulationState) -> str:
    """Describe rectangular dimensions or UHA row-count parameters."""

    coordinates = state.coordinates
    if coordinates.geometry == "UHA":
        return (
            f"Nmin={state.config.vertical_count}, "
            f"Nmax={state.config.horizontal_count}, "
            f"{coordinates.rows}개 행 / {coordinates.element_count}개 소자"
        )
    return f"{coordinates.rows} × {coordinates.columns}"


def wavelength_cm_text(wavelength_value: float, centimeter_value: float) -> str:
    """Format one physical dimension in wavelength and centimeter units."""

    return f"{wavelength_value:.3f} λ / {centimeter_value:.3f} cm"


def render_diagnostics(state: SimulationState) -> None:
    result = state.weight_result
    config = state.config
    if config.enable_null_steering and not result.null_applied:
        rank = (
            f"{result.constraint_rank}/{result.constraint_count}"
            if result.constraint_rank is not None
            else "N/A"
        )
        condition = (
            f"{result.condition_number:.3e}"
            if result.condition_number is not None
            and np.isfinite(result.condition_number)
            else "∞"
        )
        st.warning(
            "⚠️ 영점 제약 행렬이 특이하거나 수치적으로 불안정해 기본 조향 "
            f"가중치를 사용합니다. rank={rank}, condition={condition}."
        )

    assessment = state.grating_assessment
    if not assessment.has_aliasing_risk:
        return
    if assessment.risk_only:
        st.warning(
            "⚠️ UCA 인접 chord 간격이 0.5λ를 초과해 공간 앨리어싱 위험이 "
            "있습니다. UCA에는 직교 주기 배열의 복제 각도 식을 적용하지 않습니다."
        )
        return
    directions = ", ".join(
        f"(p,q)=({item.order_y},{item.order_z}) → "
        f"Az {item.azimuth_deg:.1f}°, El {item.elevation_deg:.1f}°"
        for item in assessment.directions
    )
    st.warning(f"⚠️ 가시 영역 격자 로브가 감지되었습니다: {directions}")


def pattern_figure(
    angles_rad: np.ndarray,
    pattern_db: np.ndarray,
    metrics,
    *,
    axis_name: str,
    color: str,
    coordinate_option: str,
    show_band: bool,
    show_band_value: bool,
    null_angle_deg: float | None,
) -> go.Figure:
    angles_deg = np.degrees(angles_rad)
    left = metrics.hpbw_left_index
    right = metrics.hpbw_right_index
    beamwidth_label = f"3dB BW: {metrics.hpbw_deg:.2f}°"
    figure = go.Figure()

    if coordinate_option.startswith("Polar"):
        figure.add_trace(
            go.Scatterpolar(
                r=np.maximum(pattern_db, -40.0),
                theta=angles_deg,
                mode="lines",
                line=dict(color=color, width=2),
                name="Gain (dB)",
                hovertemplate="Angle: %{theta:.2f}°<br>Gain: %{r:.2f} dB<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatterpolar(
                r=np.full(angles_deg.shape, -3.0),
                theta=angles_deg,
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if null_angle_deg is not None:
            figure.add_trace(
                go.Scatterpolar(
                    r=[-40.0, 0.0],
                    theta=[null_angle_deg, null_angle_deg],
                    mode="lines",
                    line=dict(color="red", width=1.5, dash="dash"),
                    name="Interferer",
                )
            )
        if show_band and left is not None and right is not None:
            fill_angles = list(angles_deg[left : right + 1])
            fill_values = list(np.maximum(pattern_db[left : right + 1], -40.0))
            fill_angles += [fill_angles[-1], fill_angles[0]]
            fill_values += [-40.0, -40.0]
            figure.add_trace(
                go.Scatterpolar(
                    r=fill_values,
                    theta=fill_angles,
                    fill="toself",
                    fillcolor="rgba(30,144,255,0.15)",
                    line=dict(color=color, width=1, dash="dash"),
                    name=beamwidth_label if show_band_value else "3dB Beamwidth",
                    hoverinfo="skip",
                )
            )
        elif show_band_value and left is not None and right is not None:
            figure.add_trace(
                go.Scatterpolar(
                    r=[None],
                    theta=[None],
                    mode="markers",
                    marker=dict(opacity=0),
                    name=beamwidth_label,
                )
            )
        figure.update_layout(
            polar=dict(
                angularaxis=dict(direction="clockwise", rotation=90, ticksuffix="°"),
                radialaxis=dict(range=[-40, 0], ticksuffix=" dB"),
            ),
            margin=dict(l=30, r=30, t=35, b=30),
            height=450,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=angles_deg,
                y=np.maximum(pattern_db, -40.0),
                mode="lines",
                line=dict(color=color, width=2),
                name="Gain (dB)",
                hovertemplate="Angle: %{x:.2f}°<br>Gain: %{y:.2f} dB<extra></extra>",
            )
        )
        figure.add_hline(y=-3.0, line_width=1, line_dash="dot", line_color="gray")
        if null_angle_deg is not None:
            figure.add_vline(
                x=null_angle_deg,
                line_width=1.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Interferer",
            )
        if show_band and left is not None and right is not None:
            fill_angles = list(angles_deg[left : right + 1])
            fill_values = list(np.maximum(pattern_db[left : right + 1], -40.0))
            figure.add_trace(
                go.Scatter(
                    x=[fill_angles[0], *fill_angles, fill_angles[-1]],
                    y=[-40.0, *fill_values, -40.0],
                    fill="toself",
                    fillcolor="rgba(30,144,255,0.15)",
                    line=dict(color=color, width=1, dash="dash"),
                    name=beamwidth_label if show_band_value else "3dB Beamwidth",
                    hoverinfo="skip",
                )
            )
        elif show_band_value and left is not None and right is not None:
            figure.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers", name=beamwidth_label
                )
            )
        figure.update_layout(
            xaxis=dict(title=f"{axis_name} Angle (°)", range=[-90, 90]),
            yaxis=dict(title="Normalized Gain (dB)", range=[-40, 0]),
            margin=dict(l=30, r=30, t=35, b=30),
            height=450,
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        )
    return figure


def render_pattern_tab(
    state: SimulationState,
    cuts: PatternCuts,
    *,
    coordinate_option: str,
    scale_option: str,
    show_band: bool,
    show_band_value: bool,
) -> None:
    st.subheader("방사 패턴 (Radiation Pattern)")
    left_column, right_column = st.columns(2)
    null_azimuth = (
        state.config.null_azimuth_deg
        if state.config.enable_null_steering
        else None
    )
    null_elevation = (
        state.config.null_elevation_deg
        if state.config.enable_null_steering
        else None
    )
    with left_column:
        st.markdown("#### 2D Azimuth 빔 패턴")
        st.plotly_chart(
            pattern_figure(
                cuts.azimuth_angles_rad,
                cuts.azimuth_pattern_db,
                cuts.azimuth_metrics,
                axis_name="Azimuth",
                color="dodgerblue",
                coordinate_option=coordinate_option,
                show_band=show_band,
                show_band_value=show_band_value,
                null_angle_deg=null_azimuth,
            ),
            width="stretch",
        )
    with right_column:
        st.markdown("#### 2D Elevation 빔 패턴")
        st.plotly_chart(
            pattern_figure(
                cuts.elevation_angles_rad,
                cuts.elevation_pattern_db,
                cuts.elevation_metrics,
                axis_name="Elevation",
                color="crimson",
                coordinate_option=coordinate_option,
                show_band=show_band,
                show_band_value=show_band_value,
                null_angle_deg=null_elevation,
            ),
            width="stretch",
        )

    st.divider()
    st.markdown("#### 3D 빔 패턴 (Spherical Surface)")
    surface = cached_surface_pattern(
        state.config,
        state.current_azimuth_deg,
        state.current_elevation_deg,
        SURFACE_CACHE_SCHEMA_VERSION,
    )
    if getattr(surface, "schema_version", 0) != SURFACE_CACHE_SCHEMA_VERSION:
        # Recover safely if a long-running server restored an object serialized
        # before SurfacePattern gained its current diagnostic fields.
        cached_surface_pattern.clear()
        surface = calculate_surface_pattern(state)
    # A hot reload can retain the previous simulation module for one rerun.
    # Keep that transition renderable until the process is restarted.
    base_resolution = getattr(surface, "base_resolution", min(surface.pattern.shape))
    sampled_peak_magnitude = getattr(
        surface,
        "sampled_peak_magnitude",
        float(np.max(np.abs(surface.pattern))),
    )
    target_response_magnitude = getattr(
        surface,
        "target_response_magnitude",
        sampled_peak_magnitude,
    )
    if scale_option.startswith("dB"):
        radius = np.clip((surface.pattern_db + 30.0) / 30.0, 0.0, 1.0)
        color_data = surface.pattern_db
        color_min, color_max = -30.0, 0.0
        colorbar_title = "강도 (dB)"
        hover_text = np.char.mod("빔 강도: %.1f dB", surface.pattern_db)
        title = "3D Normalized Beam Pattern (dB Scale, Min -30 dB)"
    else:
        radius = surface.pattern_linear
        color_data = surface.pattern_linear
        color_min, color_max = 0.0, 1.0
        colorbar_title = "강도 (Linear)"
        hover_text = np.char.mod("빔 강도: %.3f", surface.pattern_linear)
        title = "3D Normalized Beam Pattern (Linear Scale)"

    polar = surface.polar_angle_rad
    azimuth = surface.azimuth_angle_rad
    x = radius * np.sin(polar) * np.cos(azimuth)
    y = radius * np.sin(polar) * np.sin(azimuth)
    z = radius * np.cos(polar)
    figure = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=color_data,
                colorscale="Viridis",
                cmin=color_min,
                cmax=color_max,
                colorbar=dict(title=colorbar_title, thickness=15),
                hovertext=hover_text,
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (Forward)", range=[-1.05, 1.05]),
            yaxis=dict(title="Y (Horizontal)", range=[-1.05, 1.05]),
            zaxis=dict(title="Z (Vertical)", range=[-1.05, 1.05]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=35),
        height=550,
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"전역 {base_resolution}×{base_resolution} 격자에 목표 방향 "
        f"적응형 표본을 추가했습니다 (실제 격자: {surface.pattern.shape[0]} × "
        f"{surface.pattern.shape[1]}). 정규화 전 최대 |AF·E|="
        f"{sampled_peak_magnitude:.3f}, 목표 방향 |AF·E|="
        f"{target_response_magnitude:.3f}; 3D 색상 최대는 정규화된 0 dB입니다."
    )


def render_metrics_tab(state: SimulationState, cuts: PatternCuts) -> None:
    st.subheader("📏 주요 성능 지표 (AESA Performance Metrics)")
    gain = state.gain_metrics
    array_gain = (
        f"{gain.array_gain_db:.2f} dBi"
        if gain.array_gain_db is not None
        else "N/A (유효 가중치 없음)"
    )
    columns = st.columns(4)
    columns[0].metric("유효 배열 이득", array_gain)
    columns[1].metric("활성 소자", f"{gain.active_elements} / {gain.total_elements}")
    columns[2].metric("테이퍼 효율", f"{100.0 * gain.taper_efficiency:.2f} %")
    columns[3].metric("위상·조향 효율", f"{100.0 * gain.phase_efficiency:.2f} %")

    weight_result = state.weight_result
    rank = (
        f"{weight_result.constraint_rank}/{weight_result.constraint_count}"
        if weight_result.constraint_rank is not None
        else "N/A"
    )
    condition = (
        f"{weight_result.condition_number:.3e}"
        if weight_result.condition_number is not None
        and np.isfinite(weight_result.condition_number)
        else "∞"
    )
    actual_null_depth = (
        weight_result.null_depths_db[0] if weight_result.null_depths_db else None
    )
    continuous_null_depth = (
        weight_result.continuous_null_depths_db[0]
        if weight_result.continuous_null_depths_db
        else None
    )
    if state.config.enable_null_steering:
        null_columns = st.columns(4)
        null_columns[0].metric("실제 Null 깊이", format_depth(actual_null_depth))
        null_columns[1].metric(
            "양자화 전 Null 깊이", format_depth(continuous_null_depth)
        )
        null_columns[2].metric(
            "Null 제약", "적용됨" if weight_result.null_applied else "적용 실패"
        )
        null_columns[3].metric("제약 rank / condition", f"{rank} / {condition}")

    azimuth_metrics = cuts.azimuth_metrics
    elevation_metrics = cuts.elevation_metrics
    st.divider()
    width_columns = st.columns(2)
    width_columns[0].metric(
        "3dB Beamwidth (Azimuth)",
        f"{azimuth_metrics.hpbw_deg:.2f} °"
        if azimuth_metrics.hpbw_deg > 0.0
        else "N/A",
    )
    width_columns[1].metric(
        "3dB Beamwidth (Elevation)",
        f"{elevation_metrics.hpbw_deg:.2f} °"
        if elevation_metrics.hpbw_deg > 0.0
        else "N/A",
    )
    null_width_columns = st.columns(2)
    null_width_columns[0].metric(
        "First Null Bandwidth (Azimuth)",
        f"{azimuth_metrics.first_null_beamwidth_deg:.2f} °",
    )
    null_width_columns[1].metric(
        "First Null Bandwidth (Elevation)",
        f"{elevation_metrics.first_null_beamwidth_deg:.2f} °",
    )
    sidelobe_columns = st.columns(2)
    sidelobe_columns[0].metric(
        "Sidelobe Level (Azimuth)",
        f"{azimuth_metrics.sidelobe_level_db:.2f} dB "
        f"(@ {azimuth_metrics.sidelobe_angle_deg:.1f}°)",
    )
    sidelobe_columns[1].metric(
        "Sidelobe Level (Elevation)",
        f"{elevation_metrics.sidelobe_level_db:.2f} dB "
        f"(@ {elevation_metrics.sidelobe_angle_deg:.1f}°)",
    )
    st.info(
        "유효 배열 이득은 실제 활성 소자와 최종 복소 가중치를 사용하며, "
        "테이퍼·위상 양자화·null 조향 손실을 반영합니다."
    )

    export_frame = pd.DataFrame(
        {
            "Angle (deg)": np.degrees(cuts.azimuth_angles_rad),
            "Azimuth Gain (dB)": cuts.azimuth_pattern_db,
            "Elevation Gain (dB)": cuts.elevation_pattern_db,
        }
    )
    report = f"""# 디지털 빔포밍 안테나 설계 리포트

## 시뮬레이션 조건

- 주파수: {state.config.frequency_ghz:.2f} GHz
- 파장: {state.wavelength_m * 1000.0:.3f} mm
- 배열: {state.coordinates.geometry}, {array_size_text(state)}
- 소자 간격: {spacing_text(state)}
- 조향 가능 축: {steering_axes_text(state)}
- 조향 방향: Az {state.current_azimuth_deg:.1f}°, El {state.current_elevation_deg:.1f}°
- 진폭 창: {state.config.taper_option}
- 소자 패턴: {state.config.element_option}
- 위상 해상도: {state.config.phase_bits}
- 결함률: {state.config.failure_rate_percent:.1f}%

## 성능 지표

- 유효 배열 이득: {array_gain}
- 활성 소자: {gain.active_elements} / {gain.total_elements}
- 테이퍼 효율: {100.0 * gain.taper_efficiency:.2f}%
- 위상·조향 효율: {100.0 * gain.phase_efficiency:.2f}%
- Azimuth HPBW / FNBW / SLL: {azimuth_metrics.hpbw_deg:.2f}° / {azimuth_metrics.first_null_beamwidth_deg:.2f}° / {azimuth_metrics.sidelobe_level_db:.2f} dB
- Elevation HPBW / FNBW / SLL: {elevation_metrics.hpbw_deg:.2f}° / {elevation_metrics.first_null_beamwidth_deg:.2f}° / {elevation_metrics.sidelobe_level_db:.2f} dB
- 실제 Null 깊이: {format_depth(actual_null_depth) if state.config.enable_null_steering else 'N/A'}
"""
    st.divider()
    st.subheader("💾 데이터 내보내기")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "📊 2D 빔 패턴 CSV 다운로드",
        export_frame.to_csv(index=False).encode("utf-8"),
        file_name="beam_pattern_data.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[1].download_button(
        "📄 설계 리포트 다운로드",
        report.encode("utf-8"),
        file_name="beamforming_design_report.md",
        mime="text/markdown",
        width="stretch",
    )


def render_elements_tab(state: SimulationState) -> None:
    st.subheader("🔴 안테나 배열 및 소자별 최종 위상·실제 진폭")
    layout = summarize_array_layout(state)
    element_count = state.coordinates.element_count
    with st.container(horizontal=True):
        st.metric("전체 소자 수", f"{layout.total_elements:,}개", border=True)
        st.metric("활성 소자 수", f"{layout.active_elements:,}개", border=True)
        st.metric("결함 소자 수", f"{layout.failed_elements:,}개", border=True)

    horizontal_spacing_label = (
        "인접 chord 간격" if state.coordinates.geometry == "UCA" else "수평 소자 간격"
    )
    vertical_spacing_label = (
        "수직 행 간격"
        if state.coordinates.geometry == "UHA"
        else "수직 소자 간격"
    )
    vertical_spacing_text = (
        wavelength_cm_text(
            layout.vertical_spacing_wavelength,
            layout.vertical_spacing_cm,
        )
        if layout.vertical_spacing_wavelength is not None
        and layout.vertical_spacing_cm is not None
        else "사용 안 함"
    )
    with st.container(horizontal=True):
        st.metric(
            horizontal_spacing_label,
            wavelength_cm_text(
                layout.horizontal_spacing_wavelength,
                layout.horizontal_spacing_cm,
            ),
            border=True,
        )
        st.metric(vertical_spacing_label, vertical_spacing_text, border=True)
        st.metric(
            "전체 수평 길이",
            wavelength_cm_text(
                layout.horizontal_extent_wavelength,
                layout.horizontal_extent_cm,
            ),
            border=True,
        )
        st.metric(
            "전체 수직 길이",
            wavelength_cm_text(
                layout.vertical_extent_wavelength,
                layout.vertical_extent_cm,
            ),
            border=True,
        )
    st.caption(
        "전체 길이는 가장 바깥쪽 소자 중심 사이의 Y/Z 투영 거리이며, "
        "소자 자체의 물리적 직경은 포함하지 않습니다."
    )

    if element_count > 1024:
        st.warning(
            f"브라우저 보호를 위해 {element_count:,}개 소자의 개별 마커 렌더링을 "
            "생략합니다. 수치 계산은 전체 소자를 사용했습니다."
        )
        st.markdown(
            f"- 배열 형상: **{state.coordinates.geometry}**\n"
            f"- 배열 크기: **{array_size_text(state)}**\n"
            f"- 활성 소자: **{state.gain_metrics.active_elements:,} / {element_count:,}**\n"
            f"- 간격: **{spacing_text(state)}**"
        )
        return

    wavelength = state.wavelength_m
    physical = state.coordinates.element_mask.ravel()
    y = (state.coordinates.y / wavelength).ravel()[physical]
    z = (state.coordinates.z / wavelength).ravel()[physical]
    y_cm = y * wavelength * 100.0
    z_cm = z * wavelength * 100.0
    active = state.active_mask.ravel()[physical]
    phases = (
        np.degrees(state.weight_result.final_phases).ravel()[physical] % 360.0
    )
    amplitudes = state.actual_amplitudes.ravel()[physical]
    normalized_amplitudes = normalize_pattern_linear(
        state.complex_weights.ravel()[physical]
    )
    rows, columns = np.indices(state.active_mask.shape)
    rows = rows.ravel()[physical]
    columns = columns.ravel()[physical]
    labels = np.asarray(
        [
            f"소자 ({row + 1}, {column + 1})<br>최종 위상: {phase:.1f}°<br>"
            f"실제 진폭 |w|: {amplitude:.4f}<br>"
            f"위치: y={y_pos:.2f}λ ({y_pos_cm:.3f} cm), "
            f"z={z_pos:.2f}λ ({z_pos_cm:.3f} cm)"
            for row, column, phase, amplitude, y_pos, z_pos, y_pos_cm, z_pos_cm in zip(
                rows,
                columns,
                phases,
                amplitudes,
                y,
                z,
                y_cm,
                z_cm,
                strict=True,
            )
        ],
        dtype=object,
    )
    figure = go.Figure()
    if np.any(active):
        figure.add_trace(
            go.Scatter(
                x=y[active],
                y=z[active],
                mode="markers+text",
                marker=dict(
                    size=16.0 + 20.0 * normalized_amplitudes[active],
                    color=phases[active],
                    colorscale="Hsv",
                    cmin=0,
                    cmax=360,
                    showscale=True,
                    colorbar=dict(title="위상 (°)", thickness=15),
                    line=dict(width=1, color="black"),
                ),
                text=[f"{phase:.0f}°" for phase in phases[active]],
                textposition="middle center",
                textfont=dict(color="white", size=9),
                hovertext=labels[active],
                hoverinfo="text",
                name="정상 소자",
            )
        )
    failed = ~active
    if np.any(failed):
        figure.add_trace(
            go.Scatter(
                x=y[failed],
                y=z[failed],
                mode="markers",
                marker=dict(symbol="x", size=18, color="gray"),
                hovertext=labels[failed],
                hoverinfo="text",
                name="결함 소자",
            )
        )
    y_margin = max(0.5, 0.05 * max(float(np.ptp(y)), 1.0))
    z_margin = max(0.5, 0.05 * max(float(np.ptp(z)), 1.0))
    figure.update_layout(
        xaxis=dict(
            title="수평 방향 (λ)",
            range=[float(np.min(y)) - y_margin, float(np.max(y)) + y_margin],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="수직 방향 (λ)",
            range=[float(np.min(z)) - z_margin, float(np.max(z)) + z_margin],
        ),
        height=600,
        margin=dict(l=10, r=10, b=10, t=30),
    )
    st.caption(
        "마커 색상·텍스트는 최종 양자화 위상, 크기는 null 제약까지 반영한 "
        "실제 복소 가중치 진폭을 나타냅니다."
    )
    st.plotly_chart(figure, width="stretch")


def apply_persistent_settings(settings: Mapping[str, object]) -> None:
    """Hydrate widget state before any persistent widget is instantiated."""

    for key, value in sanitize_device_settings(settings).items():
        st.session_state[key] = value


def next_storage_command(action: str, payload: object | None = None) -> None:
    """Queue one idempotent browser-storage command for the next rerun."""

    previous = st.session_state.get("_device_storage_command", {})
    command_id = int(previous.get("id", 0)) + 1
    command: dict[str, object] = {"id": command_id, "action": action}
    if payload is not None:
        command["payload"] = payload
    st.session_state["_device_storage_command"] = command


def request_device_settings_save() -> None:
    """Save the submitted widget state in this browser only."""

    settings = collect_device_settings(st.session_state)
    next_storage_command("save", settings_envelope(settings))
    st.session_state["_device_settings_applied"] = True
    if "settings" in st.query_params:
        st.query_params.clear()
        st.session_state["_applied_query_signature"] = None


def request_device_settings_clear() -> None:
    """Clear browser storage and return persistent widgets to defaults."""

    for key in DEVICE_SETTING_KEYS:
        st.session_state.pop(key, None)
    st.session_state["is_scanning"] = False
    st.session_state["scan_idx"] = 0
    st.session_state["_device_settings_applied"] = True
    st.session_state["_applied_query_signature"] = None
    st.query_params.clear()
    next_storage_command("clear")


def request_share_link() -> None:
    """Put a validated snapshot in one explicit URL query parameter."""

    token = encode_share_token(collect_device_settings(st.session_state))
    st.query_params.clear()
    st.query_params["settings"] = token
    st.session_state["_applied_query_signature"] = f"share:{token}"
    st.session_state["_settings_notice"] = (
        "info",
        "현재 주소가 공유 링크로 갱신되었습니다. 주소창의 URL을 복사하세요.",
    )


# Session state is deliberately small; numerical arrays live in bounded caches.
st.session_state.setdefault("is_scanning", False)
st.session_state.setdefault("scan_idx", 0)
st.session_state.setdefault("scan_completed", False)
st.session_state.setdefault(
    "_device_storage_command",
    {"id": 0, "action": "load"},
)
st.session_state.setdefault("_device_settings_applied", False)
for setting_key, default_value in DEFAULT_DEVICE_SETTINGS.items():
    st.session_state.setdefault(setting_key, default_value)

# Explicit share links take precedence over this browser's stored defaults.
share_token = st.query_params.get("settings")
query_signature: str | None = None
query_settings: dict[str, object] = {}
if isinstance(share_token, str) and share_token:
    query_signature = f"share:{share_token}"
    query_settings = decode_share_token(share_token)
else:
    legacy_query = {
        key: st.query_params.get(key)
        for key in DEVICE_SETTING_KEYS
        if key in st.query_params
    }
    if legacy_query:
        query_signature = f"legacy:{sorted(legacy_query.items())!r}"
        query_settings = sanitize_device_settings(legacy_query)

if (
    query_signature is not None
    and st.session_state.get("_applied_query_signature") != query_signature
):
    if query_settings:
        apply_persistent_settings(query_settings)
        st.session_state["_device_settings_applied"] = True
        st.session_state["_settings_notice"] = (
            "info",
            "URL에서 시뮬레이터 설정을 불러왔습니다.",
        )
        if query_signature.startswith("legacy:"):
            st.query_params.clear()
    else:
        st.session_state["_settings_notice"] = (
            "warning",
            "공유 링크 설정이 손상되었거나 지원하지 않는 형식입니다.",
        )
    st.session_state["_applied_query_signature"] = query_signature

storage_result = mount_device_storage(st.session_state["_device_storage_command"])
loaded_settings = getattr(storage_result, "loaded_settings", None)
if (
    not st.session_state["_device_settings_applied"]
    and isinstance(loaded_settings, Mapping)
):
    apply_persistent_settings(loaded_settings)
    st.session_state["_device_settings_applied"] = True

storage_status = getattr(storage_result, "status", None)
if isinstance(storage_status, Mapping):
    status_id = storage_status.get("id")
    if status_id != st.session_state.get("_device_storage_status_seen"):
        st.session_state["_device_storage_status_seen"] = status_id
        action = storage_status.get("action")
        if storage_status.get("ok") and action == "save":
            st.session_state["_settings_notice"] = (
                "success",
                "현재 설정을 이 기기의 브라우저에 저장했습니다.",
            )
        elif storage_status.get("ok") and action == "clear":
            st.session_state["_settings_notice"] = (
                "success",
                "이 기기에 저장된 설정을 삭제하고 기본값으로 초기화했습니다.",
            )
        elif not storage_status.get("ok"):
            st.session_state["_settings_notice"] = (
                "warning",
                "브라우저 저장소를 사용할 수 없습니다. 브라우저 개인정보 보호 설정을 확인하세요.",
            )


def handle_geometry_change() -> None:
    """Stop an active sweep before applying an immediately changed geometry."""

    st.session_state.is_scanning = False
    st.session_state.scan_idx = 0
    st.session_state.scan_completed = False


st.title("📡 디지털 빔포밍 시뮬레이터 v1.4")
st.markdown(
    "배열·조향 조건을 제출한 뒤 활성 탭의 빔 패턴과 안테나 상태를 확인하세요."
)
if st.session_state.pop("scan_completed", False):
    st.success("🔄 설정한 전체 영역의 자동 스캔을 완료했습니다.")

st.sidebar.header("⚙️ 시뮬레이터 입력 설정")
settings_notice = st.session_state.pop("_settings_notice", None)
if settings_notice:
    notice_level, notice_text = settings_notice
    if notice_level == "success":
        st.sidebar.success(notice_text)
    elif notice_level == "warning":
        st.sidebar.warning(notice_text)
    else:
        st.sidebar.info(notice_text)
geometry = st.sidebar.selectbox(
    "안테나 배열 형상",
    GEOMETRY_OPTIONS,
    key="array_geometry",
    on_change=handle_geometry_change,
    persist_state="session",
)
azimuth_only_geometry = geometry.startswith(("ULA", "UCA"))
is_uha = geometry.startswith("UHA")
with st.sidebar.form("simulation_settings", border=True):
    frequency_ghz = st.slider(
        "주파수 (GHz)",
        1.0,
        60.0,
        step=0.5,
        key="frequency_ghz",
        persist_state="session",
    )
    if is_uha:
        horizontal_count = st.slider(
            "중앙 행 소자 수 (Nmax)",
            1,
            128,
            key="uha_max_count",
            persist_state="session",
            help="UHA의 가장 긴 중앙 행에 배치할 소자 수입니다.",
        )
        if "uha_min_count" in st.session_state:
            st.session_state.uha_min_count = min(
                int(st.session_state.uha_min_count), horizontal_count
            )
        vertical_count = st.slider(
            "최소 행 소자 수 (Nmin)",
            1,
            horizontal_count,
            key="uha_min_count",
            persist_state="session",
            help="UHA의 최하단·최상단 행에 배치할 소자 수입니다.",
        )
    else:
        horizontal_count = st.slider(
            "수평 안테나 수 (N)",
            1,
            128,
            key="horizontal_count",
            persist_state="session",
        )
    if azimuth_only_geometry:
        vertical_count = st.slider(
            "수직 안테나 수 (M)",
            1,
            128,
            1,
            disabled=True,
            key="fixed_vertical_count",
            help="ULA/UCA에서는 수직 소자 수를 1로 고정합니다.",
        )
    elif not is_uha:
        vertical_count = st.slider(
            "수직 안테나 수 (M)",
            1,
            128,
            key="vertical_count",
            persist_state="session",
        )
    horizontal_spacing_wavelength = st.slider(
        "수평 소자 간격 (dy/λ)",
        0.1,
        1.0,
        step=0.05,
        key="horizontal_spacing",
        persist_state="session",
    )
    if is_uha:
        vertical_spacing_wavelength = float(
            horizontal_spacing_wavelength * np.sin(np.pi / 3.0)
        )
        st.number_input(
            "수직 행 간격 (dz/λ)",
            value=vertical_spacing_wavelength,
            format="%.4f",
            disabled=True,
            help="공식 UHA 삼각 격자에 따라 dz = dy × sin(60°)로 자동 계산합니다.",
        )
    else:
        vertical_spacing_wavelength = st.slider(
            "수직 소자 간격 (dz/λ)",
            0.1,
            1.0,
            step=0.05,
            disabled=azimuth_only_geometry,
            help="UPA의 Z축 간격입니다. ULA와 UCA에서는 사용하지 않습니다.",
            key="vertical_spacing",
            persist_state="session",
        )
    taper_option = st.selectbox(
        "진폭 테이퍼링",
        TAPER_OPTIONS,
        key="taper_option",
        persist_state="session",
    )
    element_option = st.selectbox(
        "안테나 소자 패턴",
        ELEMENT_OPTIONS,
        key="element_option",
        persist_state="session",
    )
    phase_bits = st.selectbox(
        "위상 천이기 해상도",
        PHASE_BIT_OPTIONS,
        key="phase_bits",
        persist_state="session",
    )
    failure_rate = st.slider(
        "안테나 소자 결함률 (%)",
        0,
        50,
        key="failure_rate",
        persist_state="session",
    )
    target_azimuth = st.slider(
        "목표 Azimuth 각도 (°)",
        -90.0,
        90.0,
        step=1.0,
        key="target_azimuth",
        persist_state="session",
    )
    elevation_locked = azimuth_only_geometry or (
        is_uha and horizontal_count == vertical_count
    )
    if elevation_locked:
        target_elevation = st.slider(
            "목표 Elevation 각도 (°)",
            -90.0,
            90.0,
            0.0,
            1.0,
            disabled=True,
            key="fixed_target_elevation",
            help=(
                "ULA/UCA 또는 한 행뿐인 UHA에서는 Elevation 조향을 0°로 "
                "고정합니다."
            ),
        )
    else:
        target_elevation = st.slider(
            "목표 Elevation 각도 (°)",
            -90.0,
            90.0,
            step=1.0,
            key="target_elevation",
            persist_state="session",
        )
    enable_null = st.checkbox(
        "영점 조향 활성화",
        key="enable_null",
        persist_state="session",
    )
    null_azimuth = st.slider(
        "간섭 Azimuth 각도 (°)",
        -90.0,
        90.0,
        step=1.0,
        key="null_azimuth",
        persist_state="session",
    )
    if elevation_locked:
        null_elevation = st.slider(
            "간섭 Elevation 각도 (°)",
            -90.0,
            90.0,
            0.0,
            1.0,
            disabled=True,
            key="fixed_null_elevation",
            help=(
                "ULA/UCA 또는 한 행뿐인 UHA에서는 간섭 Elevation을 0°로 "
                "고정합니다."
            ),
        )
    else:
        null_elevation = st.slider(
            "간섭 Elevation 각도 (°)",
            -90.0,
            90.0,
            step=1.0,
            key="null_elevation",
            persist_state="session",
        )
    st.markdown("##### 시각화")
    scale_option = st.radio(
        "3D 빔 패턴 스케일",
        SCALE_OPTIONS,
        key="scale_option",
        persist_state="session",
    )
    coordinate_option = st.radio(
        "2D 패턴 좌표계",
        COORDINATE_OPTIONS,
        key="coordinate_option",
        persist_state="session",
    )
    show_3db = st.checkbox(
        "3dB 대역폭 범위 표시",
        key="show_3db",
        persist_state="session",
    )
    show_3db_value = st.checkbox(
        "3dB 대역폭 값 표시",
        key="show_3db_value",
        persist_state="session",
    )
    apply_settings = st.form_submit_button(
        "설정 적용 및 계산",
        type="primary",
        width="stretch",
        on_click=request_device_settings_save,
    )

st.sidebar.caption(
    "입력값은 서버가 아닌 현재 기기의 이 브라우저에만 저장됩니다."
)
with st.sidebar.container(horizontal=True):
    st.button(
        "공유 링크 생성",
        icon=":material/link:",
        on_click=request_share_link,
    )
    st.button(
        "저장 설정 초기화",
        icon=":material/restart_alt:",
        on_click=request_device_settings_clear,
    )

effective_vertical_count = (
    2 * (horizontal_count - vertical_count) + 1 if is_uha else vertical_count
)
steering_limits = get_steering_limits(
    effective_vertical_count,
    horizontal_count,
    geometry,
)
if not steering_limits.azimuth_controllable:
    target_azimuth = 0.0
    null_azimuth = 0.0
if not steering_limits.elevation_controllable:
    target_elevation = 0.0
    null_elevation = 0.0
if azimuth_only_geometry:
    st.sidebar.info(
        "ULA/UCA에서는 수직 안테나 수(M)를 1, 목표·간섭 Elevation을 0°로 "
        "고정합니다."
    )
elif is_uha:
    st.sidebar.info(
        "UHA 행별 소자 수는 Nmin부터 Nmax까지 증가한 뒤 대칭으로 감소하며, "
        "행 간격은 dz = dy × sin(60°)로 자동 설정됩니다."
    )
if apply_settings:
    st.session_state.is_scanning = False
    st.session_state.scan_idx = 0

with st.sidebar.expander("📡 자동 빔 스캔", expanded=False):
    with st.form("scan_settings", border=False):
        azimuth_range = st.slider(
            "Azimuth 스캔 범위 (°)", -90.0, 90.0, step=1.0,
            disabled=not steering_limits.azimuth_controllable,
            key="scan_azimuth_range",
            persist_state="session",
        )
        azimuth_steps = st.slider(
            "Azimuth 스텝 수", 3, 50,
            disabled=not steering_limits.azimuth_controllable,
            key="scan_azimuth_steps",
            persist_state="session",
        )
        elevation_range = st.slider(
            "Elevation 스캔 범위 (°)", -90.0, 90.0, step=1.0,
            disabled=not steering_limits.elevation_controllable,
            key="scan_elevation_range",
            persist_state="session",
        )
        elevation_steps = st.slider(
            "Elevation 스텝 수", 2, 20,
            disabled=not steering_limits.elevation_controllable,
            key="scan_elevation_steps",
            persist_state="session",
        )
        scan_delay = st.slider(
            "프레임 간격 (초)",
            0.1,
            2.0,
            step=0.1,
            key="scan_delay",
            persist_state="session",
        )
        scan_buttons = st.columns(2)
        start_scan = scan_buttons[0].form_submit_button(
            "▶️ 시작",
            disabled=st.session_state.is_scanning
            or not (
                steering_limits.azimuth_controllable
                or steering_limits.elevation_controllable
            ),
            width="stretch",
            on_click=request_device_settings_save,
        )
        stop_scan = scan_buttons[1].form_submit_button(
            "⏹️ 중지",
            disabled=not st.session_state.is_scanning,
            width="stretch",
            on_click=request_device_settings_save,
        )

if not steering_limits.azimuth_controllable:
    azimuth_range, azimuth_steps = (0.0, 0.0), 1
if not steering_limits.elevation_controllable:
    elevation_range, elevation_steps = (0.0, 0.0), 1
if start_scan:
    st.session_state.is_scanning = True
    st.session_state.scan_idx = 0
if stop_scan:
    st.session_state.is_scanning = False

config = SimulationConfig(
    frequency_ghz=frequency_ghz,
    vertical_count=vertical_count,
    horizontal_count=horizontal_count,
    horizontal_spacing_wavelength=horizontal_spacing_wavelength,
    vertical_spacing_wavelength=vertical_spacing_wavelength,
    geometry=geometry,
    taper_option=taper_option,
    element_option=element_option,
    phase_bits=phase_bits,
    failure_rate_percent=failure_rate,
    target_azimuth_deg=target_azimuth,
    target_elevation_deg=target_elevation,
    enable_null_steering=enable_null,
    null_azimuth_deg=null_azimuth,
    null_elevation_deg=null_elevation,
)

tab_labels = ["📊 빔 패턴 (2D/3D)", "🔍 성능 지표", "🔴 안테나 배치 및 위상"]
pattern_tab, metrics_tab, elements_tab = st.tabs(
    tab_labels,
    key="active_result_tab",
    on_change="rerun",
)

fragment_interval = scan_delay if st.session_state.is_scanning else None


@st.fragment(run_every=fragment_interval)
def render_active_result(view_name: str) -> None:
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
            cuts = cached_pattern_cuts(config, current_azimuth, current_elevation)
            render_pattern_tab(
                state,
                cuts,
                coordinate_option=coordinate_option,
                scale_option=scale_option,
                show_band=show_3db,
                show_band_value=show_3db_value,
            )
        elif view_name == "metrics":
            cuts = cached_pattern_cuts(config, current_azimuth, current_elevation)
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
