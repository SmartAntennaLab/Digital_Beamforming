"""Streamlit renderers for diagnostics, patterns, metrics, and elements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from beamforming import normalize_pattern_linear
from exporters import build_export_artifacts
from simulation import (
    SURFACE_PATTERN_SCHEMA_VERSION,
    PatternCuts,
    SimulationState,
    SurfacePattern,
    summarize_array_layout,
)
from simulation_cache import cached_surface_pattern
from ui_formatters import (
    format_absolute_residual,
    format_angle_metric,
    format_degradation,
    format_depth,
    format_residual_db,
    format_response_error,
    format_sidelobe_metric,
    null_solver_label,
    wavelength_cm_text,
)


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
    beamwidth_label = f"3dB BW: {format_angle_metric(metrics.hpbw_deg)}"
    figure = go.Figure()

    if coordinate_option == "polar":
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
    st.caption(
        f"전역 {cuts.base_sample_count}개 각도 표본에 목표 방향 투영 개구 기반 "
        f"국부 {cuts.local_sample_count}개 표본을 추가했습니다 "
        f"(실제 Azimuth {cuts.azimuth_angles_rad.size}개, "
        f"Elevation {cuts.elevation_angles_rad.size}개; 세분화 반폭 "
        f"Az ±{cuts.azimuth_refinement_half_width_deg:.2f}°, "
        f"El ±{cuts.elevation_refinement_half_width_deg:.2f}°). "
        "각 컷은 정확한 목표 각도를 포함합니다."
    )

    st.divider()
    st.markdown("#### 3D 빔 패턴 (Spherical Surface)")
    surface = cached_surface_pattern(
        state.config,
        state.current_azimuth_deg,
        state.current_elevation_deg,
        SURFACE_PATTERN_SCHEMA_VERSION,
    )
    if getattr(surface, "schema_version", 0) != SURFACE_PATTERN_SCHEMA_VERSION:
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
    if scale_option == "db":
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
    relative_array_gain = (
        f"{gain.relative_array_gain_db:.2f} dB"
        if gain.relative_array_gain_db is not None
        else "N/A (유효 가중치 없음)"
    )
    with st.container(horizontal=True):
        st.metric("상대 배열 이득", relative_array_gain, border=True)
        st.metric(
            "활성 소자",
            f"{gain.active_elements} / {gain.total_elements}",
            border=True,
        )
        st.metric(
            "테이퍼 효율",
            f"{100.0 * gain.taper_efficiency:.2f} %",
            border=True,
        )
        st.metric(
            "위상·조향 효율",
            f"{100.0 * gain.phase_efficiency:.2f} %",
            border=True,
        )

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

        continuous_diagnostics = weight_result.continuous_diagnostics
        final_diagnostics = weight_result.final_diagnostics
        st.caption(
            f"제약 해법: {null_solver_label(weight_result.solver_method)}. "
            "상대 잔차는 요청한 목표 응답 크기로 정규화하며, 양자화 열화는 "
            "양수일수록 제약이 나빠졌음을 뜻합니다."
        )
        with st.container(horizontal=True):
            st.metric(
                "목표 응답 오차 · 연속해",
                format_response_error(continuous_diagnostics),
                border=True,
                help="절대 오차와 요청 목표 응답 대비 상대 오차입니다.",
            )
            st.metric(
                "목표 응답 오차 · 최종",
                format_response_error(final_diagnostics),
                delta=(
                    format_degradation(
                        weight_result.quantization_target_degradation_db
                    )
                    if weight_result.phase_quantization_applied
                    else None
                ),
                delta_color="inverse",
                border=True,
                help="최종 위상 양자화 후 절대/상대 목표 응답 오차입니다.",
            )
            st.metric(
                "전체 제약 잔차 열화",
                (
                    format_degradation(
                        weight_result.quantization_constraint_degradation_db
                    )
                    if weight_result.phase_quantization_applied
                    else "양자화 미적용"
                ),
                border=True,
            )

        with st.container(horizontal=True):
            st.metric(
                "연속해 최대 진폭",
                f"{continuous_diagnostics.max_amplitude:.6g}",
                border=True,
            )
            st.metric(
                "최종 최대 진폭",
                f"{final_diagnostics.max_amplitude:.6g}",
                border=True,
            )
            st.metric(
                "연속해 총 가중치 전력",
                f"{continuous_diagnostics.total_weight_power:.6g}",
                border=True,
                help="Σ|wₙ|²",
            )
            st.metric(
                "최종 총 가중치 전력",
                f"{final_diagnostics.total_weight_power:.6g}",
                border=True,
                help="Σ|wₙ|²",
            )

        null_rows = []
        for index, (azimuth_rad, elevation_rad) in enumerate(
            weight_result.null_directions_rad
        ):
            continuous_absolute = (
                continuous_diagnostics.null_constraint_residuals[index]
            )
            final_absolute = final_diagnostics.null_constraint_residuals[index]
            continuous_relative = (
                continuous_diagnostics.null_relative_residuals[index]
            )
            final_relative = final_diagnostics.null_relative_residuals[index]
            degradation = weight_result.quantization_null_degradation_db[index]
            null_rows.append(
                {
                    "제약": f"Null {index + 1}",
                    "방향": (
                        f"Az {np.degrees(azimuth_rad):.3f}°, "
                        f"El {np.degrees(elevation_rad):.3f}°"
                    ),
                    "연속 절대 잔차": (
                        format_absolute_residual(continuous_absolute)
                    ),
                    "연속 상대 잔차": format_residual_db(continuous_relative),
                    "최종 절대 잔차": (
                        format_absolute_residual(final_absolute)
                    ),
                    "최종 상대 잔차": format_residual_db(final_relative),
                    "양자화 열화": (
                        format_degradation(degradation)
                        if weight_result.phase_quantization_applied
                        else "미적용"
                    ),
                    "최종 Null 깊이": format_depth(
                        weight_result.null_depths_db[index]
                    ),
                }
            )
        if null_rows:
            st.dataframe(
                pd.DataFrame(null_rows),
                hide_index=True,
                width="stretch",
            )

    azimuth_metrics = cuts.azimuth_metrics
    elevation_metrics = cuts.elevation_metrics
    st.divider()
    width_columns = st.columns(2)
    width_columns[0].metric(
        "3dB Beamwidth (Azimuth)",
        format_angle_metric(azimuth_metrics.hpbw_deg),
    )
    width_columns[1].metric(
        "3dB Beamwidth (Elevation)",
        format_angle_metric(elevation_metrics.hpbw_deg),
    )
    null_width_columns = st.columns(2)
    null_width_columns[0].metric(
        "First Null Bandwidth (Azimuth)",
        format_angle_metric(azimuth_metrics.first_null_beamwidth_deg),
    )
    null_width_columns[1].metric(
        "First Null Bandwidth (Elevation)",
        format_angle_metric(elevation_metrics.first_null_beamwidth_deg),
    )
    sidelobe_columns = st.columns(2)
    sidelobe_columns[0].metric(
        "Sidelobe Level (Azimuth)",
        format_sidelobe_metric(
            azimuth_metrics.sidelobe_level_db,
            azimuth_metrics.sidelobe_angle_deg,
        ),
    )
    sidelobe_columns[1].metric(
        "Sidelobe Level (Elevation)",
        format_sidelobe_metric(
            elevation_metrics.sidelobe_level_db,
            elevation_metrics.sidelobe_angle_deg,
        ),
    )
    st.info(
        "상대 배열 이득은 N_eff의 coherent combining 지표로 테이퍼·위상 "
        "양자화·null 조향 손실을 반영합니다. 전구 패턴을 적분한 "
        "directivity나 실제 안테나 이득(dBi)은 아닙니다."
    )

    export_artifacts = build_export_artifacts(state, cuts)
    st.divider()
    st.subheader("💾 데이터 내보내기")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "📊 2D 빔 패턴 CSV 다운로드",
        export_artifacts.pattern_csv,
        file_name="beam_pattern_data.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[1].download_button(
        "📄 설계 리포트 다운로드",
        export_artifacts.design_report,
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
        st.metric(
            "요청 결함률",
            f"{layout.requested_failure_rate_percent:.2f}%",
            border=True,
        )
        st.metric(
            "실제 결함률",
            f"{layout.actual_failure_rate_percent:.2f}%",
            border=True,
        )

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
        "소자 자체의 물리적 직경은 포함하지 않습니다. 결함 소자 수는 "
        "요청값 N×rate/100을 0.5에서 올림하는 round-half-up 정책으로 정합니다."
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
