"""Radiation-pattern figures and the 2D/3D pattern tab."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from interferer_sampling import InterfererGreatCircleCut
from pattern_sampling import (
    GreatCircleCuts,
    PatternCuts,
    SurfacePattern,
)
from simulation import SimulationState
from ui_formatters import format_angle_metric


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
    null_angle_deg: float | Sequence[float] | None,
    axis_range_deg: tuple[float, float] = (-90.0, 90.0),
) -> go.Figure:
    angles_deg = np.degrees(angles_rad)
    left = metrics.hpbw_left_index
    right = metrics.hpbw_right_index
    beamwidth_label = f"3dB BW: {format_angle_metric(metrics.hpbw_deg)}"
    figure = go.Figure()
    if null_angle_deg is None:
        null_angles_deg: tuple[float, ...] = ()
    elif np.isscalar(null_angle_deg):
        null_angles_deg = (float(null_angle_deg),)
    else:
        null_angles_deg = tuple(float(value) for value in null_angle_deg)

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
        for null_index, interferer_angle in enumerate(null_angles_deg, start=1):
            figure.add_trace(
                go.Scatterpolar(
                    r=[-40.0, 0.0],
                    theta=[interferer_angle, interferer_angle],
                    mode="lines",
                    line=dict(color="red", width=1.5, dash="dash"),
                    name=f"Interferer {null_index}",
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
        for null_index, interferer_angle in enumerate(null_angles_deg, start=1):
            figure.add_vline(
                x=interferer_angle,
                line_width=1.5,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Interferer {null_index}",
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
                go.Scatter(x=[None], y=[None], mode="markers", name=beamwidth_label)
            )
        figure.update_layout(
            xaxis=dict(title=f"{axis_name} (°)", range=list(axis_range_deg)),
            yaxis=dict(title="Normalized Gain (dB)", range=[-40, 0]),
            margin=dict(l=30, r=30, t=35, b=30),
            height=450,
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        )
    return figure


def interferer_comparison_figure(cut: InterfererGreatCircleCut) -> go.Figure:
    """Overlay Null-off/on array responses on the exact interferer plane."""

    offsets_deg = np.degrees(cut.offsets_rad)
    floor_db = -80.0
    comparison = cut.comparison
    interferer_offset_deg = float(np.degrees(comparison.angular_distance_rad))
    figure = go.Figure()
    for label, values, color, dash in (
        ("적용 전", cut.before_pattern_db, "gray", "dash"),
        ("적용 후", cut.after_pattern_db, "purple", "solid"),
    ):
        figure.add_trace(
            go.Scatter(
                x=offsets_deg,
                y=np.maximum(values, floor_db),
                customdata=values,
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
                name=label,
                hovertemplate=(
                    "실제 각거리: %{x:.2f}°<br>"
                    "목표 대비 배열 응답: %{customdata:.2f} dB<extra>"
                    + label
                    + "</extra>"
                ),
            )
        )
    figure.add_vline(
        x=interferer_offset_deg,
        line_width=1.5,
        line_dash="dot",
        line_color="red",
        annotation_text=f"간섭원 {comparison.interferer_index}",
    )
    figure.add_trace(
        go.Scatter(
            x=[interferer_offset_deg, interferer_offset_deg],
            y=[
                max(comparison.before_relative_db, floor_db),
                max(comparison.after_relative_db, floor_db),
            ],
            customdata=[
                comparison.before_relative_db,
                comparison.after_relative_db,
            ],
            mode="markers",
            marker=dict(color=["gray", "purple"], size=9),
            name="간섭 방향 정확값",
            hovertemplate=(
                "간섭 방향: %{x:.2f}°<br>"
                "목표 대비 배열 응답: %{customdata:.2f} dB<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis=dict(
            title="목표 방향으로부터의 부호 있는 실제 각거리 (°)",
            range=[-180.0, 180.0],
        ),
        yaxis=dict(title="목표 대비 배열 응답 (dB)", range=[floor_db, 5.0]),
        margin=dict(l=30, r=30, t=25, b=30),
        height=420,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    return figure


def render_pattern_tab(
    state: SimulationState,
    cuts: PatternCuts,
    great_circle_cuts: GreatCircleCuts,
    interferer_great_circle_cuts: tuple[InterfererGreatCircleCut, ...],
    *,
    coordinate_option: str,
    scale_option: str,
    show_band: bool,
    show_band_value: bool,
    render_3d: bool,
    surface: SurfacePattern | None,
    surface_quality: str,
) -> None:
    st.subheader("방사 패턴 (Radiation Pattern)")
    left_column, right_column = st.columns(2)
    target_azimuth_rad = np.radians(state.current_azimuth_deg)
    target_elevation_rad = np.radians(state.current_elevation_deg)
    null_azimuth = (
        tuple(
            float(np.degrees(azimuth))
            for azimuth, elevation in state.weight_result.null_directions_rad
            if np.isclose(elevation, target_elevation_rad, atol=1e-10)
        )
        if state.config.enable_null_steering
        else ()
    )
    null_elevation = (
        tuple(
            float(np.degrees(elevation))
            for azimuth, elevation in state.weight_result.null_directions_rad
            if np.isclose(
                np.angle(np.exp(1j * (azimuth - target_azimuth_rad))),
                0.0,
                atol=1e-10,
            )
        )
        if state.config.enable_null_steering
        else ()
    )
    with left_column:
        st.markdown("#### 좌표각 컷 · Azimuth")
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
        st.markdown("#### 좌표각 컷 · Elevation")
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
    st.markdown("#### 목표 방향 Great-circle 주평면 컷")
    horizontal_column, vertical_column = st.columns(2)
    with horizontal_column:
        st.markdown("##### 수평 주평면 · 실제 각거리")
        st.plotly_chart(
            pattern_figure(
                great_circle_cuts.horizontal_offsets_rad,
                great_circle_cuts.horizontal_pattern_db,
                great_circle_cuts.horizontal_metrics,
                axis_name="Signed angular distance",
                color="seagreen",
                coordinate_option=coordinate_option,
                show_band=show_band,
                show_band_value=show_band_value,
                null_angle_deg=None,
                axis_range_deg=(-180.0, 180.0),
            ),
            width="stretch",
        )
    with vertical_column:
        st.markdown("##### 수직 주평면 · 실제 각거리")
        st.plotly_chart(
            pattern_figure(
                great_circle_cuts.vertical_offsets_rad,
                great_circle_cuts.vertical_pattern_db,
                great_circle_cuts.vertical_metrics,
                axis_name="Signed angular distance",
                color="darkorange",
                coordinate_option=coordinate_option,
                show_band=show_band,
                show_band_value=show_band_value,
                null_angle_deg=None,
                axis_range_deg=(-180.0, 180.0),
            ),
            width="stretch",
        )
    st.caption(
        "위 두 컷의 가로축은 목표 방향으로부터의 부호 있는 구면 각거리입니다. "
        "따라서 큰 조향각에서도 1°가 실제 구면상 1°를 뜻합니다. "
        f"전역 {great_circle_cuts.base_sample_count}개 표본과 목표 주변 "
        f"국부 {great_circle_cuts.local_sample_count}개 표본을 사용했습니다."
    )

    if interferer_great_circle_cuts:
        st.divider()
        st.markdown("#### 간섭원 전용 Great-circle · Null 적용 전후 비교")
        st.caption(
            "각 컷은 목표 방향(0°)과 실제 간섭 방향을 정확히 지나는 구면 대원입니다. "
            "동일한 배열·조향·결함·위상 양자화 조건에서 Null만 끈 결과를 적용 전으로 "
            "사용합니다. 그래프는 배열 응답을 각 상태의 목표 응답으로 정규화하며, "
            "-80 dB 아래 값은 화면 바닥에 표시하고 hover에는 실제 값을 제공합니다."
        )
        for cut in interferer_great_circle_cuts:
            comparison = cut.comparison
            azimuth_deg = np.degrees(comparison.azimuth_rad)
            elevation_deg = np.degrees(comparison.elevation_rad)
            with st.expander(
                f"간섭원 {comparison.interferer_index} · "
                f"Az {azimuth_deg:.1f}°, El {elevation_deg:.1f}°",
                expanded=comparison.interferer_index == 1,
            ):
                with st.container(horizontal=True):
                    st.metric(
                        "적용 전 상대 응답",
                        f"{comparison.before_relative_db:.2f} dB",
                        border=True,
                    )
                    st.metric(
                        "적용 후 상대 응답",
                        f"{comparison.after_relative_db:.2f} dB",
                        border=True,
                    )
                    st.metric(
                        "추가 억압량",
                        f"{comparison.additional_suppression_db:.2f} dB",
                        border=True,
                    )
                st.plotly_chart(interferer_comparison_figure(cut), width="stretch")
                st.caption(
                    f"전역 {cut.base_sample_count}개 표본과 목표·간섭 방향 주변 "
                    f"각 {cut.local_sample_count}개 국부 표본을 사용했습니다 "
                    f"(세분화 반폭 ±{cut.refinement_half_width_deg:.2f}°)."
                )

    st.divider()
    st.markdown("#### 3D 빔 패턴 (Spherical Surface)")
    if not render_3d:
        st.info(
            "2D 전용 스캔 중에는 3D 표면 계산을 생략합니다. "
            "스캔을 중지하거나 완료하면 마지막 방향을 전체 품질 3D로 계산합니다."
        )
        return
    if surface is None:
        raise ValueError("3D rendering requires a calculated surface pattern.")
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
    quality_label = "미리보기" if surface_quality == "preview" else "전체 품질"
    st.caption(
        f"{quality_label}: 전역 {base_resolution}×{base_resolution} 격자에 목표 방향 "
        f"적응형 표본을 추가했습니다 (실제 격자: {surface.pattern.shape[0]} × "
        f"{surface.pattern.shape[1]}). 정규화 전 최대 |AF·E|="
        f"{sampled_peak_magnitude:.3f}, 목표 방향 |AF·E|="
        f"{target_response_magnitude:.3f}; 3D 색상 최대는 정규화된 0 dB입니다."
    )


__all__ = [
    "interferer_comparison_figure",
    "pattern_figure",
    "render_pattern_tab",
]
