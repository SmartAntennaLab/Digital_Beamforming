"""Radiation-pattern figures and the 2D/3D pattern tab."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pattern_sampling import GreatCircleCuts, PatternCuts, SurfacePattern
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


def render_pattern_tab(
    state: SimulationState,
    cuts: PatternCuts,
    great_circle_cuts: GreatCircleCuts,
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
    null_azimuth = (
        tuple(
            float(np.degrees(azimuth))
            for azimuth, _ in state.weight_result.null_directions_rad
        )
        if state.config.enable_null_steering
        else ()
    )
    null_elevation = (
        tuple(
            float(np.degrees(elevation))
            for _, elevation in state.weight_result.null_directions_rad
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


__all__ = ["pattern_figure", "render_pattern_tab"]
