"""Element-layout metrics and per-element phase/amplitude rendering."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from beamforming import normalize_pattern_linear
from simulation import SimulationState, summarize_array_layout
from ui_formatters import array_size_text, spacing_text, wavelength_cm_text


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
        "수직 행 간격" if state.coordinates.geometry == "UHA" else "수직 소자 간격"
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
    saturated = state.weight_result.saturated_element_mask.ravel()[physical] & active
    phases = np.degrees(state.weight_result.final_phases).ravel()[physical] % 360.0
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
    if np.any(saturated):
        figure.add_trace(
            go.Scatter(
                x=y[saturated],
                y=z[saturated],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=24.0 + 20.0 * normalized_amplitudes[saturated],
                    color="darkorange",
                    line=dict(width=3, color="darkorange"),
                ),
                hovertext=labels[saturated],
                hoverinfo="text",
                name="진폭 포화 소자",
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
        "실제 복소 가중치 진폭을 나타냅니다. 주황색 외곽선은 진폭 포화 소자입니다."
    )
    st.plotly_chart(figure, width="stretch")


__all__ = ["render_elements_tab"]
