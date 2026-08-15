"""Compact summary of the configuration currently driving result views."""

from __future__ import annotations

import streamlit as st

from model_options import (
    DIRECTIVITY_MODE_LABELS,
    GEOMETRY_LABELS,
    SCAN_MODE_LABELS,
    option_label,
)
from resource_policy import estimate_element_count
from simulation import SimulationConfig


def _array_size_text(config: SimulationConfig) -> str:
    if config.geometry == "UPA":
        return f"{config.vertical_count} × {config.horizontal_count}"
    if config.geometry == "UHA":
        return f"Nmax {config.horizontal_count} / Nmin {config.vertical_count}"
    return f"N {config.horizontal_count}"


def render_calculation_summary(
    config: SimulationConfig,
    *,
    scan_mode: str,
    scanning: bool,
) -> None:
    """Render the applied settings directly above the result tabs."""

    element_count = estimate_element_count(
        config.geometry,
        config.vertical_count,
        config.horizontal_count,
    )
    null_text = (
        f"간섭원 {len(config.null_constraints_deg)}개"
        if config.enable_null_steering
        else "사용 안 함"
    )
    phase_text = (
        "무한 해상도" if config.phase_bits is None else f"{config.phase_bits}-bit"
    )
    amplitude_text = (
        "진폭 제한 없음"
        if config.maximum_element_amplitude is None
        else f"|w|max {config.maximum_element_amplitude:.2f}"
    )
    execution_text = "자동 스캔 중" if scanning else "단일 방향 계산"
    model_flags = []
    if config.element_pattern_grid is not None:
        model_flags.append("실측 소자 패턴")
    if config.mutual_coupling_db is not None:
        model_flags.append("상호 결합")
    if config.position_error_rms_wavelength > 0.0:
        model_flags.append("위치 오차")
    if config.amplitude_error_rms_db > 0.0 or config.phase_error_rms_deg > 0.0:
        model_flags.append("보정 오차")
    if config.wideband_bandwidth_percent > 0.0:
        model_flags.append("Wideband")
    if config.near_field_focus_range_m is not None:
        model_flags.append("Near-field")
    if config.enable_channel_analysis:
        model_flags.append("채널/SINR")
    if config.adaptive_beamforming_method != "none":
        model_flags.append(config.adaptive_beamforming_method.upper())

    with st.container(border=True, key="current_calculation_summary"):
        with st.container(
            horizontal=True,
            vertical_alignment="center",
            gap="small",
        ):
            st.markdown("#### 현재 계산 조건")
            st.badge(
                "현재 적용값",
                icon=":material/check_circle:",
                color="green",
            )
            st.badge(
                execution_text,
                icon=":material/radar:" if scanning else ":material/target:",
                color="primary",
            )
        st.markdown(
            f"**배열** {option_label(config.geometry, GEOMETRY_LABELS)} · "
            f"{_array_size_text(config)} · {element_count:,}개 · "
            f"{config.frequency_ghz:.1f} GHz  \n"
            f"**조향** Az {config.target_azimuth_deg:.1f}° / "
            f"El {config.target_elevation_deg:.1f}° · **Null** {null_text}  \n"
            f"**하드웨어** 위상 {phase_text} · 결함률 "
            f"{config.failure_rate_percent:.0f}% · {amplitude_text}"
        )
        st.caption(
            "계산 모드 · Directivity "
            f"{option_label(config.directivity_mode, DIRECTIVITY_MODE_LABELS)} · "
            f"스캔 {option_label(scan_mode, SCAN_MODE_LABELS)}"
        )
        st.caption(
            f"실제 시스템 모델: {', '.join(model_flags) if model_flags else '정적 협대역 기본 모델'} · "
            f"난수 시드 {config.random_seed}"
        )


__all__ = ["render_calculation_summary"]
