"""Automatic scan controls, sampling preview, and timing estimates."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from beamforming import SteeringLimits
from device_settings import SCAN_MODE_OPTIONS
from model_options import SCAN_MODE_LABELS, option_label
from resource_policy import ResourcePolicy, estimate_element_count
from settings_storage import request_device_settings_save
from pattern_sampling import scan_surface_sampling
from simulation import estimate_scan_timing


@dataclass(frozen=True)
class ScanControlsResult:
    azimuth_range: tuple[float, float]
    azimuth_steps: int
    elevation_range: tuple[float, float]
    elevation_steps: int
    scan_delay: float
    scan_mode: str


def _format_duration(seconds: float) -> str:
    """Format a short, approximate duration for the scan controls."""

    if seconds < 1.0:
        return f"{seconds:.2f}초"
    if seconds < 60.0:
        return f"{seconds:.1f}초"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}분 {remaining_seconds}초"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}시간 {remaining_minutes}분"


def render_scan_controls(
    policy: ResourcePolicy,
    *,
    geometry: str,
    vertical_count: int,
    horizontal_count: int,
    steering_limits: SteeringLimits,
) -> ScanControlsResult:
    with st.sidebar.expander("📡 자동 빔 스캔", expanded=False):
        with st.form("scan_settings", border=False):
            azimuth_range = st.slider(
                "Azimuth 스캔 범위 (°)",
                -90.0,
                90.0,
                step=1.0,
                disabled=not steering_limits.azimuth_controllable,
                key="scan_azimuth_range",
                persist_state="session",
            )
            azimuth_steps = st.slider(
                "Azimuth 스텝 수",
                3,
                50,
                disabled=not steering_limits.azimuth_controllable,
                key="scan_azimuth_steps",
                persist_state="session",
            )
            elevation_range = st.slider(
                "Elevation 스캔 범위 (°)",
                -90.0,
                90.0,
                step=1.0,
                disabled=not steering_limits.elevation_controllable,
                key="scan_elevation_range",
                persist_state="session",
            )
            elevation_steps = st.slider(
                "Elevation 스텝 수",
                2,
                20,
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
            scan_mode = st.segmented_control(
                "스캔 표시 모드",
                SCAN_MODE_OPTIONS,
                format_func=lambda value: option_label(value, SCAN_MODE_LABELS),
                key="scan_mode",
                persist_state="session",
                help=(
                    "2D 전용은 스캔 중 3D를 생략하고, 3D 미리보기는 낮은 "
                    "해상도를 사용합니다. 전체 품질 3D는 모든 프레임을 정상 "
                    "해상도로 계산합니다."
                ),
            )
            if scan_mode is None:
                scan_mode = "preview_3d"

            estimated_azimuth_steps = (
                azimuth_steps if steering_limits.azimuth_controllable else 1
            )
            estimated_elevation_steps = (
                elevation_steps if steering_limits.elevation_controllable else 1
            )
            estimated_frames = estimated_azimuth_steps * estimated_elevation_steps
            estimated_elements = estimate_element_count(
                geometry,
                vertical_count,
                horizontal_count,
            )
            scan_estimate = estimate_scan_timing(
                estimated_elements,
                estimated_frames,
                scan_mode,
                scan_delay,
                session_calculations_per_minute=(
                    policy.session_calculations_per_minute
                ),
                session_burst=policy.session_burst,
            )
            sampling = scan_surface_sampling(
                estimated_elements,
                scan_mode,
                scanning=True,
            )
            if sampling.quality == "2d":
                quality_text = "3D 계산 생략"
            elif sampling.quality == "preview":
                quality_text = (
                    f"3D 미리보기 {sampling.resolution}×{sampling.resolution} + "
                    f"지역 {sampling.local_sample_count}표본"
                )
            else:
                quality_text = "전체 품질 3D"
            estimate_text = (
                f"{quality_text} · 프레임 계산 약 "
                f"{_format_duration(scan_estimate.frame_seconds)} · "
                f"{estimated_frames:,}프레임 총 약 "
                f"{_format_duration(scan_estimate.total_seconds)}"
            )
            if scan_estimate.finalization_seconds > 0.0:
                estimate_text += (
                    " (종료 후 전체 품질 재계산 약 "
                    f"{_format_duration(scan_estimate.finalization_seconds)} 포함)"
                )
            st.caption(estimate_text)
            st.caption(
                "시간은 64×64 전체 품질 1프레임≈0.85초 기준의 경험적 "
                "추정치이며 현재 세션 계산 빈도 정책을 포함합니다. 실제 시간은 "
                "CPU, 다른 사용자 수와 활성 탭에 따라 달라집니다."
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
        st.session_state.scan_show_last_frame = False
    if stop_scan:
        st.session_state.is_scanning = False
    return ScanControlsResult(
        azimuth_range=azimuth_range,
        azimuth_steps=azimuth_steps,
        elevation_range=elevation_range,
        elevation_steps=elevation_steps,
        scan_delay=scan_delay,
        scan_mode=scan_mode,
    )


__all__ = ["ScanControlsResult", "render_scan_controls"]
