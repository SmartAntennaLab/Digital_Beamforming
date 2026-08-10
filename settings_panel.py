"""Persistent settings, sidebar controls, and scan policy UI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from beamforming import get_steering_limits
from device_settings import (
    COORDINATE_OPTIONS,
    ELEMENT_OPTIONS,
    GEOMETRY_OPTIONS,
    PHASE_BIT_OPTIONS,
    SCALE_OPTIONS,
    TAPER_OPTIONS,
)
from directivity_controls import (
    render_directivity_mode_control,
    render_directivity_policy_notice,
)
from model_options import (
    COORDINATE_LABELS,
    ELEMENT_PATTERN_LABELS,
    GEOMETRY_LABELS,
    NULL_OPTIMIZATION_MODE_LABELS,
    NULL_OPTIMIZATION_MODE_OPTIONS,
    PHASE_BIT_LABELS,
    SCALE_LABELS,
    TAPER_LABELS,
    option_label,
)
from null_controls import (
    applied_null_constraints,
    apply_draft_null_count,
    render_null_count_control,
)
from resource_policy import ResourcePolicy, resource_limit_message
from scan_controls import render_scan_controls
from settings_storage import (
    initialize_settings_storage,
    request_device_settings_clear,
    request_device_settings_save,
    request_share_link,
)
from simulation import SimulationConfig


@dataclass(frozen=True)
class SettingsPanelResult:
    config: SimulationConfig
    scale_option: str
    coordinate_option: str
    show_3db: bool
    show_3db_value: bool
    azimuth_range: tuple[float, float]
    azimuth_steps: int
    elevation_range: tuple[float, float]
    elevation_steps: int
    scan_delay: float
    scan_mode: str
    resource_error: str | None


def _apply_settings_and_save() -> None:
    """Apply the draft interferer count before persisting submitted settings."""

    apply_draft_null_count()
    request_device_settings_save()


def render_settings_panel(policy: ResourcePolicy) -> SettingsPanelResult:
    initialize_settings_storage()

    def handle_geometry_change() -> None:
        """Stop an active sweep before applying an immediately changed geometry."""

        st.session_state.is_scanning = False
        st.session_state.scan_idx = 0
        st.session_state.scan_completed = False
        st.session_state.scan_show_last_frame = False

    st.title("📡 디지털 빔포밍 시뮬레이터 v1.5")
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
        format_func=lambda value: option_label(value, GEOMETRY_LABELS),
        key="array_geometry",
        on_change=handle_geometry_change,
        persist_state="session",
    )
    azimuth_only_geometry = geometry in {"ULA", "UCA"}
    is_uha = geometry == "UHA"
    draft_null_count = render_null_count_control()
    st.sidebar.caption(
        "간섭원 수 변경은 입력 영역만 바꾸며, 아래 적용 버튼을 눌러야 계산됩니다."
    )
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
            format_func=lambda value: option_label(value, TAPER_LABELS),
            key="taper_option",
            persist_state="session",
        )
        element_option = st.selectbox(
            "안테나 소자 패턴",
            ELEMENT_OPTIONS,
            format_func=lambda value: option_label(value, ELEMENT_PATTERN_LABELS),
            key="element_option",
            persist_state="session",
        )
        directivity_mode = render_directivity_mode_control()
        phase_bits = st.selectbox(
            "위상 천이기 해상도",
            PHASE_BIT_OPTIONS,
            format_func=lambda value: option_label(value, PHASE_BIT_LABELS),
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
        st.markdown("##### 간섭원 1")
        st.slider(
            "간섭 Azimuth 각도 (°)",
            -90.0,
            90.0,
            step=1.0,
            key="null_azimuth",
            persist_state="session",
        )
        if elevation_locked:
            st.slider(
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
            st.slider(
                "간섭 Elevation 각도 (°)",
                -90.0,
                90.0,
                step=1.0,
                key="null_elevation",
                persist_state="session",
            )
        st.slider(
            "간섭원 1 요구 억압량 (dB)",
            0.0,
            120.0,
            step=1.0,
            key="null_1_suppression_db",
            persist_state="session",
        )
        for null_index in range(2, draft_null_count + 1):
            st.markdown(f"##### 간섭원 {null_index}")
            st.slider(
                f"간섭원 {null_index} Azimuth 각도 (°)",
                -90.0,
                90.0,
                step=1.0,
                key=f"null_{null_index}_azimuth",
                persist_state="session",
            )
            if elevation_locked:
                st.slider(
                    f"간섭원 {null_index} Elevation 각도 (°)",
                    -90.0,
                    90.0,
                    0.0,
                    1.0,
                    disabled=True,
                    key=f"fixed_null_{null_index}_elevation",
                    help="이 배열 형상에서는 간섭 Elevation을 0°로 고정합니다.",
                )
            else:
                st.slider(
                    f"간섭원 {null_index} Elevation 각도 (°)",
                    -90.0,
                    90.0,
                    step=1.0,
                    key=f"null_{null_index}_elevation",
                    persist_state="session",
                )
            st.slider(
                f"간섭원 {null_index} 요구 억압량 (dB)",
                0.0,
                120.0,
                step=1.0,
                key=f"null_{null_index}_suppression_db",
                persist_state="session",
            )
        null_optimization_mode = st.selectbox(
            "Null 최적화 방식",
            NULL_OPTIMIZATION_MODE_OPTIONS,
            format_func=lambda value: option_label(
                value,
                NULL_OPTIMIZATION_MODE_LABELS,
            ),
            key="null_optimization_mode",
            persist_state="session",
            help=(
                "위상 전용은 테이퍼 진폭을 고정하고 위상만 반복 최적화합니다. "
                "진폭·위상 방식은 SVD 복소 가중치를 사용합니다."
            ),
        )
        enable_amplitude_limit = st.checkbox(
            "최대 소자 진폭 제한",
            key="enable_amplitude_limit",
            persist_state="session",
        )
        max_element_amplitude_value = st.number_input(
            "최대 소자 진폭 |w|max",
            min_value=0.05,
            max_value=10.0,
            step=0.05,
            format="%.2f",
            disabled=not enable_amplitude_limit,
            key="max_element_amplitude",
            persist_state="session",
            help="복소 가중치의 정규화 진폭 상한입니다. 상한에 도달한 소자를 별도 표시합니다.",
        )
        maximum_element_amplitude = (
            float(max_element_amplitude_value)
            if enable_null and enable_amplitude_limit
            else None
        )
        with st.expander("Null 최적화 고급 설정", expanded=False):
            null_optimizer_max_iterations = int(
                st.number_input(
                    "최대 반복 횟수",
                    min_value=50,
                    max_value=2000,
                    step=50,
                    key="null_optimizer_max_iterations",
                    persist_state="session",
                    help="위상 전용 또는 진폭 제한 반복 최적화의 restart별 상한입니다.",
                )
            )
            null_optimizer_tolerance = float(
                st.select_slider(
                    "수렴 허용오차",
                    options=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
                    format_func=lambda value: f"{value:.0e}",
                    key="null_optimizer_tolerance",
                    persist_state="session",
                    help="목적함수 개선량, gradient 또는 투영 이동량의 종료 기준입니다.",
                )
            )
            null_optimizer_restart_count = int(
                st.number_input(
                    "Deterministic restart 수",
                    min_value=1,
                    max_value=8,
                    step=1,
                    disabled=null_optimization_mode != "phase_only",
                    key="null_optimizer_restart_count",
                    persist_state="session",
                    help="위상 전용 비선형 최적화를 고정된 여러 초기 위상에서 반복합니다.",
                )
            )
        st.markdown("##### 시각화")
        scale_option = st.radio(
            "3D 빔 패턴 스케일",
            SCALE_OPTIONS,
            format_func=lambda value: option_label(value, SCALE_LABELS),
            key="scale_option",
            persist_state="session",
        )
        coordinate_option = st.radio(
            "2D 패턴 좌표계",
            COORDINATE_OPTIONS,
            format_func=lambda value: option_label(value, COORDINATE_LABELS),
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
            on_click=_apply_settings_and_save,
        )

    st.sidebar.caption("입력값은 서버가 아닌 현재 기기의 이 브라우저에만 저장됩니다.")
    st.sidebar.caption(
        f"세션 계산 상한: 소자 {policy.max_elements:,}개, 스캔 "
        f"{policy.max_scan_frames:,}프레임, 누적 "
        f"{policy.max_scan_element_frames:,} element-frames"
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
    null_constraints = applied_null_constraints(st.session_state)
    if not steering_limits.azimuth_controllable:
        target_azimuth = 0.0
        null_constraints = [
            (0.0, elevation, suppression)
            for _, elevation, suppression in null_constraints
        ]
    if not steering_limits.elevation_controllable:
        target_elevation = 0.0
        null_constraints = [
            (azimuth, 0.0, suppression) for azimuth, _, suppression in null_constraints
        ]
    null_azimuth, null_elevation, _ = null_constraints[0]
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
        st.session_state.scan_show_last_frame = False

    scan_controls = render_scan_controls(
        policy,
        geometry=geometry,
        vertical_count=vertical_count,
        horizontal_count=horizontal_count,
        steering_limits=steering_limits,
    )
    azimuth_range = scan_controls.azimuth_range
    azimuth_steps = scan_controls.azimuth_steps
    elevation_range = scan_controls.elevation_range
    elevation_steps = scan_controls.elevation_steps
    scan_delay = scan_controls.scan_delay
    scan_mode = scan_controls.scan_mode

    render_directivity_policy_notice(
        policy,
        mode=directivity_mode,
        geometry=geometry,
        vertical_count=vertical_count,
        horizontal_count=horizontal_count,
    )

    config = SimulationConfig(
        frequency_ghz=frequency_ghz,
        vertical_count=vertical_count,
        horizontal_count=horizontal_count,
        horizontal_spacing_wavelength=horizontal_spacing_wavelength,
        vertical_spacing_wavelength=vertical_spacing_wavelength,
        geometry=geometry,
        taper_option=taper_option,
        element_option=element_option,
        directivity_mode=directivity_mode,
        directivity_warning_elements=policy.directivity_warning_elements,
        directivity_exact_max_elements=policy.directivity_exact_max_elements,
        phase_bits=phase_bits,
        failure_rate_percent=failure_rate,
        target_azimuth_deg=target_azimuth,
        target_elevation_deg=target_elevation,
        enable_null_steering=enable_null,
        null_azimuth_deg=null_azimuth,
        null_elevation_deg=null_elevation,
        null_constraints_deg=tuple(null_constraints),
        null_optimization_mode=null_optimization_mode,
        maximum_element_amplitude=maximum_element_amplitude,
        null_optimizer_tolerance=null_optimizer_tolerance,
        null_optimizer_max_iterations=null_optimizer_max_iterations,
        null_optimizer_restart_count=null_optimizer_restart_count,
    )
    requested_scan_frames = (
        azimuth_steps * elevation_steps if st.session_state.is_scanning else 1
    )
    resource_error = resource_limit_message(
        policy,
        geometry=config.geometry,
        vertical_count=config.vertical_count,
        horizontal_count=config.horizontal_count,
        scan_frames=requested_scan_frames,
    )
    if resource_error is not None:
        st.session_state.is_scanning = False
        st.session_state.scan_idx = 0
        st.sidebar.error(resource_error)

    return SettingsPanelResult(
        config=config,
        scale_option=scale_option,
        coordinate_option=coordinate_option,
        show_3db=show_3db,
        show_3db_value=show_3db_value,
        azimuth_range=azimuth_range,
        azimuth_steps=azimuth_steps,
        elevation_range=elevation_range,
        elevation_steps=elevation_steps,
        scan_delay=scan_delay,
        scan_mode=scan_mode,
        resource_error=resource_error,
    )
