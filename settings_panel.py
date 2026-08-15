"""Persistent sidebar settings orchestration and scan policy UI."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from advanced_model_controls import render_real_system_section
from beamforming import get_steering_limits
from directivity_controls import render_directivity_policy_notice
from null_controls import (
    applied_null_constraints,
    apply_draft_null_activation,
    apply_draft_null_count,
)
from provenance import APP_VERSION
from resource_policy import ResourcePolicy, resource_limit_message
from scan_controls import render_scan_controls
from settings_sections import (
    render_advanced_section,
    render_basic_section,
    render_hardware_section,
    render_input_area_controls,
    render_null_section,
    render_steering_section,
    render_visualization_section,
)
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
    """Promote immediate Null controls before persisting the form."""

    apply_draft_null_activation()
    apply_draft_null_count()
    request_device_settings_save()


def _stop_scan() -> None:
    st.session_state.is_scanning = False
    st.session_state.scan_idx = 0
    st.session_state.scan_completed = False
    st.session_state.scan_show_last_frame = False


def _render_settings_notice() -> None:
    settings_notice = st.session_state.pop("_settings_notice", None)
    if not settings_notice:
        return
    notice_level, notice_text = settings_notice
    if notice_level == "success":
        st.sidebar.success(notice_text)
    elif notice_level == "warning":
        st.sidebar.warning(notice_text)
    else:
        st.sidebar.info(notice_text)


def render_settings_panel(policy: ResourcePolicy) -> SettingsPanelResult:
    initialize_settings_storage()

    st.title(f"📡 디지털 빔포밍 시뮬레이터 v{APP_VERSION}")
    st.markdown(
        "배열·조향 조건을 제출한 뒤 활성 탭의 빔 패턴과 안테나 상태를 확인하세요."
    )
    if st.session_state.pop("scan_completed", False):
        st.success("🔄 설정한 전체 영역의 자동 스캔을 완료했습니다.")

    st.sidebar.header("⚙️ 시뮬레이터 입력 설정")
    _render_settings_notice()
    input_controls = render_input_area_controls(_stop_scan)
    geometry = input_controls.geometry
    draft_null_enabled = input_controls.draft_null_enabled

    with st.sidebar.form("simulation_settings", border=False):
        basic = render_basic_section(geometry)
        steering = render_steering_section(basic.elevation_locked)
        render_null_section(
            enabled=draft_null_enabled,
            count=input_controls.draft_null_count,
            elevation_locked=basic.elevation_locked,
        )
        hardware = render_hardware_section(draft_null_enabled)
        visualization = render_visualization_section()
        advanced = render_advanced_section(draft_null_enabled)
        real_system = render_real_system_section(null_enabled=draft_null_enabled)
        apply_settings = st.form_submit_button(
            "설정 적용 및 계산",
            type="primary",
            width="stretch",
            icon=":material/play_arrow:",
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

    azimuth_only_geometry = geometry in {"ULA", "UCA"}
    is_uha = geometry == "UHA"
    effective_vertical_count = (
        2 * (basic.horizontal_count - basic.vertical_count) + 1
        if is_uha
        else basic.vertical_count
    )
    steering_limits = get_steering_limits(
        effective_vertical_count,
        basic.horizontal_count,
        geometry,
    )
    null_constraints = applied_null_constraints(st.session_state)
    target_azimuth = steering.target_azimuth
    target_elevation = steering.target_elevation
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
        vertical_count=basic.vertical_count,
        horizontal_count=basic.horizontal_count,
        steering_limits=steering_limits,
    )

    render_directivity_policy_notice(
        policy,
        mode=advanced.directivity_mode,
        geometry=geometry,
        vertical_count=basic.vertical_count,
        horizontal_count=basic.horizontal_count,
    )

    enable_null = bool(st.session_state.get("enable_null", False))
    maximum_element_amplitude = (
        hardware.max_element_amplitude
        if enable_null and hardware.enable_amplitude_limit
        else None
    )
    config = SimulationConfig(
        frequency_ghz=basic.frequency_ghz,
        vertical_count=basic.vertical_count,
        horizontal_count=basic.horizontal_count,
        horizontal_spacing_wavelength=basic.horizontal_spacing_wavelength,
        vertical_spacing_wavelength=basic.vertical_spacing_wavelength,
        geometry=geometry,
        taper_option=basic.taper_option,
        element_option=basic.element_option,
        directivity_mode=advanced.directivity_mode,
        directivity_warning_elements=policy.directivity_warning_elements,
        directivity_exact_max_elements=policy.directivity_exact_max_elements,
        phase_bits=hardware.phase_bits,
        failure_rate_percent=hardware.failure_rate,
        target_azimuth_deg=target_azimuth,
        target_elevation_deg=target_elevation,
        enable_null_steering=enable_null,
        null_azimuth_deg=null_azimuth,
        null_elevation_deg=null_elevation,
        null_constraints_deg=tuple(null_constraints),
        null_optimization_mode=advanced.null_optimization_mode,
        maximum_element_amplitude=maximum_element_amplitude,
        null_optimizer_tolerance=advanced.null_optimizer_tolerance,
        null_optimizer_max_iterations=advanced.null_optimizer_max_iterations,
        null_optimizer_restart_count=advanced.null_optimizer_restart_count,
        random_seed=real_system.random_seed,
        position_error_rms_wavelength=real_system.position_error_rms_wavelength,
        amplitude_error_rms_db=real_system.amplitude_error_rms_db,
        phase_error_rms_deg=real_system.phase_error_rms_deg,
        mutual_coupling_db=real_system.mutual_coupling_db,
        mutual_coupling_phase_deg=real_system.mutual_coupling_phase_deg,
        polarization_angle_deg=real_system.polarization_angle_deg,
        element_pattern_grid=real_system.element_pattern_grid,
        wideband_bandwidth_percent=real_system.wideband_bandwidth_percent,
        wideband_frequency_samples=real_system.wideband_frequency_samples,
        near_field_focus_range_m=real_system.near_field_focus_range_m,
        enable_channel_analysis=real_system.enable_channel_analysis,
        channel_snapshots=real_system.channel_snapshots,
        multipath_count=real_system.multipath_count,
        signal_power_dbm=real_system.signal_power_dbm,
        interference_power_dbm=real_system.interference_power_dbm,
        noise_power_dbm=real_system.noise_power_dbm,
        adaptive_beamforming_method=real_system.adaptive_beamforming_method,
        diagonal_loading=real_system.diagonal_loading,
        enable_doa_estimation=real_system.enable_doa_estimation,
        golden_dataset=real_system.golden_dataset,
    )
    requested_scan_frames = (
        scan_controls.azimuth_steps * scan_controls.elevation_steps
        if st.session_state.is_scanning
        else 1
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
    if real_system.input_error is not None:
        resource_error = real_system.input_error
        st.sidebar.error(resource_error)

    return SettingsPanelResult(
        config=config,
        scale_option=visualization.scale_option,
        coordinate_option=visualization.coordinate_option,
        show_3db=visualization.show_3db,
        show_3db_value=visualization.show_3db_value,
        azimuth_range=scan_controls.azimuth_range,
        azimuth_steps=scan_controls.azimuth_steps,
        elevation_range=scan_controls.elevation_range,
        elevation_steps=scan_controls.elevation_steps,
        scan_delay=scan_controls.scan_delay,
        scan_mode=scan_controls.scan_mode,
        resource_error=resource_error,
    )
