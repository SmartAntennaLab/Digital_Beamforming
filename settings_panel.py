"""Persistent settings, sidebar controls, and scan policy UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import streamlit as st

from beamforming import get_steering_limits
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
from model_options import (
    COORDINATE_LABELS,
    ELEMENT_PATTERN_LABELS,
    GEOMETRY_LABELS,
    PHASE_BIT_LABELS,
    SCALE_LABELS,
    TAPER_LABELS,
    option_label,
)
from resource_policy import ResourcePolicy, resource_limit_message
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
    resource_error: str | None


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


def render_settings_panel(policy: ResourcePolicy) -> SettingsPanelResult:
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
        format_func=lambda value: option_label(value, GEOMETRY_LABELS),
        key="array_geometry",
        on_change=handle_geometry_change,
        persist_state="session",
    )
    azimuth_only_geometry = geometry in {"ULA", "UCA"}
    is_uha = geometry == "UHA"
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
            on_click=request_device_settings_save,
        )

    st.sidebar.caption(
        "입력값은 서버가 아닌 현재 기기의 이 브라우저에만 저장됩니다."
    )
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
        resource_error=resource_error,
    )
