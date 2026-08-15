"""Section renderers for the staged simulator settings form."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import streamlit as st

from device_settings import (
    COORDINATE_OPTIONS,
    ELEMENT_OPTIONS,
    GEOMETRY_OPTIONS,
    PHASE_BIT_OPTIONS,
    SCALE_OPTIONS,
    TAPER_OPTIONS,
)
from directivity_controls import render_directivity_mode_control
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
    DRAFT_NULL_COUNT_KEY,
    render_null_activation_control,
    render_null_count_control,
)


@dataclass(frozen=True)
class InputAreaControls:
    geometry: str
    draft_null_enabled: bool
    draft_null_count: int


@dataclass(frozen=True)
class BasicSettings:
    frequency_ghz: float
    vertical_count: int
    horizontal_count: int
    horizontal_spacing_wavelength: float
    vertical_spacing_wavelength: float
    taper_option: str
    element_option: str
    elevation_locked: bool


@dataclass(frozen=True)
class SteeringSettings:
    target_azimuth: float
    target_elevation: float


@dataclass(frozen=True)
class HardwareSettings:
    phase_bits: int | None
    failure_rate: int
    enable_amplitude_limit: bool
    max_element_amplitude: float


@dataclass(frozen=True)
class VisualizationSettings:
    scale_option: str
    coordinate_option: str
    show_3db: bool
    show_3db_value: bool


@dataclass(frozen=True)
class AdvancedSettings:
    directivity_mode: str
    null_optimization_mode: str
    null_optimizer_tolerance: float
    null_optimizer_max_iterations: int
    null_optimizer_restart_count: int


def render_input_area_controls(
    on_geometry_change: Callable[[], None],
) -> InputAreaControls:
    """Render controls that must rerun immediately to change visible inputs."""

    with st.sidebar.container(border=True):
        st.markdown("**입력 영역 제어**")
        geometry = st.selectbox(
            "안테나 배열 형상",
            GEOMETRY_OPTIONS,
            format_func=lambda value: option_label(value, GEOMETRY_LABELS),
            key="array_geometry",
            on_change=on_geometry_change,
            persist_state="session",
        )
        draft_null_enabled = render_null_activation_control()
        if draft_null_enabled:
            draft_null_count = render_null_count_control()
        else:
            draft_null_count = int(
                st.session_state.get(
                    DRAFT_NULL_COUNT_KEY,
                    st.session_state.get("null_count", 1),
                )
            )
            st.caption("활성화하면 간섭원 방향과 억압량 입력이 표시됩니다.")

        applied_null_enabled = bool(st.session_state.get("enable_null", False))
        if draft_null_enabled != applied_null_enabled:
            st.caption("변경 대기 중 · 아래 적용 버튼을 누르면 결과에 반영됩니다.")

    st.sidebar.caption(
        "형상과 Null 표시 상태는 입력 영역만 즉시 바꾸며, 계산은 적용 버튼을 "
        "누른 뒤 갱신됩니다."
    )
    return InputAreaControls(
        geometry=geometry,
        draft_null_enabled=draft_null_enabled,
        draft_null_count=draft_null_count,
    )


def render_basic_section(geometry: str) -> BasicSettings:
    """Render array shape details, frequency, element counts, and spacing."""

    azimuth_only_geometry = geometry in {"ULA", "UCA"}
    is_uha = geometry == "UHA"
    with st.expander(
        "1. 기본 설정",
        expanded=True,
        icon=":material/grid_view:",
    ):
        st.caption(f"배열 형상 · {option_label(geometry, GEOMETRY_LABELS)}")
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

    elevation_locked = azimuth_only_geometry or (
        is_uha and horizontal_count == vertical_count
    )
    return BasicSettings(
        frequency_ghz=float(frequency_ghz),
        vertical_count=int(vertical_count),
        horizontal_count=int(horizontal_count),
        horizontal_spacing_wavelength=float(horizontal_spacing_wavelength),
        vertical_spacing_wavelength=float(vertical_spacing_wavelength),
        taper_option=taper_option,
        element_option=element_option,
        elevation_locked=elevation_locked,
    )


def render_steering_section(elevation_locked: bool) -> SteeringSettings:
    """Render the target beam direction controls."""

    with st.expander(
        "2. 조향 설정",
        expanded=True,
        icon=":material/explore:",
    ):
        target_azimuth = st.slider(
            "목표 Azimuth 각도 (°)",
            -90.0,
            90.0,
            step=1.0,
            key="target_azimuth",
            persist_state="session",
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
    return SteeringSettings(
        target_azimuth=float(target_azimuth),
        target_elevation=float(target_elevation),
    )


def _render_interferer(null_index: int, elevation_locked: bool) -> None:
    st.markdown(f"**간섭원 {null_index}**")
    if null_index == 1:
        azimuth_label = "간섭 Azimuth 각도 (°)"
        elevation_label = "간섭 Elevation 각도 (°)"
        azimuth_key = "null_azimuth"
        elevation_key = "null_elevation"
        fixed_elevation_key = "fixed_null_elevation"
    else:
        azimuth_label = f"간섭원 {null_index} Azimuth 각도 (°)"
        elevation_label = f"간섭원 {null_index} Elevation 각도 (°)"
        azimuth_key = f"null_{null_index}_azimuth"
        elevation_key = f"null_{null_index}_elevation"
        fixed_elevation_key = f"fixed_null_{null_index}_elevation"

    st.slider(
        azimuth_label,
        -90.0,
        90.0,
        step=1.0,
        key=azimuth_key,
        persist_state="session",
    )
    if elevation_locked:
        st.slider(
            elevation_label,
            -90.0,
            90.0,
            0.0,
            1.0,
            disabled=True,
            key=fixed_elevation_key,
            help="이 배열 형상에서는 간섭 Elevation을 0°로 고정합니다.",
        )
    else:
        st.slider(
            elevation_label,
            -90.0,
            90.0,
            step=1.0,
            key=elevation_key,
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


def render_null_section(
    *,
    enabled: bool,
    count: int,
    elevation_locked: bool,
) -> None:
    """Render interferer inputs only while the draft Null switch is enabled."""

    with st.expander(
        "3. Null 설정",
        expanded=enabled,
        icon=":material/block:",
    ):
        if not enabled:
            st.caption(
                "입력 영역 제어에서 영점 조향을 활성화하면 간섭원 방향과 "
                "요구 억압량을 설정할 수 있습니다."
            )
            return
        st.caption(f"현재 입력할 간섭원 · {count}개")
        for null_index in range(1, count + 1):
            _render_interferer(null_index, elevation_locked)


def render_hardware_section(null_enabled: bool) -> HardwareSettings:
    """Render fault, phase quantization, and amplitude realism controls."""

    with st.expander(
        "4. 하드웨어 현실성",
        expanded=False,
        icon=":material/memory:",
    ):
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
        enable_amplitude_limit = st.checkbox(
            "최대 소자 진폭 제한",
            disabled=not null_enabled,
            key="enable_amplitude_limit",
            persist_state="session",
            help="Null 가중치에 적용할 정규화 진폭 상한입니다.",
        )
        max_element_amplitude = st.number_input(
            "최대 소자 진폭 |w|max",
            min_value=0.05,
            max_value=10.0,
            step=0.05,
            format="%.2f",
            disabled=not (null_enabled and enable_amplitude_limit),
            key="max_element_amplitude",
            persist_state="session",
            help="복소 가중치의 정규화 진폭 상한입니다.",
        )
        if not null_enabled:
            st.caption("진폭 제한은 영점 조향을 활성화할 때 사용할 수 있습니다.")
    return HardwareSettings(
        phase_bits=phase_bits,
        failure_rate=int(failure_rate),
        enable_amplitude_limit=bool(enable_amplitude_limit),
        max_element_amplitude=float(max_element_amplitude),
    )


def render_visualization_section() -> VisualizationSettings:
    """Render display scale, coordinates, and beamwidth annotations."""

    with st.expander(
        "5. 시각화 설정",
        expanded=False,
        icon=":material/visibility:",
    ):
        scale_option = st.segmented_control(
            "3D 빔 패턴 스케일",
            SCALE_OPTIONS,
            format_func=lambda value: option_label(value, SCALE_LABELS),
            key="scale_option",
            persist_state="session",
        )
        coordinate_option = st.segmented_control(
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
    return VisualizationSettings(
        scale_option=scale_option or "db",
        coordinate_option=coordinate_option or "polar",
        show_3db=bool(show_3db),
        show_3db_value=bool(show_3db_value),
    )


def render_advanced_section(null_enabled: bool) -> AdvancedSettings:
    """Render calculation mode and optional Null optimizer controls."""

    with st.expander(
        "6. 고급 계산·스캔 설정",
        expanded=False,
        icon=":material/calculate:",
    ):
        directivity_mode = render_directivity_mode_control()
        if null_enabled:
            st.markdown("**Null 최적화**")
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
            null_optimizer_max_iterations = int(
                st.number_input(
                    "최대 반복 횟수",
                    min_value=50,
                    max_value=2000,
                    step=50,
                    key="null_optimizer_max_iterations",
                    persist_state="session",
                    help="반복 최적화의 restart별 상한입니다.",
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
                    help="고정된 여러 초기 위상에서 비선형 최적화를 반복합니다.",
                )
            )
        else:
            st.caption("Null 최적화 입력은 영점 조향을 활성화할 때 표시됩니다.")
            null_optimization_mode = str(
                st.session_state.get("null_optimization_mode", "amplitude_phase")
            )
            null_optimizer_max_iterations = int(
                st.session_state.get("null_optimizer_max_iterations", 400)
            )
            null_optimizer_tolerance = float(
                st.session_state.get("null_optimizer_tolerance", 1e-8)
            )
            null_optimizer_restart_count = int(
                st.session_state.get("null_optimizer_restart_count", 4)
            )
        st.caption("자동 빔 스캔 범위와 실행은 적용 버튼 아래에서 설정합니다.")

    return AdvancedSettings(
        directivity_mode=directivity_mode,
        null_optimization_mode=null_optimization_mode,
        null_optimizer_tolerance=null_optimizer_tolerance,
        null_optimizer_max_iterations=null_optimizer_max_iterations,
        null_optimizer_restart_count=null_optimizer_restart_count,
    )


__all__ = [
    "AdvancedSettings",
    "BasicSettings",
    "HardwareSettings",
    "InputAreaControls",
    "SteeringSettings",
    "VisualizationSettings",
    "render_advanced_section",
    "render_basic_section",
    "render_hardware_section",
    "render_input_area_controls",
    "render_null_section",
    "render_steering_section",
    "render_visualization_section",
]
