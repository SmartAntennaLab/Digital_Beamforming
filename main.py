"""Streamlit UI for the digital beamforming simulator."""

from __future__ import annotations

import uuid

import streamlit as st

from compute_governor import (
    ComputeBusyError,
    ComputeCancelled,
    ComputeDeadlineExceeded,
    SessionRateLimitError,
    get_compute_governor,
)
from directivity import DIRECTIVITY_SCHEMA_VERSION
from model_options import SCAN_MODE_LABELS, option_label
from resource_policy import ResourcePolicy
from settings_panel import render_settings_panel
from pattern_sampling import (
    GREAT_CIRCLE_CUT_SCHEMA_VERSION,
    PATTERN_CUT_SCHEMA_VERSION,
    SURFACE_PATTERN_SCHEMA_VERSION,
    calculate_surface_pattern,
    scan_surface_sampling,
)
from simulation import scan_direction
from simulation_cache import (
    cached_directivity,
    cached_great_circle_cuts,
    cached_pattern_cuts,
    cached_state,
    cached_surface_pattern,
)
from ui_elements import render_elements_tab
from ui_metrics import render_metrics_tab
from ui_pattern import render_pattern_tab
from ui_renderers import render_diagnostics


RESOURCE_POLICY = ResourcePolicy.from_environment()
COMPUTE_GOVERNOR = get_compute_governor(RESOURCE_POLICY)


st.set_page_config(page_title="Digital Beamforming Simulator", layout="wide")
st.session_state.setdefault("_compute_session_id", uuid.uuid4().hex)
compute_session_id = str(st.session_state["_compute_session_id"])


def request_compute_cancel() -> None:
    """Cancel an in-flight lease and suppress the callback-triggered rerun."""

    COMPUTE_GOVERNOR.cancel_session(compute_session_id)
    st.session_state.is_scanning = False
    st.session_state["_skip_next_compute"] = True


panel = render_settings_panel(RESOURCE_POLICY)
config = panel.config
scale_option = panel.scale_option
coordinate_option = panel.coordinate_option
show_3db = panel.show_3db
show_3db_value = panel.show_3db_value
azimuth_range = panel.azimuth_range
azimuth_steps = panel.azimuth_steps
elevation_range = panel.elevation_range
elevation_steps = panel.elevation_steps
scan_delay = panel.scan_delay
scan_mode = panel.scan_mode
resource_error = panel.resource_error

st.sidebar.button(
    "현재 계산 취소",
    icon=":material/cancel:",
    width="stretch",
    on_click=request_compute_cancel,
    help=(
        "실행 중인 계산에 취소 신호를 보내고 자동 스캔을 중지합니다. "
        "NumPy 청크 경계에서 안전하게 중단됩니다."
    ),
)
compute_snapshot = COMPUTE_GOVERNOR.log_health_if_due()
with st.sidebar.expander("서버 계산 상태", expanded=False):
    st.caption(
        f"동시 계산 {compute_snapshot.active_calculations}/"
        f"{compute_snapshot.max_concurrent_calculations} · "
        f"대기 {compute_snapshot.queued_calculations}"
    )
    st.caption(
        f"프로세스 CPU {compute_snapshot.process_cpu_percent:.1f}% · "
        f"RSS {compute_snapshot.process_rss_bytes / (1024**2):.1f} MiB"
    )
    st.caption(
        f"시스템 CPU {compute_snapshot.system_cpu_percent:.1f}% · 메모리 "
        f"{compute_snapshot.system_memory_percent:.1f}%"
    )
    st.caption(
        f"완료 {compute_snapshot.completed_calculations:,} · "
        f"평균 {compute_snapshot.average_duration_seconds:.2f}초 · "
        f"혼잡 거절 {compute_snapshot.busy_rejections:,} · "
        f"빈도 제한 {compute_snapshot.rate_rejections:,} · "
        f"시간 초과 {compute_snapshot.timed_out_calculations:,} · "
        f"취소 {compute_snapshot.cancelled_calculations:,}"
    )


tab_labels = ["📊 빔 패턴 (2D/3D)", "🔍 성능 지표", "🔴 안테나 배치 및 위상"]
pattern_tab, metrics_tab, elements_tab = st.tabs(
    tab_labels,
    key="active_result_tab",
    on_change="rerun",
)

fragment_interval = scan_delay if st.session_state.is_scanning else None


@st.fragment(run_every=fragment_interval)
def render_active_result(view_name: str) -> None:
    if resource_error is not None:
        st.error(
            "계산 요청이 리소스 정책을 초과했습니다. 배열 크기 또는 스캔 "
            f"스텝을 줄이세요. {resource_error}"
        )
        return
    if st.session_state.pop("_skip_next_compute", False):
        st.info("현재 계산과 자동 스캔을 취소했습니다.")
        return
    scanning = bool(st.session_state.is_scanning)
    scan_index = int(st.session_state.scan_idx)
    if scanning:
        total_steps = azimuth_steps * elevation_steps
        scan_index = min(scan_index, total_steps - 1)
        current_azimuth, current_elevation, total_steps = scan_direction(
            scan_index,
            azimuth_range,
            elevation_range,
            azimuth_steps,
            elevation_steps,
        )
        st.session_state.scan_last_azimuth = current_azimuth
        st.session_state.scan_last_elevation = current_elevation
        st.session_state.scan_show_last_frame = True
        st.info(
            f"🔄 자동 스캔: Az {current_azimuth:.1f}°, El {current_elevation:.1f}° "
            f"({scan_index + 1}/{total_steps}) · "
            f"{option_label(scan_mode, SCAN_MODE_LABELS)}"
        )
        st.progress((scan_index + 1) / total_steps)
    else:
        show_last_frame = bool(st.session_state.get("scan_show_last_frame", False))
        last_azimuth = st.session_state.get("scan_last_azimuth")
        last_elevation = st.session_state.get("scan_last_elevation")
        if (
            show_last_frame
            and isinstance(last_azimuth, (int, float))
            and isinstance(last_elevation, (int, float))
        ):
            current_azimuth = float(last_azimuth)
            current_elevation = float(last_elevation)
        else:
            current_azimuth = config.target_azimuth_deg
            current_elevation = config.target_elevation_deg
        total_steps = 0

    try:
        with st.spinner("활성 탭 계산 중…", show_time=True):
            with COMPUTE_GOVERNOR.lease(
                compute_session_id,
                f"{view_name}:{config.geometry}",
            ) as compute_lease:
                state = cached_state(config, current_azimuth, current_elevation)
                cuts = None
                great_circle_cuts = None
                directivity = None
                surface = None
                surface_sampling = None
                if view_name in {"pattern", "metrics"}:
                    cuts = cached_pattern_cuts(
                        config,
                        current_azimuth,
                        current_elevation,
                        PATTERN_CUT_SCHEMA_VERSION,
                    )
                    great_circle_cuts = cached_great_circle_cuts(
                        config,
                        current_azimuth,
                        current_elevation,
                        GREAT_CIRCLE_CUT_SCHEMA_VERSION,
                    )
                if view_name == "metrics":
                    directivity = cached_directivity(
                        config,
                        current_azimuth,
                        current_elevation,
                        DIRECTIVITY_SCHEMA_VERSION,
                    )
                if view_name == "pattern":
                    surface_sampling = scan_surface_sampling(
                        state.coordinates.element_count,
                        scan_mode,
                        scanning=scanning,
                    )
                    if surface_sampling.render_3d:
                        surface = cached_surface_pattern(
                            config,
                            current_azimuth,
                            current_elevation,
                            int(surface_sampling.resolution),
                            int(surface_sampling.local_sample_count),
                            SURFACE_PATTERN_SCHEMA_VERSION,
                        )
                        if (
                            getattr(surface, "schema_version", 0)
                            != SURFACE_PATTERN_SCHEMA_VERSION
                        ):
                            cached_surface_pattern.clear()
                            surface = calculate_surface_pattern(
                                state,
                                resolution=surface_sampling.resolution,
                                local_sample_count=(
                                    surface_sampling.local_sample_count
                                ),
                                cancel_check=compute_lease.check,
                            )
                compute_lease.check()
    except SessionRateLimitError as error:
        st.warning(
            "이 세션의 계산 요청이 너무 빠릅니다. "
            f"약 {error.retry_after_seconds:.1f}초 후 다시 시도하세요. "
            "자동 스캔 중에는 다음 fragment 주기에 재시도합니다."
        )
        return
    except ComputeBusyError:
        st.warning(
            "서버의 동시 계산 슬롯이 모두 사용 중입니다. 잠시 후 다시 "
            "시도합니다. 배열 크기나 스캔 품질을 낮추면 대기 가능성이 줄어듭니다."
        )
        return
    except ComputeDeadlineExceeded:
        st.session_state.is_scanning = False
        st.error(
            "계산이 서버 제한 시간을 초과하여 안전하게 중단됐습니다. 배열 "
            f"크기나 해상도를 낮추세요. 현재 제한: "
            f"{RESOURCE_POLICY.compute_timeout_seconds:.1f}초"
        )
        return
    except ComputeCancelled:
        st.session_state.is_scanning = False
        st.info("사용자 요청으로 계산과 자동 스캔을 중단했습니다.")
        return
    finally:
        COMPUTE_GOVERNOR.log_health_if_due()

    render_diagnostics(state)
    if view_name == "pattern":
        if cuts is None or great_circle_cuts is None or surface_sampling is None:
            raise RuntimeError("Pattern calculation did not produce render data.")
        render_pattern_tab(
            state,
            cuts,
            great_circle_cuts,
            coordinate_option=coordinate_option,
            scale_option=scale_option,
            show_band=show_3db,
            show_band_value=show_3db_value,
            render_3d=surface_sampling.render_3d,
            surface=surface,
            surface_quality=surface_sampling.quality,
        )
    elif view_name == "metrics":
        if cuts is None or great_circle_cuts is None or directivity is None:
            raise RuntimeError("Metric calculation did not produce pattern cuts.")
        render_metrics_tab(state, cuts, great_circle_cuts, directivity)
    else:
        render_elements_tab(state)

    if scanning:
        if scan_index < total_steps - 1:
            st.session_state.scan_idx = scan_index + 1
        else:
            st.session_state.is_scanning = False
            st.session_state.scan_completed = True
            # One app rerun releases the fragment timer after the final frame.
            st.rerun(scope="app")


# Streamlit 1.60 dynamic tabs expose `.open`; only the visible branch executes.
if pattern_tab.open:
    with pattern_tab:
        render_active_result("pattern")
elif metrics_tab.open:
    with metrics_tab:
        render_active_result("metrics")
elif elements_tab.open:
    with elements_tab:
        render_active_result("elements")
