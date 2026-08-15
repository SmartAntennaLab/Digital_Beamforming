"""Streamlit controls for real-system propagation and validation models."""

from __future__ import annotations

from dataclasses import dataclass, replace

import streamlit as st

from element_pattern_data import ElementPatternGrid, parse_element_pattern_csv
from golden_validation import GoldenDataset, parse_golden_dataset


@dataclass(frozen=True)
class RealSystemSettings:
    random_seed: int
    position_error_rms_wavelength: float
    amplitude_error_rms_db: float
    phase_error_rms_deg: float
    mutual_coupling_db: float | None
    mutual_coupling_phase_deg: float
    polarization_angle_deg: float
    element_pattern_grid: ElementPatternGrid | None
    wideband_bandwidth_percent: float
    wideband_frequency_samples: int
    near_field_focus_range_m: float | None
    enable_channel_analysis: bool
    channel_snapshots: int
    multipath_count: int
    signal_power_dbm: float
    interference_power_dbm: float
    noise_power_dbm: float
    adaptive_beamforming_method: str
    diagonal_loading: float
    enable_doa_estimation: bool
    golden_dataset: GoldenDataset | None
    input_error: str | None


def _parse_pattern_upload(uploaded) -> tuple[ElementPatternGrid | None, str | None]:
    if uploaded is None:
        return None, None
    try:
        return parse_element_pattern_csv(uploaded.getvalue(), name=uploaded.name), None
    except ValueError as error:
        return None, f"소자 패턴 파일: {error}"


def _parse_golden_upload(uploaded, source: str):
    if uploaded is None:
        return None, None
    try:
        dataset = parse_golden_dataset(uploaded.getvalue(), filename=uploaded.name)
        return replace(dataset, source=source), None
    except ValueError as error:
        return None, f"Golden Dataset: {error}"


def render_real_system_section(*, null_enabled: bool) -> RealSystemSettings:
    """Render optional advanced models; defaults preserve the legacy model."""

    with st.expander(
        "7. 실제 시스템 모델·검증",
        expanded=False,
        icon=":material/science:",
    ):
        st.caption("모든 오차·채널 모델은 기본값에서 꺼져 있어 기존 결과를 유지합니다.")
        random_seed = int(
            st.number_input(
                "난수 시드",
                min_value=0,
                max_value=4_294_967_295,
                step=1,
                key="random_seed",
                persist_state="session",
            )
        )
        st.markdown("**배열·RF 현실성**")
        position_error = st.slider(
            "소자 위치 오차 RMS (λ)",
            0.0,
            0.20,
            step=0.001,
            key="position_error_rms_wavelength",
            persist_state="session",
        )
        amplitude_error = st.slider(
            "진폭 보정 오차 RMS (dB)",
            0.0,
            6.0,
            step=0.1,
            key="amplitude_error_rms_db",
            persist_state="session",
        )
        phase_error = st.slider(
            "위상 보정 오차 RMS (°)",
            0.0,
            60.0,
            step=0.5,
            key="phase_error_rms_deg",
            persist_state="session",
        )
        coupling_enabled = st.checkbox(
            "최근접 상호 결합 적용",
            key="enable_mutual_coupling",
            persist_state="session",
        )
        coupling_db = st.slider(
            "상호 결합 크기 (dB)",
            -60.0,
            -6.0,
            step=1.0,
            key="mutual_coupling_db",
            disabled=not coupling_enabled,
            persist_state="session",
        )
        coupling_phase = st.slider(
            "상호 결합 위상 (°)",
            -180.0,
            180.0,
            step=5.0,
            key="mutual_coupling_phase_deg",
            disabled=not coupling_enabled,
            persist_state="session",
        )
        polarization = st.slider(
            "수신 편파 회전각 (°)",
            -90.0,
            90.0,
            step=1.0,
            key="polarization_angle_deg",
            persist_state="session",
        )
        pattern_upload = st.file_uploader(
            "실측 소자 패턴 CSV",
            type=("csv",),
            key="element_pattern_upload",
            help=(
                "필수 열: azimuth_deg,elevation_deg,copol_gain_db. "
                "선택 열: copol_phase_deg,crosspol_gain_db,crosspol_phase_deg."
            ),
            width="stretch",
        )
        element_pattern, pattern_error = _parse_pattern_upload(pattern_upload)

        st.markdown("**Wideband·근거리장**")
        bandwidth = st.slider(
            "Wideband 대역폭 (%)",
            0.0,
            40.0,
            step=1.0,
            key="wideband_bandwidth_percent",
            persist_state="session",
        )
        frequency_samples = st.slider(
            "주파수 샘플 수",
            3,
            33,
            step=2,
            key="wideband_frequency_samples",
            disabled=bandwidth <= 0.0,
            persist_state="session",
        )
        near_field_enabled = st.checkbox(
            "Near-field Beam Focusing",
            key="enable_near_field_focus",
            disabled=null_enabled,
            help="Far-field Null 조향과 동시에 사용할 수 없습니다.",
            persist_state="session",
        )
        focus_range = st.number_input(
            "초점 거리 (m)",
            min_value=0.01,
            max_value=100_000.0,
            step=0.1,
            key="near_field_focus_range_m",
            disabled=not near_field_enabled,
            persist_state="session",
        )

        st.markdown("**채널·적응 빔포밍·DOA**")
        channel_enabled = st.checkbox(
            "채널·잡음·다중경로 분석",
            key="enable_channel_analysis",
            persist_state="session",
        )
        snapshots = st.slider(
            "Snapshot 수",
            16,
            1024,
            step=16,
            key="channel_snapshots",
            disabled=not channel_enabled,
            persist_state="session",
        )
        multipath_count = st.slider(
            "원하는 신호 다중경로 수",
            0,
            8,
            key="multipath_count",
            disabled=not channel_enabled,
            persist_state="session",
        )
        signal_power = st.number_input(
            "신호 전력 (dBm)",
            -120.0,
            60.0,
            step=1.0,
            key="signal_power_dbm",
            disabled=not channel_enabled,
            persist_state="session",
        )
        interference_power = st.number_input(
            "간섭원별 전력 (dBm)",
            -120.0,
            60.0,
            step=1.0,
            key="interference_power_dbm",
            disabled=not channel_enabled,
            persist_state="session",
        )
        noise_power = st.number_input(
            "소자별 잡음 전력 (dBm)",
            -180.0,
            60.0,
            step=1.0,
            key="noise_power_dbm",
            disabled=not channel_enabled,
            persist_state="session",
        )
        adaptive_method = st.selectbox(
            "적응 빔포밍",
            ("none", "mvdr", "lcmv"),
            format_func=lambda value: {
                "none": "사용 안 함",
                "mvdr": "MVDR",
                "lcmv": "LCMV",
            }[value],
            key="adaptive_beamforming_method",
            disabled=not channel_enabled,
            persist_state="session",
        )
        loading = st.number_input(
            "Diagonal loading",
            1e-6,
            1.0,
            format="%.6f",
            key="diagonal_loading",
            disabled=not channel_enabled or adaptive_method == "none",
            persist_state="session",
        )
        doa_enabled = st.checkbox(
            "MUSIC DOA 추정",
            key="enable_doa_estimation",
            disabled=not channel_enabled,
            persist_state="session",
        )

        st.markdown("**Golden Dataset 교차 검증**")
        golden_source = st.selectbox(
            "기준 데이터 출처",
            ("matlab", "measurement", "other"),
            format_func=lambda value: {
                "matlab": "MATLAB",
                "measurement": "측정",
                "other": "기타",
            }[value],
            key="golden_dataset_source",
            persist_state="session",
        )
        golden_upload = st.file_uploader(
            "Golden Dataset (JSON/CSV)",
            type=("json", "csv"),
            key="golden_dataset_upload",
            width="stretch",
        )
        golden_dataset, golden_error = _parse_golden_upload(
            golden_upload, golden_source
        )
        input_error = pattern_error or golden_error
        if input_error:
            st.error(input_error)

    return RealSystemSettings(
        random_seed=random_seed,
        position_error_rms_wavelength=float(position_error),
        amplitude_error_rms_db=float(amplitude_error),
        phase_error_rms_deg=float(phase_error),
        mutual_coupling_db=float(coupling_db) if coupling_enabled else None,
        mutual_coupling_phase_deg=float(coupling_phase),
        polarization_angle_deg=float(polarization),
        element_pattern_grid=element_pattern,
        wideband_bandwidth_percent=float(bandwidth),
        wideband_frequency_samples=int(frequency_samples),
        near_field_focus_range_m=(
            float(focus_range) if near_field_enabled and not null_enabled else None
        ),
        enable_channel_analysis=bool(channel_enabled),
        channel_snapshots=int(snapshots),
        multipath_count=int(multipath_count),
        signal_power_dbm=float(signal_power),
        interference_power_dbm=float(interference_power),
        noise_power_dbm=float(noise_power),
        adaptive_beamforming_method=str(adaptive_method) if channel_enabled else "none",
        diagonal_loading=float(loading),
        enable_doa_estimation=bool(doa_enabled and channel_enabled),
        golden_dataset=golden_dataset,
        input_error=input_error,
    )


__all__ = ["RealSystemSettings", "render_real_system_section"]
