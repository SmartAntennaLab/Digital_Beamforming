"""Optional real-system, channel, adaptive, DOA, and Golden metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from golden_validation import GoldenValidationResult
from signal_processing import AdvancedAnalysis
from simulation import SimulationState


def _hardware_enabled(state: SimulationState) -> bool:
    config = state.config
    return any(
        (
            config.position_error_rms_wavelength > 0.0,
            config.amplitude_error_rms_db > 0.0,
            config.phase_error_rms_deg > 0.0,
            config.mutual_coupling_db is not None,
            config.element_pattern_grid is not None,
            abs(config.polarization_angle_deg) > 0.0,
        )
    )


def render_advanced_metrics(
    state: SimulationState,
    analysis: AdvancedAnalysis | None,
    golden: GoldenValidationResult | None,
) -> None:
    """Render only the optional analyses enabled by the applied configuration."""

    has_analysis = analysis is not None and any(
        (
            analysis.wideband,
            analysis.near_field,
            analysis.channel,
            analysis.adaptive,
            analysis.doa,
        )
    )
    if not _hardware_enabled(state) and not has_analysis and golden is None:
        return
    st.divider()
    st.markdown("#### 실제 시스템 모델 분석")
    if _hardware_enabled(state):
        diagnostics = state.hardware_diagnostics
        with st.container(horizontal=True):
            st.metric(
                "실현 위치 오차 RMS",
                f"{diagnostics.realized_position_error_rms_wavelength:.4f} λ",
                border=True,
            )
            st.metric(
                "실현 진폭 오차 RMS",
                f"{diagnostics.realized_amplitude_error_rms_db:.3f} dB",
                border=True,
            )
            st.metric(
                "실현 위상 오차 RMS",
                f"{diagnostics.realized_phase_error_rms_deg:.3f}°",
                border=True,
            )
            st.metric(
                "결합 이웃 링크",
                f"{diagnostics.coupled_neighbor_links:,}",
                border=True,
            )
    if analysis is not None and analysis.wideband is not None:
        wideband = analysis.wideband
        with st.container(horizontal=True):
            st.metric(
                "최대 Beam Squint", f"{wideband.maximum_squint_deg:.3f}°", border=True
            )
            st.metric(
                "대역 가장자리 목표 손실",
                f"{wideband.edge_target_loss_db:.3f} dB",
                border=True,
            )
            st.metric("분석 대역폭", f"{wideband.bandwidth_percent:.1f}%", border=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "Frequency (GHz)": [
                        point.frequency_ghz for point in wideband.points
                    ],
                    "Peak Azimuth (deg)": [
                        point.peak_azimuth_deg for point in wideband.points
                    ],
                    "Peak Elevation (deg)": [
                        point.peak_elevation_deg for point in wideband.points
                    ],
                    "Squint (deg)": [
                        point.squint_angle_deg for point in wideband.points
                    ],
                    "Target loss (dB)": [
                        point.target_loss_db for point in wideband.points
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    if analysis is not None and analysis.near_field is not None:
        near = analysis.near_field
        with st.container(horizontal=True):
            st.metric("초점 거리", f"{near.focus_range_m:.3f} m", border=True)
            st.metric("Rayleigh 거리", f"{near.rayleigh_distance_m:.3f} m", border=True)
            st.metric(
                "초점 Coherence", f"{near.focus_coherence_db:.3f} dB", border=True
            )
            st.metric(
                "Near-field 판정",
                "Rayleigh 내부"
                if near.focus_inside_rayleigh_distance
                else "Far-field 영역",
                border=True,
            )
    if analysis is not None and analysis.channel is not None:
        channel = analysis.channel
        with st.container(horizontal=True):
            st.metric("입력 SNR", f"{channel.input_snr_db:.2f} dB", border=True)
            st.metric(
                "Conventional SINR",
                f"{channel.conventional_sinr_db:.2f} dB",
                border=True,
            )
            st.metric("Snapshot", f"{channel.snapshots:,}", border=True)
            st.metric("분석 소자", f"{channel.analysis_elements:,}", border=True)
    if analysis is not None and analysis.adaptive is not None:
        adaptive = analysis.adaptive
        with st.container(horizontal=True):
            st.metric("적응 방식", adaptive.method.upper(), border=True)
            st.metric(
                "적응 출력 SINR", f"{adaptive.output_sinr_db:.2f} dB", border=True
            )
            st.metric("SINR 개선", f"{adaptive.improvement_db:+.2f} dB", border=True)
            st.metric("제약 오차", f"{adaptive.constraint_error:.2e}", border=True)
    if analysis is not None and analysis.doa is not None:
        doa = analysis.doa
        st.markdown("**MUSIC DOA 추정 피크**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Azimuth (deg)": [peak.azimuth_deg for peak in doa.peaks],
                    "Spectrum (dB)": [peak.spectrum_db for peak in doa.peaks],
                    "Elevation assumption (deg)": doa.elevation_deg,
                }
            ),
            hide_index=True,
            width="stretch",
        )
    if golden is not None:
        st.markdown("**Golden Dataset 교차 검증**")
        with st.container(horizontal=True):
            st.metric("검증 결과", "PASS" if golden.passed else "FAIL", border=True)
            st.metric("RMSE", f"{golden.rmse_db:.3f} dB", border=True)
            st.metric("최대 오차", f"{golden.maximum_error_db:.3f} dB", border=True)
            st.metric("허용 오차", f"{golden.tolerance_db:.3f} dB", border=True)
        st.caption(
            f"{golden.dataset_name} · {golden.source} · {golden.sample_count:,} samples"
        )


__all__ = ["render_advanced_metrics"]
