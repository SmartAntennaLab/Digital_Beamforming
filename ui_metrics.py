"""Performance metrics, null diagnostics, and data exports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from directivity import DirectivityResult
from exporters import build_export_artifacts
from golden_validation import GoldenValidationResult
from interferer_sampling import InterfererResponseComparison
from pattern_sampling import (
    GreatCircleCuts,
    PatternCuts,
)
from signal_processing import AdvancedAnalysis
from simulation import SimulationState
from ui_advanced_metrics import render_advanced_metrics
from ui_formatters import (
    format_absolute_residual,
    format_angle_metric,
    format_degradation,
    format_depth,
    format_residual_db,
    format_response_error,
    format_sidelobe_metric,
    null_solver_label,
    optimizer_convergence_label,
)


def render_metrics_tab(
    state: SimulationState,
    cuts: PatternCuts,
    great_circle_cuts: GreatCircleCuts,
    directivity: DirectivityResult,
    interferer_comparisons: tuple[InterfererResponseComparison, ...],
    advanced_analysis: AdvancedAnalysis | None = None,
    golden_validation: GoldenValidationResult | None = None,
) -> None:
    st.subheader("📏 주요 성능 지표 (AESA Performance Metrics)")
    gain = state.gain_metrics
    relative_array_gain = (
        f"{gain.relative_array_gain_db:.2f} dB"
        if gain.relative_array_gain_db is not None
        else "N/A (유효 가중치 없음)"
    )
    directivity_value = (
        f"{directivity.directivity_dbi:.2f} dBi"
        if directivity.directivity_dbi is not None
        and np.isfinite(directivity.directivity_dbi)
        else ("-∞ dBi" if directivity.directivity_dbi is not None else "N/A")
    )
    with st.container(horizontal=True):
        st.metric("상대 배열 이득", relative_array_gain, border=True)
        st.metric("목표 방향 Directivity", directivity_value, border=True)
        st.metric(
            "활성 소자",
            f"{gain.active_elements} / {gain.total_elements}",
            border=True,
        )
        st.metric(
            "테이퍼 효율",
            f"{100.0 * gain.taper_efficiency:.2f} %",
            border=True,
        )
        st.metric(
            "위상·조향 효율",
            f"{100.0 * gain.phase_efficiency:.2f} %",
            border=True,
        )

    mode_label = (
        "정확"
        if directivity.effective_mode == "exact"
        else "고속 근사"
        if directivity.effective_mode == "fast"
        else "사용 불가"
    )
    if directivity.warning_message is not None:
        st.warning(directivity.warning_message)
    elif directivity.effective_mode == "fast":
        st.info("Directivity는 고속 근사 전구 적분 결과입니다.")

    weight_result = state.weight_result
    actual_null_depth = (
        state.realized_null_depths_db[0] if state.realized_null_depths_db else None
    )
    continuous_null_depth = (
        weight_result.continuous_null_depths_db[0]
        if weight_result.continuous_null_depths_db
        else None
    )
    if state.config.enable_null_steering:
        met_count = sum(status is True for status in weight_result.null_requirement_met)
        null_columns = st.columns(6)
        null_columns[0].metric("실제 Null 깊이", format_depth(actual_null_depth))
        null_columns[1].metric(
            "양자화 전 Null 깊이", format_depth(continuous_null_depth)
        )
        null_columns[2].metric(
            "Null 제약", "적용됨" if weight_result.null_applied else "적용 실패"
        )
        null_columns[3].metric(
            "억압 요구 충족",
            f"{met_count} / {len(weight_result.null_requirement_met)}",
        )
        null_columns[4].metric(
            "포화 소자",
            f"{weight_result.saturated_element_count}개",
        )
        null_columns[5].metric(
            "최적화 수렴",
            optimizer_convergence_label(weight_result.optimizer_convergence_reason),
        )

        continuous_diagnostics = weight_result.continuous_diagnostics
        final_diagnostics = weight_result.final_diagnostics
        st.caption(
            f"제약 해법: {null_solver_label(weight_result.solver_method)}. "
            "상대 잔차는 요청한 목표 응답 크기로 정규화하며, 양자화 열화는 "
            "양수일수록 제약이 나빠졌음을 뜻합니다."
        )
        if weight_result.optimizer_selected_restart is not None:
            st.caption(
                f"선택 restart {weight_result.optimizer_selected_restart}/"
                f"{weight_result.optimizer_restart_count}, 선택해 "
                f"{weight_result.optimizer_iterations}회·전체 "
                f"{weight_result.optimizer_total_iterations}회 반복, "
                f"허용오차 {weight_result.optimizer_tolerance:.1e}, "
                f"restart별 상한 {weight_result.optimizer_max_iterations}회."
            )
        with st.container(horizontal=True):
            st.metric(
                "목표 응답 오차 · 연속해",
                format_response_error(continuous_diagnostics),
                border=True,
                help="절대 오차와 요청 목표 응답 대비 상대 오차입니다.",
            )
            st.metric(
                "목표 응답 오차 · 최종",
                format_response_error(final_diagnostics),
                delta=(
                    format_degradation(weight_result.quantization_target_degradation_db)
                    if weight_result.phase_quantization_applied
                    else None
                ),
                delta_color="inverse",
                border=True,
                help="최종 위상 양자화 후 절대/상대 목표 응답 오차입니다.",
            )
            st.metric(
                "전체 제약 잔차 열화",
                (
                    format_degradation(
                        weight_result.quantization_constraint_degradation_db
                    )
                    if weight_result.phase_quantization_applied
                    else "양자화 미적용"
                ),
                border=True,
            )

        with st.container(horizontal=True):
            st.metric(
                "연속해 최대 진폭",
                f"{continuous_diagnostics.max_amplitude:.6g}",
                border=True,
            )
            st.metric(
                "최종 최대 진폭",
                f"{final_diagnostics.max_amplitude:.6g}",
                border=True,
            )
            st.metric(
                "연속해 총 가중치 전력",
                f"{continuous_diagnostics.total_weight_power:.6g}",
                border=True,
                help="Σ|wₙ|²",
            )
            st.metric(
                "최종 총 가중치 전력",
                f"{final_diagnostics.total_weight_power:.6g}",
                border=True,
                help="Σ|wₙ|²",
            )

        null_rows = []
        comparison_by_index = {
            comparison.interferer_index: comparison
            for comparison in interferer_comparisons
        }
        for index, (azimuth_rad, elevation_rad) in enumerate(
            weight_result.null_directions_rad
        ):
            response_comparison = comparison_by_index.get(index + 1)
            continuous_absolute = continuous_diagnostics.null_constraint_residuals[
                index
            ]
            final_absolute = final_diagnostics.null_constraint_residuals[index]
            continuous_relative = continuous_diagnostics.null_relative_residuals[index]
            final_relative = final_diagnostics.null_relative_residuals[index]
            degradation = weight_result.quantization_null_degradation_db[index]
            null_rows.append(
                {
                    "제약": f"Null {index + 1}",
                    "방향": (
                        f"Az {np.degrees(azimuth_rad):.3f}°, "
                        f"El {np.degrees(elevation_rad):.3f}°"
                    ),
                    "요구 억압": (
                        f"{weight_result.null_required_suppression_db[index]:.1f} dB"
                    ),
                    "적용 전 상대 응답": (
                        f"{response_comparison.before_relative_db:.2f} dB"
                        if response_comparison is not None
                        else "N/A"
                    ),
                    "적용 후 상대 응답": (
                        f"{response_comparison.after_relative_db:.2f} dB"
                        if response_comparison is not None
                        else "N/A"
                    ),
                    "추가 억압량": (
                        f"{response_comparison.additional_suppression_db:+.2f} dB"
                        if response_comparison is not None
                        else "N/A"
                    ),
                    "충족 여부": (
                        "충족"
                        if weight_result.null_requirement_met[index] is True
                        else (
                            "미달"
                            if weight_result.null_requirement_met[index] is False
                            else "N/A"
                        )
                    ),
                    "연속 절대 잔차": (format_absolute_residual(continuous_absolute)),
                    "연속 상대 잔차": format_residual_db(continuous_relative),
                    "최종 절대 잔차": (format_absolute_residual(final_absolute)),
                    "최종 상대 잔차": format_residual_db(final_relative),
                    "양자화 열화": (
                        format_degradation(degradation)
                        if weight_result.phase_quantization_applied
                        else "미적용"
                    ),
                    "최종 Null 깊이": format_depth(
                        state.realized_null_depths_db[index]
                    ),
                }
            )
        if null_rows:
            st.dataframe(
                pd.DataFrame(null_rows),
                hide_index=True,
                width="stretch",
            )
        if weight_result.optimizer_trace:
            with st.expander("Null 최적화 반복 이력", expanded=False):
                trace_rows = [
                    {
                        "Restart": point.restart_index,
                        "선택": (
                            point.restart_index
                            == weight_result.optimizer_selected_restart
                        ),
                        "반복": point.iteration,
                        "목적함수": point.objective,
                        "최악 Null 상대 잔차 (dB)": point.worst_null_residual_db,
                        "목표 방향 손실 (dB)": point.target_loss_db,
                    }
                    for point in weight_result.optimizer_trace
                ]
                st.dataframe(
                    pd.DataFrame(trace_rows),
                    hide_index=True,
                    width="stretch",
                    height=320,
                )

    azimuth_metrics = cuts.azimuth_metrics
    elevation_metrics = cuts.elevation_metrics
    horizontal_metrics = great_circle_cuts.horizontal_metrics
    vertical_metrics = great_circle_cuts.vertical_metrics
    st.divider()
    st.markdown("#### 좌표각 빔폭")
    width_columns = st.columns(2)
    width_columns[0].metric(
        "좌표각 HPBW (Azimuth)",
        format_angle_metric(azimuth_metrics.hpbw_deg),
    )
    width_columns[1].metric(
        "좌표각 HPBW (Elevation)",
        format_angle_metric(elevation_metrics.hpbw_deg),
    )
    physical_width_columns = st.columns(2)
    physical_width_columns[0].metric(
        "실제 각거리 HPBW (수평 주평면)",
        format_angle_metric(horizontal_metrics.hpbw_deg),
    )
    physical_width_columns[1].metric(
        "실제 각거리 HPBW (수직 주평면)",
        format_angle_metric(vertical_metrics.hpbw_deg),
    )
    null_width_columns = st.columns(2)
    null_width_columns[0].metric(
        "First Null Bandwidth (Azimuth)",
        format_angle_metric(azimuth_metrics.first_null_beamwidth_deg),
    )
    null_width_columns[1].metric(
        "First Null Bandwidth (Elevation)",
        format_angle_metric(elevation_metrics.first_null_beamwidth_deg),
    )
    sidelobe_columns = st.columns(2)
    sidelobe_columns[0].metric(
        "Sidelobe Level (Azimuth)",
        format_sidelobe_metric(
            azimuth_metrics.sidelobe_level_db,
            azimuth_metrics.sidelobe_angle_deg,
        ),
    )
    sidelobe_columns[1].metric(
        "Sidelobe Level (Elevation)",
        format_sidelobe_metric(
            elevation_metrics.sidelobe_level_db,
            elevation_metrics.sidelobe_angle_deg,
        ),
    )
    st.info(
        "상대 배열 이득은 N_eff 기반 coherent combining 지표(dB)입니다. "
        "목표 방향 Directivity는 동일한 최종 가중치와 소자 패턴의 전구 "
        "방사전력을 적분한 물리적 지향도(dBi)이며, RF 손실을 포함한 실제 "
        "안테나 이득은 아닙니다."
    )
    integration_detail = directivity.integration_method
    if directivity.azimuth_samples is not None:
        integration_detail += (
            f" · {directivity.azimuth_samples}×"
            f"{directivity.elevation_samples} 구면 표본"
        )
    cache_detail = ""
    if directivity.kernel_cache_used:
        cache_detail = (
            " · 기하 kernel cache hit"
            if directivity.kernel_cache_hit
            else " · 기하 kernel cache 생성"
        )
    st.caption(
        f"Directivity 계산 모드: {mode_label} · 소자 {directivity.element_count:,}개 · "
        f"pairwise 규모 {directivity.pair_count:,}{cache_detail}. "
        f"적분: {integration_detail}; "
        f"∫U dΩ={directivity.radiated_power_integral:.6g}, "
        f"목표 U={directivity.target_radiation_intensity:.6g}."
    )

    render_advanced_metrics(state, advanced_analysis, golden_validation)
    export_artifacts = build_export_artifacts(
        state,
        cuts,
        great_circle_cuts=great_circle_cuts,
        directivity=directivity,
    )
    st.divider()
    st.subheader("💾 데이터 내보내기")
    download_columns = st.columns(4)
    download_columns[0].download_button(
        "📊 2D 빔 패턴 CSV 다운로드",
        export_artifacts.pattern_csv,
        file_name="beam_pattern_data.csv",
        mime="text/csv",
        width="stretch",
        icon=":material/table_view:",
        on_click="ignore",
    )
    download_columns[1].download_button(
        "📄 설계 리포트 다운로드",
        export_artifacts.design_report,
        file_name="beamforming_design_report.md",
        mime="text/markdown",
        width="stretch",
        icon=":material/description:",
        on_click="ignore",
    )
    download_columns[2].download_button(
        "설정 JSON",
        export_artifacts.settings_json,
        file_name="simulation_settings.json",
        mime="application/json",
        width="stretch",
        icon=":material/data_object:",
        on_click="ignore",
    )
    download_columns[3].download_button(
        "재현성 패키지",
        export_artifacts.reproducibility_zip,
        file_name="beamforming_reproducibility.zip",
        mime="application/zip",
        width="stretch",
        icon=":material/folder_zip:",
        on_click="ignore",
    )


__all__ = ["render_metrics_tab"]
