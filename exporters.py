"""CSV and Markdown exports for one completed simulation frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model_options import (
    ELEMENT_PATTERN_LABELS,
    PHASE_BIT_LABELS,
    TAPER_LABELS,
    option_label,
)
from simulation import PatternCuts, SimulationState, summarize_array_layout
from ui_formatters import (
    array_size_text,
    format_absolute_residual,
    format_degradation,
    format_depth,
    format_pattern_metric_summary,
    format_residual_db,
    format_response_error,
    null_solver_label,
    spacing_text,
    steering_axes_text,
)


@dataclass(frozen=True)
class ExportArtifacts:
    pattern_csv: bytes
    design_report: bytes


def build_pattern_export_frame(cuts: PatternCuts) -> pd.DataFrame:
    """Return aligned azimuth/elevation cut data without UI dependencies."""

    return pd.DataFrame(
        {
            "Azimuth Angle (deg)": pd.Series(
                np.degrees(cuts.azimuth_angles_rad), dtype=float
            ),
            "Azimuth Gain (dB)": pd.Series(
                cuts.azimuth_pattern_db, dtype=float
            ),
            "Elevation Angle (deg)": pd.Series(
                np.degrees(cuts.elevation_angles_rad), dtype=float
            ),
            "Elevation Gain (dB)": pd.Series(
                cuts.elevation_pattern_db, dtype=float
            ),
        }
    )


def build_design_report(state: SimulationState, cuts: PatternCuts) -> str:
    """Build the human-readable Markdown design report."""

    gain = state.gain_metrics
    layout = summarize_array_layout(state)
    relative_array_gain = (
        f"{gain.relative_array_gain_db:.2f} dB"
        if gain.relative_array_gain_db is not None
        else "N/A (유효 가중치 없음)"
    )
    weight_result = state.weight_result
    rank = (
        f"{weight_result.constraint_rank}/{weight_result.constraint_count}"
        if weight_result.constraint_rank is not None
        else "N/A"
    )
    condition = (
        f"{weight_result.condition_number:.3e}"
        if weight_result.condition_number is not None
        and np.isfinite(weight_result.condition_number)
        else "∞"
    )
    actual_null_depth = (
        weight_result.null_depths_db[0] if weight_result.null_depths_db else None
    )

    if state.config.enable_null_steering:
        continuous = weight_result.continuous_diagnostics
        final = weight_result.final_diagnostics
        lines = [
            f"- 해법: {null_solver_label(weight_result.solver_method)}",
            f"- 제약 rank / condition: {rank} / {condition}",
            (
                "- 목표 응답 오차(연속 / 최종): "
                f"{format_response_error(continuous)} / "
                f"{format_response_error(final)}"
            ),
            (
                "- 최대 진폭(연속 / 최종): "
                f"{continuous.max_amplitude:.6g} / {final.max_amplitude:.6g}"
            ),
            (
                "- 총 가중치 전력 Σ|wₙ|²(연속 / 최종): "
                f"{continuous.total_weight_power:.6g} / "
                f"{final.total_weight_power:.6g}"
            ),
            (
                "- 전체 제약 잔차 양자화 열화: "
                + (
                    format_degradation(
                        weight_result.quantization_constraint_degradation_db
                    )
                    if weight_result.phase_quantization_applied
                    else "양자화 미적용"
                )
            ),
            (
                "- 목표 응답 오차 양자화 열화: "
                + (
                    format_degradation(
                        weight_result.quantization_target_degradation_db
                    )
                    if weight_result.phase_quantization_applied
                    else "양자화 미적용"
                )
            ),
        ]
        for index, (azimuth_rad, elevation_rad) in enumerate(
            weight_result.null_directions_rad
        ):
            continuous_absolute = continuous.null_constraint_residuals[index]
            final_absolute = final.null_constraint_residuals[index]
            continuous_relative = continuous.null_relative_residuals[index]
            final_relative = final.null_relative_residuals[index]
            degradation = weight_result.quantization_null_degradation_db[index]
            lines.append(
                f"- Null {index + 1} (Az {np.degrees(azimuth_rad):.3f}°, "
                f"El {np.degrees(elevation_rad):.3f}°): 절대 잔차 "
                f"{format_absolute_residual(continuous_absolute)} → "
                f"{format_absolute_residual(final_absolute)}, 상대 잔차 "
                f"{format_residual_db(continuous_relative)} → "
                f"{format_residual_db(final_relative)}, 열화 "
                + (
                    format_degradation(degradation)
                    if weight_result.phase_quantization_applied
                    else "미적용"
                )
            )
        null_report = "\n".join(lines)
    else:
        null_report = "- Null 조향 미사용"

    return f"""# 디지털 빔포밍 안테나 설계 리포트

## 시뮬레이션 조건

- 주파수: {state.config.frequency_ghz:.2f} GHz
- 파장: {state.wavelength_m * 1000.0:.3f} mm
- 배열: {state.coordinates.geometry}, {array_size_text(state)}
- 소자 간격: {spacing_text(state)}
- 조향 가능 축: {steering_axes_text(state)}
- 조향 방향: Az {state.current_azimuth_deg:.1f}°, El {state.current_elevation_deg:.1f}°
- 진폭 창: {option_label(state.config.taper_option, TAPER_LABELS)}
- 소자 패턴: {option_label(state.config.element_option, ELEMENT_PATTERN_LABELS)}
- 위상 해상도: {option_label(state.config.phase_bits, PHASE_BIT_LABELS)}
- 요청 결함률: {layout.requested_failure_rate_percent:.2f}%
- 실제 결함률: {layout.actual_failure_rate_percent:.2f}% ({layout.failed_elements} / {layout.total_elements}개)

## 성능 지표

- 상대 배열 이득: {relative_array_gain}
- 활성 소자: {gain.active_elements} / {gain.total_elements}
- 테이퍼 효율: {100.0 * gain.taper_efficiency:.2f}%
- 위상·조향 효율: {100.0 * gain.phase_efficiency:.2f}%
- Azimuth HPBW / FNBW / SLL: {format_pattern_metric_summary(cuts.azimuth_metrics)}
- Elevation HPBW / FNBW / SLL: {format_pattern_metric_summary(cuts.elevation_metrics)}
- 실제 Null 깊이: {format_depth(actual_null_depth) if state.config.enable_null_steering else 'N/A'}

## Null 제약 진단

{null_report}
"""


def build_export_artifacts(
    state: SimulationState,
    cuts: PatternCuts,
) -> ExportArtifacts:
    frame = build_pattern_export_frame(cuts)
    return ExportArtifacts(
        pattern_csv=frame.to_csv(index=False).encode("utf-8"),
        design_report=build_design_report(state, cuts).encode("utf-8"),
    )
