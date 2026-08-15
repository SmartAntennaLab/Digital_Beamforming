"""CSV and Markdown exports for one completed simulation frame."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from directivity import DirectivityResult
from model_options import (
    ELEMENT_PATTERN_LABELS,
    PHASE_BIT_LABELS,
    TAPER_LABELS,
    option_label,
)
from pattern_sampling import GreatCircleCuts, PatternCuts
from provenance import APP_VERSION, git_commit
from simulation import SimulationState, summarize_array_layout
from ui_formatters import (
    array_size_text,
    format_absolute_residual,
    format_degradation,
    format_depth,
    format_pattern_metric_summary,
    format_residual_db,
    format_response_error,
    null_solver_label,
    optimizer_convergence_label,
    spacing_text,
    steering_axes_text,
)


@dataclass(frozen=True)
class ExportArtifacts:
    pattern_csv: bytes
    design_report: bytes
    settings_json: bytes
    reproducibility_zip: bytes


def _build_reproducibility_files(
    state: SimulationState,
    pattern_csv: bytes,
    design_report: bytes,
    directivity: DirectivityResult | None,
) -> tuple[bytes, bytes]:
    settings_document = {
        "schema_version": 1,
        "app_version": APP_VERSION,
        "git_commit": git_commit(),
        "random_seed": state.config.random_seed,
        "simulation_config": asdict(state.config),
    }
    settings_json = json.dumps(
        settings_document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    files = {
        "beam_pattern_data.csv": pattern_csv,
        "beamforming_design_report.md": design_report,
        "simulation_settings.json": settings_json,
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "app_version": APP_VERSION,
        "git_commit": settings_document["git_commit"],
        "random_seed": state.config.random_seed,
        "calculation": {
            "directivity_requested_mode": state.config.directivity_mode,
            "directivity_effective_mode": (
                directivity.effective_mode if directivity is not None else None
            ),
            "directivity_is_approximate": (
                directivity.is_approximate if directivity is not None else None
            ),
            "directivity_method": (
                directivity.integration_method if directivity is not None else None
            ),
            "wideband_model": "fixed_phase_shifter_far_field",
            "near_field_model": "scalar_spherical_wave",
            "channel_model": "deterministic_seeded_complex_gaussian",
        },
        "files": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in files.items()
        },
    }
    manifest_json = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr("manifest.json", manifest_json)
    return settings_json, archive_buffer.getvalue()


def build_pattern_export_frame(
    cuts: PatternCuts,
    great_circle_cuts: GreatCircleCuts | None = None,
) -> pd.DataFrame:
    """Return aligned azimuth/elevation cut data without UI dependencies."""

    columns = {
        "Azimuth Angle (deg)": pd.Series(
            np.degrees(cuts.azimuth_angles_rad), dtype=float
        ),
        "Azimuth Gain (dB)": pd.Series(cuts.azimuth_pattern_db, dtype=float),
        "Elevation Angle (deg)": pd.Series(
            np.degrees(cuts.elevation_angles_rad), dtype=float
        ),
        "Elevation Gain (dB)": pd.Series(cuts.elevation_pattern_db, dtype=float),
    }
    if great_circle_cuts is not None:
        columns.update(
            {
                "Horizontal Great-circle Offset (deg)": pd.Series(
                    np.degrees(great_circle_cuts.horizontal_offsets_rad),
                    dtype=float,
                ),
                "Horizontal Great-circle Gain (dB)": pd.Series(
                    great_circle_cuts.horizontal_pattern_db,
                    dtype=float,
                ),
                "Vertical Great-circle Offset (deg)": pd.Series(
                    np.degrees(great_circle_cuts.vertical_offsets_rad),
                    dtype=float,
                ),
                "Vertical Great-circle Gain (dB)": pd.Series(
                    great_circle_cuts.vertical_pattern_db,
                    dtype=float,
                ),
            }
        )
    return pd.DataFrame(columns)


def build_design_report(
    state: SimulationState,
    cuts: PatternCuts,
    *,
    great_circle_cuts: GreatCircleCuts | None = None,
    directivity: DirectivityResult | None = None,
) -> str:
    """Build the human-readable Markdown design report."""

    gain = state.gain_metrics
    layout = summarize_array_layout(state)
    relative_array_gain = (
        f"{gain.relative_array_gain_db:.2f} dB"
        if gain.relative_array_gain_db is not None
        else "N/A (유효 가중치 없음)"
    )
    directivity_text = (
        f"{directivity.directivity_dbi:.2f} dBi"
        if directivity is not None
        and directivity.directivity_dbi is not None
        and np.isfinite(directivity.directivity_dbi)
        else (
            "-∞ dBi"
            if directivity is not None and directivity.directivity_dbi is not None
            else "N/A"
        )
    )
    great_circle_report = (
        "- 실제 각거리 수평 주평면 HPBW / FNBW / SLL: "
        f"{format_pattern_metric_summary(great_circle_cuts.horizontal_metrics)}\n"
        "- 실제 각거리 수직 주평면 HPBW / FNBW / SLL: "
        f"{format_pattern_metric_summary(great_circle_cuts.vertical_metrics)}"
        if great_circle_cuts is not None
        else "- 실제 각거리 Great-circle 지표: N/A"
    )
    directivity_method = (
        directivity.integration_method if directivity is not None else "N/A"
    )
    directivity_mode = (
        "정확"
        if directivity is not None and directivity.effective_mode == "exact"
        else "고속 근사"
        if directivity is not None and directivity.effective_mode == "fast"
        else "N/A"
    )
    directivity_warning = (
        directivity.warning_message
        if directivity is not None and directivity.warning_message is not None
        else "없음"
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
        state.realized_null_depths_db[0] if state.realized_null_depths_db else None
    )
    amplitude_limit_text = (
        f"{weight_result.maximum_element_amplitude:.6g}"
        if weight_result.maximum_element_amplitude is not None
        else "미사용"
    )
    met_null_count = sum(
        status is True for status in weight_result.null_requirement_met
    )
    hardware = state.hardware_diagnostics
    coupling_text = (
        "사용 안 함"
        if hardware.mutual_coupling_db is None
        else (
            f"{hardware.mutual_coupling_db:.1f} dB / "
            f"{hardware.mutual_coupling_phase_deg:.1f}° / "
            f"{hardware.coupled_neighbor_links:,} links"
        )
    )
    pattern_source = (
        "내장 패턴"
        if state.config.element_pattern_grid is None
        else (
            f"{state.config.element_pattern_grid.name} "
            f"(SHA-256 {state.config.element_pattern_grid.source_sha256})"
        )
    )
    focus_text = (
        "사용 안 함"
        if state.config.near_field_focus_range_m is None
        else f"{state.config.near_field_focus_range_m:.6g} m"
    )

    if state.config.enable_null_steering:
        continuous = weight_result.continuous_diagnostics
        final = weight_result.final_diagnostics
        lines = [
            f"- 해법: {null_solver_label(weight_result.solver_method)}",
            (
                "- 최적화 수렴: "
                f"{optimizer_convergence_label(weight_result.optimizer_convergence_reason)}"
            ),
            (
                "- 최적화 반복: 선택해 "
                f"{weight_result.optimizer_iterations}회 / 전체 "
                f"{weight_result.optimizer_total_iterations}회"
            ),
            (
                "- 최적화 설정: 허용오차 "
                f"{weight_result.optimizer_tolerance:.1e}, restart별 상한 "
                f"{weight_result.optimizer_max_iterations}회, 선택 restart "
                f"{weight_result.optimizer_selected_restart or 'N/A'} / "
                f"{weight_result.optimizer_restart_count}"
            ),
            (
                "- 최종 최적화 목적함수: "
                + (
                    f"{weight_result.optimizer_final_objective:.6e}"
                    if weight_result.optimizer_final_objective is not None
                    else "N/A"
                )
            ),
            f"- 최대 소자 진폭 제한: {amplitude_limit_text}",
            f"- 포화 소자: {weight_result.saturated_element_count}개",
            (
                f"- 요구 억압 충족: {met_null_count} / {len(weight_result.null_requirement_met)}"
            ),
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
                    format_degradation(weight_result.quantization_target_degradation_db)
                    if weight_result.phase_quantization_applied
                    else "양자화 미적용"
                )
            ),
        ]
        if weight_result.optimizer_trace:
            selected_trace = [
                point
                for point in weight_result.optimizer_trace
                if point.restart_index == weight_result.optimizer_selected_restart
            ]
            if selected_trace:
                first_point = selected_trace[0]
                last_point = selected_trace[-1]
                lines.append(
                    "- 선택 restart 이력(초기 → 최종): 최악 Null 상대 잔차 "
                    f"{first_point.worst_null_residual_db!s} dB → "
                    f"{last_point.worst_null_residual_db!s} dB, 목표 방향 손실 "
                    f"{first_point.target_loss_db!s} dB → "
                    f"{last_point.target_loss_db!s} dB"
                )
        for index, (azimuth_rad, elevation_rad) in enumerate(
            weight_result.null_directions_rad
        ):
            continuous_absolute = continuous.null_constraint_residuals[index]
            final_absolute = final.null_constraint_residuals[index]
            continuous_relative = continuous.null_relative_residuals[index]
            final_relative = final.null_relative_residuals[index]
            degradation = weight_result.quantization_null_degradation_db[index]
            requirement_status = (
                "충족" if weight_result.null_requirement_met[index] is True else "미달"
            )
            required_db = weight_result.null_required_suppression_db[index]
            lines.append(
                f"- Null {index + 1} (Az {np.degrees(azimuth_rad):.3f}°, "
                f"El {np.degrees(elevation_rad):.3f}°): 요구 {required_db:.1f} dB "
                f"({requirement_status}), 절대 잔차 "
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
- 난수 시드: {state.config.random_seed}
- 위치 오차 RMS 요청 / 실현: {hardware.position_error_rms_wavelength:.6g} λ / {hardware.realized_position_error_rms_wavelength:.6g} λ
- 진폭 보정 오차 RMS 요청 / 실현: {hardware.amplitude_error_rms_db:.6g} dB / {hardware.realized_amplitude_error_rms_db:.6g} dB
- 위상 보정 오차 RMS 요청 / 실현: {hardware.phase_error_rms_deg:.6g}° / {hardware.realized_phase_error_rms_deg:.6g}°
- 상호 결합: {coupling_text}
- 편파 회전각: {state.config.polarization_angle_deg:.3f}°
- 소자 패턴 소스: {pattern_source}
- Wideband 대역폭 / 표본: {state.config.wideband_bandwidth_percent:.3f}% / {state.config.wideband_frequency_samples}
- Near-field 초점: {focus_text}
- 채널 / 적응 빔포밍 / MUSIC DOA: {state.config.enable_channel_analysis} / {state.config.adaptive_beamforming_method.upper()} / {state.config.enable_doa_estimation}

## 성능 지표

- 상대 배열 이득: {relative_array_gain}
- 목표 방향 Directivity: {directivity_text}
- Directivity 계산 모드: {directivity_mode}
- Directivity 전구 적분: {directivity_method}
- Directivity 계산 경고: {directivity_warning}
- 활성 소자: {gain.active_elements} / {gain.total_elements}
- 테이퍼 효율: {100.0 * gain.taper_efficiency:.2f}%
- 위상·조향 효율: {100.0 * gain.phase_efficiency:.2f}%
- 좌표각 Azimuth HPBW / FNBW / SLL: {format_pattern_metric_summary(cuts.azimuth_metrics)}
- 좌표각 Elevation HPBW / FNBW / SLL: {format_pattern_metric_summary(cuts.elevation_metrics)}
{great_circle_report}
- 실제 Null 깊이: {format_depth(actual_null_depth) if state.config.enable_null_steering else "N/A"}

## Null 제약 진단

{null_report}
"""


def build_export_artifacts(
    state: SimulationState,
    cuts: PatternCuts,
    *,
    great_circle_cuts: GreatCircleCuts | None = None,
    directivity: DirectivityResult | None = None,
) -> ExportArtifacts:
    frame = build_pattern_export_frame(cuts, great_circle_cuts)
    pattern_csv = frame.to_csv(index=False).encode("utf-8")
    design_report = build_design_report(
        state,
        cuts,
        great_circle_cuts=great_circle_cuts,
        directivity=directivity,
    ).encode("utf-8")
    settings_json, reproducibility_zip = _build_reproducibility_files(
        state, pattern_csv, design_report, directivity
    )
    return ExportArtifacts(
        pattern_csv=pattern_csv,
        design_report=design_report,
        settings_json=settings_json,
        reproducibility_zip=reproducibility_zip,
    )
