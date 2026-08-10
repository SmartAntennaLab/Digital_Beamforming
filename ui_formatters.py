"""Presentation-only formatters shared by Streamlit UI modules."""

from __future__ import annotations

import numpy as np

from beamforming import ConstraintDiagnostics, get_steering_limits
from simulation import SimulationState


def format_depth(depth_db: float | None) -> str:
    if depth_db is None:
        return "N/A"
    if depth_db >= 299.95:
        return "≥ 300 dB"
    displayed = 0.0 if abs(depth_db) < 0.005 else depth_db
    return f"{displayed:.2f} dB"


def format_response_error(diagnostics: ConstraintDiagnostics) -> str:
    absolute = diagnostics.target_response_error
    relative = diagnostics.target_relative_error
    if absolute is None or relative is None:
        return "N/A"
    return f"{absolute:.3e} ({100.0 * relative:.3e} %)"


def relative_residual_db(relative_residual: float | None) -> float | None:
    if relative_residual is None or not np.isfinite(relative_residual):
        return None
    return float(20.0 * np.log10(max(relative_residual, 1e-15)))


def format_residual_db(relative_residual: float | None) -> str:
    level_db = relative_residual_db(relative_residual)
    return f"{level_db:.2f} dB" if level_db is not None else "N/A"


def format_absolute_residual(residual: float | None) -> str:
    if residual is None or not np.isfinite(residual):
        return "N/A"
    return f"{residual:.3e}"


def format_angle_metric(value_deg: float | None) -> str:
    return f"{value_deg:.2f}°" if value_deg is not None else "N/A"


def format_sidelobe_metric(
    level_db: float | None,
    angle_deg: float | None,
) -> str:
    if level_db is None or angle_deg is None:
        return "N/A"
    return f"{level_db:.2f} dB (@ {angle_deg:.1f}°)"


def format_pattern_metric_summary(metrics: object) -> str:
    return " / ".join(
        (
            format_angle_metric(metrics.hpbw_deg),
            format_angle_metric(metrics.first_null_beamwidth_deg),
            format_sidelobe_metric(
                metrics.sidelobe_level_db,
                metrics.sidelobe_angle_deg,
            ),
        )
    )


def format_degradation(degradation_db: float | None) -> str:
    if degradation_db is None or not np.isfinite(degradation_db):
        return "N/A"
    displayed = 0.0 if abs(degradation_db) < 0.005 else degradation_db
    return f"{displayed:+.2f} dB"


def null_solver_label(method: str) -> str:
    return {
        "svd_minimum_norm": "SVD 최소노름",
        "phase_only_projected_gradient": "위상 전용 반복 최적화",
        "svd_rejected": "SVD 조건 불량",
        "svd_failed": "SVD 계산 실패",
        "not_requested": "미사용",
    }.get(method, method)


def steering_axes_text(state: SimulationState) -> str:
    limits = get_steering_limits(
        state.coordinates.rows,
        state.coordinates.columns,
        state.coordinates.geometry,
    )
    if limits.azimuth_controllable and limits.elevation_controllable:
        return "Azimuth 및 Elevation"
    if limits.azimuth_controllable:
        return "Azimuth (Elevation 0° 고정)"
    if limits.elevation_controllable:
        return "Elevation (Azimuth 0° 고정)"
    return "전자 조향 불가 (Azimuth/Elevation 0° 고정)"


def spacing_text(state: SimulationState) -> str:
    horizontal = state.config.horizontal_spacing_wavelength
    vertical = state.config.vertical_spacing_wavelength
    if state.coordinates.geometry == "UCA":
        return f"인접 chord dy={horizontal:.2f} λ"
    if state.coordinates.geometry == "ULA":
        return f"수평 dy={horizontal:.2f} λ"
    if state.coordinates.geometry == "UHA":
        row_spacing = horizontal * np.sin(np.pi / 3.0)
        return f"최근접 dy={horizontal:.2f} λ / 행 간격 dz={row_spacing:.3f} λ"
    return f"수평 dy={horizontal:.2f} λ / 수직 dz={vertical:.2f} λ"


def array_size_text(state: SimulationState) -> str:
    coordinates = state.coordinates
    if coordinates.geometry == "UHA":
        return (
            f"Nmin={state.config.vertical_count}, "
            f"Nmax={state.config.horizontal_count}, "
            f"{coordinates.rows}개 행 / {coordinates.element_count}개 소자"
        )
    return f"{coordinates.rows} × {coordinates.columns}"


def wavelength_cm_text(wavelength_value: float, centimeter_value: float) -> str:
    return f"{wavelength_value:.3f} λ / {centimeter_value:.3f} cm"


__all__ = [name for name in globals() if name.startswith("format_")] + [
    "array_size_text",
    "null_solver_label",
    "relative_residual_db",
    "spacing_text",
    "steering_axes_text",
    "wavelength_cm_text",
]
