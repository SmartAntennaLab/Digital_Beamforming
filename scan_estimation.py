"""Raster scan direction and bounded-work timing estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_options import SCAN_MODE_OPTIONS
from pattern_sampling import (
    GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT,
    PATTERN_CUT_BASE_SAMPLE_COUNT,
    pattern_cut_local_sample_count,
    scan_surface_sampling,
)

SCAN_REFERENCE_ELEMENT_COUNT = 4096
SCAN_REFERENCE_FULL_FRAME_SECONDS = 0.85


@dataclass(frozen=True)
class ScanTimingEstimate:
    frame_seconds: float
    effective_interval_seconds: float
    finalization_seconds: float
    total_seconds: float
    frame_count: int


def _scan_render_work_units(element_count: int, scan_mode: str) -> int:
    sampling = scan_surface_sampling(element_count, scan_mode, scanning=True)
    cut_points = 2 * (
        PATTERN_CUT_BASE_SAMPLE_COUNT
        + GREAT_CIRCLE_CUT_BASE_SAMPLE_COUNT
        + 2 * pattern_cut_local_sample_count(element_count)
    )
    surface_points = 0
    if sampling.render_3d:
        side = int(sampling.resolution or 0) + int(sampling.local_sample_count or 0)
        surface_points = side**2
    return element_count * (cut_points + surface_points)


def estimate_scan_timing(
    element_count: int,
    frame_count: int,
    scan_mode: str,
    frame_interval_seconds: float,
    *,
    session_calculations_per_minute: int | None = None,
    session_burst: int = 1,
) -> ScanTimingEstimate:
    """Estimate scan duration from relative array-factor work."""

    if element_count < 1 or frame_count < 1:
        raise ValueError("Element and frame counts must be positive.")
    if not np.isfinite(frame_interval_seconds) or frame_interval_seconds < 0.0:
        raise ValueError("Frame interval must be finite and non-negative.")
    if scan_mode not in SCAN_MODE_OPTIONS:
        raise ValueError("Unsupported scan mode.")
    if (
        session_calculations_per_minute is not None
        and session_calculations_per_minute < 1
    ):
        raise ValueError("Session calculation rate must be positive.")
    if session_burst < 1:
        raise ValueError("Session burst must be positive.")
    reference = _scan_render_work_units(SCAN_REFERENCE_ELEMENT_COUNT, "full_3d")
    frame_seconds = max(
        0.02,
        SCAN_REFERENCE_FULL_FRAME_SECONDS
        * _scan_render_work_units(element_count, scan_mode)
        / reference,
    )
    effective_interval = max(frame_seconds, frame_interval_seconds)
    finalization = 0.0
    if scan_mode != "full_3d":
        finalization = max(
            0.02,
            SCAN_REFERENCE_FULL_FRAME_SECONDS
            * _scan_render_work_units(element_count, "full_3d")
            / reference,
        )
    total = frame_count * effective_interval
    if session_calculations_per_minute is not None:
        refill_rate = session_calculations_per_minute / 60.0
        capacity = float(min(session_burst, session_calculations_per_minute))
        tokens, total, updated_at = capacity, 0.0, 0.0
        for _ in range(frame_count):
            tokens = min(capacity, tokens + (total - updated_at) * refill_rate)
            updated_at = total
            if tokens < 1.0:
                total += (1.0 - tokens) / refill_rate
                tokens, updated_at = 1.0, total
            tokens -= 1.0
            total += effective_interval
    return ScanTimingEstimate(
        frame_seconds=float(frame_seconds),
        effective_interval_seconds=float(effective_interval),
        finalization_seconds=float(finalization),
        total_seconds=float(total + finalization),
        frame_count=frame_count,
    )


def scan_direction(
    index: int,
    azimuth_range_deg: tuple[float, float],
    elevation_range_deg: tuple[float, float],
    azimuth_steps: int,
    elevation_steps: int,
) -> tuple[float, float, int]:
    """Resolve one azimuth-first raster index."""

    if azimuth_steps < 1 or elevation_steps < 1:
        raise ValueError("Scan step counts must be positive.")
    total_steps = azimuth_steps * elevation_steps
    if not 0 <= index < total_steps:
        raise ValueError("Scan index is outside the configured raster.")
    elevation_index, azimuth_index = divmod(index, azimuth_steps)
    azimuth_fraction = azimuth_index / (azimuth_steps - 1) if azimuth_steps > 1 else 0.0
    elevation_fraction = (
        elevation_index / (elevation_steps - 1) if elevation_steps > 1 else 0.0
    )
    azimuth = azimuth_range_deg[0] + azimuth_fraction * (
        azimuth_range_deg[1] - azimuth_range_deg[0]
    )
    elevation = elevation_range_deg[0] + elevation_fraction * (
        elevation_range_deg[1] - elevation_range_deg[0]
    )
    return float(azimuth), float(elevation), total_steps


__all__ = ["ScanTimingEstimate", "estimate_scan_timing", "scan_direction"]
