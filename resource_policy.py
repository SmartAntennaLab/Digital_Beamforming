"""Per-session compute limits for interactive and public deployments."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

HARD_MAX_ELEMENTS = 16_384
HARD_MAX_DIRECTIVITY_EXACT_ELEMENTS = 4_096
HARD_MAX_SCAN_FRAMES = 1_000
HARD_MAX_SCAN_ELEMENT_FRAMES = 4_000_000
HARD_MAX_CONCURRENT_CALCULATIONS = 32
HARD_MAX_COMPUTE_SECONDS = 120.0
HARD_MAX_SESSION_CALCULATIONS_PER_MINUTE = 600
HARD_MAX_SESSION_BURST = 60
HARD_MAX_QUEUE_TIMEOUT_SECONDS = 30.0
HARD_MAX_HEALTH_LOG_INTERVAL_SECONDS = 300.0


def _bounded_environment_int(name: str, default: int, hard_maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(1, value), hard_maximum)


def _bounded_environment_float(
    name: str,
    default: float,
    hard_maximum: float,
    *,
    minimum: float = 0.01,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), hard_maximum)


@dataclass(frozen=True)
class ResourcePolicy:
    """Maximum work accepted from one browser session request."""

    max_elements: int = 4_096
    directivity_warning_elements: int = 1_024
    directivity_exact_max_elements: int = 4_096
    max_scan_frames: int = 400
    max_scan_element_frames: int = 1_000_000
    max_concurrent_calculations: int = 2
    compute_queue_timeout_seconds: float = 1.0
    compute_timeout_seconds: float = 10.0
    session_calculations_per_minute: int = 120
    session_burst: int = 8
    health_log_interval_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> ResourcePolicy:
        """Load bounded operator overrides without exceeding hard limits."""

        calculations_per_minute = _bounded_environment_int(
            "DBF_SESSION_CALCULATIONS_PER_MINUTE",
            cls.session_calculations_per_minute,
            HARD_MAX_SESSION_CALCULATIONS_PER_MINUTE,
        )
        session_burst = min(
            _bounded_environment_int(
                "DBF_SESSION_BURST",
                cls.session_burst,
                HARD_MAX_SESSION_BURST,
            ),
            calculations_per_minute,
        )
        directivity_exact_max_elements = _bounded_environment_int(
            "DBF_DIRECTIVITY_EXACT_MAX_ELEMENTS",
            cls.directivity_exact_max_elements,
            HARD_MAX_DIRECTIVITY_EXACT_ELEMENTS,
        )
        directivity_warning_elements = min(
            _bounded_environment_int(
                "DBF_DIRECTIVITY_WARNING_ELEMENTS",
                cls.directivity_warning_elements,
                HARD_MAX_DIRECTIVITY_EXACT_ELEMENTS,
            ),
            directivity_exact_max_elements,
        )
        return cls(
            max_elements=_bounded_environment_int(
                "DBF_MAX_ELEMENTS",
                cls.max_elements,
                HARD_MAX_ELEMENTS,
            ),
            directivity_warning_elements=directivity_warning_elements,
            directivity_exact_max_elements=directivity_exact_max_elements,
            max_scan_frames=_bounded_environment_int(
                "DBF_MAX_SCAN_FRAMES",
                cls.max_scan_frames,
                HARD_MAX_SCAN_FRAMES,
            ),
            max_scan_element_frames=_bounded_environment_int(
                "DBF_MAX_SCAN_ELEMENT_FRAMES",
                cls.max_scan_element_frames,
                HARD_MAX_SCAN_ELEMENT_FRAMES,
            ),
            max_concurrent_calculations=_bounded_environment_int(
                "DBF_MAX_CONCURRENT_CALCULATIONS",
                cls.max_concurrent_calculations,
                HARD_MAX_CONCURRENT_CALCULATIONS,
            ),
            compute_queue_timeout_seconds=_bounded_environment_float(
                "DBF_COMPUTE_QUEUE_TIMEOUT_SECONDS",
                cls.compute_queue_timeout_seconds,
                HARD_MAX_QUEUE_TIMEOUT_SECONDS,
            ),
            compute_timeout_seconds=_bounded_environment_float(
                "DBF_COMPUTE_TIMEOUT_SECONDS",
                cls.compute_timeout_seconds,
                HARD_MAX_COMPUTE_SECONDS,
            ),
            session_calculations_per_minute=calculations_per_minute,
            session_burst=session_burst,
            health_log_interval_seconds=_bounded_environment_float(
                "DBF_HEALTH_LOG_INTERVAL_SECONDS",
                cls.health_log_interval_seconds,
                HARD_MAX_HEALTH_LOG_INTERVAL_SECONDS,
                minimum=1.0,
            ),
        )


def estimate_element_count(
    geometry: str,
    vertical_count: int,
    horizontal_count: int,
) -> int:
    """Return physical element count without allocating coordinate arrays."""

    if vertical_count < 1 or horizontal_count < 1:
        raise ValueError("Element counts must be positive.")
    geometry_id = geometry.strip().upper().split(maxsplit=1)[0]
    if geometry_id in {"ULA", "UCA"}:
        return horizontal_count
    if geometry_id == "UPA":
        return vertical_count * horizontal_count
    if geometry_id == "UHA":
        if vertical_count > horizontal_count:
            raise ValueError("UHA requires Nmin <= Nmax.")
        return horizontal_count**2 - vertical_count * (vertical_count - 1)
    raise ValueError(f"Unsupported geometry: {geometry!r}")


def resource_limit_message(
    policy: ResourcePolicy,
    *,
    geometry: str,
    vertical_count: int,
    horizontal_count: int,
    scan_frames: int,
) -> str | None:
    """Return a user-facing rejection reason, or ``None`` when allowed."""

    element_count = estimate_element_count(
        geometry,
        vertical_count,
        horizontal_count,
    )
    if element_count > policy.max_elements:
        return (
            f"배열 소자 {element_count:,}개가 세션 상한 "
            f"{policy.max_elements:,}개를 초과합니다."
        )
    if scan_frames > policy.max_scan_frames:
        return (
            f"스캔 {scan_frames:,}프레임이 세션 상한 "
            f"{policy.max_scan_frames:,}프레임을 초과합니다."
        )
    element_frames = element_count * scan_frames
    if element_frames > policy.max_scan_element_frames:
        return (
            f"스캔 작업량 {element_frames:,} element-frames가 세션 상한 "
            f"{policy.max_scan_element_frames:,}을 초과합니다."
        )
    return None
