"""Per-session compute limits for interactive and public deployments."""

from __future__ import annotations

from dataclasses import dataclass
import os


HARD_MAX_ELEMENTS = 16_384
HARD_MAX_SCAN_FRAMES = 1_000
HARD_MAX_SCAN_ELEMENT_FRAMES = 4_000_000


def _bounded_environment_int(name: str, default: int, hard_maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(1, value), hard_maximum)


@dataclass(frozen=True)
class ResourcePolicy:
    """Maximum work accepted from one browser session request."""

    max_elements: int = 4_096
    max_scan_frames: int = 400
    max_scan_element_frames: int = 1_000_000

    @classmethod
    def from_environment(cls) -> "ResourcePolicy":
        """Load bounded operator overrides without exceeding hard limits."""

        return cls(
            max_elements=_bounded_environment_int(
                "DBF_MAX_ELEMENTS",
                cls.max_elements,
                HARD_MAX_ELEMENTS,
            ),
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
