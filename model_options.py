"""Stable simulator option IDs and their user-facing Korean labels."""

from __future__ import annotations

from collections.abc import Mapping

GEOMETRY_LABELS: dict[str, str] = {
    "UPA": "UPA (사각형 평면형)",
    "UHA": "UHA (균일 육각 평면형)",
    "ULA": "ULA (수평 선형)",
    "UCA": "UCA (수평 원형)",
}
TAPER_LABELS: dict[str, str] = {
    "uniform": "Uniform (균일)",
    "hamming": "Hamming",
    "hanning": "Hanning",
    "blackman": "Blackman",
    "bartlett": "Bartlett",
}
ELEMENT_PATTERN_LABELS: dict[str, str] = {
    "isotropic": "Isotropic (등방성)",
    "cosine": "Cosine (코사인)",
    "cosine_squared": "Cosine² (코사인 제곱)",
    "dipole": "Dipole (다이폴)",
}
DIRECTIVITY_MODE_LABELS: dict[str, str] = {
    "auto": "자동 (대형 배열은 고속 근사)",
    "exact": "정확 (해석적 pairwise)",
    "fast": "고속 근사 (비균일 전구 적분)",
}
PHASE_BIT_LABELS: dict[int | None, str] = {
    None: "Infinite (무한)",
    2: "2-bit",
    3: "3-bit",
    4: "4-bit",
    5: "5-bit",
    6: "6-bit",
}
SCALE_LABELS: dict[str, str] = {
    "db": "dB Scale",
    "linear": "Linear Scale",
}
COORDINATE_LABELS: dict[str, str] = {
    "polar": "Polar (극좌표)",
    "rectangular": "Rectangular (직각좌표)",
}
SCAN_MODE_LABELS: dict[str, str] = {
    "2d": "2D 전용",
    "preview_3d": "3D 미리보기",
    "full_3d": "전체 품질 3D",
}
NULL_OPTIMIZATION_MODE_LABELS: dict[str, str] = {
    "amplitude_phase": "진폭·위상 동시 제어",
    "phase_only": "위상 전용 최적화",
}

GEOMETRY_OPTIONS = tuple(GEOMETRY_LABELS)
TAPER_OPTIONS = tuple(TAPER_LABELS)
ELEMENT_OPTIONS = tuple(ELEMENT_PATTERN_LABELS)
DIRECTIVITY_MODE_OPTIONS = tuple(DIRECTIVITY_MODE_LABELS)
PHASE_BIT_OPTIONS = tuple(PHASE_BIT_LABELS)
SCALE_OPTIONS = tuple(SCALE_LABELS)
COORDINATE_OPTIONS = tuple(COORDINATE_LABELS)
SCAN_MODE_OPTIONS = tuple(SCAN_MODE_LABELS)
NULL_OPTIMIZATION_MODE_OPTIONS = tuple(NULL_OPTIMIZATION_MODE_LABELS)


def option_label(value: object, labels: Mapping[object, str]) -> str:
    """Return a UI label for one stable option ID."""

    return labels.get(value, str(value))


def normalize_option_id(value: object, labels: Mapping[object, str]) -> object:
    """Accept a stable ID or one legacy translated label and return the ID."""

    try:
        if value in labels:
            return value
    except TypeError:
        pass
    text = str(value).strip()
    for option_id, label in labels.items():
        if text == label:
            return option_id
    raise ValueError("Unsupported option.")
