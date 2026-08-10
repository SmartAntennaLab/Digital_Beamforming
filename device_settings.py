"""Validation and sharing helpers for per-browser simulator settings."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from model_options import (
    COORDINATE_LABELS,
    COORDINATE_OPTIONS,
    ELEMENT_OPTIONS,
    ELEMENT_PATTERN_LABELS,
    GEOMETRY_LABELS,
    GEOMETRY_OPTIONS,
    PHASE_BIT_LABELS,
    NULL_OPTIMIZATION_MODE_LABELS,
    NULL_OPTIMIZATION_MODE_OPTIONS,
    PHASE_BIT_OPTIONS,
    SCALE_LABELS,
    SCALE_OPTIONS,
    SCAN_MODE_LABELS,
    SCAN_MODE_OPTIONS,
    TAPER_LABELS,
    TAPER_OPTIONS,
    normalize_option_id,
)

DEVICE_SETTINGS_SCHEMA_VERSION = 4
DEVICE_STORAGE_KEY = "digital_beamforming.settings.v1"

DEFAULT_DEVICE_SETTINGS: dict[str, object] = {
    "array_geometry": "UPA",
    "frequency_ghz": 28.0,
    "horizontal_count": 4,
    "vertical_count": 4,
    "uha_max_count": 4,
    "uha_min_count": 2,
    "horizontal_spacing": 0.5,
    "vertical_spacing": 0.5,
    "taper_option": "uniform",
    "element_option": "isotropic",
    "phase_bits": None,
    "failure_rate": 0,
    "target_azimuth": 0.0,
    "target_elevation": 0.0,
    "enable_null": False,
    "null_azimuth": 30.0,
    "null_elevation": 0.0,
    "null_count": 1,
    "null_1_suppression_db": 40.0,
    "null_optimization_mode": "amplitude_phase",
    "enable_amplitude_limit": False,
    "max_element_amplitude": 1.0,
    "scale_option": "db",
    "coordinate_option": "polar",
    "show_3db": True,
    "show_3db_value": True,
    "scan_azimuth_range": (-45.0, 45.0),
    "scan_azimuth_steps": 10,
    "scan_elevation_range": (-15.0, 15.0),
    "scan_elevation_steps": 5,
    "scan_delay": 0.2,
    "scan_mode": "preview_3d",
}

for null_index, default_azimuth in enumerate(
    (-30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0),
    start=2,
):
    DEFAULT_DEVICE_SETTINGS[f"null_{null_index}_azimuth"] = default_azimuth
    DEFAULT_DEVICE_SETTINGS[f"null_{null_index}_elevation"] = 0.0
    DEFAULT_DEVICE_SETTINGS[f"null_{null_index}_suppression_db"] = 40.0


def _choice(options: Sequence[str]) -> Callable[[object], str]:
    def validate(value: object) -> str:
        text = str(value)
        if text not in options:
            raise ValueError("Unsupported choice.")
        return text

    return validate


def _option(labels: Mapping[object, str]) -> Callable[[object], object]:
    def validate(value: object) -> object:
        return normalize_option_id(value, labels)

    return validate


def _finite_float(minimum: float, maximum: float) -> Callable[[object], float]:
    def validate(value: object) -> float:
        number = float(value)
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError("Number is outside the supported range.")
        return number

    return validate


def _bounded_int(minimum: int, maximum: int) -> Callable[[object], int]:
    def validate(value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("Boolean is not an integer setting.")
        number = int(value)
        if float(value) != number or not minimum <= number <= maximum:
            raise ValueError("Integer is outside the supported range.")
        return number

    return validate


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError("Invalid Boolean setting.")


def _range_pair(
    minimum: float, maximum: float
) -> Callable[[object], tuple[float, float]]:
    def validate(value: object) -> tuple[float, float]:
        candidate: object = value
        if isinstance(candidate, str):
            text = candidate.strip()
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                candidate = [part.strip() for part in text.split(",")]
        if (
            not isinstance(candidate, Sequence)
            or isinstance(candidate, (str, bytes))
            or len(candidate) != 2
        ):
            raise ValueError("Range setting must contain two values.")
        start = float(candidate[0])
        stop = float(candidate[1])
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or start < minimum
            or stop > maximum
            or start > stop
        ):
            raise ValueError("Range setting is outside the supported bounds.")
        return (start, stop)

    return validate


SETTING_VALIDATORS: dict[str, Callable[[object], object]] = {
    "array_geometry": _option(GEOMETRY_LABELS),
    "frequency_ghz": _finite_float(1.0, 60.0),
    "horizontal_count": _bounded_int(1, 128),
    "vertical_count": _bounded_int(1, 128),
    "uha_max_count": _bounded_int(1, 128),
    "uha_min_count": _bounded_int(1, 128),
    "horizontal_spacing": _finite_float(0.1, 1.0),
    "vertical_spacing": _finite_float(0.1, 1.0),
    "taper_option": _option(TAPER_LABELS),
    "element_option": _option(ELEMENT_PATTERN_LABELS),
    "phase_bits": _option(PHASE_BIT_LABELS),
    "failure_rate": _bounded_int(0, 50),
    "target_azimuth": _finite_float(-90.0, 90.0),
    "target_elevation": _finite_float(-90.0, 90.0),
    "enable_null": _boolean,
    "null_azimuth": _finite_float(-90.0, 90.0),
    "null_elevation": _finite_float(-90.0, 90.0),
    "null_count": _bounded_int(1, 8),
    "null_1_suppression_db": _finite_float(0.0, 120.0),
    "null_optimization_mode": _option(NULL_OPTIMIZATION_MODE_LABELS),
    "enable_amplitude_limit": _boolean,
    "max_element_amplitude": _finite_float(0.05, 10.0),
    "scale_option": _option(SCALE_LABELS),
    "coordinate_option": _option(COORDINATE_LABELS),
    "show_3db": _boolean,
    "show_3db_value": _boolean,
    "scan_azimuth_range": _range_pair(-90.0, 90.0),
    "scan_azimuth_steps": _bounded_int(3, 50),
    "scan_elevation_range": _range_pair(-90.0, 90.0),
    "scan_elevation_steps": _bounded_int(2, 20),
    "scan_delay": _finite_float(0.1, 2.0),
    "scan_mode": _option(SCAN_MODE_LABELS),
}

for null_index in range(2, 9):
    SETTING_VALIDATORS[f"null_{null_index}_azimuth"] = _finite_float(-90.0, 90.0)
    SETTING_VALIDATORS[f"null_{null_index}_elevation"] = _finite_float(-90.0, 90.0)
    SETTING_VALIDATORS[f"null_{null_index}_suppression_db"] = _finite_float(0.0, 120.0)

DEVICE_SETTING_KEYS = tuple(SETTING_VALIDATORS)


def sanitize_device_settings(settings: object) -> dict[str, object]:
    """Return only recognized, type-safe settings from untrusted browser data."""

    if not isinstance(settings, Mapping):
        return {}
    if "settings" in settings:
        version = settings.get("schema_version")
        if version not in {1, 2, 3, DEVICE_SETTINGS_SCHEMA_VERSION}:
            return {}
        settings = settings.get("settings")
        if not isinstance(settings, Mapping):
            return {}

    sanitized: dict[str, object] = {}
    for key, validator in SETTING_VALIDATORS.items():
        if key not in settings:
            continue
        try:
            sanitized[key] = validator(settings[key])
        except (TypeError, ValueError, OverflowError):
            continue

    maximum = sanitized.get("uha_max_count")
    minimum = sanitized.get("uha_min_count")
    if isinstance(maximum, int) and isinstance(minimum, int) and minimum > maximum:
        sanitized["uha_min_count"] = maximum
    return sanitized


def collect_device_settings(state: Mapping[str, Any]) -> dict[str, object]:
    """Collect and validate all persistent widget values present in state."""

    return sanitize_device_settings(
        {key: state[key] for key in DEVICE_SETTING_KEYS if key in state}
    )


def settings_envelope(settings: Mapping[str, Any]) -> dict[str, object]:
    """Wrap validated settings in a versioned local-storage payload."""

    return {
        "schema_version": DEVICE_SETTINGS_SCHEMA_VERSION,
        "settings": sanitize_device_settings(settings),
    }


def encode_share_token(settings: Mapping[str, Any]) -> str:
    """Encode settings as URL-safe, versioned JSON without padding."""

    serialized = json.dumps(
        settings_envelope(settings),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(serialized).decode("ascii").rstrip("=")


def decode_share_token(token: str) -> dict[str, object]:
    """Decode and validate one URL-safe settings token."""

    if not isinstance(token, str) or not token or len(token) > 16_384:
        return {}
    try:
        padding = "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode((token + padding).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return {}
    return sanitize_device_settings(payload)
