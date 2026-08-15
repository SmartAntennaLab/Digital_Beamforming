"""Validated measured element-pattern grids and polarization evaluation."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import element_pattern_factor

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
MAX_PATTERN_FILE_BYTES = 1_000_000
MAX_PATTERN_SAMPLES = 16_384
REQUIRED_PATTERN_COLUMNS = (
    "azimuth_deg",
    "elevation_deg",
    "copol_gain_db",
)


@dataclass(frozen=True)
class ElementPatternGrid:
    """One rectangular co/cross-polar complex far-field pattern grid."""

    name: str
    source_sha256: str
    azimuth_deg: tuple[float, ...]
    elevation_deg: tuple[float, ...]
    copol_gain_db: tuple[tuple[float, ...], ...]
    copol_phase_deg: tuple[tuple[float, ...], ...]
    crosspol_gain_db: tuple[tuple[float, ...], ...]
    crosspol_phase_deg: tuple[tuple[float, ...], ...]

    @property
    def sample_count(self) -> int:
        return len(self.azimuth_deg) * len(self.elevation_deg)


def _finite_field(
    row: dict[str, str],
    name: str,
    *,
    minimum: float,
    maximum: float,
    default: float | None = None,
) -> float:
    text = row.get(name, "").strip()
    if not text and default is not None:
        return default
    value = float(text)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}].")
    return value


def parse_element_pattern_csv(
    payload: bytes,
    *,
    name: str = "uploaded_element_pattern.csv",
) -> ElementPatternGrid:
    """Parse a bounded rectangular pattern grid from untrusted CSV bytes."""

    if not payload:
        raise ValueError("Element-pattern CSV is empty.")
    if len(payload) > MAX_PATTERN_FILE_BYTES:
        raise ValueError("Element-pattern CSV exceeds the 1 MB limit.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Element-pattern CSV must be UTF-8 encoded.") from error
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = tuple(reader.fieldnames or ())
    missing = [name for name in REQUIRED_PATTERN_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError("Missing element-pattern columns: " + ", ".join(missing))

    samples: dict[tuple[float, float], tuple[float, float, float, float]] = {}
    for row_index, row in enumerate(reader, start=2):
        if len(samples) >= MAX_PATTERN_SAMPLES:
            raise ValueError("Element-pattern CSV has too many samples.")
        try:
            azimuth = _finite_field(
                row,
                "azimuth_deg",
                minimum=-180.0,
                maximum=180.0,
            )
            elevation = _finite_field(
                row,
                "elevation_deg",
                minimum=-90.0,
                maximum=90.0,
            )
            copol_gain = _finite_field(
                row,
                "copol_gain_db",
                minimum=-300.0,
                maximum=100.0,
            )
            copol_phase = _finite_field(
                row,
                "copol_phase_deg",
                minimum=-3600.0,
                maximum=3600.0,
                default=0.0,
            )
            crosspol_gain = _finite_field(
                row,
                "crosspol_gain_db",
                minimum=-300.0,
                maximum=100.0,
                default=-300.0,
            )
            crosspol_phase = _finite_field(
                row,
                "crosspol_phase_deg",
                minimum=-3600.0,
                maximum=3600.0,
                default=0.0,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid element-pattern row {row_index}: {error}"
            ) from error
        key = (azimuth, elevation)
        if key in samples:
            raise ValueError(f"Duplicate element-pattern direction at row {row_index}.")
        samples[key] = (
            copol_gain,
            copol_phase,
            crosspol_gain,
            crosspol_phase,
        )

    if not samples:
        raise ValueError("Element-pattern CSV has no data rows.")
    azimuth_axis = tuple(sorted({key[0] for key in samples}))
    elevation_axis = tuple(sorted({key[1] for key in samples}))
    expected = len(azimuth_axis) * len(elevation_axis)
    if expected != len(samples):
        raise ValueError(
            "Element-pattern samples must form a complete Azimuth/Elevation grid."
        )

    grids = [np.empty((len(elevation_axis), len(azimuth_axis))) for _ in range(4)]
    for elevation_index, elevation in enumerate(elevation_axis):
        for azimuth_index, azimuth in enumerate(azimuth_axis):
            values = samples[(azimuth, elevation)]
            for grid, value in zip(grids, values, strict=True):
                grid[elevation_index, azimuth_index] = value
    return ElementPatternGrid(
        name=name[:128],
        source_sha256=hashlib.sha256(payload).hexdigest(),
        azimuth_deg=azimuth_axis,
        elevation_deg=elevation_axis,
        copol_gain_db=tuple(tuple(float(value) for value in row) for row in grids[0]),
        copol_phase_deg=tuple(tuple(float(value) for value in row) for row in grids[1]),
        crosspol_gain_db=tuple(
            tuple(float(value) for value in row) for row in grids[2]
        ),
        crosspol_phase_deg=tuple(
            tuple(float(value) for value in row) for row in grids[3]
        ),
    )


def _complex_grid(gain_db, phase_deg) -> ComplexArray:
    gain = np.asarray(gain_db, dtype=float)
    phase = np.radians(np.asarray(phase_deg, dtype=float))
    return np.asarray(10.0 ** (gain / 20.0) * np.exp(1j * phase), dtype=complex)


def _axis_indices(axis: FloatArray, values: FloatArray):
    if axis.size == 1:
        zeros = np.zeros(values.size, dtype=int)
        return zeros, zeros, np.zeros(values.size, dtype=float)
    upper = np.searchsorted(axis, values, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    denominator = axis[upper] - axis[lower]
    fraction = np.divide(
        values - axis[lower],
        denominator,
        out=np.zeros_like(values),
        where=np.abs(denominator) > np.finfo(float).eps,
    )
    return lower, upper, np.clip(fraction, 0.0, 1.0)


def _bilinear_complex(
    grid: ComplexArray,
    azimuth_axis: FloatArray,
    elevation_axis: FloatArray,
    azimuth_deg: FloatArray,
    elevation_deg: FloatArray,
) -> ComplexArray:
    azimuth = np.clip(azimuth_deg.ravel(), azimuth_axis[0], azimuth_axis[-1])
    elevation = np.clip(elevation_deg.ravel(), elevation_axis[0], elevation_axis[-1])
    a0, a1, af = _axis_indices(azimuth_axis, azimuth)
    e0, e1, ef = _axis_indices(elevation_axis, elevation)
    lower = grid[e0, a0] * (1.0 - af) + grid[e0, a1] * af
    upper = grid[e1, a0] * (1.0 - af) + grid[e1, a1] * af
    return np.asarray((lower * (1.0 - ef) + upper * ef).reshape(azimuth_deg.shape))


def evaluate_element_pattern(
    element_option: str,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
    *,
    pattern_grid: ElementPatternGrid | None = None,
    polarization_angle_deg: float = 0.0,
) -> ComplexArray:
    """Evaluate built-in or measured patterns with linear polarization mismatch."""

    azimuth, elevation = np.broadcast_arrays(
        np.asarray(azimuth_rad, dtype=float),
        np.asarray(elevation_rad, dtype=float),
    )
    polarization = np.radians(float(polarization_angle_deg))
    if pattern_grid is None:
        factor = element_pattern_factor(element_option, azimuth, elevation)
        return np.asarray(factor * np.cos(polarization), dtype=complex)

    azimuth_axis = np.asarray(pattern_grid.azimuth_deg, dtype=float)
    elevation_axis = np.asarray(pattern_grid.elevation_deg, dtype=float)
    azimuth_deg = (np.degrees(azimuth) + 180.0) % 360.0 - 180.0
    elevation_deg = np.degrees(elevation)
    copol = _bilinear_complex(
        _complex_grid(pattern_grid.copol_gain_db, pattern_grid.copol_phase_deg),
        azimuth_axis,
        elevation_axis,
        azimuth_deg,
        elevation_deg,
    )
    crosspol = _bilinear_complex(
        _complex_grid(
            pattern_grid.crosspol_gain_db,
            pattern_grid.crosspol_phase_deg,
        ),
        azimuth_axis,
        elevation_axis,
        azimuth_deg,
        elevation_deg,
    )
    return np.asarray(
        copol * np.cos(polarization) + crosspol * np.sin(polarization),
        dtype=complex,
    )


__all__ = [
    "ElementPatternGrid",
    "evaluate_element_pattern",
    "parse_element_pattern_csv",
]
