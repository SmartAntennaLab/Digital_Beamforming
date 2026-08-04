"""Shared direction, element-pattern, phase, and steering-vector primitives."""

from __future__ import annotations

import re
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


def parse_phase_bits(phase_bits: int | str | None) -> int | None:
    """Convert phase-resolution values to an integer or ideal ``None``."""

    if phase_bits is None:
        return None
    if isinstance(phase_bits, (int, np.integer)):
        bits = int(phase_bits)
    else:
        text = str(phase_bits).strip().lower()
        if text.startswith("infinite") or text in {"none", "ideal"}:
            return None
        match = re.fullmatch(r"(\d+)\s*-?\s*bit", text)
        if not match:
            raise ValueError(f"Invalid phase-bit value: {phase_bits!r}")
        bits = int(match.group(1))
    if bits < 1:
        raise ValueError("Phase resolution must be at least one bit.")
    return bits


def quantize_phases(
    phases_rad: ArrayLike,
    phase_bits: int | str | None,
) -> FloatArray:
    """Quantize phases to equally spaced states over 2π."""

    phases = np.asarray(phases_rad, dtype=float)
    bits = parse_phase_bits(phase_bits)
    if bits is None:
        return phases.copy()
    step = 2.0 * np.pi / (2**bits)
    return np.round(phases / step) * step


def direction_cosines(
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return X, Y, and Z direction cosines for the simulator convention."""

    azimuth = np.asarray(azimuth_rad, dtype=float)
    elevation = np.asarray(elevation_rad, dtype=float)
    azimuth, elevation = np.broadcast_arrays(azimuth, elevation)
    cosine_elevation = np.cos(elevation)
    return (
        cosine_elevation * np.cos(azimuth),
        cosine_elevation * np.sin(azimuth),
        np.sin(elevation),
    )


def element_pattern_factor(
    option: str,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
) -> FloatArray:
    """Evaluate one element-pattern model from shared direction cosines."""

    u_x, _, u_z = direction_cosines(azimuth_rad, elevation_rad)
    key = option.strip().lower()
    if key == "isotropic" or key.startswith("isotropic "):
        factor = np.ones_like(u_x, dtype=float)
    elif key == "cosine_squared" or key.startswith("cosine²") or "제곱" in key:
        factor = np.maximum(0.0, u_x) ** 2
    elif key == "cosine" or key.startswith("cosine "):
        factor = np.maximum(0.0, u_x)
    elif key == "dipole" or key.startswith("dipole "):
        transverse = np.sqrt(np.maximum(0.0, 1.0 - u_z**2))
        numerator = np.cos((np.pi / 2.0) * u_z)
        factor = np.divide(
            numerator,
            transverse,
            out=np.zeros_like(transverse, dtype=float),
            where=transverse > 1e-12,
        )
        factor = np.maximum(0.0, factor)
    else:
        raise ValueError(f"Unsupported element pattern: {option!r}")
    return np.asarray(factor, dtype=float)


def steering_phases(
    y: ArrayLike,
    z: ArrayLike,
    wavelength_m: float,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
) -> FloatArray:
    """Return vectorized array-response phases for the shared convention."""

    if wavelength_m <= 0 or not np.isfinite(wavelength_m):
        raise ValueError("Wavelength must be a finite positive value.")
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    if y_array.shape != z_array.shape:
        raise ValueError("Y and Z coordinate arrays must have matching shapes.")
    if np.any(~np.isfinite(y_array)) or np.any(~np.isfinite(z_array)):
        raise ValueError("Element coordinates must be finite.")
    _, u_y, u_z = direction_cosines(azimuth_rad, elevation_rad)
    phase = 2.0 * np.pi / wavelength_m * (
        u_y[..., None] * y_array.ravel()
        + u_z[..., None] * z_array.ravel()
    )
    return np.asarray(phase.reshape(u_y.shape + y_array.shape), dtype=float)


def steering_vector(
    y: ArrayLike,
    z: ArrayLike,
    wavelength_m: float,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
) -> ComplexArray:
    """Return the response vector used by beamforming and array-factor sums."""

    phases = steering_phases(
        y,
        z,
        wavelength_m,
        azimuth_rad,
        elevation_rad,
    )
    return np.asarray(np.exp(1j * phases), dtype=complex)
