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


def great_circle_directions(
    target_azimuth_rad: float,
    target_elevation_rad: float,
    angular_offset_rad: ArrayLike,
    *,
    plane: str,
) -> tuple[FloatArray, FloatArray]:
    """Return directions on a target-centered great-circle principal plane.

    ``angular_offset_rad`` is signed spherical angular distance from the
    target.  The horizontal plane starts along increasing azimuth and the
    vertical plane starts along increasing elevation.  Unlike a fixed
    azimuth/elevation coordinate cut, one radian on either returned curve is
    always one radian of physical angular distance on the unit sphere.
    """

    target_azimuth = float(target_azimuth_rad)
    target_elevation = float(target_elevation_rad)
    if not np.isfinite(target_azimuth) or not np.isfinite(target_elevation):
        raise ValueError("Target angles must be finite.")
    if not -np.pi / 2.0 <= target_elevation <= np.pi / 2.0:
        raise ValueError("Target elevation must be between -pi/2 and pi/2.")

    offsets = np.asarray(angular_offset_rad, dtype=float)
    if np.any(~np.isfinite(offsets)):
        raise ValueError("Great-circle offsets must be finite.")

    cosine_elevation = np.cos(target_elevation)
    target = np.array(
        [
            cosine_elevation * np.cos(target_azimuth),
            cosine_elevation * np.sin(target_azimuth),
            np.sin(target_elevation),
        ],
        dtype=float,
    )
    if plane == "horizontal":
        tangent = np.array(
            [-np.sin(target_azimuth), np.cos(target_azimuth), 0.0],
            dtype=float,
        )
    elif plane == "vertical":
        tangent = np.array(
            [
                -np.sin(target_elevation) * np.cos(target_azimuth),
                -np.sin(target_elevation) * np.sin(target_azimuth),
                np.cos(target_elevation),
            ],
            dtype=float,
        )
    else:
        raise ValueError("Great-circle plane must be 'horizontal' or 'vertical'.")

    directions = (
        np.cos(offsets)[..., None] * target + np.sin(offsets)[..., None] * tangent
    )
    azimuth = np.arctan2(directions[..., 1], directions[..., 0])
    elevation = np.arcsin(np.clip(directions[..., 2], -1.0, 1.0))
    return (
        np.asarray(azimuth, dtype=float),
        np.asarray(elevation, dtype=float),
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
    phase = (
        2.0
        * np.pi
        / wavelength_m
        * (u_y[..., None] * y_array.ravel() + u_z[..., None] * z_array.ravel())
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
