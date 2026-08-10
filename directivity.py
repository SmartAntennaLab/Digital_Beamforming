"""Full-sphere radiation-power integration and directional directivity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import element_pattern_factor, steering_phases
from pattern_metrics import array_factor


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
DIRECTIVITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DirectivityResult:
    """Target-direction directivity and its full-sphere integration details."""

    schema_version: int
    directivity_linear: float | None
    directivity_dbi: float | None
    target_radiation_intensity: float
    radiated_power_integral: float
    integration_method: str
    azimuth_samples: int | None = None
    elevation_samples: int | None = None


def _pattern_id(option: str) -> str:
    key = option.strip().lower()
    if key in {"isotropic", "cosine", "cosine_squared", "dipole"}:
        return key
    if key.startswith("isotropic "):
        return "isotropic"
    if key.startswith("cosine²"):
        return "cosine_squared"
    if key.startswith("cosine "):
        return "cosine"
    if key.startswith("dipole "):
        return "dipole"
    raise ValueError(f"Unsupported element pattern: {option!r}")


def _isotropic_kernel(q: FloatArray) -> FloatArray:
    return np.asarray(4.0 * np.pi * np.sinc(q / np.pi), dtype=float)


def _cosine_kernel(q: FloatArray) -> FloatArray:
    result = np.empty_like(q, dtype=float)
    small = np.abs(q) < 1.0e-3
    q_small = q[small]
    result[small] = (
        2.0
        * np.pi
        * (1.0 / 3.0 - q_small**2 / 30.0 + q_small**4 / 840.0 - q_small**6 / 45_360.0)
    )
    q_large = q[~small]
    result[~small] = (
        2.0 * np.pi * (np.sin(q_large) - q_large * np.cos(q_large)) / q_large**3
    )
    return result


def _cosine_squared_kernel(q: FloatArray) -> FloatArray:
    result = np.empty_like(q, dtype=float)
    small = np.abs(q) < 1.0e-2
    q_small = q[small]
    result[small] = (
        6.0
        * np.pi
        * (
            1.0 / 15.0
            - q_small**2 / 210.0
            + q_small**4 / 7_560.0
            - q_small**6 / 498_960.0
        )
    )
    q_large = q[~small]
    result[~small] = (
        6.0
        * np.pi
        * (
            -(q_large**2) * np.sin(q_large)
            - 3.0 * q_large * np.cos(q_large)
            + 3.0 * np.sin(q_large)
        )
        / q_large**5
    )
    return result


def _pairwise_power_integral(
    y: FloatArray,
    z: FloatArray,
    weights: ComplexArray,
    wavelength_m: float,
    pattern_id: str,
    *,
    max_chunk_entries: int,
    cancel_check: Callable[[], None] | None,
) -> float:
    """Evaluate the full-sphere integral through exact pairwise kernels."""

    kernel_function = {
        "isotropic": _isotropic_kernel,
        "cosine": _cosine_kernel,
        "cosine_squared": _cosine_squared_kernel,
    }[pattern_id]
    element_count = weights.size
    row_count = max(1, max_chunk_entries // element_count)
    conjugate_weights = np.conjugate(weights)
    wave_number = 2.0 * np.pi / wavelength_m
    total = 0.0 + 0.0j
    for start in range(0, element_count, row_count):
        if cancel_check is not None:
            cancel_check()
        stop = min(element_count, start + row_count)
        delta_y = y[start:stop, None] - y[None, :]
        delta_z = z[start:stop, None] - z[None, :]
        q = wave_number * np.hypot(delta_y, delta_z)
        kernel = kernel_function(np.asarray(q, dtype=float))
        total += np.sum(weights[start:stop, None] * conjugate_weights[None, :] * kernel)
    return float(np.real(total))


def _dipole_power_integral(
    y: FloatArray,
    z: FloatArray,
    weights: ComplexArray,
    wavelength_m: float,
    *,
    azimuth_samples: int,
    elevation_samples: int,
    max_chunk_entries: int,
    cancel_check: Callable[[], None] | None,
) -> float:
    """Numerically integrate dipole power with azimuth/Gauss-Legendre nodes."""

    mu, mu_weights = np.polynomial.legendre.leggauss(elevation_samples)
    elevation = np.arcsin(mu)
    azimuth = np.linspace(-np.pi, np.pi, azimuth_samples, endpoint=False)
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="ij")
    pattern = array_factor(
        y,
        z,
        weights,
        wavelength_m,
        azimuth_grid,
        elevation_grid,
        max_chunk_entries=max_chunk_entries,
        cancel_check=cancel_check,
    )
    pattern *= element_pattern_factor("dipole", azimuth_grid, elevation_grid)
    intensity = np.abs(pattern) ** 2
    integral = (2.0 * np.pi / azimuth_samples) * np.sum(intensity * mu_weights[None, :])
    return float(integral)


def calculate_directivity(
    y: ArrayLike,
    z: ArrayLike,
    complex_weights: ArrayLike,
    wavelength_m: float,
    target_azimuth_rad: float,
    target_elevation_rad: float,
    element_option: str,
    *,
    element_mask: ArrayLike | None = None,
    max_chunk_entries: int = 1_000_000,
    dipole_azimuth_samples: int = 144,
    dipole_elevation_samples: int = 96,
    cancel_check: Callable[[], None] | None = None,
) -> DirectivityResult:
    """Calculate physical target-direction directivity from full-sphere power.

    Isotropic and front-hemisphere cosine models use analytic full-sphere
    pairwise kernels.  The half-wave dipole model uses Gauss-Legendre
    quadrature in ``sin(elevation)`` and a periodic azimuth rule.
    """

    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    weights_array = np.asarray(complex_weights, dtype=complex)
    if y_array.shape != z_array.shape or y_array.shape != weights_array.shape:
        raise ValueError("Coordinates and weights must have matching shapes.")
    if wavelength_m <= 0.0 or not np.isfinite(wavelength_m):
        raise ValueError("Wavelength must be finite and positive.")
    if max_chunk_entries < 1:
        raise ValueError("Maximum chunk entries must be positive.")
    if dipole_azimuth_samples < 8 or dipole_elevation_samples < 8:
        raise ValueError("Dipole integration requires at least eight samples per axis.")
    if element_mask is None:
        physical_mask = np.ones(y_array.shape, dtype=bool)
    else:
        physical_mask = np.asarray(element_mask, dtype=bool)
        if physical_mask.shape != y_array.shape:
            raise ValueError("Element mask must match coordinate shape.")
    if cancel_check is not None:
        cancel_check()

    selected = physical_mask.ravel()
    y_flat = y_array.ravel()[selected]
    z_flat = z_array.ravel()[selected]
    weights_flat = weights_array.ravel()[selected]
    nonzero = np.abs(weights_flat) > 0.0
    y_flat = np.asarray(y_flat[nonzero], dtype=float)
    z_flat = np.asarray(z_flat[nonzero], dtype=float)
    weights_flat = np.asarray(weights_flat[nonzero], dtype=complex)
    if weights_flat.size == 0:
        return DirectivityResult(
            schema_version=DIRECTIVITY_SCHEMA_VERSION,
            directivity_linear=None,
            directivity_dbi=None,
            target_radiation_intensity=0.0,
            radiated_power_integral=0.0,
            integration_method="unavailable",
        )

    # Normalization prevents overflow without changing the dimensionless ratio.
    weights_flat = weights_flat / float(np.max(np.abs(weights_flat)))
    pattern_id = _pattern_id(element_option)
    target_phase = steering_phases(
        y_flat,
        z_flat,
        wavelength_m,
        float(target_azimuth_rad),
        float(target_elevation_rad),
    )
    target_response = np.sum(np.exp(1j * target_phase) * weights_flat)
    target_element_factor = float(
        element_pattern_factor(
            pattern_id,
            float(target_azimuth_rad),
            float(target_elevation_rad),
        )
    )
    target_intensity = float(np.abs(target_response * target_element_factor) ** 2)

    if pattern_id == "dipole":
        power_integral = _dipole_power_integral(
            y_flat,
            z_flat,
            weights_flat,
            wavelength_m,
            azimuth_samples=dipole_azimuth_samples,
            elevation_samples=dipole_elevation_samples,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        method = "spherical Gauss-Legendre quadrature"
        azimuth_samples: int | None = dipole_azimuth_samples
        elevation_samples: int | None = dipole_elevation_samples
        self_kernel = 2.0 * np.pi * 0.6090
    else:
        power_integral = _pairwise_power_integral(
            y_flat,
            z_flat,
            weights_flat,
            wavelength_m,
            pattern_id,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        method = f"analytic full-sphere pairwise kernel ({pattern_id})"
        azimuth_samples = None
        elevation_samples = None
        self_kernel = {
            "isotropic": 4.0 * np.pi,
            "cosine": 2.0 * np.pi / 3.0,
            "cosine_squared": 2.0 * np.pi / 5.0,
        }[pattern_id]

    if cancel_check is not None:
        cancel_check()
    tolerance = (
        128.0
        * np.finfo(float).eps
        * max(1.0, float(np.sum(np.abs(weights_flat) ** 2)) * self_kernel)
    )
    if not np.isfinite(power_integral) or power_integral <= tolerance:
        return DirectivityResult(
            schema_version=DIRECTIVITY_SCHEMA_VERSION,
            directivity_linear=None,
            directivity_dbi=None,
            target_radiation_intensity=target_intensity,
            radiated_power_integral=float(power_integral),
            integration_method=method,
            azimuth_samples=azimuth_samples,
            elevation_samples=elevation_samples,
        )

    directivity_linear = float(4.0 * np.pi * target_intensity / power_integral)
    directivity_dbi = (
        float(10.0 * np.log10(directivity_linear))
        if directivity_linear > 0.0
        else float("-inf")
    )
    return DirectivityResult(
        schema_version=DIRECTIVITY_SCHEMA_VERSION,
        directivity_linear=directivity_linear,
        directivity_dbi=directivity_dbi,
        target_radiation_intensity=target_intensity,
        radiated_power_integral=float(power_integral),
        integration_method=method,
        azimuth_samples=azimuth_samples,
        elevation_samples=elevation_samples,
    )
