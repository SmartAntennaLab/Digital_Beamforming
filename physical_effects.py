"""Deterministic array-position, calibration, coupling, and near-field models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_geometry import ArrayCoordinates
from array_math import direction_cosines

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class HardwareEffectDiagnostics:
    """Realized deterministic impairment values for one simulation frame."""

    random_seed: int
    position_error_rms_wavelength: float
    realized_position_error_rms_wavelength: float
    amplitude_error_rms_db: float
    realized_amplitude_error_rms_db: float
    phase_error_rms_deg: float
    realized_phase_error_rms_deg: float
    mutual_coupling_db: float | None
    mutual_coupling_phase_deg: float
    coupled_neighbor_links: int


def _validate_nonnegative(name: str, value: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= maximum:
        raise ValueError(f"{name} must be finite and between 0 and {maximum}.")
    return number


def apply_position_errors(
    coordinates: ArrayCoordinates,
    wavelength_m: float,
    rms_wavelength: float,
    *,
    random_seed: int,
) -> tuple[ArrayCoordinates, float]:
    """Apply zero-mean deterministic Gaussian Y/Z placement errors."""

    rms = _validate_nonnegative("Position-error RMS", rms_wavelength, 0.5)
    if wavelength_m <= 0.0 or not math.isfinite(wavelength_m):
        raise ValueError("Wavelength must be finite and positive.")
    if rms == 0.0:
        return coordinates, 0.0
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 101]))
    mask = coordinates.element_mask
    y_error = np.zeros_like(coordinates.y, dtype=float)
    z_error = np.zeros_like(coordinates.z, dtype=float)
    sigma_m = rms * wavelength_m
    y_error[mask] = rng.normal(0.0, sigma_m, int(np.count_nonzero(mask)))
    z_error[mask] = rng.normal(0.0, sigma_m, int(np.count_nonzero(mask)))
    realized = float(
        np.sqrt(np.mean((y_error[mask] ** 2 + z_error[mask] ** 2) / 2.0)) / wavelength_m
    )
    return (
        ArrayCoordinates(
            geometry=coordinates.geometry,
            rows=coordinates.rows,
            columns=coordinates.columns,
            y=np.asarray(coordinates.y + y_error, dtype=float),
            z=np.asarray(coordinates.z + z_error, dtype=float),
            element_mask=np.asarray(mask, dtype=bool),
        ),
        realized,
    )


def _apply_nearest_neighbor_coupling(
    y: FloatArray,
    z: FloatArray,
    weights: ComplexArray,
    active_mask: BoolArray,
    reference_spacing_m: float,
    coupling_db: float | None,
    coupling_phase_deg: float,
    *,
    cancel_check=None,
) -> tuple[ComplexArray, int]:
    if coupling_db is None:
        return weights.copy(), 0
    if not math.isfinite(coupling_db) or coupling_db > 0.0 or coupling_db < -120.0:
        raise ValueError("Mutual coupling must be between -120 and 0 dB.")
    if reference_spacing_m <= 0.0 or not math.isfinite(reference_spacing_m):
        raise ValueError("Coupling reference spacing must be finite and positive.")
    if not math.isfinite(coupling_phase_deg):
        raise ValueError("Mutual-coupling phase must be finite.")

    active_indices = np.flatnonzero(active_mask.ravel())
    if active_indices.size <= 1:
        return weights.copy(), 0
    y_active = y.ravel()[active_indices]
    z_active = z.ravel()[active_indices]
    weights_active = weights.ravel()[active_indices]
    coefficient = 10.0 ** (coupling_db / 20.0) * np.exp(
        1j * np.radians(coupling_phase_deg)
    )
    radius_squared = (1.2 * reference_spacing_m) ** 2
    coupled = weights_active.copy()
    links = 0
    chunk_size = 128
    for start in range(0, active_indices.size, chunk_size):
        if cancel_check is not None:
            cancel_check()
        stop = min(start + chunk_size, active_indices.size)
        distance_squared = (y_active[start:stop, None] - y_active[None, :]) ** 2 + (
            z_active[start:stop, None] - z_active[None, :]
        ) ** 2
        neighbor_mask = (distance_squared > 1e-24) & (
            distance_squared <= radius_squared
        )
        links += int(np.count_nonzero(neighbor_mask))
        coupled[start:stop] += coefficient * (
            neighbor_mask.astype(float) @ weights_active
        )
    result = np.zeros_like(weights, dtype=complex)
    result.ravel()[active_indices] = coupled
    return result, links // 2


def apply_weight_errors_and_coupling(
    coordinates: ArrayCoordinates,
    weights: ArrayLike,
    active_mask: ArrayLike,
    *,
    amplitude_error_rms_db: float,
    phase_error_rms_deg: float,
    mutual_coupling_db: float | None,
    mutual_coupling_phase_deg: float,
    reference_spacing_m: float,
    random_seed: int,
    position_error_rms_wavelength: float,
    realized_position_error_rms_wavelength: float,
    cancel_check=None,
) -> tuple[ComplexArray, HardwareEffectDiagnostics]:
    """Apply repeatable per-channel calibration errors then reciprocal coupling."""

    amplitude_rms = _validate_nonnegative(
        "Amplitude calibration-error RMS", amplitude_error_rms_db, 20.0
    )
    phase_rms = _validate_nonnegative(
        "Phase calibration-error RMS", phase_error_rms_deg, 180.0
    )
    weight_array = np.asarray(weights, dtype=complex)
    mask = np.asarray(active_mask, dtype=bool)
    if weight_array.shape != coordinates.y.shape or mask.shape != weight_array.shape:
        raise ValueError("Weights, mask, and coordinates must have matching shapes.")
    active_count = int(np.count_nonzero(mask))
    amplitude_error = np.zeros(weight_array.shape, dtype=float)
    phase_error = np.zeros(weight_array.shape, dtype=float)
    if active_count:
        rng = np.random.default_rng(np.random.SeedSequence([random_seed, 202]))
        amplitude_error[mask] = rng.normal(0.0, amplitude_rms, active_count)
        phase_error[mask] = rng.normal(0.0, phase_rms, active_count)
    impaired = (
        weight_array
        * 10.0 ** (amplitude_error / 20.0)
        * np.exp(1j * np.radians(phase_error))
    )
    impaired[~mask] = 0.0
    coupled, links = _apply_nearest_neighbor_coupling(
        coordinates.y,
        coordinates.z,
        impaired,
        mask,
        reference_spacing_m,
        mutual_coupling_db,
        mutual_coupling_phase_deg,
        cancel_check=cancel_check,
    )
    diagnostics = HardwareEffectDiagnostics(
        random_seed=int(random_seed),
        position_error_rms_wavelength=float(position_error_rms_wavelength),
        realized_position_error_rms_wavelength=float(
            realized_position_error_rms_wavelength
        ),
        amplitude_error_rms_db=amplitude_rms,
        realized_amplitude_error_rms_db=(
            float(np.sqrt(np.mean(amplitude_error[mask] ** 2))) if active_count else 0.0
        ),
        phase_error_rms_deg=phase_rms,
        realized_phase_error_rms_deg=(
            float(np.sqrt(np.mean(phase_error[mask] ** 2))) if active_count else 0.0
        ),
        mutual_coupling_db=(
            None if mutual_coupling_db is None else float(mutual_coupling_db)
        ),
        mutual_coupling_phase_deg=float(mutual_coupling_phase_deg),
        coupled_neighbor_links=links,
    )
    return np.asarray(coupled, dtype=complex), diagnostics


def near_field_focusing_weights(
    y: ArrayLike,
    z: ArrayLike,
    wavelength_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    range_m: float,
    amplitudes: ArrayLike,
) -> ComplexArray:
    """Return phase-conjugate spherical-wave transmit focusing weights."""

    if range_m <= 0.0 or not math.isfinite(range_m):
        raise ValueError("Near-field focus range must be finite and positive.")
    if wavelength_m <= 0.0 or not math.isfinite(wavelength_m):
        raise ValueError("Wavelength must be finite and positive.")
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    amplitude_array = np.asarray(amplitudes, dtype=float)
    if y_array.shape != z_array.shape or y_array.shape != amplitude_array.shape:
        raise ValueError("Near-field arrays must have matching shapes.")
    u_x, u_y, u_z = direction_cosines(azimuth_rad, elevation_rad)
    target_x = range_m * float(u_x)
    target_y = range_m * float(u_y)
    target_z = range_m * float(u_z)
    distances = np.sqrt(
        target_x**2 + (target_y - y_array) ** 2 + (target_z - z_array) ** 2
    )
    phase = 2.0 * np.pi / wavelength_m * (distances - range_m)
    return np.asarray(amplitude_array * np.exp(1j * phase), dtype=complex)


def near_field_response(
    y: ArrayLike,
    z: ArrayLike,
    weights: ArrayLike,
    wavelength_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    range_m: float,
) -> complex:
    """Evaluate normalized scalar spherical-wave field at one focus point."""

    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    weight_array = np.asarray(weights, dtype=complex)
    u_x, u_y, u_z = direction_cosines(azimuth_rad, elevation_rad)
    target_x = range_m * float(u_x)
    target_y = range_m * float(u_y)
    target_z = range_m * float(u_z)
    distances = np.sqrt(
        target_x**2 + (target_y - y_array) ** 2 + (target_z - z_array) ** 2
    )
    propagation = np.exp(-1j * 2.0 * np.pi / wavelength_m * distances)
    propagation *= range_m / np.maximum(distances, np.finfo(float).tiny)
    return complex(np.sum(weight_array * propagation))


__all__ = [
    "HardwareEffectDiagnostics",
    "apply_position_errors",
    "apply_weight_errors_and_coupling",
    "near_field_focusing_weights",
    "near_field_response",
]
