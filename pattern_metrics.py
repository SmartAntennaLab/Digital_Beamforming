"""Chunked array-factor evaluation and derived beam-pattern metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import steering_phases


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class PatternMetrics:
    """Beam-pattern metrics with ``None`` for undetected measurements."""

    peak_index: int
    hpbw_deg: float | None
    hpbw_left_index: int | None
    hpbw_right_index: int | None
    hpbw_left_angle_deg: float | None
    hpbw_right_angle_deg: float | None
    first_null_beamwidth_deg: float | None
    first_null_left_index: int | None
    first_null_right_index: int | None
    sidelobe_level_db: float | None
    sidelobe_angle_deg: float | None


@dataclass(frozen=True)
class ArrayGainMetrics:
    """Relative coherent-array metrics for the selected target direction."""

    total_elements: int
    active_elements: int
    taper_efficiency: float
    phase_efficiency: float
    effective_element_count: float
    relative_array_gain_db: float | None


def array_factor(
    y: ArrayLike,
    z: ArrayLike,
    complex_weights: ArrayLike,
    wavelength_m: float,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
    *,
    angle_chunk_size: int | None = None,
    element_chunk_size: int | None = None,
    max_chunk_entries: int = 1_000_000,
) -> ComplexArray:
    """Evaluate the array factor with bounded angle/element working sets."""

    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    weights = np.asarray(complex_weights, dtype=complex)
    if y_array.shape != z_array.shape or y_array.shape != weights.shape:
        raise ValueError("Coordinates and weights must have matching shapes.")
    if angle_chunk_size is not None and angle_chunk_size < 1:
        raise ValueError("Angle chunk size must be positive.")
    if element_chunk_size is not None and element_chunk_size < 1:
        raise ValueError("Element chunk size must be positive.")
    if max_chunk_entries < 1:
        raise ValueError("Maximum chunk entries must be positive.")

    azimuth = np.asarray(azimuth_rad, dtype=float)
    elevation = np.asarray(elevation_rad, dtype=float)
    azimuth, elevation = np.broadcast_arrays(azimuth, elevation)
    angle_shape = azimuth.shape
    azimuth_flat = azimuth.ravel()
    elevation_flat = elevation.ravel()
    weights_flat = weights.ravel()
    element_count = weights_flat.size
    angle_count = azimuth_flat.size

    if angle_count == 0:
        return np.empty(angle_shape, dtype=complex)
    if element_count == 0:
        return np.zeros(angle_shape, dtype=complex)

    effective_angle_chunk = angle_chunk_size
    effective_element_chunk = element_chunk_size
    if effective_angle_chunk is None and effective_element_chunk is None:
        if angle_count * element_count <= max_chunk_entries:
            effective_angle_chunk = angle_count
            effective_element_chunk = element_count
        elif angle_count >= element_count:
            effective_angle_chunk = max(1, max_chunk_entries // element_count)
            effective_element_chunk = element_count
        else:
            effective_angle_chunk = angle_count
            effective_element_chunk = max(1, max_chunk_entries // angle_count)
    elif effective_angle_chunk is None:
        effective_element_chunk = min(effective_element_chunk, element_count)
        effective_angle_chunk = max(
            1,
            min(angle_count, max_chunk_entries // effective_element_chunk),
        )
    elif effective_element_chunk is None:
        effective_angle_chunk = min(effective_angle_chunk, angle_count)
        effective_element_chunk = max(
            1,
            min(element_count, max_chunk_entries // effective_angle_chunk),
        )
    else:
        effective_angle_chunk = min(effective_angle_chunk, angle_count)
        effective_element_chunk = min(effective_element_chunk, element_count)
        if effective_angle_chunk * effective_element_chunk > max_chunk_entries:
            effective_element_chunk = max(
                1,
                max_chunk_entries // effective_angle_chunk,
            )

    result = np.zeros(angle_count, dtype=complex)
    y_flat = y_array.ravel()
    z_flat = z_array.ravel()
    for angle_start in range(0, angle_count, effective_angle_chunk):
        angle_stop = min(angle_count, angle_start + effective_angle_chunk)
        partial = np.zeros(angle_stop - angle_start, dtype=complex)
        for element_start in range(0, element_count, effective_element_chunk):
            element_stop = min(
                element_count,
                element_start + effective_element_chunk,
            )
            phases = steering_phases(
                y_flat[element_start:element_stop],
                z_flat[element_start:element_stop],
                wavelength_m,
                azimuth_flat[angle_start:angle_stop],
                elevation_flat[angle_start:angle_stop],
            )
            partial += np.exp(1j * phases) @ weights_flat[
                element_start:element_stop
            ]
        result[angle_start:angle_stop] = partial
    return np.asarray(result.reshape(angle_shape), dtype=complex)


def normalize_pattern_linear(pattern: ArrayLike) -> FloatArray:
    """Normalize magnitudes without dividing by a zero/non-finite maximum."""

    magnitude = np.abs(np.asarray(pattern, dtype=complex))
    if magnitude.size == 0:
        return np.asarray(magnitude, dtype=float)
    maximum = float(np.max(magnitude))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros(magnitude.shape, dtype=float)
    return np.asarray(magnitude / maximum, dtype=float)


def normalize_pattern_db(
    pattern: ArrayLike,
    *,
    floor_db: float = -120.0,
) -> FloatArray:
    """Normalize a complex pattern to a finite dB scale."""

    if floor_db >= 0:
        raise ValueError("The dB floor must be negative.")
    normalized = normalize_pattern_linear(pattern)
    if not np.any(normalized > 0.0):
        return np.full(normalized.shape, floor_db, dtype=float)
    ratio_floor = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(normalized, ratio_floor))


def calculate_array_gain_metrics(
    y: ArrayLike,
    z: ArrayLike,
    complex_weights: ArrayLike,
    active_mask: ArrayLike,
    wavelength_m: float,
    target_azimuth_rad: float,
    target_elevation_rad: float,
    *,
    element_mask: ArrayLike | None = None,
    zero_tolerance: float = 1e-12,
) -> ArrayGainMetrics:
    """Calculate relative coherent-array gain and efficiency components."""

    if zero_tolerance < 0 or not np.isfinite(zero_tolerance):
        raise ValueError("Zero tolerance must be finite and non-negative.")
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    weights = np.asarray(complex_weights, dtype=complex)
    mask = np.asarray(active_mask, dtype=bool)
    if not (y_array.shape == z_array.shape == weights.shape == mask.shape):
        raise ValueError("Coordinates, weights, and active mask must have matching shapes.")
    if weights.size == 0:
        raise ValueError("Relative array gain requires at least one element.")
    if np.any(~np.isfinite(np.abs(weights))):
        raise ValueError("Complex weights must be finite.")
    if element_mask is None:
        physical_mask = np.ones(mask.shape, dtype=bool)
    else:
        physical_mask = np.asarray(element_mask, dtype=bool)
        if physical_mask.shape != mask.shape:
            raise ValueError("Element mask must match the active mask shape.")
        if not np.any(physical_mask):
            raise ValueError(
                "Relative array gain requires at least one physical element."
            )
    if np.any(mask & ~physical_mask):
        raise ValueError("Active mask cannot include non-physical element slots.")
    if np.any(np.abs(weights[~mask]) > zero_tolerance):
        raise ValueError("Inactive elements must have zero complex weight.")

    total_elements = int(np.count_nonzero(physical_mask))
    active_elements = int(np.count_nonzero(mask))
    magnitudes = np.abs(weights)
    weight_scale = float(np.max(magnitudes))
    if active_elements == 0 or weight_scale <= 0.0:
        return ArrayGainMetrics(
            total_elements=total_elements,
            active_elements=active_elements,
            taper_efficiency=0.0,
            phase_efficiency=0.0,
            effective_element_count=0.0,
            relative_array_gain_db=None,
        )

    scaled_weights = weights / weight_scale
    scaled_magnitudes = magnitudes / weight_scale
    weight_power = float(np.sum(scaled_magnitudes**2))
    magnitude_sum = float(np.sum(scaled_magnitudes))
    if weight_power <= 0.0 or magnitude_sum <= 0.0:
        return ArrayGainMetrics(
            total_elements=total_elements,
            active_elements=active_elements,
            taper_efficiency=0.0,
            phase_efficiency=0.0,
            effective_element_count=0.0,
            relative_array_gain_db=None,
        )

    target_response = array_factor(
        y_array,
        z_array,
        scaled_weights,
        wavelength_m,
        target_azimuth_rad,
        target_elevation_rad,
    )
    target_power = float(np.abs(target_response.item()) ** 2)
    taper_efficiency = magnitude_sum**2 / (active_elements * weight_power)
    phase_efficiency = target_power / magnitude_sum**2
    effective_element_count = target_power / weight_power
    taper_efficiency = float(np.clip(taper_efficiency, 0.0, 1.0))
    phase_efficiency = float(np.clip(phase_efficiency, 0.0, 1.0))
    effective_element_count = float(
        np.clip(effective_element_count, 0.0, float(active_elements))
    )
    relative_array_gain_db = (
        float(10.0 * np.log10(effective_element_count))
        if effective_element_count > 0.0
        else None
    )
    return ArrayGainMetrics(
        total_elements=total_elements,
        active_elements=active_elements,
        taper_efficiency=taper_efficiency,
        phase_efficiency=phase_efficiency,
        effective_element_count=effective_element_count,
        relative_array_gain_db=relative_array_gain_db,
    )


def find_first_null(
    pattern_db: ArrayLike,
    peak_index: int,
) -> tuple[int | None, int | None]:
    """Find nearest local minima, returning ``None`` on an undetected side."""

    values = np.asarray(pattern_db, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Pattern must be a non-empty one-dimensional array.")
    if not 0 <= peak_index < values.size:
        raise ValueError("Peak index is outside the pattern.")
    null_left: int | None = None
    for index in range(peak_index - 1, -1, -1):
        if values[index] > values[index + 1]:
            null_left = index + 1
            break
    null_right: int | None = None
    for index in range(peak_index + 1, values.size):
        if values[index] > values[index - 1]:
            null_right = index - 1
            break
    return null_left, null_right


def _interpolate_crossing(
    angle_0: float,
    angle_1: float,
    value_0: float,
    value_1: float,
    threshold: float,
) -> float:
    delta = value_1 - value_0
    scale = max(abs(value_0), abs(value_1), abs(threshold), 1.0)
    if abs(delta) <= np.finfo(float).eps * scale:
        return float((angle_0 + angle_1) / 2.0)
    fraction = float(np.clip((threshold - value_0) / delta, 0.0, 1.0))
    return float(angle_0 + fraction * (angle_1 - angle_0))


def calculate_pattern_metrics(
    pattern: ArrayLike,
    angles_rad: ArrayLike,
) -> PatternMetrics:
    """Calculate interpolated HPBW, FNBW, and peak sidelobe level."""

    values = np.asarray(pattern, dtype=complex)
    angles = np.asarray(angles_rad, dtype=float)
    if values.ndim != 1 or angles.ndim != 1 or values.size != angles.size:
        raise ValueError("Pattern and angle arrays must be one-dimensional and equal in size.")
    if values.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("Pattern metrics require finite, non-empty angle samples.")
    if values.size > 1 and np.any(np.diff(angles) <= 0):
        raise ValueError("Angle samples must be strictly increasing.")
    magnitude = np.abs(values)
    if np.any(~np.isfinite(magnitude)):
        raise ValueError("Pattern contains non-finite values.")
    peak_index = int(np.argmax(magnitude))
    peak = float(magnitude[peak_index])

    hpbw_left: int | None = None
    hpbw_right: int | None = None
    hpbw_left_angle: float | None = None
    hpbw_right_angle: float | None = None
    hpbw_deg: float | None = None
    if peak > 0.0:
        half_power = peak / np.sqrt(2.0)
        left_matches = np.where(magnitude[:peak_index] <= half_power)[0]
        right_matches = np.where(magnitude[peak_index:] <= half_power)[0]
        if left_matches.size:
            hpbw_left = int(left_matches[-1])
            hpbw_left_angle = _interpolate_crossing(
                angles[hpbw_left],
                angles[hpbw_left + 1],
                magnitude[hpbw_left],
                magnitude[hpbw_left + 1],
                half_power,
            )
        if right_matches.size:
            hpbw_right = int(peak_index + right_matches[0])
            hpbw_right_angle = _interpolate_crossing(
                angles[hpbw_right - 1],
                angles[hpbw_right],
                magnitude[hpbw_right - 1],
                magnitude[hpbw_right],
                half_power,
            )
        if hpbw_left_angle is not None and hpbw_right_angle is not None:
            hpbw_deg = float(np.degrees(hpbw_right_angle - hpbw_left_angle))
        elif hpbw_left_angle is not None:
            hpbw_right_angle = 2.0 * angles[peak_index] - hpbw_left_angle
            hpbw_right = min(
                values.size - 1,
                peak_index + (peak_index - int(hpbw_left)),
            )
            hpbw_deg = float(
                2.0 * np.degrees(angles[peak_index] - hpbw_left_angle)
            )
        elif hpbw_right_angle is not None:
            hpbw_left_angle = 2.0 * angles[peak_index] - hpbw_right_angle
            hpbw_left = max(
                0,
                peak_index - (int(hpbw_right) - peak_index),
            )
            hpbw_deg = float(
                2.0 * np.degrees(hpbw_right_angle - angles[peak_index])
            )

    pattern_db = normalize_pattern_db(values)
    null_left, null_right = find_first_null(pattern_db, peak_index)
    if null_left is not None and null_right is not None:
        fnbw_deg = float(
            np.degrees(angles[null_right] - angles[null_left])
        )
    elif null_right is not None:
        fnbw_deg = float(
            2.0 * np.degrees(angles[null_right] - angles[peak_index])
        )
    elif null_left is not None:
        fnbw_deg = float(
            2.0 * np.degrees(angles[peak_index] - angles[null_left])
        )
    else:
        fnbw_deg = None

    sidelobe_values: list[FloatArray] = []
    sidelobe_indices: list[NDArray[np.int_]] = []
    if null_left is not None and null_left > 0:
        sidelobe_values.append(pattern_db[:null_left])
        sidelobe_indices.append(np.arange(0, null_left))
    if null_right is not None and null_right < values.size - 1:
        sidelobe_values.append(pattern_db[null_right + 1 :])
        sidelobe_indices.append(np.arange(null_right + 1, values.size))
    if sidelobe_values:
        combined_values = np.concatenate(sidelobe_values)
        combined_indices = np.concatenate(sidelobe_indices)
        sidelobe_position = int(np.argmax(combined_values))
        sidelobe_level_db = float(combined_values[sidelobe_position])
        sidelobe_angle_deg = float(
            np.degrees(angles[combined_indices[sidelobe_position]])
        )
    else:
        sidelobe_level_db = None
        sidelobe_angle_deg = None

    return PatternMetrics(
        peak_index=peak_index,
        hpbw_deg=hpbw_deg,
        hpbw_left_index=hpbw_left,
        hpbw_right_index=hpbw_right,
        hpbw_left_angle_deg=(
            float(np.degrees(hpbw_left_angle))
            if hpbw_left_angle is not None
            else None
        ),
        hpbw_right_angle_deg=(
            float(np.degrees(hpbw_right_angle))
            if hpbw_right_angle is not None
            else None
        ),
        first_null_beamwidth_deg=fnbw_deg,
        first_null_left_index=null_left,
        first_null_right_index=null_right,
        sidelobe_level_db=sidelobe_level_db,
        sidelobe_angle_deg=sidelobe_angle_deg,
    )
