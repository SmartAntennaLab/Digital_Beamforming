"""Pure numerical helpers for the digital beamforming simulator.

This module intentionally has no Streamlit dependency so its behavior can be
tested and reused independently from the web user interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class ArrayCoordinates:
    """Physical Y-Z coordinates for an antenna array."""

    geometry: str
    rows: int
    columns: int
    y: FloatArray
    z: FloatArray
    element_mask: BoolArray

    @property
    def element_count(self) -> int:
        """Return the number of physical elements, excluding padded slots."""

        return int(np.count_nonzero(self.element_mask))

    @property
    def row_lengths(self) -> tuple[int, ...]:
        """Return the physical element count in each displayed row."""

        return tuple(int(value) for value in np.count_nonzero(self.element_mask, axis=1))


@dataclass(frozen=True)
class SteeringLimits:
    """Independent steering capabilities of the effective array geometry."""

    geometry: str
    azimuth_controllable: bool
    elevation_controllable: bool
    azimuth_min_deg: float = -90.0
    azimuth_max_deg: float = 90.0
    elevation_min_deg: float = -90.0
    elevation_max_deg: float = 90.0


@dataclass(frozen=True)
class GratingLobeDirection:
    """One visible spatial-alias direction for a periodic array."""

    order_y: int
    order_z: int
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class GratingLobeAssessment:
    """Geometry-aware grating-lobe or spatial-aliasing assessment."""

    geometry: str
    has_aliasing_risk: bool
    risk_only: bool
    directions: tuple[GratingLobeDirection, ...]
    criterion: str


@dataclass(frozen=True)
class BeamformingWeights:
    """Complex element weights and null-steering diagnostics."""

    weights: ComplexArray
    continuous_weights: ComplexArray
    target_phases: FloatArray
    final_phases: FloatArray
    null_applied: bool
    determinant: float | None
    constraint_rank: int | None
    constraint_count: int
    condition_number: float | None
    null_directions_rad: tuple[tuple[float, float], ...]
    continuous_null_depths_db: tuple[float | None, ...]
    null_depths_db: tuple[float | None, ...]
    diagnostic_message: str | None


@dataclass(frozen=True)
class PatternMetrics:
    """Beam-pattern metrics, including interpolated half-power crossings."""

    peak_index: int
    hpbw_deg: float
    hpbw_left_index: int | None
    hpbw_right_index: int | None
    hpbw_left_angle_deg: float | None
    hpbw_right_angle_deg: float | None
    first_null_beamwidth_deg: float
    first_null_left_index: int
    first_null_right_index: int
    sidelobe_level_db: float
    sidelobe_angle_deg: float


@dataclass(frozen=True)
class ArrayGainMetrics:
    """Array-only gain metrics referenced to an isotropic element.

    The effective array gain includes the actual active-element mask, final
    amplitude taper, phase quantization, and any null-steering distortion.
    Element-pattern directivity and RF losses are intentionally excluded.
    """

    total_elements: int
    active_elements: int
    taper_efficiency: float
    phase_efficiency: float
    effective_element_count: float
    array_gain_db: float | None


def _geometry_key(geometry: str) -> str:
    key = geometry.strip().upper().split(maxsplit=1)[0]
    if key not in {"ULA", "UPA", "UCA", "UHA"}:
        raise ValueError(f"Unsupported array geometry: {geometry!r}")
    return key


def create_array_coordinates(
    vertical_count: int,
    horizontal_count: int,
    horizontal_spacing_m: float,
    geometry: str,
    *,
    vertical_spacing_m: float | None = None,
) -> ArrayCoordinates:
    """Create Y-Z element coordinates for ULA, UPA, UCA, or UHA geometry.

    UPA uses independent horizontal (Y) and vertical (Z) spacings.  Omitting
    ``vertical_spacing_m`` preserves the legacy square-grid behavior.  ULA
    and UCA use only the horizontal spacing; for UCA it is the physical chord
    distance between adjacent element centers, giving
    ``radius = horizontal_spacing_m / (2*sin(pi/N))`` for two or more elements.

    For UHA, ``vertical_count`` is MATLAB's ``Nmin`` (the bottom/top row
    length) and ``horizontal_count`` is ``Nmax`` (the middle row length).
    Row lengths are ``Nmin:Nmax, Nmax-1:-1:Nmin`` and the triangular-lattice
    row spacing is fixed to ``horizontal_spacing_m*sin(pi/3)``.  Rectangular
    arrays pad shorter rows internally; ``element_mask`` identifies the real
    elements so those padded slots never contribute to a calculation.
    """

    if vertical_count < 1 or horizontal_count < 1:
        raise ValueError("Element counts must be positive integers.")
    vertical_spacing = (
        horizontal_spacing_m
        if vertical_spacing_m is None
        else vertical_spacing_m
    )
    if horizontal_spacing_m < 0 or not np.isfinite(horizontal_spacing_m):
        raise ValueError(
            "Horizontal element spacing must be finite and non-negative."
        )
    if vertical_spacing < 0 or not np.isfinite(vertical_spacing):
        raise ValueError(
            "Vertical element spacing must be finite and non-negative."
        )

    key = _geometry_key(geometry)
    if key == "ULA":
        rows = 1
        horizontal_indices = np.arange(horizontal_count, dtype=float)
        horizontal_indices -= (horizontal_count - 1) / 2.0
        y, z = np.meshgrid(
            horizontal_indices * horizontal_spacing_m,
            np.zeros(rows, dtype=float),
        )
        element_mask = np.ones_like(y, dtype=bool)
    elif key == "UCA":
        rows = 1
        if horizontal_count == 1:
            radius = 0.0
        else:
            radius = horizontal_spacing_m / (
                2.0 * np.sin(np.pi / horizontal_count)
            )
        alpha = 2.0 * np.pi * np.arange(horizontal_count) / horizontal_count
        y = (radius * np.cos(alpha))[None, :]
        z = (radius * np.sin(alpha))[None, :]
        element_mask = np.ones_like(y, dtype=bool)
    elif key == "UHA":
        minimum_row_count = vertical_count
        maximum_row_count = horizontal_count
        if minimum_row_count > maximum_row_count:
            raise ValueError("UHA requires Nmin to be less than or equal to Nmax.")

        increasing_lengths = np.arange(
            minimum_row_count,
            maximum_row_count + 1,
            dtype=int,
        )
        row_lengths = np.concatenate(
            (increasing_lengths, increasing_lengths[-2::-1])
        )
        rows = int(row_lengths.size)
        y = np.zeros((rows, maximum_row_count), dtype=float)
        z = np.zeros_like(y)
        element_mask = np.zeros_like(y, dtype=bool)
        row_spacing = horizontal_spacing_m * np.sin(np.pi / 3.0)
        row_indices = np.arange(rows, dtype=float) - (rows - 1) / 2.0
        for row_index, (row_length, centered_row_index) in enumerate(
            zip(row_lengths, row_indices, strict=True)
        ):
            element_mask[row_index, :row_length] = True
            horizontal_indices = (
                np.arange(row_length, dtype=float) - (row_length - 1) / 2.0
            )
            y[row_index, :row_length] = (
                horizontal_indices * horizontal_spacing_m
            )
            z[row_index, :row_length] = centered_row_index * row_spacing
    else:
        rows = vertical_count
        vertical_indices = np.arange(rows, dtype=float) - (rows - 1) / 2.0
        horizontal_indices = (
            np.arange(horizontal_count, dtype=float)
            - (horizontal_count - 1) / 2.0
        )
        y, z = np.meshgrid(
            horizontal_indices * horizontal_spacing_m,
            vertical_indices * vertical_spacing,
        )
        element_mask = np.ones_like(y, dtype=bool)

    return ArrayCoordinates(
        geometry=key,
        rows=rows,
        columns=horizontal_count,
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        element_mask=np.asarray(element_mask, dtype=bool),
    )


def get_steering_limits(
    vertical_count: int,
    horizontal_count: int,
    geometry: str,
) -> SteeringLimits:
    """Return independently controllable angle axes for the effective array."""

    if vertical_count < 1 or horizontal_count < 1:
        raise ValueError("Element counts must be positive integers.")
    key = _geometry_key(geometry)

    if key == "ULA":
        azimuth_controllable = horizontal_count > 1
        elevation_controllable = False
    elif key == "UCA":
        azimuth_controllable = horizontal_count > 1
        elevation_controllable = False
    else:
        azimuth_controllable = horizontal_count > 1
        elevation_controllable = vertical_count > 1

    return SteeringLimits(
        geometry=key,
        azimuth_controllable=azimuth_controllable,
        elevation_controllable=elevation_controllable,
    )


def get_window_weights(length: int, option: str) -> FloatArray:
    """Return normalized amplitude-window weights.

    Windows that degenerate to all zeros for very short arrays (notably
    Hanning and Bartlett at length two) fall back to uniform weights rather
    than propagating NaN values through the simulator.
    """

    if length < 1:
        raise ValueError("Window length must be at least one.")
    if length == 1:
        return np.ones(1, dtype=float)

    key = option.strip().lower().split(maxsplit=1)[0]
    factories = {
        "hamming": np.hamming,
        "hanning": np.hanning,
        "blackman": np.blackman,
        "bartlett": np.bartlett,
    }
    factory = factories.get(key)
    weights = (
        np.ones(length, dtype=float)
        if factory is None
        else np.asarray(factory(length), dtype=float)
    )

    peak = float(np.max(np.abs(weights)))
    if not np.isfinite(peak) or peak <= 1e-12:
        return np.ones(length, dtype=float)

    normalized = weights / peak
    normalized[np.abs(normalized) < 1e-15] = 0.0
    return normalized


def create_failure_mask(
    rows: int,
    columns: int,
    failure_rate_percent: float,
    *,
    seed: int = 42,
    element_mask: ArrayLike | None = None,
) -> BoolArray:
    """Create a deterministic active-element mask for regression stability.

    ``element_mask`` can exclude padded/non-physical slots in an irregular
    geometry.  The requested failure percentage is then applied only to real
    elements.
    """

    if rows < 1 or columns < 1:
        raise ValueError("Mask dimensions must be positive integers.")
    if not 0.0 <= failure_rate_percent <= 100.0:
        raise ValueError("Failure rate must be between 0 and 100 percent.")

    if element_mask is None:
        physical_mask = np.ones((rows, columns), dtype=bool)
    else:
        physical_mask = np.asarray(element_mask, dtype=bool)
        if physical_mask.shape != (rows, columns):
            raise ValueError("Element mask dimensions must match rows and columns.")

    physical_indices = np.flatnonzero(physical_mask.ravel())
    total = int(physical_indices.size)
    if total < 1:
        raise ValueError("Failure mask requires at least one physical element.")
    failure_count = int(np.round(total * failure_rate_percent / 100.0))
    mask = physical_mask.ravel().copy()
    if failure_count:
        rng = np.random.RandomState(seed)
        failed_indices = rng.choice(physical_indices, failure_count, replace=False)
        mask[failed_indices] = False
    return mask.reshape(rows, columns)


def create_array_taper(coordinates: ArrayCoordinates, option: str) -> FloatArray:
    """Create geometry-aware separable amplitude weights.

    Rectangular and one-row geometries preserve the original row/column outer
    product.  For UHA, every physical row receives the vertical window value
    and its own centered horizontal window.  This keeps each unequal row
    symmetric while padded slots remain exactly zero.
    """

    row_taper = get_window_weights(coordinates.rows, option)
    if coordinates.geometry != "UHA":
        column_taper = get_window_weights(coordinates.columns, option)
        return np.asarray(
            np.outer(row_taper, column_taper) * coordinates.element_mask,
            dtype=float,
        )

    taper = np.zeros((coordinates.rows, coordinates.columns), dtype=float)
    for row_index, row_length in enumerate(coordinates.row_lengths):
        if row_length:
            taper[row_index, :row_length] = (
                row_taper[row_index] * get_window_weights(row_length, option)
            )
    return taper


def parse_phase_bits(phase_bits: int | str | None) -> int | None:
    """Convert UI phase-bit values to an integer or ``None`` for ideal phase."""

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


def quantize_phases(phases_rad: ArrayLike, phase_bits: int | str | None) -> FloatArray:
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
    """Evaluate one element-pattern model from shared direction cosines.

    Cosine patterns are referenced to the +X array broadside.  The dipole is
    a Z-directed half-wave dipole and uses a stable analytic endpoint limit.
    """

    u_x, _, u_z = direction_cosines(azimuth_rad, elevation_rad)
    key = option.strip().lower()
    if key.startswith("isotropic"):
        factor = np.ones_like(u_x, dtype=float)
    elif key.startswith("cosine²") or "제곱" in key:
        factor = np.maximum(0.0, u_x) ** 2
    elif key.startswith("cosine"):
        factor = np.maximum(0.0, u_x)
    elif key.startswith("dipole"):
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


def assess_grating_lobes(
    geometry: str,
    horizontal_spacing_over_wavelength: float,
    steering_azimuth_rad: float,
    steering_elevation_rad: float,
    *,
    vertical_count: int,
    horizontal_count: int,
    vertical_spacing_over_wavelength: float | None = None,
    order_limit: int = 2,
    tolerance: float = 1e-9,
) -> GratingLobeAssessment:
    """Assess visible spatial aliases using geometry-specific conditions.

    ULA and UPA use their periodic Y/Z sampling axes, their independent axis
    spacings, and the visible-disk condition ``u_y**2 + u_z**2 <= 1``.  UHA
    uses the reciprocal lattice of its triangular element grid.  A UCA is not
    a separable lattice, so for three or more elements this returns a
    conservative local sampling risk when its physical adjacent chord spacing
    exceeds half a wavelength.
    """

    key = _geometry_key(geometry)
    vertical_spacing = (
        horizontal_spacing_over_wavelength
        if vertical_spacing_over_wavelength is None
        else vertical_spacing_over_wavelength
    )
    if horizontal_spacing_over_wavelength <= 0.0 or not np.isfinite(
        horizontal_spacing_over_wavelength
    ):
        raise ValueError("Normalized horizontal spacing must be finite and positive.")
    if vertical_spacing <= 0.0 or not np.isfinite(vertical_spacing):
        raise ValueError("Normalized vertical spacing must be finite and positive.")
    if vertical_count < 1 or horizontal_count < 1:
        raise ValueError("Element counts must be positive integers.")
    if order_limit < 1:
        raise ValueError("Order limit must be at least one.")

    if key == "UCA" and horizontal_count >= 3:
        has_risk = horizontal_spacing_over_wavelength > 0.5 + tolerance
        return GratingLobeAssessment(
            geometry=key,
            has_aliasing_risk=has_risk,
            risk_only=True,
            directions=(),
            criterion="UCA adjacent chord spacing exceeds 0.5 wavelength.",
        )

    if key == "ULA" or (key == "UCA" and horizontal_count == 2):
        has_y_axis = horizontal_count > 1
        has_z_axis = False
    elif key in {"UPA", "UHA"}:
        has_y_axis = horizontal_count > 1
        has_z_axis = vertical_count > 1
    else:
        has_y_axis = False
        has_z_axis = False

    if not has_y_axis and not has_z_axis:
        return GratingLobeAssessment(
            geometry=key,
            has_aliasing_risk=False,
            risk_only=False,
            directions=(),
            criterion="No periodically sampled steering axis is present.",
        )

    _, target_u_y, target_u_z = direction_cosines(
        steering_azimuth_rad,
        steering_elevation_rad,
    )
    target_u_y = float(target_u_y)
    target_u_z = float(target_u_z)
    y_orders = range(-order_limit, order_limit + 1) if has_y_axis else (0,)
    z_orders = range(-order_limit, order_limit + 1) if has_z_axis else (0,)
    directions: list[GratingLobeDirection] = []
    seen: set[tuple[float, float]] = set()

    for order_y in y_orders:
        for order_z in z_orders:
            if order_y == 0 and order_z == 0:
                continue
            candidate_u_y = (
                target_u_y + order_y / horizontal_spacing_over_wavelength
            )
            if key == "UHA" and has_z_axis:
                # Triangular lattice basis a1=(dy, 0), a2=(dy/2, dz).
                # Solving A^T*delta_u=(p,q) gives the reciprocal shift below.
                candidate_u_z = target_u_z + (
                    order_z - 0.5 * order_y
                ) / vertical_spacing
            else:
                candidate_u_z = target_u_z + order_z / vertical_spacing
            visible_radius_squared = candidate_u_y**2 + candidate_u_z**2
            if visible_radius_squared > 1.0 + tolerance:
                continue

            candidate_u_y = float(np.clip(candidate_u_y, -1.0, 1.0))
            candidate_u_z = float(np.clip(candidate_u_z, -1.0, 1.0))
            candidate_u_x = float(
                np.sqrt(max(0.0, 1.0 - candidate_u_y**2 - candidate_u_z**2))
            )
            azimuth_deg = float(np.degrees(np.arctan2(candidate_u_y, candidate_u_x)))
            elevation_deg = float(np.degrees(np.arcsin(candidate_u_z)))
            direction_key = (round(azimuth_deg, 9), round(elevation_deg, 9))
            if direction_key in seen:
                continue
            seen.add(direction_key)
            directions.append(
                GratingLobeDirection(
                    order_y=int(order_y),
                    order_z=int(order_z),
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                )
            )

    criterion = (
        "Visible aliases of the triangular UHA reciprocal lattice."
        if key == "UHA" and has_z_axis
        else "Visible aliases of the periodic Y/Z sampling lattice."
    )
    return GratingLobeAssessment(
        geometry=key,
        has_aliasing_risk=bool(directions),
        risk_only=False,
        directions=tuple(directions),
        criterion=criterion,
    )


def steering_phases(
    y: ArrayLike,
    z: ArrayLike,
    wavelength_m: float,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
) -> FloatArray:
    """Return the vectorized array-response phases used by the array factor."""

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
    """Return the exact response vector used in the array-factor sum.

    Scalar angles return the same shape as the coordinate arrays.  Broadcast
    angle arrays prepend their shape to the coordinate-array shape.
    """

    phases = steering_phases(
        y,
        z,
        wavelength_m,
        azimuth_rad,
        elevation_rad,
    )
    return np.asarray(np.exp(1j * phases), dtype=complex)


def _relative_null_depths_db(
    constraint_matrix: ComplexArray,
    weights_flat: ComplexArray,
    *,
    maximum_depth_db: float = 300.0,
) -> tuple[float | None, ...]:
    """Measure null depths relative to the target response."""

    if constraint_matrix.shape[0] <= 1:
        return ()
    responses = constraint_matrix @ weights_flat
    target_magnitude = float(np.abs(responses[0]))
    if target_magnitude <= 0.0 or not np.isfinite(target_magnitude):
        return tuple(None for _ in responses[1:])

    minimum_ratio = 10.0 ** (-maximum_depth_db / 20.0)
    depths: list[float | None] = []
    for response in responses[1:]:
        null_magnitude = float(np.abs(response))
        if not np.isfinite(null_magnitude):
            depths.append(None)
            continue
        ratio = max(null_magnitude / target_magnitude, minimum_ratio)
        depths.append(float(-20.0 * np.log10(ratio)))
    return tuple(depths)


def compute_beamforming_weights(
    y: ArrayLike,
    z: ArrayLike,
    wavelength_m: float,
    target_azimuth_rad: float,
    target_elevation_rad: float,
    amplitude_weights: ArrayLike,
    *,
    phase_bits: int | str | None = None,
    null_direction_rad: tuple[float, float] | None = None,
    null_directions_rad: Sequence[tuple[float, float]] | None = None,
    singular_tolerance: float = 1e-6,
) -> BeamformingWeights:
    """Compute tapered weights with target and optional null constraints.

    The constraint rows use the exact same steering-vector definition as
    :func:`array_factor`.  A minimum-norm correction is solved in a
    taper-weighted control space, so failed or zero-taper elements remain at
    zero.  Hardware phase quantization is applied only after the continuous
    constrained solution is complete.

    ``null_direction_rad`` preserves the single-null API.  The plural
    ``null_directions_rad`` accepts multiple directions for future UI
    expansion; callers must not provide both.
    """

    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    amplitudes = np.asarray(amplitude_weights, dtype=float)
    if y_array.shape != z_array.shape or y_array.shape != amplitudes.shape:
        raise ValueError("Coordinates and amplitude weights must have matching shapes.")
    if np.any(~np.isfinite(amplitudes)) or np.any(amplitudes < 0):
        raise ValueError("Amplitude weights must be finite and non-negative.")
    if singular_tolerance <= 0.0 or not np.isfinite(singular_tolerance):
        raise ValueError("Singular tolerance must be finite and positive.")
    if null_direction_rad is not None and null_directions_rad is not None:
        raise ValueError("Use either null_direction_rad or null_directions_rad, not both.")

    if null_directions_rad is not None:
        null_directions = tuple(
            (float(azimuth), float(elevation))
            for azimuth, elevation in null_directions_rad
        )
    elif null_direction_rad is not None:
        null_directions = (
            (float(null_direction_rad[0]), float(null_direction_rad[1])),
        )
    else:
        null_directions = ()
    if any(
        not np.isfinite(azimuth) or not np.isfinite(elevation)
        for azimuth, elevation in null_directions
    ):
        raise ValueError("Null directions must contain finite angles.")

    target_response_vector = steering_vector(
        y_array,
        z_array,
        wavelength_m,
        target_azimuth_rad,
        target_elevation_rad,
    ).ravel()
    target_phases = np.angle(np.conjugate(target_response_vector)).reshape(
        y_array.shape
    )
    amplitude_flat = amplitudes.ravel()
    reference_control = np.conjugate(target_response_vector)
    reference_weights = amplitude_flat * reference_control

    response_rows = [target_response_vector]
    response_rows.extend(
        steering_vector(
            y_array,
            z_array,
            wavelength_m,
            null_azimuth,
            null_elevation,
        ).ravel()
        for null_azimuth, null_elevation in null_directions
    )
    constraint_matrix = np.asarray(np.vstack(response_rows), dtype=complex)

    continuous_flat = np.asarray(reference_weights, dtype=complex)
    null_applied = False
    determinant: float | None = None
    constraint_rank: int | None = None
    condition_number: float | None = None
    diagnostic_message: str | None = None

    if null_directions:
        control_matrix = constraint_matrix * amplitude_flat[None, :]
        singular_values = np.linalg.svd(control_matrix, compute_uv=False)
        largest_singular = float(singular_values[0]) if singular_values.size else 0.0
        rank_threshold = singular_tolerance * largest_singular
        constraint_rank = int(np.count_nonzero(singular_values > rank_threshold))
        constraint_count = int(constraint_matrix.shape[0])
        smallest_singular = float(singular_values[-1]) if singular_values.size else 0.0
        condition_number = (
            float(largest_singular / smallest_singular)
            if smallest_singular > 0.0
            else float("inf")
        )
        gram_matrix = control_matrix @ control_matrix.conjugate().T
        determinant = float(np.abs(np.linalg.det(gram_matrix)))
        condition_limit = 1.0 / singular_tolerance

        if constraint_rank < constraint_count or condition_number > condition_limit:
            diagnostic_message = (
                "Null constraint matrix is singular or ill-conditioned "
                f"(rank {constraint_rank}/{constraint_count}, "
                f"condition {condition_number:.3e})."
            )
        else:
            desired_response = np.zeros(constraint_count, dtype=complex)
            desired_response[0] = constraint_matrix[0] @ reference_weights
            residual = desired_response - control_matrix @ reference_control
            try:
                lagrange_multipliers = np.linalg.solve(gram_matrix, residual)
                corrected_control = (
                    reference_control
                    + control_matrix.conjugate().T @ lagrange_multipliers
                )
                continuous_flat = amplitude_flat * corrected_control
                null_applied = True
            except np.linalg.LinAlgError:
                diagnostic_message = (
                    "Null constraint solve failed because the Gram matrix is singular "
                    f"(rank {constraint_rank}/{constraint_count})."
                )
    else:
        constraint_count = 1

    continuous_weights = continuous_flat.reshape(y_array.shape)
    final_phases = quantize_phases(np.angle(continuous_weights), phase_bits)
    final_weights = np.abs(continuous_weights) * np.exp(1j * final_phases)
    final_weights[np.abs(continuous_weights) == 0.0] = 0.0

    continuous_depths = _relative_null_depths_db(
        constraint_matrix,
        np.asarray(continuous_flat, dtype=complex),
    )
    final_depths = _relative_null_depths_db(
        constraint_matrix,
        np.asarray(final_weights.ravel(), dtype=complex),
    )
    return BeamformingWeights(
        weights=np.asarray(final_weights, dtype=complex),
        continuous_weights=np.asarray(continuous_weights, dtype=complex),
        target_phases=np.asarray(target_phases, dtype=float),
        final_phases=np.asarray(final_phases, dtype=float),
        null_applied=null_applied,
        determinant=determinant,
        constraint_rank=constraint_rank,
        constraint_count=constraint_count,
        condition_number=condition_number,
        null_directions_rad=null_directions,
        continuous_null_depths_db=continuous_depths,
        null_depths_db=final_depths,
        diagnostic_message=diagnostic_message,
    )


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
    """Evaluate the array factor with bounded angle/element working sets.

    The unchunked phase matrix has ``direction_count * element_count``
    entries.  When that exceeds ``max_chunk_entries``, the function chooses an
    angle or element chunk automatically.  Explicit chunk sizes can be used
    for deterministic benchmarking and are safely nested when both are set.
    """

    if wavelength_m <= 0 or not np.isfinite(wavelength_m):
        raise ValueError("Wavelength must be a finite positive value.")
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    weights = np.asarray(complex_weights, dtype=complex)
    if y_array.shape != z_array.shape or y_array.shape != weights.shape:
        raise ValueError("Coordinates and complex weights must have matching shapes.")
    if max_chunk_entries < 1:
        raise ValueError("Maximum chunk entries must be a positive integer.")
    if angle_chunk_size is not None and angle_chunk_size < 1:
        raise ValueError("Angle chunk size must be a positive integer.")
    if element_chunk_size is not None and element_chunk_size < 1:
        raise ValueError("Element chunk size must be a positive integer.")

    azimuth = np.asarray(azimuth_rad, dtype=float)
    elevation = np.asarray(elevation_rad, dtype=float)
    azimuth, elevation = np.broadcast_arrays(azimuth, elevation)
    angle_shape = azimuth.shape
    _, u_y, u_z = direction_cosines(azimuth, elevation)
    u_y_flat = u_y.ravel()
    u_z_flat = u_z.ravel()
    direction_count = int(u_y_flat.size)
    element_count = int(weights.size)
    if direction_count == 0:
        return np.empty(angle_shape, dtype=complex)

    angle_chunk = min(direction_count, angle_chunk_size or direction_count)
    element_chunk = min(element_count, element_chunk_size or element_count)
    if angle_chunk_size is None and element_chunk_size is None:
        if direction_count * element_count > max_chunk_entries:
            if element_count >= direction_count:
                angle_chunk = max(1, max_chunk_entries // element_count)
            else:
                element_chunk = max(1, max_chunk_entries // direction_count)

    # If one axis alone exceeds the working-set limit, nest the other axis too.
    if angle_chunk * element_chunk > max_chunk_entries:
        if element_chunk_size is None:
            element_chunk = max(1, max_chunk_entries // angle_chunk)
        else:
            angle_chunk = max(1, max_chunk_entries // element_chunk)

    y_flat = y_array.ravel()
    z_flat = z_array.ravel()
    weights_flat = weights.ravel()
    wave_number = 2.0 * np.pi / wavelength_m
    result_flat = np.zeros(direction_count, dtype=complex)

    for angle_start in range(0, direction_count, angle_chunk):
        angle_stop = min(direction_count, angle_start + angle_chunk)
        chunk_response = np.zeros(angle_stop - angle_start, dtype=complex)
        chunk_u_y = u_y_flat[angle_start:angle_stop]
        chunk_u_z = u_z_flat[angle_start:angle_stop]
        for element_start in range(0, element_count, element_chunk):
            element_stop = min(element_count, element_start + element_chunk)
            phase = wave_number * (
                chunk_u_y[:, None] * y_flat[element_start:element_stop]
                + chunk_u_z[:, None] * z_flat[element_start:element_stop]
            )
            chunk_response += np.sum(
                weights_flat[element_start:element_stop] * np.exp(1j * phase),
                axis=1,
            )
        result_flat[angle_start:angle_stop] = chunk_response

    return np.asarray(result_flat.reshape(angle_shape), dtype=complex)


def normalize_pattern_linear(pattern: ArrayLike) -> FloatArray:
    """Normalize a pattern magnitude without ever dividing by a zero peak."""

    magnitude = np.abs(np.asarray(pattern, dtype=complex))
    if magnitude.size == 0:
        raise ValueError("Pattern must contain at least one sample.")
    if np.any(~np.isfinite(magnitude)):
        raise ValueError("Pattern contains non-finite values.")

    denominator = float(np.max(magnitude))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.zeros(magnitude.shape, dtype=float)

    return np.divide(
        magnitude,
        denominator,
        out=np.zeros(magnitude.shape, dtype=float),
        where=denominator > 0.0,
    )


def normalize_pattern_db(pattern: ArrayLike, *, floor_db: float = -120.0) -> FloatArray:
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
    """Calculate effective array gain and its efficiency components.

    For final weights ``w`` and the target-direction response ``A``::

        taper efficiency = (sum(abs(w)))**2 / (N_active * sum(abs(w)**2))
        phase efficiency = abs(A)**2 / (sum(abs(w)))**2
        effective gain   = abs(A)**2 / sum(abs(w)**2)

    Thus effective gain equals ``N_active * taper_efficiency *
    phase_efficiency`` and reduces to ``N_active`` for coherent uniform
    weights.  A ``None`` dB value represents an array with no usable weight.
    """

    if zero_tolerance < 0 or not np.isfinite(zero_tolerance):
        raise ValueError("Zero tolerance must be finite and non-negative.")

    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    weights = np.asarray(complex_weights, dtype=complex)
    mask = np.asarray(active_mask, dtype=bool)
    if not (
        y_array.shape == z_array.shape == weights.shape == mask.shape
    ):
        raise ValueError("Coordinates, weights, and active mask must have matching shapes.")
    if weights.size == 0:
        raise ValueError("Array gain requires at least one element.")
    if np.any(~np.isfinite(np.abs(weights))):
        raise ValueError("Complex weights must be finite.")
    if element_mask is None:
        physical_mask = np.ones(mask.shape, dtype=bool)
    else:
        physical_mask = np.asarray(element_mask, dtype=bool)
        if physical_mask.shape != mask.shape:
            raise ValueError("Element mask must match the active mask shape.")
        if not np.any(physical_mask):
            raise ValueError("Array gain requires at least one physical element.")
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
            array_gain_db=None,
        )

    # Work with scaled weights so gain stays invariant and finite even when
    # every non-zero weight is close to floating-point underflow or overflow.
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
            array_gain_db=None,
        )

    taper_efficiency = magnitude_sum**2 / (active_elements * weight_power)
    target_response = array_factor(
        y_array,
        z_array,
        scaled_weights,
        wavelength_m,
        target_azimuth_rad,
        target_elevation_rad,
    )
    target_power = float(np.abs(target_response.item()) ** 2)
    phase_efficiency = target_power / magnitude_sum**2
    effective_element_count = target_power / weight_power

    # Cauchy-Schwarz bounds both efficiencies by one; clip only round-off.
    taper_efficiency = float(np.clip(taper_efficiency, 0.0, 1.0))
    phase_efficiency = float(np.clip(phase_efficiency, 0.0, 1.0))
    effective_element_count = float(
        np.clip(effective_element_count, 0.0, float(active_elements))
    )
    array_gain_db = (
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
        array_gain_db=array_gain_db,
    )


def find_first_null(pattern_db: ArrayLike, peak_index: int) -> tuple[int, int]:
    """Find the nearest discrete local minima on both sides of the main lobe."""

    values = np.asarray(pattern_db, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Pattern must be a non-empty one-dimensional array.")
    if not 0 <= peak_index < values.size:
        raise ValueError("Peak index is outside the pattern.")

    null_left = 0
    for index in range(peak_index - 1, -1, -1):
        if values[index] > values[index + 1]:
            null_left = index + 1
            break

    null_right = values.size - 1
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
    """Linearly interpolate an angle where two samples cross a threshold."""

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
    """Calculate discrete HPBW, FNBW, and peak sidelobe level."""

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
    hpbw_deg = 0.0
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
    has_left_null = null_left > 0
    has_right_null = null_right < values.size - 1
    if has_left_null and has_right_null:
        fnbw_deg = float(np.degrees(angles[null_right] - angles[null_left]))
    elif has_right_null:
        fnbw_deg = float(2.0 * np.degrees(angles[null_right] - angles[peak_index]))
    elif has_left_null:
        fnbw_deg = float(2.0 * np.degrees(angles[peak_index] - angles[null_left]))
    else:
        fnbw_deg = 0.0

    sidelobe_values: list[FloatArray] = []
    sidelobe_indices: list[NDArray[np.int_]] = []
    if null_left > 0:
        sidelobe_values.append(pattern_db[:null_left])
        sidelobe_indices.append(np.arange(0, null_left))
    if null_right < values.size - 1:
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
        sidelobe_level_db = -99.0
        sidelobe_angle_deg = 0.0

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
