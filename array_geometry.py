"""Array geometry, taper, failure-mask, and grating-lobe models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import direction_cosines

FloatArray: TypeAlias = NDArray[np.float64]
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
        return int(np.count_nonzero(self.element_mask))

    @property
    def row_lengths(self) -> tuple[int, ...]:
        return tuple(
            int(value) for value in np.count_nonzero(self.element_mask, axis=1)
        )


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


def geometry_id(geometry: str) -> str:
    """Normalize a stable geometry ID or legacy translated label."""

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
    """Create Y-Z element coordinates for ULA, UPA, UCA, or UHA geometry."""

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

    key = geometry_id(geometry)
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
        radius = (
            0.0
            if horizontal_count == 1
            else horizontal_spacing_m
            / (2.0 * np.sin(np.pi / horizontal_count))
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
    """Return independently controllable axes for the effective geometry."""

    if vertical_count < 1 or horizontal_count < 1:
        raise ValueError("Element counts must be positive integers.")
    key = geometry_id(geometry)
    azimuth_controllable = horizontal_count > 1
    elevation_controllable = key in {"UPA", "UHA"} and vertical_count > 1
    return SteeringLimits(
        geometry=key,
        azimuth_controllable=azimuth_controllable,
        elevation_controllable=elevation_controllable,
    )


def get_window_weights(length: int, option: str) -> FloatArray:
    """Return normalized amplitude-window weights."""

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
    """Create a deterministic mask using explicit round-half-up counting."""

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
    requested_failure_count = total * failure_rate_percent / 100.0
    failure_count = min(total, int(np.floor(requested_failure_count + 0.5)))
    mask = physical_mask.ravel().copy()
    if failure_count:
        rng = np.random.RandomState(seed)
        failed_indices = rng.choice(physical_indices, failure_count, replace=False)
        mask[failed_indices] = False
    return mask.reshape(rows, columns)


def create_array_taper(
    coordinates: ArrayCoordinates,
    option: str,
) -> FloatArray:
    """Create geometry-aware separable amplitude weights."""

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
    """Assess visible spatial aliases using geometry-specific conditions."""

    key = geometry_id(geometry)
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

    _, target_u_y_array, target_u_z_array = direction_cosines(
        steering_azimuth_rad,
        steering_elevation_rad,
    )
    target_u_y = float(target_u_y_array)
    target_u_z = float(target_u_z_array)
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
                candidate_u_z = target_u_z + (
                    order_z - 0.5 * order_y
                ) / vertical_spacing
            else:
                candidate_u_z = target_u_z + order_z / vertical_spacing
            if candidate_u_y**2 + candidate_u_z**2 > 1.0 + tolerance:
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
