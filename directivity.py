"""Full-sphere radiation-power integration and directional directivity."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import element_pattern_factor, steering_phases
from pattern_metrics import array_factor

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
DIRECTIVITY_SCHEMA_VERSION = 2
DIRECTIVITY_MODE_OPTIONS = ("auto", "exact", "fast")
DEFAULT_DIRECTIVITY_WARNING_ELEMENTS = 1_024
DEFAULT_DIRECTIVITY_EXACT_MAX_ELEMENTS = 4_096
DEFAULT_FAST_AZIMUTH_SAMPLES = 64
DEFAULT_FAST_ELEVATION_SAMPLES = 33
DEFAULT_FAST_LOCAL_SAMPLES = 9
PAIRWISE_KERNEL_CACHE_MAX_ELEMENTS = 1_024
PAIRWISE_KERNEL_CACHE_MAX_BYTES = 32 * 1024 * 1024
_PAIRWISE_KERNEL_CACHE: OrderedDict[tuple[object, ...], FloatArray] = OrderedDict()
_PAIRWISE_KERNEL_CACHE_BYTES = 0
_PAIRWISE_KERNEL_CACHE_LOCK = threading.RLock()


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
    requested_mode: str = "auto"
    effective_mode: str = "exact"
    is_approximate: bool = False
    element_count: int = 0
    pair_count: int = 0
    kernel_cache_used: bool = False
    kernel_cache_hit: bool = False
    warning_message: str | None = None


@dataclass(frozen=True)
class PairwiseKernelCacheInfo:
    entries: int
    bytes: int
    maximum_bytes: int
    maximum_elements: int


def clear_pairwise_kernel_cache() -> None:
    """Clear bounded process-wide analytic geometry kernels."""

    global _PAIRWISE_KERNEL_CACHE_BYTES
    with _PAIRWISE_KERNEL_CACHE_LOCK:
        _PAIRWISE_KERNEL_CACHE.clear()
        _PAIRWISE_KERNEL_CACHE_BYTES = 0


def pairwise_kernel_cache_info() -> PairwiseKernelCacheInfo:
    with _PAIRWISE_KERNEL_CACHE_LOCK:
        return PairwiseKernelCacheInfo(
            entries=len(_PAIRWISE_KERNEL_CACHE),
            bytes=_PAIRWISE_KERNEL_CACHE_BYTES,
            maximum_bytes=PAIRWISE_KERNEL_CACHE_MAX_BYTES,
            maximum_elements=PAIRWISE_KERNEL_CACHE_MAX_ELEMENTS,
        )


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


def _kernel_function(pattern_id: str) -> Callable[[FloatArray], FloatArray]:
    return {
        "isotropic": _isotropic_kernel,
        "cosine": _cosine_kernel,
        "cosine_squared": _cosine_squared_kernel,
    }[pattern_id]


def _pairwise_kernel_cache_key(
    y: FloatArray,
    z: FloatArray,
    wavelength_m: float,
    pattern_id: str,
) -> tuple[object, ...]:
    normalized = np.ascontiguousarray(
        np.column_stack((y / wavelength_m, z / wavelength_m)), dtype="<f8"
    )
    digest = hashlib.blake2b(normalized.tobytes(), digest_size=16).digest()
    return pattern_id, int(y.size), digest


def _build_pairwise_kernel(
    y: FloatArray,
    z: FloatArray,
    wavelength_m: float,
    pattern_id: str,
) -> FloatArray:
    delta_y = y[:, None] - y[None, :]
    delta_z = z[:, None] - z[None, :]
    q = (2.0 * np.pi / wavelength_m) * np.hypot(delta_y, delta_z)
    kernel = _kernel_function(pattern_id)(np.asarray(q, dtype=float))
    kernel.setflags(write=False)
    return kernel


def _cached_pairwise_kernel(
    y: FloatArray,
    z: FloatArray,
    wavelength_m: float,
    pattern_id: str,
    cancel_check: Callable[[], None] | None,
) -> tuple[FloatArray, bool]:
    """Return a bounded LRU geometry kernel and whether it was a cache hit."""

    global _PAIRWISE_KERNEL_CACHE_BYTES
    key = _pairwise_kernel_cache_key(y, z, wavelength_m, pattern_id)
    with _PAIRWISE_KERNEL_CACHE_LOCK:
        cached = _PAIRWISE_KERNEL_CACHE.get(key)
        if cached is not None:
            _PAIRWISE_KERNEL_CACHE.move_to_end(key)
            return cached, True

    if cancel_check is not None:
        cancel_check()
    kernel = _build_pairwise_kernel(y, z, wavelength_m, pattern_id)
    if cancel_check is not None:
        cancel_check()

    with _PAIRWISE_KERNEL_CACHE_LOCK:
        cached = _PAIRWISE_KERNEL_CACHE.get(key)
        if cached is not None:
            _PAIRWISE_KERNEL_CACHE.move_to_end(key)
            return cached, True
        while (
            _PAIRWISE_KERNEL_CACHE
            and _PAIRWISE_KERNEL_CACHE_BYTES + kernel.nbytes
            > PAIRWISE_KERNEL_CACHE_MAX_BYTES
        ):
            _, evicted = _PAIRWISE_KERNEL_CACHE.popitem(last=False)
            _PAIRWISE_KERNEL_CACHE_BYTES -= evicted.nbytes
        if kernel.nbytes <= PAIRWISE_KERNEL_CACHE_MAX_BYTES:
            _PAIRWISE_KERNEL_CACHE[key] = kernel
            _PAIRWISE_KERNEL_CACHE_BYTES += kernel.nbytes
    return kernel, False


def _pairwise_power_integral(
    y: FloatArray,
    z: FloatArray,
    weights: ComplexArray,
    wavelength_m: float,
    pattern_id: str,
    *,
    max_chunk_entries: int,
    cancel_check: Callable[[], None] | None,
) -> tuple[float, bool, bool]:
    """Evaluate the full-sphere integral through exact pairwise kernels."""

    element_count = weights.size
    cache_eligible = (
        element_count <= PAIRWISE_KERNEL_CACHE_MAX_ELEMENTS
        and element_count * element_count <= max_chunk_entries
        and element_count * element_count * np.dtype(float).itemsize
        <= PAIRWISE_KERNEL_CACHE_MAX_BYTES
    )
    if cache_eligible:
        kernel, cache_hit = _cached_pairwise_kernel(
            y, z, wavelength_m, pattern_id, cancel_check
        )
        if cancel_check is not None:
            cancel_check()
        power = float(np.real(np.vdot(weights, kernel @ weights)))
        return power, True, cache_hit

    kernel_function = _kernel_function(pattern_id)
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
    return float(np.real(total)), False, False


def _periodic_cell_weights(nodes: FloatArray) -> FloatArray:
    previous = np.roll(nodes, 1)
    following = np.roll(nodes, -1)
    previous[0] -= 2.0 * np.pi
    following[-1] += 2.0 * np.pi
    return np.asarray(0.5 * (following - previous), dtype=float)


def _bounded_cell_weights(nodes: FloatArray) -> FloatArray:
    edges = np.empty(nodes.size + 1, dtype=float)
    edges[0] = -1.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    return np.diff(edges)


def _fast_sampling_axes(
    y: FloatArray,
    z: FloatArray,
    wavelength_m: float,
    target_azimuth_rad: float,
    target_elevation_rad: float,
    *,
    azimuth_samples: int,
    elevation_samples: int,
    local_samples: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build global nodes plus deterministic main/back-lobe local nodes."""

    azimuth = np.linspace(-np.pi, np.pi, azimuth_samples, endpoint=False)
    mu = np.linspace(-1.0, 1.0, elevation_samples)
    aperture_wavelengths = max(float(np.ptp(y)), float(np.ptp(z))) / wavelength_m
    angular_scale = min(
        np.deg2rad(15.0),
        max(np.deg2rad(0.05), 0.8 / max(aperture_wavelengths, 1.0)),
    )
    offsets = np.linspace(-2.0, 2.0, local_samples) * angular_scale
    target_mu = float(np.sin(target_elevation_rad))
    centers = (
        (float(target_azimuth_rad), target_mu),
        (float(target_azimuth_rad) + np.pi, -target_mu),
    )
    azimuth_extra: list[FloatArray] = []
    mu_extra: list[FloatArray] = []
    mu_scale = max(abs(float(np.cos(target_elevation_rad))), 0.1)
    for center_azimuth, center_mu in centers:
        azimuth_extra.append(
            (center_azimuth + offsets + np.pi) % (2.0 * np.pi) - np.pi
        )
        mu_extra.append(np.clip(center_mu + offsets * mu_scale, -1.0, 1.0))
    azimuth = np.unique(np.concatenate((azimuth, *azimuth_extra)))
    mu = np.unique(np.concatenate((mu, *mu_extra, np.array([-1.0, 1.0]))))
    return azimuth, mu, _periodic_cell_weights(azimuth), _bounded_cell_weights(mu)


def _fast_power_integral(
    y: FloatArray,
    z: FloatArray,
    weights: ComplexArray,
    wavelength_m: float,
    pattern_id: str,
    target_azimuth_rad: float,
    target_elevation_rad: float,
    *,
    azimuth_samples: int,
    elevation_samples: int,
    local_samples: int,
    max_chunk_entries: int,
    cancel_check: Callable[[], None] | None,
) -> tuple[float, int, int]:
    azimuth, mu, azimuth_weights, mu_weights = _fast_sampling_axes(
        y,
        z,
        wavelength_m,
        target_azimuth_rad,
        target_elevation_rad,
        azimuth_samples=azimuth_samples,
        elevation_samples=elevation_samples,
        local_samples=local_samples,
    )
    azimuth_grid, mu_grid = np.meshgrid(azimuth, mu, indexing="ij")
    elevation_grid = np.arcsin(mu_grid)
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
    pattern *= element_pattern_factor(pattern_id, azimuth_grid, elevation_grid)
    intensity = np.abs(pattern) ** 2
    integral = np.sum(
        intensity * azimuth_weights[:, None] * mu_weights[None, :]
    )
    return float(integral), int(azimuth.size), int(mu.size)


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
    directivity_mode: str = "auto",
    warning_element_count: int = DEFAULT_DIRECTIVITY_WARNING_ELEMENTS,
    exact_max_elements: int = DEFAULT_DIRECTIVITY_EXACT_MAX_ELEMENTS,
    fast_azimuth_samples: int = DEFAULT_FAST_AZIMUTH_SAMPLES,
    fast_elevation_samples: int = DEFAULT_FAST_ELEVATION_SAMPLES,
    fast_local_samples: int = DEFAULT_FAST_LOCAL_SAMPLES,
    cancel_check: Callable[[], None] | None = None,
) -> DirectivityResult:
    """Calculate physical target-direction directivity from full-sphere power.

    ``auto`` selects exact integration for small arrays and the O(N*S) fast
    quadrature for larger arrays. Explicit exact requests above the hard cap
    safely fall back to fast integration.
    """

    y_array: FloatArray = np.asarray(y, dtype=np.float64)
    z_array: FloatArray = np.asarray(z, dtype=np.float64)
    weights_array: ComplexArray = np.asarray(complex_weights, dtype=np.complex128)
    if y_array.shape != z_array.shape or y_array.shape != weights_array.shape:
        raise ValueError("Coordinates and weights must have matching shapes.")
    if wavelength_m <= 0.0 or not np.isfinite(wavelength_m):
        raise ValueError("Wavelength must be finite and positive.")
    if max_chunk_entries < 1:
        raise ValueError("Maximum chunk entries must be positive.")
    if dipole_azimuth_samples < 8 or dipole_elevation_samples < 8:
        raise ValueError("Dipole integration requires at least eight samples per axis.")
    requested_mode = directivity_mode.strip().lower()
    if requested_mode not in DIRECTIVITY_MODE_OPTIONS:
        raise ValueError(f"Unsupported directivity mode: {directivity_mode!r}")
    if warning_element_count < 1 or exact_max_elements < 1:
        raise ValueError("Directivity element limits must be positive.")
    if fast_azimuth_samples < 16 or fast_elevation_samples < 9:
        raise ValueError("Fast integration requires at least 16 x 9 global samples.")
    if fast_local_samples < 1:
        raise ValueError("Fast local sample count must be positive.")
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
    weights_flat = np.asarray(weights_flat[nonzero], dtype=np.complex128)
    element_count = int(weights_flat.size)
    pair_count = element_count * element_count
    if weights_flat.size == 0:
        return DirectivityResult(
            schema_version=DIRECTIVITY_SCHEMA_VERSION,
            directivity_linear=None,
            directivity_dbi=None,
            target_radiation_intensity=0.0,
            radiated_power_integral=0.0,
            integration_method="unavailable",
            requested_mode=requested_mode,
            effective_mode="unavailable",
        )

    warning_message: str | None = None
    if requested_mode == "auto":
        effective_mode = (
            "exact" if element_count <= warning_element_count else "fast"
        )
        if effective_mode == "fast":
            warning_message = (
                f"소자 {element_count:,}개가 자동 정확 모드 기준 "
                f"{warning_element_count:,}개를 넘어 고속 근사를 사용했습니다."
            )
    elif requested_mode == "exact" and element_count > exact_max_elements:
        effective_mode = "fast"
        warning_message = (
            f"소자 {element_count:,}개가 정확 모드 상한 {exact_max_elements:,}개를 "
            "넘어 고속 근사로 전환했습니다."
        )
    else:
        effective_mode = requested_mode
        if requested_mode == "exact" and element_count > warning_element_count:
            warning_message = (
                f"정확 모드는 {pair_count:,}개 pairwise 항을 계산하므로 시간이 오래 걸릴 수 있습니다."
            )

    # Normalization prevents overflow without changing the dimensionless ratio.
    normalized_weights: ComplexArray = np.asarray(
        weights_flat / float(np.max(np.abs(weights_flat))),
        dtype=np.complex128,
    )
    pattern_id = _pattern_id(element_option)
    target_phase = steering_phases(
        y_flat,
        z_flat,
        wavelength_m,
        float(target_azimuth_rad),
        float(target_elevation_rad),
    )
    target_response = np.sum(np.exp(1j * target_phase) * normalized_weights)
    target_element_factor = float(
        element_pattern_factor(
            pattern_id,
            float(target_azimuth_rad),
            float(target_elevation_rad),
        )
    )
    target_intensity = float(np.abs(target_response * target_element_factor) ** 2)

    kernel_cache_used = False
    kernel_cache_hit = False
    azimuth_samples: int | None
    elevation_samples: int | None
    if effective_mode == "fast":
        power_integral, azimuth_samples, elevation_samples = _fast_power_integral(
            y_flat,
            z_flat,
            normalized_weights,
            wavelength_m,
            pattern_id,
            float(target_azimuth_rad),
            float(target_elevation_rad),
            azimuth_samples=fast_azimuth_samples,
            elevation_samples=fast_elevation_samples,
            local_samples=fast_local_samples,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        method = "nonuniform full-sphere quadrature (fast approximation)"
        self_kernel = {
            "isotropic": 4.0 * np.pi,
            "cosine": 2.0 * np.pi / 3.0,
            "cosine_squared": 2.0 * np.pi / 5.0,
            "dipole": 2.0 * np.pi * 0.6090,
        }[pattern_id]
    elif pattern_id == "dipole":
        power_integral = _dipole_power_integral(
            y_flat,
            z_flat,
            normalized_weights,
            wavelength_m,
            azimuth_samples=dipole_azimuth_samples,
            elevation_samples=dipole_elevation_samples,
            max_chunk_entries=max_chunk_entries,
            cancel_check=cancel_check,
        )
        method = "spherical Gauss-Legendre quadrature"
        azimuth_samples = dipole_azimuth_samples
        elevation_samples = dipole_elevation_samples
        self_kernel = 2.0 * np.pi * 0.6090
    else:
        power_integral, kernel_cache_used, kernel_cache_hit = _pairwise_power_integral(
            y_flat,
            z_flat,
            normalized_weights,
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
        * max(1.0, float(np.sum(np.abs(normalized_weights) ** 2)) * self_kernel)
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
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            is_approximate=effective_mode == "fast",
            element_count=element_count,
            pair_count=pair_count,
            kernel_cache_used=kernel_cache_used,
            kernel_cache_hit=kernel_cache_hit,
            warning_message=warning_message,
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
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        is_approximate=effective_mode == "fast",
        element_count=element_count,
        pair_count=pair_count,
        kernel_cache_used=kernel_cache_used,
        kernel_cache_hit=kernel_cache_hit,
        warning_message=warning_message,
    )
