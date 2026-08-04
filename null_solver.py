"""SVD-based constrained null steering and quantization diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import parse_phase_bits, quantize_phases, steering_vector


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class ConstraintDiagnostics:
    """Constraint residuals and weight-resource use for one weight vector."""

    target_response_error: float | None
    target_relative_error: float | None
    null_constraint_residuals: tuple[float | None, ...]
    null_relative_residuals: tuple[float | None, ...]
    constraint_residual_norm: float | None
    constraint_relative_residual_norm: float | None
    max_amplitude: float
    total_weight_power: float


@dataclass(frozen=True)
class BeamformingWeights:
    """Complex element weights and null-steering diagnostics."""

    weights: ComplexArray
    continuous_weights: ComplexArray
    target_phases: FloatArray
    final_phases: FloatArray
    null_applied: bool
    solver_method: str
    phase_quantization_applied: bool
    determinant: float | None
    constraint_rank: int | None
    constraint_count: int
    condition_number: float | None
    null_directions_rad: tuple[tuple[float, float], ...]
    continuous_null_depths_db: tuple[float | None, ...]
    null_depths_db: tuple[float | None, ...]
    continuous_diagnostics: ConstraintDiagnostics
    final_diagnostics: ConstraintDiagnostics
    quantization_target_degradation_db: float | None
    quantization_null_degradation_db: tuple[float | None, ...]
    quantization_constraint_degradation_db: float | None
    diagnostic_message: str | None


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


def _finite_magnitude(value: complex) -> float | None:
    magnitude = float(np.abs(value))
    return magnitude if np.isfinite(magnitude) else None


def _constraint_diagnostics(
    constraint_matrix: ComplexArray,
    weights_flat: ComplexArray,
    desired_response: ComplexArray,
) -> ConstraintDiagnostics:
    residual = constraint_matrix @ weights_flat - desired_response
    residual_magnitudes = tuple(_finite_magnitude(value) for value in residual)
    target_error = residual_magnitudes[0] if residual_magnitudes else None
    null_residuals = residual_magnitudes[1:]

    residual_norm_value = float(np.linalg.norm(residual))
    residual_norm = (
        residual_norm_value if np.isfinite(residual_norm_value) else None
    )
    target_scale = _finite_magnitude(desired_response[0])
    if target_scale is None or target_scale <= 0.0:
        target_relative_error = None
        null_relative_residuals = tuple(None for _ in null_residuals)
        relative_residual_norm = None
    else:
        target_relative_error = (
            target_error / target_scale if target_error is not None else None
        )
        null_relative_residuals = tuple(
            value / target_scale if value is not None else None
            for value in null_residuals
        )
        relative_residual_norm = (
            residual_norm / target_scale if residual_norm is not None else None
        )

    amplitudes = np.abs(weights_flat)
    max_amplitude = float(np.max(amplitudes)) if amplitudes.size else 0.0
    total_weight_power = float(np.sum(np.square(amplitudes)))
    return ConstraintDiagnostics(
        target_response_error=target_error,
        target_relative_error=target_relative_error,
        null_constraint_residuals=null_residuals,
        null_relative_residuals=null_relative_residuals,
        constraint_residual_norm=residual_norm,
        constraint_relative_residual_norm=relative_residual_norm,
        max_amplitude=max_amplitude,
        total_weight_power=total_weight_power,
    )


def _residual_degradation_db(
    continuous_residual: float | None,
    final_residual: float | None,
    *,
    residual_floor: float = 1e-15,
) -> float | None:
    if continuous_residual is None or final_residual is None:
        return None
    if not np.isfinite(continuous_residual) or not np.isfinite(final_residual):
        return None
    continuous_level = max(float(continuous_residual), residual_floor)
    final_level = max(float(final_residual), residual_floor)
    return float(20.0 * np.log10(final_level / continuous_level))


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
    """Solve target/null constraints with a direct SVD minimum-norm update."""

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
    constraint_count = int(constraint_matrix.shape[0])
    desired_response = np.zeros(constraint_count, dtype=complex)
    desired_response[0] = constraint_matrix[0] @ reference_weights

    continuous_flat = np.asarray(reference_weights, dtype=complex).copy()
    null_applied = False
    solver_method = "not_requested"
    constraint_rank: int | None = None
    condition_number: float | None = None
    diagnostic_message: str | None = None

    if null_directions:
        control_matrix = constraint_matrix * amplitude_flat[None, :]
        try:
            left_vectors, singular_values, right_vectors_h = np.linalg.svd(
                control_matrix,
                full_matrices=False,
            )
        except np.linalg.LinAlgError:
            solver_method = "svd_failed"
            diagnostic_message = (
                "Null constraint SVD failed to converge; steering-only weights "
                "were retained."
            )
        else:
            largest_singular = (
                float(singular_values[0]) if singular_values.size else 0.0
            )
            rank_threshold = singular_tolerance * largest_singular
            constraint_rank = int(
                np.count_nonzero(singular_values > rank_threshold)
            )
            smallest_singular = (
                float(singular_values[-1]) if singular_values.size else 0.0
            )
            condition_number = (
                float(largest_singular / smallest_singular)
                if constraint_rank == constraint_count
                and smallest_singular > rank_threshold
                else float("inf")
            )
            condition_limit = 1.0 / singular_tolerance
            if (
                constraint_rank < constraint_count
                or condition_number > condition_limit
            ):
                solver_method = "svd_rejected"
                diagnostic_message = (
                    "Null constraint matrix is singular or ill-conditioned "
                    f"(rank {constraint_rank}/{constraint_count}, "
                    f"condition {condition_number:.3e})."
                )
            else:
                residual = desired_response - control_matrix @ reference_control
                projected_residual = left_vectors.conjugate().T @ residual
                minimum_norm_correction = right_vectors_h.conjugate().T @ (
                    projected_residual / singular_values
                )
                corrected_control = reference_control + minimum_norm_correction
                continuous_flat = amplitude_flat * corrected_control
                null_applied = True
                solver_method = "svd_minimum_norm"

    continuous_weights = continuous_flat.reshape(y_array.shape)
    parsed_phase_bits = parse_phase_bits(phase_bits)
    phase_quantization_applied = parsed_phase_bits is not None
    final_phases = quantize_phases(
        np.angle(continuous_weights),
        parsed_phase_bits,
    )
    if phase_quantization_applied:
        final_weights = np.abs(continuous_weights) * np.exp(1j * final_phases)
        final_weights[np.abs(continuous_weights) == 0.0] = 0.0
    else:
        final_weights = continuous_weights.copy()

    continuous_depths = _relative_null_depths_db(
        constraint_matrix,
        np.asarray(continuous_flat, dtype=complex),
    )
    final_depths = _relative_null_depths_db(
        constraint_matrix,
        np.asarray(final_weights.ravel(), dtype=complex),
    )
    continuous_diagnostics = _constraint_diagnostics(
        constraint_matrix,
        np.asarray(continuous_flat, dtype=complex),
        desired_response,
    )
    final_diagnostics = _constraint_diagnostics(
        constraint_matrix,
        np.asarray(final_weights.ravel(), dtype=complex),
        desired_response,
    )
    quantization_target_degradation = _residual_degradation_db(
        continuous_diagnostics.target_relative_error,
        final_diagnostics.target_relative_error,
    )
    quantization_null_degradation = tuple(
        _residual_degradation_db(continuous, final)
        for continuous, final in zip(
            continuous_diagnostics.null_relative_residuals,
            final_diagnostics.null_relative_residuals,
            strict=True,
        )
    )
    quantization_constraint_degradation = _residual_degradation_db(
        continuous_diagnostics.constraint_relative_residual_norm,
        final_diagnostics.constraint_relative_residual_norm,
    )
    return BeamformingWeights(
        weights=np.asarray(final_weights, dtype=complex),
        continuous_weights=np.asarray(continuous_weights, dtype=complex),
        target_phases=np.asarray(target_phases, dtype=float),
        final_phases=np.asarray(final_phases, dtype=float),
        null_applied=null_applied,
        solver_method=solver_method,
        phase_quantization_applied=phase_quantization_applied,
        determinant=None,
        constraint_rank=constraint_rank,
        constraint_count=constraint_count,
        condition_number=condition_number,
        null_directions_rad=null_directions,
        continuous_null_depths_db=continuous_depths,
        null_depths_db=final_depths,
        continuous_diagnostics=continuous_diagnostics,
        final_diagnostics=final_diagnostics,
        quantization_target_degradation_db=quantization_target_degradation,
        quantization_null_degradation_db=quantization_null_degradation,
        quantization_constraint_degradation_db=quantization_constraint_degradation,
        diagnostic_message=diagnostic_message,
    )
