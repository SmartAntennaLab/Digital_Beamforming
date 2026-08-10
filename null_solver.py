"""SVD-based constrained null steering and quantization diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from array_math import parse_phase_bits, quantize_phases, steering_vector

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]


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
class OptimizationTracePoint:
    """One deterministic optimizer checkpoint."""

    restart_index: int
    iteration: int
    objective: float
    worst_null_residual_db: float | None
    target_loss_db: float | None


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
    null_required_suppression_db: tuple[float, ...]
    continuous_null_depths_db: tuple[float | None, ...]
    null_depths_db: tuple[float | None, ...]
    continuous_null_requirement_met: tuple[bool | None, ...]
    null_requirement_met: tuple[bool | None, ...]
    continuous_diagnostics: ConstraintDiagnostics
    final_diagnostics: ConstraintDiagnostics
    quantization_target_degradation_db: float | None
    quantization_null_degradation_db: tuple[float | None, ...]
    quantization_constraint_degradation_db: float | None
    optimization_mode: str
    maximum_element_amplitude: float | None
    saturated_element_mask: BoolArray
    saturated_element_count: int
    optimizer_iterations: int
    optimizer_total_iterations: int
    optimizer_max_iterations: int
    optimizer_tolerance: float
    optimizer_restart_count: int
    optimizer_selected_restart: int | None
    optimizer_convergence_reason: str
    optimizer_final_objective: float | None
    optimizer_trace: tuple[OptimizationTracePoint, ...]
    diagnostic_message: str | None


@dataclass(frozen=True)
class _OptimizationResult:
    weights: ComplexArray
    iterations: int
    total_iterations: int
    selected_restart: int
    convergence_reason: str
    final_objective: float
    trace: tuple[OptimizationTracePoint, ...]


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
    residual_norm = residual_norm_value if np.isfinite(residual_norm_value) else None
    target_scale = _finite_magnitude(desired_response[0])
    null_relative_residuals: tuple[float | None, ...]
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


def _requirement_status(
    depths_db: tuple[float | None, ...],
    required_db: tuple[float, ...],
) -> tuple[bool | None, ...]:
    return tuple(
        None if depth is None else bool(depth + 1.0e-9 >= requirement)
        for depth, requirement in zip(depths_db, required_db, strict=True)
    )


def _weighted_constraint_system(
    constraint_matrix: ComplexArray,
    desired_response: ComplexArray,
    required_suppression_db: tuple[float, ...],
) -> tuple[ComplexArray, ComplexArray]:
    """Scale target/null rows without allowing large dB weights to overflow."""

    target_scale = float(np.abs(desired_response[0]))
    if target_scale <= 0.0 or not np.isfinite(target_scale):
        return (
            np.asarray(constraint_matrix, dtype=complex).copy(),
            np.asarray(desired_response, dtype=complex).copy(),
        )

    null_weights = np.asarray(
        [
            10.0 ** (min(requirement, 120.0) / 20.0)
            for requirement in required_suppression_db
        ],
        dtype=float,
    )
    target_weight = max(100.0, float(np.max(null_weights, initial=1.0)))
    row_weights = np.concatenate(([target_weight], null_weights))
    row_weights /= float(np.max(row_weights, initial=1.0))
    return (
        constraint_matrix * row_weights[:, None] / target_scale,
        desired_response * row_weights / target_scale,
    )


def _optimization_trace_point(
    constraint_matrix: ComplexArray,
    desired_response: ComplexArray,
    weights: ComplexArray,
    *,
    restart_index: int,
    iteration: int,
    objective: float,
) -> OptimizationTracePoint:
    responses = constraint_matrix @ weights
    desired_target = float(np.abs(desired_response[0]))
    actual_target = float(np.abs(responses[0])) if responses.size else 0.0
    if (
        desired_target <= 0.0
        or not np.isfinite(desired_target)
        or not np.isfinite(actual_target)
    ):
        target_loss_db = None
    else:
        target_ratio = max(actual_target / desired_target, 1.0e-15)
        target_loss_db = float(-20.0 * np.log10(target_ratio))

    if responses.size <= 1 or actual_target <= 0.0 or not np.isfinite(actual_target):
        worst_null_residual_db = None
    else:
        worst_null = float(np.max(np.abs(responses[1:])))
        if np.isfinite(worst_null):
            ratio = max(worst_null / actual_target, 1.0e-15)
            worst_null_residual_db = float(20.0 * np.log10(ratio))
        else:
            worst_null_residual_db = None

    return OptimizationTracePoint(
        restart_index=restart_index,
        iteration=iteration,
        objective=float(objective),
        worst_null_residual_db=worst_null_residual_db,
        target_loss_db=target_loss_db,
    )


def _objective(
    weighted_matrix: ComplexArray,
    weighted_desired: ComplexArray,
    weights: ComplexArray,
) -> float:
    residual = weighted_matrix @ weights - weighted_desired
    return float(np.vdot(residual, residual).real)


def _phase_only_single_restart(
    constraint_matrix: ComplexArray,
    desired_response: ComplexArray,
    weighted_matrix: ComplexArray,
    weighted_desired: ComplexArray,
    fixed_amplitudes: FloatArray,
    initial_weights: ComplexArray,
    *,
    restart_index: int,
    max_iterations: int,
    tolerance: float,
    cancel_check: Callable[[], None] | None,
) -> _OptimizationResult:
    """Run one projected phase-gradient solve from a supplied initial phase."""

    phases = np.angle(initial_weights)
    weights = fixed_amplitudes * np.exp(1j * phases)
    best_weights = np.asarray(weights, dtype=complex).copy()
    best_objective = _objective(weighted_matrix, weighted_desired, best_weights)
    trace = [
        _optimization_trace_point(
            constraint_matrix,
            desired_response,
            best_weights,
            restart_index=restart_index,
            iteration=0,
            objective=best_objective,
        )
    ]
    step = 0.25
    completed_iterations = 0
    convergence_reason = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        if cancel_check is not None:
            cancel_check()
        residual = weighted_matrix @ weights - weighted_desired
        gradient = -2.0 * np.imag(
            weights * (weighted_matrix.T @ np.conjugate(residual))
        )
        gradient_scale = float(np.max(np.abs(gradient), initial=0.0))
        if not np.isfinite(gradient_scale):
            convergence_reason = "non_finite_gradient"
            break
        if gradient_scale <= tolerance:
            convergence_reason = "gradient_tolerance"
            break

        trial_step = step
        candidate_weights = weights
        candidate_phases = phases
        candidate_objective = best_objective
        while trial_step >= 1.0e-10:
            if cancel_check is not None:
                cancel_check()
            trial_phases = phases - trial_step * gradient / gradient_scale
            trial_weights = fixed_amplitudes * np.exp(1j * trial_phases)
            trial_objective = _objective(
                weighted_matrix,
                weighted_desired,
                trial_weights,
            )
            if np.isfinite(trial_objective) and trial_objective < best_objective:
                candidate_phases = trial_phases
                candidate_weights = trial_weights
                candidate_objective = trial_objective
                break
            trial_step *= 0.5

        if candidate_objective >= best_objective:
            convergence_reason = "step_tolerance"
            break

        previous_objective = best_objective
        phases = candidate_phases
        weights = candidate_weights
        best_weights = np.asarray(candidate_weights, dtype=complex).copy()
        best_objective = candidate_objective
        step = min(0.5, trial_step * 1.05)
        completed_iterations = iteration
        trace.append(
            _optimization_trace_point(
                constraint_matrix,
                desired_response,
                best_weights,
                restart_index=restart_index,
                iteration=iteration,
                objective=best_objective,
            )
        )
        improvement = previous_objective - best_objective
        if improvement <= tolerance * max(1.0, abs(previous_objective)):
            convergence_reason = "objective_tolerance"
            break

    return _OptimizationResult(
        weights=best_weights,
        iterations=completed_iterations,
        total_iterations=completed_iterations,
        selected_restart=restart_index,
        convergence_reason=convergence_reason,
        final_objective=best_objective,
        trace=tuple(trace),
    )


def _phase_only_optimize(
    constraint_matrix: ComplexArray,
    desired_response: ComplexArray,
    fixed_amplitudes: FloatArray,
    initial_weights: ComplexArray,
    required_suppression_db: tuple[float, ...],
    *,
    max_iterations: int,
    tolerance: float,
    restart_count: int,
    cancel_check: Callable[[], None] | None,
) -> _OptimizationResult:
    """Minimize fixed-amplitude residuals using deterministic multi-starts."""

    weighted_matrix, weighted_desired = _weighted_constraint_system(
        constraint_matrix,
        desired_response,
        required_suppression_db,
    )
    base_phases = np.angle(initial_weights)
    target_phases = np.angle(np.conjugate(constraint_matrix[0]))
    initial_candidates = [
        fixed_amplitudes * np.exp(1j * base_phases),
        fixed_amplitudes * np.exp(1j * target_phases),
    ]
    rng = np.random.default_rng(0xD1BF)
    while len(initial_candidates) < restart_count:
        restart_number = len(initial_candidates)
        scale = min(1.0, 0.25 + 0.25 * (restart_number - 1))
        perturbation = rng.uniform(-np.pi, np.pi, size=base_phases.shape) * scale
        initial_candidates.append(
            fixed_amplitudes * np.exp(1j * (base_phases + perturbation))
        )

    results: list[_OptimizationResult] = []
    all_trace: list[OptimizationTracePoint] = []
    total_iterations = 0
    for restart_index, candidate in enumerate(
        initial_candidates[:restart_count],
        start=1,
    ):
        if cancel_check is not None:
            cancel_check()
        result = _phase_only_single_restart(
            constraint_matrix,
            desired_response,
            weighted_matrix,
            weighted_desired,
            fixed_amplitudes,
            candidate,
            restart_index=restart_index,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cancel_check=cancel_check,
        )
        results.append(result)
        all_trace.extend(result.trace)
        total_iterations += result.iterations

    selected = min(results, key=lambda item: item.final_objective)
    return _OptimizationResult(
        weights=selected.weights,
        iterations=selected.iterations,
        total_iterations=total_iterations,
        selected_restart=selected.selected_restart,
        convergence_reason=selected.convergence_reason,
        final_objective=selected.final_objective,
        trace=tuple(all_trace),
    )


def _project_complex_bounds(
    weights: ComplexArray,
    bounds: FloatArray,
) -> ComplexArray:
    magnitudes = np.abs(weights)
    scale = np.ones_like(magnitudes, dtype=float)
    nonzero = magnitudes > 0.0
    scale[nonzero] = np.minimum(1.0, bounds[nonzero] / magnitudes[nonzero])
    projected = np.asarray(weights * scale, dtype=complex)
    projected[bounds <= 0.0] = 0.0
    return projected


def _bounded_complex_optimize(
    constraint_matrix: ComplexArray,
    desired_response: ComplexArray,
    initial_weights: ComplexArray,
    bounds: FloatArray,
    required_suppression_db: tuple[float, ...],
    *,
    max_iterations: int,
    tolerance: float,
    cancel_check: Callable[[], None] | None,
) -> _OptimizationResult:
    """Solve weighted complex least squares under per-element disk bounds."""

    weighted_matrix, weighted_desired = _weighted_constraint_system(
        constraint_matrix,
        desired_response,
        required_suppression_db,
    )
    weights = _project_complex_bounds(initial_weights, bounds)
    best_objective = _objective(weighted_matrix, weighted_desired, weights)
    trace = [
        _optimization_trace_point(
            constraint_matrix,
            desired_response,
            weights,
            restart_index=1,
            iteration=0,
            objective=best_objective,
        )
    ]
    spectral_norm = float(np.linalg.norm(weighted_matrix, ord=2))
    if not np.isfinite(spectral_norm) or spectral_norm <= 0.0:
        return _OptimizationResult(
            weights=weights,
            iterations=0,
            total_iterations=0,
            selected_restart=1,
            convergence_reason="degenerate_system",
            final_objective=best_objective,
            trace=tuple(trace),
        )

    step = 1.0 / (spectral_norm * spectral_norm)
    completed_iterations = 0
    convergence_reason = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        if cancel_check is not None:
            cancel_check()
        residual = weighted_matrix @ weights - weighted_desired
        gradient = weighted_matrix.conjugate().T @ residual
        if np.any(~np.isfinite(gradient)):
            convergence_reason = "non_finite_gradient"
            break

        trial_step = step
        candidate = weights
        candidate_objective = best_objective
        while trial_step >= 1.0e-14:
            if cancel_check is not None:
                cancel_check()
            trial = _project_complex_bounds(weights - trial_step * gradient, bounds)
            trial_objective = _objective(
                weighted_matrix,
                weighted_desired,
                trial,
            )
            if np.isfinite(trial_objective) and trial_objective <= best_objective:
                candidate = trial
                candidate_objective = trial_objective
                break
            trial_step *= 0.5

        projected_step = float(np.linalg.norm(candidate - weights))
        reference_norm = max(1.0, float(np.linalg.norm(weights)))
        if projected_step <= tolerance * reference_norm:
            convergence_reason = "projected_step_tolerance"
            break

        previous_objective = best_objective
        weights = np.asarray(candidate, dtype=complex)
        best_objective = candidate_objective
        step = min(1.0 / (spectral_norm * spectral_norm), trial_step * 1.1)
        completed_iterations = iteration
        trace.append(
            _optimization_trace_point(
                constraint_matrix,
                desired_response,
                weights,
                restart_index=1,
                iteration=iteration,
                objective=best_objective,
            )
        )
        improvement = previous_objective - best_objective
        if improvement <= tolerance * max(1.0, abs(previous_objective)):
            convergence_reason = "objective_tolerance"
            break

    return _OptimizationResult(
        weights=weights,
        iterations=completed_iterations,
        total_iterations=completed_iterations,
        selected_restart=1,
        convergence_reason=convergence_reason,
        final_objective=best_objective,
        trace=tuple(trace),
    )


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
    null_required_suppression_db: Sequence[float] | None = None,
    maximum_element_amplitude: float | None = None,
    optimization_mode: str = "amplitude_phase",
    singular_tolerance: float = 1e-6,
    optimizer_tolerance: float = 1e-8,
    optimizer_max_iterations: int = 400,
    optimizer_restart_count: int = 4,
    cancel_check: Callable[[], None] | None = None,
) -> BeamformingWeights:
    """Solve practical target/null constraints with cooperative cancellation."""

    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    amplitudes = np.asarray(amplitude_weights, dtype=float)
    if y_array.shape != z_array.shape or y_array.shape != amplitudes.shape:
        raise ValueError("Coordinates and amplitude weights must have matching shapes.")
    if np.any(~np.isfinite(amplitudes)) or np.any(amplitudes < 0):
        raise ValueError("Amplitude weights must be finite and non-negative.")
    if singular_tolerance <= 0.0 or not np.isfinite(singular_tolerance):
        raise ValueError("Singular tolerance must be finite and positive.")
    if optimizer_tolerance <= 0.0 or not np.isfinite(optimizer_tolerance):
        raise ValueError("Optimizer tolerance must be finite and positive.")
    if (
        isinstance(optimizer_max_iterations, bool)
        or int(optimizer_max_iterations) != optimizer_max_iterations
        or not 1 <= int(optimizer_max_iterations) <= 10_000
    ):
        raise ValueError("Optimizer maximum iterations must be between 1 and 10000.")
    optimizer_max_iterations = int(optimizer_max_iterations)
    if (
        isinstance(optimizer_restart_count, bool)
        or int(optimizer_restart_count) != optimizer_restart_count
        or not 1 <= int(optimizer_restart_count) <= 16
    ):
        raise ValueError("Optimizer restart count must be between 1 and 16.")
    optimizer_restart_count = int(optimizer_restart_count)
    if optimization_mode not in {"amplitude_phase", "phase_only"}:
        raise ValueError("Unsupported null optimization mode.")
    if maximum_element_amplitude is not None:
        maximum_element_amplitude = float(maximum_element_amplitude)
        if (
            not np.isfinite(maximum_element_amplitude)
            or maximum_element_amplitude <= 0.0
        ):
            raise ValueError("Maximum element amplitude must be finite and positive.")
    if null_direction_rad is not None and null_directions_rad is not None:
        raise ValueError(
            "Use either null_direction_rad or null_directions_rad, not both."
        )

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

    if null_required_suppression_db is None:
        required_suppression = tuple(40.0 for _ in null_directions)
    else:
        required_suppression = tuple(
            float(value) for value in null_required_suppression_db
        )
    if len(required_suppression) != len(null_directions):
        raise ValueError("Each null direction requires one suppression value.")
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 300.0
        for value in required_suppression
    ):
        raise ValueError("Required null suppression must be between 0 and 300 dB.")

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
    fixed_amplitudes = np.asarray(amplitude_flat, dtype=float).copy()
    if maximum_element_amplitude is not None:
        fixed_amplitudes = np.minimum(
            fixed_amplitudes,
            maximum_element_amplitude,
        )
    reference_control = np.conjugate(target_response_vector)
    reference_weights = fixed_amplitudes * reference_control

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
    optimizer_iterations = 0
    optimizer_total_iterations = 0
    optimizer_selected_restart: int | None = None
    optimizer_convergence_reason = "not_run"
    optimizer_final_objective: float | None = None
    optimizer_trace: tuple[OptimizationTracePoint, ...] = ()
    diagnostic_message: str | None = None

    if null_directions:
        if cancel_check is not None:
            cancel_check()
        control_matrix = constraint_matrix * fixed_amplitudes[None, :]
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
            constraint_rank = int(np.count_nonzero(singular_values > rank_threshold))
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
            if constraint_rank < constraint_count or condition_number > condition_limit:
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
                unconstrained_weights = fixed_amplitudes * corrected_control
                if optimization_mode == "phase_only":
                    initial_weights = fixed_amplitudes * np.exp(
                        1j * np.angle(unconstrained_weights)
                    )
                    optimization = _phase_only_optimize(
                        constraint_matrix,
                        desired_response,
                        fixed_amplitudes,
                        initial_weights,
                        required_suppression,
                        max_iterations=optimizer_max_iterations,
                        tolerance=optimizer_tolerance,
                        restart_count=optimizer_restart_count,
                        cancel_check=cancel_check,
                    )
                    continuous_flat = optimization.weights
                    solver_method = "phase_only_projected_gradient"
                else:
                    continuous_flat = np.asarray(
                        unconstrained_weights,
                        dtype=complex,
                    )
                    if maximum_element_amplitude is not None:
                        amplitude_bounds = np.where(
                            amplitude_flat > 0.0,
                            maximum_element_amplitude,
                            0.0,
                        )
                        optimization = _bounded_complex_optimize(
                            constraint_matrix,
                            desired_response,
                            continuous_flat,
                            np.asarray(amplitude_bounds, dtype=float),
                            required_suppression,
                            max_iterations=optimizer_max_iterations,
                            tolerance=optimizer_tolerance,
                            cancel_check=cancel_check,
                        )
                        continuous_flat = optimization.weights
                        solver_method = "bounded_projected_gradient"
                    else:
                        optimization = None
                        solver_method = "svd_minimum_norm"
                if optimization_mode == "phase_only" or (
                    optimization_mode == "amplitude_phase"
                    and maximum_element_amplitude is not None
                ):
                    assert optimization is not None
                    optimizer_iterations = optimization.iterations
                    optimizer_total_iterations = optimization.total_iterations
                    optimizer_selected_restart = optimization.selected_restart
                    optimizer_convergence_reason = optimization.convergence_reason
                    optimizer_final_objective = optimization.final_objective
                    optimizer_trace = optimization.trace
                null_applied = True

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

    if maximum_element_amplitude is None:
        saturated_flat = np.zeros(amplitude_flat.shape, dtype=bool)
    else:
        saturation_threshold = maximum_element_amplitude * (1.0 - 1.0e-9)
        saturated_flat = (fixed_amplitudes > 0.0) & (
            np.abs(final_weights.ravel()) >= saturation_threshold
        )
    saturated_mask = saturated_flat.reshape(y_array.shape)

    continuous_depths = _relative_null_depths_db(
        constraint_matrix,
        np.asarray(continuous_flat, dtype=complex),
    )
    final_depths = _relative_null_depths_db(
        constraint_matrix,
        np.asarray(final_weights.ravel(), dtype=complex),
    )
    continuous_requirement_met = _requirement_status(
        continuous_depths,
        required_suppression,
    )
    final_requirement_met = _requirement_status(
        final_depths,
        required_suppression,
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

    unmet_count = sum(status is False for status in final_requirement_met)
    if null_applied and unmet_count:
        diagnostic_message = (
            f"{unmet_count}/{len(final_requirement_met)} null directions did not "
            "meet the requested suppression after practical constraints."
        )
    elif null_applied and optimizer_convergence_reason == "max_iterations":
        diagnostic_message = (
            "Null optimizer reached the configured iteration limit; inspect the "
            "convergence trace before accepting the solution."
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
        null_required_suppression_db=required_suppression,
        continuous_null_depths_db=continuous_depths,
        null_depths_db=final_depths,
        continuous_null_requirement_met=continuous_requirement_met,
        null_requirement_met=final_requirement_met,
        continuous_diagnostics=continuous_diagnostics,
        final_diagnostics=final_diagnostics,
        quantization_target_degradation_db=quantization_target_degradation,
        quantization_null_degradation_db=quantization_null_degradation,
        quantization_constraint_degradation_db=quantization_constraint_degradation,
        optimization_mode=optimization_mode,
        maximum_element_amplitude=maximum_element_amplitude,
        saturated_element_mask=np.asarray(saturated_mask, dtype=bool),
        saturated_element_count=int(np.count_nonzero(saturated_mask)),
        optimizer_iterations=optimizer_iterations,
        optimizer_total_iterations=optimizer_total_iterations,
        optimizer_max_iterations=optimizer_max_iterations,
        optimizer_tolerance=float(optimizer_tolerance),
        optimizer_restart_count=(
            optimizer_restart_count if optimization_mode == "phase_only" else 1
        ),
        optimizer_selected_restart=optimizer_selected_restart,
        optimizer_convergence_reason=optimizer_convergence_reason,
        optimizer_final_objective=optimizer_final_objective,
        optimizer_trace=optimizer_trace,
        diagnostic_message=diagnostic_message,
    )
