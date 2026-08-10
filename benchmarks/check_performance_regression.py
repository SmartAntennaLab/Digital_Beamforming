"""Fail CI when representative 64x64 directivity work exceeds its budget."""

from __future__ import annotations

import os
import sys

from benchmark_directivity import BenchmarkResult, run_case


def _positive_budget(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _print_result(result: BenchmarkResult, budget_seconds: float) -> None:
    status = "PASS" if result.seconds <= budget_seconds else "FAIL"
    print(
        f"{status} {result.array} {result.requested_mode}->{result.effective_mode}: "
        f"{result.seconds:.3f}s <= {budget_seconds:.3f}s, "
        f"peak RSS delta {result.peak_rss_delta_mib:.1f} MiB"
    )


def main() -> int:
    exact_budget = _positive_budget("DBF_PERF_EXACT_64_SECONDS", 5.0)
    fast_budget = _positive_budget("DBF_PERF_FAST_64_SECONDS", 6.0)

    # Warm imports, ufunc dispatch and BLAS initialization outside timed gates.
    run_case(8, "fast")
    cases = (
        (run_case(64, "exact"), exact_budget, "exact"),
        (run_case(64, "fast"), fast_budget, "fast"),
    )
    failed = False
    for result, budget, required_mode in cases:
        _print_result(result, budget)
        if result.effective_mode != required_mode:
            print(
                f"FAIL expected effective mode {required_mode!r}, "
                f"got {result.effective_mode!r}"
            )
            failed = True
        if result.seconds > budget:
            failed = True
        if result.directivity_dbi is None:
            print("FAIL directivity result is unavailable")
            failed = True

    exact_result, _, _ = cases[0]
    fast_result, _, _ = cases[1]
    if (
        exact_result.directivity_dbi is not None
        and fast_result.directivity_dbi is not None
        and abs(exact_result.directivity_dbi - fast_result.directivity_dbi) > 0.5
    ):
        print("FAIL fast directivity differs from exact by more than 0.5 dB")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
