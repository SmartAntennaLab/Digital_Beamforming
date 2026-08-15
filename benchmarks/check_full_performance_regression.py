"""Fail CI when any representative full calculation workload regresses."""

from __future__ import annotations

import os
import sys

from benchmark_workloads import WORKLOAD_NAMES, run_workload

DEFAULT_BUDGET_SECONDS = {
    "directivity": 5.0,
    "surface": 5.0,
    "multi_null": 8.0,
    "automatic_scan": 12.0,
    "advanced_models": 10.0,
}


def _positive_budget(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return value


def main() -> int:
    max_rss_mib = _positive_budget("DBF_PERF_MAX_RSS_DELTA_MIB", 512.0)
    failed = False
    # Warm NumPy dispatch and imports outside the timed gates.
    run_workload("surface")
    for workload in WORKLOAD_NAMES:
        budget_name = f"DBF_PERF_{workload.upper()}_SECONDS"
        budget = _positive_budget(
            budget_name,
            DEFAULT_BUDGET_SECONDS[workload],
        )
        result = run_workload(workload)
        passed = (
            result.valid
            and result.seconds <= budget
            and result.peak_rss_delta_mib <= max_rss_mib
        )
        print(
            f"{'PASS' if passed else 'FAIL'} {workload}: "
            f"{result.seconds:.3f}s <= {budget:.3f}s, "
            f"peak RSS delta {result.peak_rss_delta_mib:.1f} MiB <= "
            f"{max_rss_mib:.1f} MiB, {result.detail}"
        )
        failed = failed or not passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
