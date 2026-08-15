"""Concurrent session soak test with parent/worker RSS leak checks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_executor import ComputeExecutor  # noqa: E402
from compute_governor import ComputeGovernor  # noqa: E402
from compute_tasks import ViewComputeRequest  # noqa: E402
from resource_policy import ResourcePolicy  # noqa: E402
from simulation import SimulationConfig  # noqa: E402


@dataclass(frozen=True)
class SoakResult:
    backend: str
    sessions: int
    iterations_per_session: int
    completed_tasks: int
    failed_tasks: int
    seconds: float
    tasks_per_second: float
    baseline_rss_mib: float
    peak_rss_mib: float
    final_rss_mib: float
    final_growth_mib: float
    memory_budget_mib: float


def _process_tree_rss() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _request(iteration: int) -> ViewComputeRequest:
    if iteration % 2 == 0:
        return ViewComputeRequest(
            SimulationConfig(vertical_count=8, horizontal_count=8),
            float((iteration % 7) * 5 - 15),
            5.0,
            "pattern",
            scan_mode="preview_3d",
            scanning=True,
        )
    config = SimulationConfig(
        vertical_count=8,
        horizontal_count=8,
        enable_null_steering=True,
        null_constraints_deg=((25.0, 0.0, 25.0), (-25.0, 5.0, 25.0)),
    )
    return ViewComputeRequest(config, 0.0, 0.0, "metrics")


def run_soak(
    *,
    sessions: int,
    iterations: int,
    backend: str,
    workers: int,
    memory_budget_mib: float,
) -> SoakResult:
    if sessions < 1 or iterations < 1 or workers < 1:
        raise ValueError("Soak concurrency values must be positive.")
    if memory_budget_mib <= 0.0:
        raise ValueError("Memory budget must be positive.")
    mode = "process" if backend == "process" else "inline"
    executor = ComputeExecutor(
        mode=mode,
        worker_count=workers,
        max_tasks_per_child=max(10, iterations),
    )
    policy = ResourcePolicy(
        max_concurrent_calculations=workers,
        compute_queue_timeout_seconds=30.0,
        compute_timeout_seconds=60.0,
        session_calculations_per_minute=600,
        session_burst=60,
    )
    governor = ComputeGovernor(policy)

    # Warm imports, BLAS dispatch, and worker creation before the leak baseline.
    with governor.lease("warmup", "elements:UPA") as lease:
        executor.execute(
            ViewComputeRequest(SimulationConfig(), 0.0, 0.0, "elements"),
            session_id="warmup",
            timeout_seconds=30.0,
            cancel_check=lease.check,
        )
    gc.collect()
    baseline_rss = _process_tree_rss()
    peak_rss = baseline_rss
    stop_sampler = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop_sampler.wait(0.05):
            peak_rss = max(peak_rss, _process_tree_rss())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    def run_session(session_index: int) -> tuple[int, int]:
        completed = 0
        failed = 0
        session_id = f"soak-{session_index}"
        for iteration in range(iterations):
            request = _request(iteration)
            try:
                with governor.lease(
                    session_id,
                    f"{request.view_name}:{request.config.geometry}",
                ) as lease:
                    executor.execute(
                        request,
                        session_id=session_id,
                        timeout_seconds=60.0,
                        cancel_check=lease.check,
                    )
                completed += 1
            except Exception:
                failed += 1
        return completed, failed

    started_at = time.perf_counter()
    completed = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=sessions) as session_pool:
            futures = [session_pool.submit(run_session, index) for index in range(sessions)]
            for future in as_completed(futures):
                session_completed, session_failed = future.result()
                completed += session_completed
                failed += session_failed
    finally:
        duration = max(0.001, time.perf_counter() - started_at)
        stop_sampler.set()
        sampler.join()
        gc.collect()
        final_rss = _process_tree_rss()
        peak_rss = max(peak_rss, final_rss)
        executor.shutdown()

    return SoakResult(
        backend=mode,
        sessions=sessions,
        iterations_per_session=iterations,
        completed_tasks=completed,
        failed_tasks=failed,
        seconds=duration,
        tasks_per_second=completed / duration,
        baseline_rss_mib=baseline_rss / (1024.0**2),
        peak_rss_mib=peak_rss / (1024.0**2),
        final_rss_mib=final_rss / (1024.0**2),
        final_growth_mib=max(0.0, (final_rss - baseline_rss) / (1024.0**2)),
        memory_budget_mib=memory_budget_mib,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backend", choices=("inline", "process"), default="process")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-budget-mib", type=float, default=256.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run_soak(
        sessions=args.sessions,
        iterations=args.iterations,
        backend=args.backend,
        workers=args.workers,
        memory_budget_mib=args.memory_budget_mib,
    )
    print(json.dumps(asdict(result), indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    if result.failed_tasks or result.final_growth_mib > result.memory_budget_mib:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
