"""Inline and Process Pool execution backends for heavy numerical work."""

from __future__ import annotations

import atexit
import math
import multiprocessing
import os
import threading
import time
import uuid
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass
from typing import Literal, Protocol

from compute_governor import (
    ComputeBusyError,
    ComputeCancelled,
    ComputeDeadlineExceeded,
)
from compute_tasks import ViewComputeRequest, ViewComputeResult, calculate_view
from resource_policy import ResourcePolicy

ComputeBackendMode = Literal["inline", "process"]


class _EventLike(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


class _WorkerCancelled(RuntimeError):
    pass


class _WorkerDeadlineExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ComputeExecutorSnapshot:
    mode: ComputeBackendMode
    worker_count: int
    inflight_tasks: int


def _positive_environment_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(1, value), maximum)


def _process_entry(
    request: ViewComputeRequest,
    cancel_event: _EventLike,
    timeout_seconds: float,
) -> ViewComputeResult:
    """Run in a child process with cooperative cancellation checkpoints."""

    deadline = time.monotonic() + timeout_seconds

    def check() -> None:
        if cancel_event.is_set():
            raise _WorkerCancelled("Worker task was cancelled.")
        if time.monotonic() > deadline:
            raise _WorkerDeadlineExceeded("Worker task exceeded its deadline.")

    return calculate_view(request, cancel_check=check)


class ComputeExecutor:
    """Execute serializable view calculations inline or in child processes."""

    def __init__(
        self,
        *,
        mode: ComputeBackendMode = "inline",
        worker_count: int = 1,
        max_tasks_per_child: int = 100,
    ) -> None:
        if mode not in {"inline", "process"}:
            raise ValueError("Unsupported compute backend.")
        if worker_count < 1 or max_tasks_per_child < 1:
            raise ValueError("Process worker limits must be positive.")
        self.mode = mode
        self.worker_count = worker_count if mode == "process" else 0
        self.max_tasks_per_child = max_tasks_per_child
        self._lock = threading.RLock()
        self._inflight: dict[str, dict[str, _EventLike]] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._manager = None
        if mode == "process":
            # Prevent every child from creating a full BLAS thread pool.
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            context = multiprocessing.get_context("spawn")
            self._manager = context.Manager()
            self._pool = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                max_tasks_per_child=max_tasks_per_child,
            )

    def execute(
        self,
        request: ViewComputeRequest,
        *,
        session_id: str,
        timeout_seconds: float,
        cancel_check,
    ) -> ViewComputeResult:
        if not session_id:
            raise ValueError("A non-empty session ID is required.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("Execution timeout must be finite and positive.")
        if self.mode == "inline":
            deadline = time.monotonic() + timeout_seconds

            def inline_check() -> None:
                cancel_check()
                if time.monotonic() > deadline:
                    raise ComputeDeadlineExceeded(
                        "Inline calculation exceeded its execution deadline."
                    )

            return calculate_view(request, cancel_check=inline_check)

        if self._pool is None or self._manager is None:
            raise ComputeBusyError("Process worker pool is not available.")
        task_id = uuid.uuid4().hex
        cancel_event = self._manager.Event()
        with self._lock:
            self._inflight.setdefault(session_id, {})[task_id] = cancel_event
        future: Future[ViewComputeResult] = self._pool.submit(
            _process_entry,
            request,
            cancel_event,
            timeout_seconds,
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    cancel_event.set()
                    future.cancel()
                    raise ComputeDeadlineExceeded(
                        "Process worker calculation exceeded its deadline."
                    )
                try:
                    return future.result(timeout=min(0.05, remaining))
                except FutureTimeoutError:
                    cancel_check()
        except (_WorkerCancelled, ComputeCancelled) as error:
            cancel_event.set()
            future.cancel()
            raise ComputeCancelled(str(error)) from error
        except (_WorkerDeadlineExceeded, ComputeDeadlineExceeded) as error:
            cancel_event.set()
            future.cancel()
            raise ComputeDeadlineExceeded(str(error)) from error
        except BaseException:
            cancel_event.set()
            future.cancel()
            raise
        finally:
            with self._lock:
                session_tasks = self._inflight.get(session_id)
                if session_tasks is not None:
                    session_tasks.pop(task_id, None)
                    if not session_tasks:
                        self._inflight.pop(session_id, None)

    def cancel_session(self, session_id: str) -> None:
        with self._lock:
            events = tuple(self._inflight.get(session_id, {}).values())
        for event in events:
            event.set()

    def snapshot(self) -> ComputeExecutorSnapshot:
        with self._lock:
            inflight = sum(len(tasks) for tasks in self._inflight.values())
        return ComputeExecutorSnapshot(
            mode=self.mode,
            worker_count=self.worker_count,
            inflight_tasks=inflight,
        )

    def shutdown(self) -> None:
        with self._lock:
            pool = self._pool
            manager = self._manager
            self._pool = None
            self._manager = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        if manager is not None:
            manager.shutdown()


_EXECUTORS: dict[tuple[ComputeBackendMode, int, int], ComputeExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def get_compute_executor(policy: ResourcePolicy) -> ComputeExecutor:
    mode_text = os.getenv("DBF_COMPUTE_BACKEND", "inline").strip().lower()
    mode: ComputeBackendMode = "process" if mode_text == "process" else "inline"
    worker_count = _positive_environment_int(
        "DBF_PROCESS_WORKERS",
        policy.max_concurrent_calculations,
        32,
    )
    max_tasks = _positive_environment_int("DBF_PROCESS_MAX_TASKS", 100, 10_000)
    key = (mode, worker_count, max_tasks)
    with _EXECUTORS_LOCK:
        executor = _EXECUTORS.get(key)
        if executor is None:
            executor = ComputeExecutor(
                mode=mode,
                worker_count=worker_count,
                max_tasks_per_child=max_tasks,
            )
            _EXECUTORS[key] = executor
        return executor


def _shutdown_executors() -> None:
    with _EXECUTORS_LOCK:
        executors = tuple(_EXECUTORS.values())
        _EXECUTORS.clear()
    for executor in executors:
        executor.shutdown()


atexit.register(_shutdown_executors)


__all__ = [
    "ComputeBackendMode",
    "ComputeExecutor",
    "ComputeExecutorSnapshot",
    "get_compute_executor",
]
