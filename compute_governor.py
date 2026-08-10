"""Process-wide compute admission, cancellation, deadlines, and telemetry."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import psutil

from resource_policy import ResourcePolicy

LOGGER = logging.getLogger("digital_beamforming.compute")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _health_handler = logging.StreamHandler()
    _health_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    LOGGER.addHandler(_health_handler)
LOGGER.propagate = False
_CURRENT_LEASE = threading.local()


class ComputeGovernorError(RuntimeError):
    """Base class for expected compute-governor rejections."""


class ComputeBusyError(ComputeGovernorError):
    """Raised when every process-wide calculation slot remains occupied."""


class SessionRateLimitError(ComputeGovernorError):
    """Raised when one browser session exhausts its token bucket."""

    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        super().__init__(
            f"Session calculation rate exceeded; retry after "
            f"{self.retry_after_seconds:.2f} seconds."
        )


class ComputeDeadlineExceeded(ComputeGovernorError):
    """Raised cooperatively after a calculation passes its deadline."""


class ComputeCancelled(ComputeGovernorError):
    """Raised cooperatively when a session invalidates a running lease."""


class _ProcessLike(Protocol):
    def cpu_percent(self, interval: float | None = None) -> float: ...

    def memory_info(self): ...


@dataclass(frozen=True)
class ComputeSnapshot:
    """One process and governor health sample suitable for UI or logs."""

    active_calculations: int
    queued_calculations: int
    max_concurrent_calculations: int
    completed_calculations: int
    busy_rejections: int
    rate_rejections: int
    timed_out_calculations: int
    cancelled_calculations: int
    average_duration_seconds: float
    process_cpu_percent: float
    system_cpu_percent: float
    process_rss_bytes: int
    system_memory_percent: float


@dataclass
class _Bucket:
    tokens: float
    updated_at: float
    last_seen_at: float


class ComputeLease:
    """One admitted calculation with a cooperative deadline and generation."""

    def __init__(
        self,
        governor: ComputeGovernor,
        session_id: str,
        task_label: str,
    ) -> None:
        self._governor = governor
        self.session_id = session_id
        self.task_label = task_label
        self.started_at = 0.0
        self.deadline = 0.0
        self.generation = 0
        self._previous_lease: ComputeLease | None = None
        self._admitted = False

    def __enter__(self) -> ComputeLease:
        self._governor._consume_session_token(self.session_id)
        with self._governor._lock:
            self._governor._queued += 1
        acquired = False
        try:
            acquired = self._governor._semaphore.acquire(
                timeout=self._governor.policy.compute_queue_timeout_seconds
            )
        finally:
            with self._governor._lock:
                self._governor._queued -= 1
        if not acquired:
            with self._governor._lock:
                self._governor._busy_rejections += 1
            raise ComputeBusyError("Server calculation slots are busy; retry shortly.")

        self._admitted = True
        self.started_at = self._governor._clock()
        self.deadline = self.started_at + self._governor.policy.compute_timeout_seconds
        with self._governor._lock:
            self.generation = self._governor._session_generations.get(
                self.session_id, 0
            )
            self._governor._active += 1
        self._previous_lease = getattr(_CURRENT_LEASE, "value", None)
        _CURRENT_LEASE.value = self
        return self

    def check(self) -> None:
        """Raise at a safe checkpoint if cancelled or past the deadline."""

        now = self._governor._clock()
        with self._governor._lock:
            current_generation = self._governor._session_generations.get(
                self.session_id, 0
            )
        if current_generation != self.generation:
            raise ComputeCancelled(
                f"Calculation {self.task_label!r} was cancelled by the session."
            )
        if now > self.deadline:
            raise ComputeDeadlineExceeded(
                f"Calculation {self.task_label!r} exceeded "
                f"{self._governor.policy.compute_timeout_seconds:.2f} seconds."
            )

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        generated_error: ComputeGovernorError | None = None
        if exc_type is None:
            try:
                self.check()
            except (ComputeDeadlineExceeded, ComputeCancelled) as error:
                generated_error = error

        _CURRENT_LEASE.value = self._previous_lease
        if self._admitted:
            self._governor._semaphore.release()
            duration = max(0.0, self._governor._clock() - self.started_at)
            effective_error = generated_error or exc_value
            with self._governor._lock:
                self._governor._active -= 1
                self._governor._finished += 1
                self._governor._total_duration_seconds += duration
                if isinstance(effective_error, ComputeDeadlineExceeded):
                    self._governor._timed_out += 1
                elif isinstance(effective_error, ComputeCancelled):
                    self._governor._cancelled += 1
                elif effective_error is None:
                    self._governor._completed += 1

        if generated_error is not None:
            raise generated_error
        return False


class ComputeGovernor:
    """Coordinate all heavy calculations inside one Python process."""

    def __init__(
        self,
        policy: ResourcePolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        process: _ProcessLike | None = None,
    ) -> None:
        if policy.max_concurrent_calculations < 1:
            raise ValueError("Concurrent calculation limit must be positive.")
        if (
            not math.isfinite(policy.compute_queue_timeout_seconds)
            or policy.compute_queue_timeout_seconds <= 0.0
        ):
            raise ValueError("Compute queue timeout must be finite and positive.")
        if (
            not math.isfinite(policy.compute_timeout_seconds)
            or policy.compute_timeout_seconds <= 0.0
        ):
            raise ValueError("Compute deadline must be finite and positive.")
        if policy.session_calculations_per_minute < 1 or policy.session_burst < 1:
            raise ValueError("Session calculation rate and burst must be positive.")
        self.policy = policy
        self._clock = clock
        self._process = process or psutil.Process(os.getpid())
        self._semaphore = threading.BoundedSemaphore(policy.max_concurrent_calculations)
        self._lock = threading.RLock()
        self._monitor_lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}
        self._session_generations: dict[str, int] = {}
        self._active = 0
        self._queued = 0
        self._completed = 0
        self._finished = 0
        self._busy_rejections = 0
        self._rate_rejections = 0
        self._timed_out = 0
        self._cancelled = 0
        self._total_duration_seconds = 0.0
        self._last_health_log_at = float("-inf")
        try:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except (psutil.Error, OSError):
            pass

    def lease(self, session_id: str, task_label: str) -> ComputeLease:
        if not session_id:
            raise ValueError("A non-empty session ID is required.")
        return ComputeLease(self, session_id, task_label)

    def cancel_session(self, session_id: str) -> None:
        """Invalidate every lease already running for one browser session."""

        if not session_id:
            return
        with self._lock:
            self._session_generations[session_id] = (
                self._session_generations.get(session_id, 0) + 1
            )

    def _consume_session_token(self, session_id: str) -> None:
        now = self._clock()
        refill_rate = self.policy.session_calculations_per_minute / 60.0
        capacity = float(self.policy.session_burst)
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                bucket = _Bucket(capacity, now, now)
                self._buckets[session_id] = bucket
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(capacity, bucket.tokens + elapsed * refill_rate)
            bucket.updated_at = now
            bucket.last_seen_at = now
            if bucket.tokens < 1.0:
                self._rate_rejections += 1
                retry_after = (1.0 - bucket.tokens) / refill_rate
                raise SessionRateLimitError(retry_after)
            bucket.tokens -= 1.0

            stale_before = now - 600.0
            if len(self._buckets) > 2_048:
                stale_ids = [
                    key
                    for key, value in self._buckets.items()
                    if value.last_seen_at < stale_before
                ]
                for key in stale_ids:
                    self._buckets.pop(key, None)
                    self._session_generations.pop(key, None)

    def snapshot(self) -> ComputeSnapshot:
        with self._lock:
            active = self._active
            queued = self._queued
            completed = self._completed
            busy_rejections = self._busy_rejections
            rate_rejections = self._rate_rejections
            timed_out = self._timed_out
            cancelled = self._cancelled
            average_duration = (
                self._total_duration_seconds / self._finished if self._finished else 0.0
            )
        with self._monitor_lock:
            try:
                process_cpu = float(self._process.cpu_percent(interval=None))
                rss_bytes = int(self._process.memory_info().rss)
                system_cpu = float(psutil.cpu_percent(interval=None))
                system_memory = float(psutil.virtual_memory().percent)
            except (psutil.Error, OSError, AttributeError):
                process_cpu = 0.0
                rss_bytes = 0
                system_cpu = 0.0
                system_memory = 0.0
        return ComputeSnapshot(
            active_calculations=active,
            queued_calculations=queued,
            max_concurrent_calculations=self.policy.max_concurrent_calculations,
            completed_calculations=completed,
            busy_rejections=busy_rejections,
            rate_rejections=rate_rejections,
            timed_out_calculations=timed_out,
            cancelled_calculations=cancelled,
            average_duration_seconds=float(average_duration),
            process_cpu_percent=process_cpu,
            system_cpu_percent=system_cpu,
            process_rss_bytes=rss_bytes,
            system_memory_percent=system_memory,
        )

    def log_health_if_due(self) -> ComputeSnapshot:
        snapshot = self.snapshot()
        now = self._clock()
        should_log = False
        with self._lock:
            if (
                now - self._last_health_log_at
                >= self.policy.health_log_interval_seconds
            ):
                self._last_health_log_at = now
                should_log = True
        if should_log:
            LOGGER.info(
                "compute_health %s",
                json.dumps(asdict(snapshot), separators=(",", ":")),
            )
        return snapshot


def check_current_computation() -> None:
    """Check the lease bound to this worker thread, if one exists."""

    lease = getattr(_CURRENT_LEASE, "value", None)
    if lease is not None:
        lease.check()


_GOVERNORS: dict[ResourcePolicy, ComputeGovernor] = {}
_GOVERNORS_LOCK = threading.Lock()


def get_compute_governor(policy: ResourcePolicy) -> ComputeGovernor:
    """Return the process-wide governor associated with one immutable policy."""

    with _GOVERNORS_LOCK:
        governor = _GOVERNORS.get(policy)
        if governor is None:
            governor = ComputeGovernor(policy)
            _GOVERNORS[policy] = governor
        return governor
