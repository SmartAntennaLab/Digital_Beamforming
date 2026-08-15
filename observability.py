"""Low-cardinality Prometheus metrics and optional OpenTelemetry tracing."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator

_LOCK = threading.Lock()
_INITIALIZED = False
_TRACER = None
_TRACER_PROVIDER = None
_METER_PROVIDER = None
_OTEL_REQUESTS = None
_OTEL_DURATION = None

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ImportError:  # Runtime-only installs intentionally keep observability optional.
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    Counter = Gauge = Histogram = None
    generate_latest = None


def _metric(factory, *args, **kwargs):
    if factory is None:
        return None
    return factory(*args, **kwargs)


COMPUTE_REQUESTS = _metric(
    Counter,
    "dbf_compute_requests_total",
    "Completed calculation attempts by bounded result category.",
    ("view", "geometry", "backend", "outcome"),
)
COMPUTE_DURATION = _metric(
    Histogram,
    "dbf_compute_duration_seconds",
    "End-to-end calculation duration including queue and cache lookup.",
    ("view", "geometry", "backend"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
LOCAL_ACTIVE = _metric(
    Gauge,
    "dbf_compute_active",
    "Calculations active in this Streamlit process.",
)
LOCAL_QUEUED = _metric(
    Gauge,
    "dbf_compute_queued",
    "Calculations queued in this Streamlit process.",
)
GLOBAL_ACTIVE = _metric(
    Gauge,
    "dbf_compute_global_active",
    "Calculations admitted globally through the shared coordinator.",
)
GLOBAL_LIMIT = _metric(
    Gauge,
    "dbf_compute_global_limit",
    "Configured global calculation concurrency limit.",
)
COORDINATOR_AVAILABLE = _metric(
    Gauge,
    "dbf_compute_coordinator_available",
    "Whether the configured global coordinator is reachable.",
    ("backend",),
)
WORKER_INFLIGHT = _metric(
    Gauge,
    "dbf_compute_worker_inflight",
    "View tasks currently submitted to this process execution backend.",
    ("backend",),
)
PROCESS_RSS = _metric(
    Gauge,
    "dbf_compute_process_rss_bytes",
    "Resident memory of the Streamlit process.",
)
REJECTIONS = _metric(
    Gauge,
    "dbf_compute_rejections_total_snapshot",
    "Current process rejection counters by reason.",
    ("reason",),
)


def configure_open_telemetry() -> bool:
    """Enable OTLP/HTTP traces and metrics when endpoints are configured."""

    global _INITIALIZED, _METER_PROVIDER, _OTEL_DURATION, _OTEL_REQUESTS
    global _TRACER, _TRACER_PROVIDER
    with _LOCK:
        if _INITIALIZED:
            return _TRACER is not None or _OTEL_REQUESTS is not None
        _INITIALIZED = True
        traces_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""
        ).strip()
        metrics_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", ""
        ).strip()
        if not traces_endpoint and not metrics_endpoint:
            return False
        try:
            from opentelemetry.sdk.resources import Resource
        except ImportError as error:
            raise RuntimeError(
                "OpenTelemetry export requires the 'ops' dependency extra."
            ) from error

        resource = Resource.create(
            {
                "service.name": os.getenv(
                    "OTEL_SERVICE_NAME",
                    "digital-beamforming",
                ),
                "service.version": "1.7.0",
            }
        )
        if traces_endpoint:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            trace_provider = TracerProvider(resource=resource)
            trace_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_endpoint))
            )
            trace.set_tracer_provider(trace_provider)
            _TRACER_PROVIDER = trace_provider
            _TRACER = trace.get_tracer("digital_beamforming.compute")
        if metrics_endpoint:
            from opentelemetry import metrics
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=metrics_endpoint)
            )
            meter_provider = MeterProvider(
                metric_readers=[reader],
                resource=resource,
            )
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter("digital_beamforming.compute")
            _METER_PROVIDER = meter_provider
            _OTEL_REQUESTS = meter.create_counter(
                "dbf.compute.requests",
                description="Calculation attempts by bounded result category.",
            )
            _OTEL_DURATION = meter.create_histogram(
                "dbf.compute.duration",
                unit="s",
                description="End-to-end calculation duration.",
            )
        return True


def shutdown_open_telemetry() -> None:
    meter_provider = _METER_PROVIDER
    if meter_provider is not None:
        meter_provider.shutdown()
    provider = _TRACER_PROVIDER
    if provider is not None:
        provider.shutdown()


def _task_dimensions(task_label: str) -> tuple[str, str]:
    view, separator, geometry = task_label.partition(":")
    if not separator:
        geometry = "unknown"
    safe_view = view if view in {"pattern", "metrics", "elements"} else "unknown"
    safe_geometry = (
        geometry if geometry in {"UPA", "UHA", "ULA", "UCA"} else "unknown"
    )
    return safe_view, safe_geometry


def _outcome(error: BaseException | None) -> str:
    if error is None:
        return "success"
    name = type(error).__name__
    if name == "SessionRateLimitError":
        return "rate_limited"
    if name in {"ComputeBusyError", "ComputeCoordinationError"}:
        return "busy"
    if name == "ComputeDeadlineExceeded":
        return "deadline"
    if name == "ComputeCancelled":
        return "cancelled"
    return "error"


@contextlib.contextmanager
def observe_calculation(task_label: str, backend: str) -> Iterator[None]:
    """Measure one bounded calculation without high-cardinality labels."""

    view, geometry = _task_dimensions(task_label)
    started_at = time.perf_counter()
    error: BaseException | None = None
    span_context = (
        _TRACER.start_as_current_span(
            "dbf.compute",
            attributes={
                "dbf.view": view,
                "dbf.geometry": geometry,
                "dbf.backend": backend,
            },
        )
        if _TRACER is not None
        else contextlib.nullcontext()
    )
    try:
        with span_context:
            yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        duration = max(0.0, time.perf_counter() - started_at)
        if COMPUTE_REQUESTS is not None:
            COMPUTE_REQUESTS.labels(
                view=view,
                geometry=geometry,
                backend=backend,
                outcome=_outcome(error),
            ).inc()
        if COMPUTE_DURATION is not None:
            COMPUTE_DURATION.labels(
                view=view,
                geometry=geometry,
                backend=backend,
            ).observe(duration)
        attributes = {
            "dbf.view": view,
            "dbf.geometry": geometry,
            "dbf.backend": backend,
            "dbf.outcome": _outcome(error),
        }
        if _OTEL_REQUESTS is not None:
            _OTEL_REQUESTS.add(1, attributes)
        if _OTEL_DURATION is not None:
            _OTEL_DURATION.record(duration, attributes)


def record_runtime_snapshot(compute_snapshot, executor_snapshot) -> None:
    """Update gauges from existing lightweight health snapshots."""

    if LOCAL_ACTIVE is None:
        return
    LOCAL_ACTIVE.set(compute_snapshot.active_calculations)
    LOCAL_QUEUED.set(compute_snapshot.queued_calculations)
    GLOBAL_ACTIVE.set(compute_snapshot.global_active_calculations)
    GLOBAL_LIMIT.set(compute_snapshot.global_max_concurrent_calculations)
    COORDINATOR_AVAILABLE.labels(
        backend=compute_snapshot.coordination_backend
    ).set(1 if compute_snapshot.global_coordination_available else 0)
    WORKER_INFLIGHT.labels(backend=executor_snapshot.mode).set(
        executor_snapshot.inflight_tasks
    )
    PROCESS_RSS.set(compute_snapshot.process_rss_bytes)
    REJECTIONS.labels(reason="busy").set(compute_snapshot.busy_rejections)
    REJECTIONS.labels(reason="rate").set(compute_snapshot.rate_rejections)
    REJECTIONS.labels(reason="deadline").set(
        compute_snapshot.timed_out_calculations
    )
    REJECTIONS.labels(reason="cancelled").set(
        compute_snapshot.cancelled_calculations
    )


def prometheus_payload() -> tuple[bytes, str] | None:
    if generate_latest is None:
        return None
    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "configure_open_telemetry",
    "observe_calculation",
    "prometheus_payload",
    "record_runtime_snapshot",
    "shutdown_open_telemetry",
]
