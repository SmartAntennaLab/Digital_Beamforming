import unittest
from types import SimpleNamespace

from observability import (
    observe_calculation,
    prometheus_payload,
    record_runtime_snapshot,
)


class ObservabilityTests(unittest.TestCase):
    def test_metrics_are_optional_and_use_bounded_labels(self):
        compute = SimpleNamespace(
            active_calculations=1,
            queued_calculations=2,
            global_active_calculations=3,
            global_max_concurrent_calculations=4,
            coordination_backend="redis",
            global_coordination_available=True,
            process_rss_bytes=123_456,
            busy_rejections=4,
            rate_rejections=5,
            timed_out_calculations=6,
            cancelled_calculations=7,
        )
        executor = SimpleNamespace(mode="process", inflight_tasks=1)

        record_runtime_snapshot(compute, executor)
        with observe_calculation("pattern:UPA", "process"):
            pass

        payload = prometheus_payload()
        if payload is None:
            return
        body, content_type = payload
        metrics = body.decode("utf-8")
        self.assertIn("dbf_compute_requests_total", metrics)
        self.assertIn("dbf_compute_duration_seconds", metrics)
        self.assertIn("dbf_compute_coordinator_available", metrics)
        self.assertIn("dbf_compute_global_limit", metrics)
        self.assertIn("view=\"pattern\"", metrics)
        self.assertIn("geometry=\"UPA\"", metrics)
        self.assertNotIn("session", metrics)
        self.assertIn("text/plain", content_type)


if __name__ == "__main__":
    unittest.main()
