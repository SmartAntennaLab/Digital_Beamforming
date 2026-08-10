import unittest

from compute_governor import (
    ComputeBusyError,
    ComputeCancelled,
    ComputeDeadlineExceeded,
    ComputeGovernor,
    SessionRateLimitError,
    check_current_computation,
)
from resource_policy import ResourcePolicy


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class FakeMemoryInfo:
    rss = 256 * 1024**2


class FakeProcess:
    def cpu_percent(self, interval=None) -> float:
        return 125.0

    def memory_info(self) -> FakeMemoryInfo:
        return FakeMemoryInfo()


def policy(**overrides) -> ResourcePolicy:
    values = {
        "max_concurrent_calculations": 1,
        "compute_queue_timeout_seconds": 0.01,
        "compute_timeout_seconds": 1.0,
        "session_calculations_per_minute": 60,
        "session_burst": 2,
        "health_log_interval_seconds": 30.0,
    }
    values.update(overrides)
    return ResourcePolicy(**values)


class ComputeGovernorTests(unittest.TestCase):
    def test_invalid_governor_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            ComputeGovernor(
                policy(max_concurrent_calculations=0),
                process=FakeProcess(),
            )
        with self.assertRaises(ValueError):
            ComputeGovernor(
                policy(compute_timeout_seconds=0.0),
                process=FakeProcess(),
            )

    def test_process_semaphore_rejects_when_slot_stays_busy(self):
        governor = ComputeGovernor(policy(), process=FakeProcess())

        with governor.lease("session-a", "first"):
            with self.assertRaises(ComputeBusyError):
                with governor.lease("session-b", "second"):
                    self.fail("A second calculation must not be admitted.")

        snapshot = governor.snapshot()
        self.assertEqual(snapshot.completed_calculations, 1)
        self.assertEqual(snapshot.busy_rejections, 1)
        self.assertEqual(snapshot.active_calculations, 0)

    def test_session_token_bucket_refills_at_configured_rate(self):
        clock = FakeClock()
        governor = ComputeGovernor(policy(), clock=clock, process=FakeProcess())

        with governor.lease("same-session", "one"):
            pass
        with governor.lease("same-session", "two"):
            pass
        with self.assertRaises(SessionRateLimitError) as raised:
            with governor.lease("same-session", "three"):
                pass
        self.assertAlmostEqual(raised.exception.retry_after_seconds, 1.0)

        clock.value = 1.0
        with governor.lease("same-session", "after-refill"):
            pass
        self.assertEqual(governor.snapshot().rate_rejections, 1)

    def test_deadline_is_checked_inside_and_when_leaving_lease(self):
        clock = FakeClock()
        governor = ComputeGovernor(policy(), clock=clock, process=FakeProcess())

        with self.assertRaises(ComputeDeadlineExceeded):
            with governor.lease("session", "slow"):
                clock.value = 1.01
                check_current_computation()

        snapshot = governor.snapshot()
        self.assertEqual(snapshot.timed_out_calculations, 1)
        self.assertEqual(snapshot.active_calculations, 0)

    def test_cancel_invalidates_only_existing_session_generation(self):
        clock = FakeClock()
        governor = ComputeGovernor(policy(), clock=clock, process=FakeProcess())

        with self.assertRaises(ComputeCancelled):
            with governor.lease("session", "cancelled") as lease:
                governor.cancel_session("session")
                lease.check()

        with governor.lease("session", "new-generation"):
            pass
        self.assertEqual(governor.snapshot().cancelled_calculations, 1)

    def test_snapshot_reports_process_memory_and_cpu(self):
        governor = ComputeGovernor(policy(), process=FakeProcess())

        snapshot = governor.log_health_if_due()

        self.assertEqual(snapshot.process_cpu_percent, 125.0)
        self.assertEqual(snapshot.process_rss_bytes, 256 * 1024**2)
        self.assertGreaterEqual(snapshot.system_cpu_percent, 0.0)
        self.assertGreaterEqual(snapshot.system_memory_percent, 0.0)


if __name__ == "__main__":
    unittest.main()
