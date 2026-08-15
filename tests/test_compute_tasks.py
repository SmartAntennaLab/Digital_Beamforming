import os
import unittest

from compute_executor import ComputeExecutor
from compute_tasks import ViewComputeRequest, calculate_view
from simulation import SimulationConfig


class ComputeTaskTests(unittest.TestCase):
    def test_pattern_bundle_contains_surface_and_cut_results(self):
        result = calculate_view(
            ViewComputeRequest(
                SimulationConfig(vertical_count=4, horizontal_count=4),
                5.0,
                2.0,
                "pattern",
            )
        )

        self.assertIsNotNone(result.cuts)
        self.assertIsNotNone(result.great_circle_cuts)
        self.assertIsNotNone(result.surface_sampling)
        self.assertIsNotNone(result.surface)
        self.assertIsNone(result.directivity)

    def test_metrics_bundle_covers_multiple_null_constraints(self):
        config = SimulationConfig(
            vertical_count=4,
            horizontal_count=4,
            enable_null_steering=True,
            null_constraints_deg=((25.0, 5.0, 30.0), (-30.0, -5.0, 35.0)),
        )
        result = calculate_view(ViewComputeRequest(config, 0.0, 0.0, "metrics"))

        self.assertIsNotNone(result.directivity)
        self.assertEqual(len(result.interferer_comparisons), 2)
        self.assertEqual(result.interferer_great_circle_cuts, ())

    def test_inline_executor_runs_serializable_bundle(self):
        executor = ComputeExecutor(mode="inline")
        result = executor.execute(
            ViewComputeRequest(SimulationConfig(), 0.0, 0.0, "elements"),
            session_id="session",
            timeout_seconds=5.0,
            cancel_check=lambda: None,
        )

        self.assertEqual(result.state.coordinates.element_count, 16)
        self.assertIsNone(result.cuts)

    @unittest.skipUnless(
        os.getenv("RUN_PROCESS_POOL_INTEGRATION") == "1",
        "Set RUN_PROCESS_POOL_INTEGRATION=1 for the spawn integration test.",
    )
    def test_process_executor_runs_in_spawned_worker(self):
        executor = ComputeExecutor(
            mode="process",
            worker_count=1,
            max_tasks_per_child=2,
        )
        try:
            result = executor.execute(
                ViewComputeRequest(SimulationConfig(), 0.0, 0.0, "elements"),
                session_id="process-session",
                timeout_seconds=15.0,
                cancel_check=lambda: None,
            )
        finally:
            executor.shutdown()

        self.assertEqual(result.state.coordinates.element_count, 16)


if __name__ == "__main__":
    unittest.main()
