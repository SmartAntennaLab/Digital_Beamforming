import os
import unittest
from unittest.mock import patch

from resource_policy import (
    HARD_MAX_COMPUTE_SECONDS,
    HARD_MAX_CONCURRENT_CALCULATIONS,
    HARD_MAX_DIRECTIVITY_EXACT_ELEMENTS,
    HARD_MAX_ELEMENTS,
    HARD_MAX_SESSION_CALCULATIONS_PER_MINUTE,
    ResourcePolicy,
    estimate_element_count,
    resource_limit_message,
)


class ResourcePolicyTests(unittest.TestCase):
    def test_geometry_aware_element_estimates(self):
        self.assertEqual(estimate_element_count("ULA", 1, 128), 128)
        self.assertEqual(estimate_element_count("UCA", 1, 128), 128)
        self.assertEqual(estimate_element_count("UPA", 16, 32), 512)
        self.assertEqual(estimate_element_count("UHA", 2, 4), 14)

    def test_limits_reject_elements_frames_and_combined_work(self):
        policy = ResourcePolicy(
            max_elements=512,
            max_scan_frames=100,
            max_scan_element_frames=10_000,
        )
        self.assertIn(
            "소자",
            resource_limit_message(
                policy,
                geometry="UPA",
                vertical_count=32,
                horizontal_count=32,
                scan_frames=1,
            ),
        )
        self.assertIn(
            "프레임",
            resource_limit_message(
                policy,
                geometry="ULA",
                vertical_count=1,
                horizontal_count=8,
                scan_frames=101,
            ),
        )
        self.assertIn(
            "element-frames",
            resource_limit_message(
                policy,
                geometry="UPA",
                vertical_count=16,
                horizontal_count=16,
                scan_frames=50,
            ),
        )

    def test_environment_overrides_cannot_exceed_hard_limit(self):
        with patch.dict(
            os.environ,
            {"DBF_MAX_ELEMENTS": str(HARD_MAX_ELEMENTS * 10)},
        ):
            policy = ResourcePolicy.from_environment()

        self.assertEqual(policy.max_elements, HARD_MAX_ELEMENTS)

    def test_compute_governor_environment_overrides_are_bounded(self):
        with patch.dict(
            os.environ,
            {
                "DBF_MAX_CONCURRENT_CALCULATIONS": str(
                    HARD_MAX_CONCURRENT_CALCULATIONS * 10
                ),
                "DBF_COMPUTE_TIMEOUT_SECONDS": str(HARD_MAX_COMPUTE_SECONDS * 10),
                "DBF_SESSION_CALCULATIONS_PER_MINUTE": str(
                    HARD_MAX_SESSION_CALCULATIONS_PER_MINUTE * 10
                ),
                "DBF_SESSION_BURST": "999",
            },
        ):
            policy = ResourcePolicy.from_environment()

        self.assertEqual(
            policy.max_concurrent_calculations,
            HARD_MAX_CONCURRENT_CALCULATIONS,
        )
        self.assertEqual(policy.compute_timeout_seconds, HARD_MAX_COMPUTE_SECONDS)
        self.assertEqual(
            policy.session_calculations_per_minute,
            HARD_MAX_SESSION_CALCULATIONS_PER_MINUTE,
        )
        self.assertLessEqual(
            policy.session_burst,
            policy.session_calculations_per_minute,
        )

    def test_directivity_thresholds_are_bounded_and_ordered(self):
        with patch.dict(
            os.environ,
            {
                "DBF_DIRECTIVITY_WARNING_ELEMENTS": "99999",
                "DBF_DIRECTIVITY_EXACT_MAX_ELEMENTS": "2048",
            },
        ):
            policy = ResourcePolicy.from_environment()

        self.assertEqual(policy.directivity_warning_elements, 2_048)
        self.assertEqual(policy.directivity_exact_max_elements, 2_048)
        self.assertLessEqual(
            policy.directivity_exact_max_elements,
            HARD_MAX_DIRECTIVITY_EXACT_ELEMENTS,
        )


if __name__ == "__main__":
    unittest.main()
