import unittest

from exporters import build_export_artifacts, build_design_report
from simulation import (
    SimulationConfig,
    build_simulation_state,
    calculate_great_circle_cuts,
    calculate_pattern_cuts,
    calculate_state_directivity,
)


class ExporterTests(unittest.TestCase):
    def test_csv_and_report_are_built_without_streamlit_ui(self):
        state = build_simulation_state(
            SimulationConfig(
                geometry="ULA",
                horizontal_count=10,
                vertical_count=1,
                failure_rate_percent=5.0,
            )
        )
        cuts = calculate_pattern_cuts(state)
        great_circle_cuts = calculate_great_circle_cuts(state)
        directivity = calculate_state_directivity(state)
        artifacts = build_export_artifacts(
            state,
            cuts,
            great_circle_cuts=great_circle_cuts,
            directivity=directivity,
        )

        csv_text = artifacts.pattern_csv.decode("utf-8")
        report = artifacts.design_report.decode("utf-8")
        self.assertIn("Azimuth Angle (deg)", csv_text)
        self.assertIn("Elevation Gain (dB)", csv_text)
        self.assertIn("Horizontal Great-circle Offset (deg)", csv_text)
        self.assertIn("요청 결함률: 5.00%", report)
        self.assertIn("실제 결함률: 10.00% (1 / 10개)", report)
        self.assertIn("목표 방향 Directivity:", report)
        self.assertIn("실제 각거리 수평 주평면 HPBW", report)

    def test_undetected_metrics_are_exported_as_na(self):
        state = build_simulation_state(
            SimulationConfig(
                geometry="ULA",
                horizontal_count=1,
                vertical_count=1,
            )
        )
        cuts = calculate_pattern_cuts(state)
        report = build_design_report(state, cuts)
        self.assertIn("Azimuth HPBW / FNBW / SLL: N/A / N/A / N/A", report)

    def test_report_includes_practical_null_constraint_status(self):
        state = build_simulation_state(
            SimulationConfig(
                geometry="ULA",
                horizontal_count=16,
                vertical_count=1,
                enable_null_steering=True,
                null_constraints_deg=(
                    (-25.0, 0.0, 40.0),
                    (32.0, 0.0, 40.0),
                ),
                maximum_element_amplitude=1.0,
            )
        )
        report = build_design_report(state, calculate_pattern_cuts(state))

        self.assertIn("최대 소자 진폭 제한: 1", report)
        self.assertIn("포화 소자:", report)
        self.assertIn("요구 억압 충족:", report)
        self.assertIn("요구 40.0 dB", report)
        self.assertIn("(미달)", report)


if __name__ == "__main__":
    unittest.main()
