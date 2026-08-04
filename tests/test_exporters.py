import unittest

from exporters import build_export_artifacts, build_design_report
from simulation import (
    SimulationConfig,
    build_simulation_state,
    calculate_pattern_cuts,
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
        artifacts = build_export_artifacts(state, cuts)

        csv_text = artifacts.pattern_csv.decode("utf-8")
        report = artifacts.design_report.decode("utf-8")
        self.assertIn("Azimuth Angle (deg)", csv_text)
        self.assertIn("Elevation Gain (dB)", csv_text)
        self.assertIn("요청 결함률: 5.00%", report)
        self.assertIn("실제 결함률: 10.00% (1 / 10개)", report)

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


if __name__ == "__main__":
    unittest.main()
