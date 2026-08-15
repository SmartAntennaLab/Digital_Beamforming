import unittest
from pathlib import Path

import interferer_sampling
import pattern_sampling
import settings_panel
import settings_storage
import simulation
import ui_elements
import ui_metrics
import ui_pattern
import ui_renderers
import ui_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModuleBoundaryTests(unittest.TestCase):
    def test_compatibility_facades_reexport_split_implementations(self):
        self.assertIs(
            simulation.calculate_pattern_cuts,
            pattern_sampling.calculate_pattern_cuts,
        )
        self.assertIs(
            simulation.calculate_surface_pattern,
            pattern_sampling.calculate_surface_pattern,
        )
        self.assertIs(
            simulation.calculate_interferer_great_circle_cuts,
            interferer_sampling.calculate_interferer_great_circle_cuts,
        )
        self.assertIs(ui_renderers.render_pattern_tab, ui_pattern.render_pattern_tab)
        self.assertIs(ui_renderers.render_metrics_tab, ui_metrics.render_metrics_tab)
        self.assertIs(ui_renderers.render_elements_tab, ui_elements.render_elements_tab)
        self.assertIs(
            settings_panel.request_device_settings_save,
            settings_storage.request_device_settings_save,
        )

    def test_original_large_modules_are_thin_orchestration_layers(self):
        limits = {
            "ui_renderers.py": 120,
            "settings_panel.py": 300,
            "settings_sections.py": 600,
            "simulation.py": 500,
        }
        for filename, maximum_lines in limits.items():
            with self.subTest(filename=filename):
                line_count = len(
                    (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
                )
                self.assertLess(line_count, maximum_lines)

    def test_result_summary_is_a_dedicated_ui_module(self):
        self.assertTrue(callable(ui_summary.render_calculation_summary))

    def test_settings_sections_follow_the_user_workflow(self):
        source = (PROJECT_ROOT / "settings_sections.py").read_text(encoding="utf-8")
        section_labels = (
            "1. 기본 설정",
            "2. 조향 설정",
            "3. Null 설정",
            "4. 하드웨어 현실성",
            "5. 시각화 설정",
            "6. 고급 계산·스캔 설정",
        )
        for label in section_labels:
            with self.subTest(label=label):
                self.assertIn(label, source)

    def test_numerical_sampling_module_has_no_streamlit_dependency(self):
        for filename in ("pattern_sampling.py", "interferer_sampling.py"):
            with self.subTest(filename=filename):
                source = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
                self.assertNotIn("import streamlit", source)


if __name__ == "__main__":
    unittest.main()
