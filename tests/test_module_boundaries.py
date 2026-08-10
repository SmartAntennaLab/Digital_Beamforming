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
            "settings_panel.py": 600,
            "simulation.py": 500,
        }
        for filename, maximum_lines in limits.items():
            with self.subTest(filename=filename):
                line_count = len(
                    (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
                )
                self.assertLess(line_count, maximum_lines)

    def test_numerical_sampling_module_has_no_streamlit_dependency(self):
        for filename in ("pattern_sampling.py", "interferer_sampling.py"):
            with self.subTest(filename=filename):
                source = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
                self.assertNotIn("import streamlit", source)


if __name__ == "__main__":
    unittest.main()
