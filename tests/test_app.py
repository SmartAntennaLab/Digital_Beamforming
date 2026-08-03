from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from device_settings import decode_share_token, encode_share_token


APP_PATH = Path(__file__).resolve().parents[1] / "main.py"


class StreamlitPerformanceArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertEqual(list(self.app.exception), [])

    def test_dynamic_tabs_render_only_the_active_view(self):
        self.assertEqual(len(self.app.get("plotly_chart")), 3)
        self.assertEqual(len(self.app.metric), 0)

        self.app.session_state["active_result_tab"] = "🔍 성능 지표"
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(len(self.app.get("plotly_chart")), 0)
        self.assertEqual(len(self.app.metric), 10)
        self.assertEqual(len(self.app.download_button), 2)

        self.app.session_state["active_result_tab"] = "🔴 안테나 배치 및 위상"
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(len(self.app.get("plotly_chart")), 1)
        self.assertEqual(len(self.app.metric), 7)
        layout_metrics = {item.label: item.value for item in self.app.metric}
        self.assertEqual(layout_metrics["전체 소자 수"], "16개")
        self.assertEqual(layout_metrics["활성 소자 수"], "16개")
        self.assertEqual(layout_metrics["결함 소자 수"], "0개")
        self.assertEqual(
            layout_metrics["수평 소자 간격"],
            "0.500 λ / 0.536 cm",
        )
        self.assertEqual(
            layout_metrics["수직 소자 간격"],
            "0.500 λ / 0.536 cm",
        )
        self.assertEqual(
            layout_metrics["전체 수평 길이"],
            "1.500 λ / 1.607 cm",
        )
        self.assertEqual(
            layout_metrics["전체 수직 길이"],
            "1.500 λ / 1.607 cm",
        )

    def test_fragment_scan_advances_and_releases_timer_at_completion(self):
        self.app.session_state["active_result_tab"] = "🔴 안테나 배치 및 위상"
        self.app.session_state["is_scanning"] = True
        self.app.session_state["scan_idx"] = 0
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertTrue(self.app.session_state["is_scanning"])
        self.assertEqual(self.app.session_state["scan_idx"], 1)

        # Default scan settings are 10 azimuth by 5 elevation frames.
        self.app.session_state["scan_idx"] = 49
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertFalse(self.app.session_state["is_scanning"])
        self.assertEqual(len(self.app.success), 1)

    def test_vertical_controls_disable_immediately_for_ula_and_uca(self):
        def geometry_widget():
            return next(
                item
                for item in self.app.selectbox
                if item.label == "안테나 배열 형상"
            )

        constrained_labels = {
            "수직 안테나 수 (M)",
            "수직 소자 간격 (dz/λ)",
            "목표 Elevation 각도 (°)",
            "간섭 Elevation 각도 (°)",
        }

        def constrained_widgets():
            return {
                item.label: item
                for item in self.app.slider
                if item.label in constrained_labels
            }

        slider_labels = [item.label for item in self.app.slider]
        self.assertLess(
            slider_labels.index("수평 안테나 수 (N)"),
            slider_labels.index("수직 안테나 수 (M)"),
        )
        self.assertTrue(
            all(not item.disabled for item in constrained_widgets().values())
        )

        geometry_widget().select("ULA (수평 선형)")
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        ula_widgets = constrained_widgets()
        self.assertTrue(all(item.disabled for item in ula_widgets.values()))
        self.assertEqual(ula_widgets["수직 안테나 수 (M)"].value, 1)
        self.assertEqual(ula_widgets["목표 Elevation 각도 (°)"].value, 0.0)
        self.assertEqual(ula_widgets["간섭 Elevation 각도 (°)"].value, 0.0)

        geometry_widget().select("UCA (수평 원형)")
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        uca_widgets = constrained_widgets()
        self.assertTrue(all(item.disabled for item in uca_widgets.values()))
        self.assertEqual(uca_widgets["수직 안테나 수 (M)"].value, 1)
        self.assertEqual(uca_widgets["목표 Elevation 각도 (°)"].value, 0.0)
        self.assertEqual(uca_widgets["간섭 Elevation 각도 (°)"].value, 0.0)

        geometry_widget().select("UPA (사각형 평면형)")
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertTrue(
            all(not item.disabled for item in constrained_widgets().values())
        )

    def test_query_params_restore_settings_in_a_new_session(self):
        restored = AppTest.from_file(str(APP_PATH), default_timeout=30)
        restored.query_params.update(
            {
                "array_geometry": "UCA (수평 원형)",
                "frequency_ghz": "35.0",
                "vertical_count": "16",
                "horizontal_count": "32",
                "horizontal_spacing": "0.65",
                "target_elevation": "25.0",
                "null_elevation": "-15.0",
                "scale_option": "Linear Scale",
            }
        )
        restored.run()

        self.assertEqual(list(restored.exception), [])
        self.assertEqual(
            next(
                item.value
                for item in restored.selectbox
                if item.label == "안테나 배열 형상"
            ),
            "UCA (수평 원형)",
        )
        expected_sliders = {
            "주파수 (GHz)": 35.0,
            "수직 안테나 수 (M)": 1,
            "수평 안테나 수 (N)": 32,
            "수평 소자 간격 (dy/λ)": 0.65,
            "목표 Elevation 각도 (°)": 0.0,
            "간섭 Elevation 각도 (°)": 0.0,
        }
        actual_sliders = {
            item.label: item.value
            for item in restored.slider
            if item.label in expected_sliders
        }
        self.assertEqual(actual_sliders, expected_sliders)
        self.assertTrue(
            next(
                item.disabled
                for item in restored.slider
                if item.label == "수직 소자 간격 (dz/λ)"
            )
        )
        self.assertEqual(
            next(
                item.value
                for item in restored.radio
                if item.label == "3D 빔 패턴 스케일"
            ),
            "Linear Scale",
        )

    def test_uniform_hexagonal_array_exposes_mathworks_parameters(self):
        geometry_widget = next(
            item for item in self.app.selectbox if item.label == "안테나 배열 형상"
        )
        geometry_widget.select("UHA (균일 육각 평면형)")
        self.app.run()

        self.assertEqual(list(self.app.exception), [])
        uha_sliders = {
            item.label: item for item in self.app.slider
            if item.label in {"중앙 행 소자 수 (Nmax)", "최소 행 소자 수 (Nmin)"}
        }
        self.assertEqual(uha_sliders["중앙 행 소자 수 (Nmax)"].value, 4)
        self.assertEqual(uha_sliders["최소 행 소자 수 (Nmin)"].value, 2)
        row_spacing = next(
            item for item in self.app.number_input
            if item.label == "수직 행 간격 (dz/λ)"
        )
        self.assertTrue(row_spacing.disabled)
        self.assertAlmostEqual(row_spacing.value, 0.5 * (3.0**0.5) / 2.0)
        self.assertFalse(
            next(
                item.disabled for item in self.app.slider
                if item.label == "목표 Elevation 각도 (°)"
            )
        )

    def test_explicit_share_token_restores_without_widget_query_binding(self):
        restored = AppTest.from_file(str(APP_PATH), default_timeout=30)
        restored.query_params.update(
            {
                "settings": encode_share_token(
                    {
                        "array_geometry": "UHA (균일 육각 평면형)",
                        "frequency_ghz": 12.0,
                        "uha_max_count": 5,
                        "uha_min_count": 2,
                    }
                )
            }
        )
        restored.run()

        self.assertEqual(list(restored.exception), [])
        self.assertEqual(
            next(
                item.value for item in restored.selectbox
                if item.label == "안테나 배열 형상"
            ),
            "UHA (균일 육각 평면형)",
        )
        slider_values = {item.label: item.value for item in restored.slider}
        self.assertEqual(slider_values["주파수 (GHz)"], 12.0)
        self.assertEqual(slider_values["중앙 행 소자 수 (Nmax)"], 5)
        self.assertEqual(slider_values["최소 행 소자 수 (Nmin)"], 2)

    def test_device_settings_save_share_and_reset_workflow(self):
        frequency = next(
            item for item in self.app.slider if item.label == "주파수 (GHz)"
        )
        apply_button = next(
            item for item in self.app.button if item.label == "설정 적용 및 계산"
        )
        frequency.set_value(4.0)
        apply_button.click()
        self.app.run()

        self.assertEqual(list(self.app.exception), [])
        command = self.app.session_state["_device_storage_command"]
        self.assertEqual(command["action"], "save")
        self.assertEqual(command["payload"]["settings"]["frequency_ghz"], 4.0)
        self.assertNotIn("frequency_ghz", self.app.query_params)

        next(
            item for item in self.app.button if item.label == "공유 링크 생성"
        ).click()
        self.app.run()
        token = self.app.query_params["settings"]
        if isinstance(token, list):
            token = token[0]
        self.assertEqual(decode_share_token(token)["frequency_ghz"], 4.0)

        next(
            item for item in self.app.button if item.label == "저장 설정 초기화"
        ).click()
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(
            next(
                item.value for item in self.app.slider
                if item.label == "주파수 (GHz)"
            ),
            28.0,
        )
        self.assertEqual(dict(self.app.query_params), {})
        self.assertEqual(
            self.app.session_state["_device_storage_command"]["action"],
            "clear",
        )


if __name__ == "__main__":
    unittest.main()
