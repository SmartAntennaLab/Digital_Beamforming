import unittest
from pathlib import Path

from device_settings import (
    DEVICE_SETTINGS_SCHEMA_VERSION,
    collect_device_settings,
    decode_share_token,
    encode_share_token,
    sanitize_device_settings,
    settings_envelope,
)


class DeviceSettingsTests(unittest.TestCase):
    def test_storage_bridge_uses_component_v2_and_browser_local_storage(self):
        source = (Path(__file__).resolve().parents[1] / "device_storage.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("st.components.v2.component", source)
        self.assertIn("window.localStorage", source)
        self.assertNotIn("components.v1", source)

    def test_browser_payload_accepts_only_known_valid_values(self):
        sanitized = sanitize_device_settings(
            {
                "frequency_ghz": 35.0,
                "horizontal_count": 32,
                "enable_null": "true",
                "target_azimuth": 999.0,
                "unknown": "ignored",
            }
        )

        self.assertEqual(
            sanitized,
            {
                "frequency_ghz": 35.0,
                "horizontal_count": 32,
                "enable_null": True,
            },
        )

    def test_uha_minimum_is_clamped_to_restored_maximum(self):
        sanitized = sanitize_device_settings({"uha_min_count": 8, "uha_max_count": 4})
        self.assertEqual(sanitized["uha_min_count"], 4)
        self.assertEqual(sanitized["uha_max_count"], 4)

    def test_versioned_envelope_rejects_unknown_schema(self):
        valid = settings_envelope({"frequency_ghz": 12.0})
        self.assertEqual(valid["schema_version"], DEVICE_SETTINGS_SCHEMA_VERSION)
        self.assertEqual(sanitize_device_settings(valid)["frequency_ghz"], 12.0)
        self.assertEqual(
            sanitize_device_settings(
                {"schema_version": 999, "settings": {"frequency_ghz": 12.0}}
            ),
            {},
        )

    def test_share_token_round_trip_is_url_safe_and_validated(self):
        settings = {
            "array_geometry": "UHA",
            "uha_min_count": 2,
            "uha_max_count": 5,
            "scan_azimuth_range": (-30.0, 45.0),
            "scan_mode": "2d",
        }
        token = encode_share_token(settings)

        self.assertNotIn("=", token)
        self.assertEqual(decode_share_token(token), settings)
        self.assertEqual(decode_share_token("not-valid-설정"), {})

    def test_scan_mode_is_validated_and_schema_two_remains_readable(self):
        self.assertEqual(
            sanitize_device_settings({"scan_mode": "preview_3d"}),
            {"scan_mode": "preview_3d"},
        )
        self.assertEqual(sanitize_device_settings({"scan_mode": "invalid"}), {})
        self.assertEqual(
            sanitize_device_settings(
                {
                    "schema_version": 2,
                    "settings": {"scan_delay": 0.4},
                }
            ),
            {"scan_delay": 0.4},
        )

    def test_legacy_translated_options_migrate_to_stable_ids(self):
        sanitized = sanitize_device_settings(
            {
                "array_geometry": "UCA (수평 원형)",
                "taper_option": "Uniform (균일)",
                "element_option": "Cosine² (코사인 제곱)",
                "phase_bits": "Infinite (무한)",
                "scale_option": "Linear Scale",
                "coordinate_option": "Rectangular (직각좌표)",
            }
        )

        self.assertEqual(
            sanitized,
            {
                "array_geometry": "UCA",
                "taper_option": "uniform",
                "element_option": "cosine_squared",
                "phase_bits": None,
                "scale_option": "linear",
                "coordinate_option": "rectangular",
            },
        )

    def test_multiple_null_and_practical_constraint_settings_are_validated(self):
        settings = {
            "null_count": 2,
            "null_1_suppression_db": 35.0,
            "null_2_azimuth": -25.0,
            "null_2_elevation": 10.0,
            "null_2_suppression_db": 50.0,
            "null_optimization_mode": "phase_only",
            "enable_amplitude_limit": True,
            "max_element_amplitude": 0.8,
            "null_optimizer_tolerance": 1e-10,
            "null_optimizer_max_iterations": 800,
            "null_optimizer_restart_count": 6,
        }

        self.assertEqual(sanitize_device_settings(settings), settings)
        self.assertEqual(decode_share_token(encode_share_token(settings)), settings)
        self.assertEqual(sanitize_device_settings({"null_count": 9}), {})
        self.assertEqual(
            sanitize_device_settings({"null_optimization_mode": "unsupported"}),
            {},
        )

    def test_directivity_mode_is_persisted_and_validated(self):
        settings = {"directivity_mode": "fast"}

        self.assertEqual(sanitize_device_settings(settings), settings)
        self.assertEqual(decode_share_token(encode_share_token(settings)), settings)
        self.assertEqual(sanitize_device_settings({"directivity_mode": "invalid"}), {})

    def test_collect_ignores_transient_session_values(self):
        collected = collect_device_settings(
            {
                "frequency_ghz": 4.0,
                "is_scanning": True,
                "scan_idx": 12,
            }
        )
        self.assertEqual(collected, {"frequency_ghz": 4.0})


if __name__ == "__main__":
    unittest.main()
