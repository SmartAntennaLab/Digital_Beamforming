import json
import unittest

import numpy as np

from element_pattern_data import parse_element_pattern_csv
from golden_validation import parse_golden_dataset, validate_golden_dataset
from signal_processing import calculate_advanced_analysis
from simulation import (
    SimulationConfig,
    build_simulation_state,
    calculate_state_directivity,
)


def _isotropic_pattern_csv() -> bytes:
    lines = ["azimuth_deg,elevation_deg,copol_gain_db,crosspol_gain_db"]
    for elevation in (-90, 0, 90):
        for azimuth in (-180, 0, 180):
            lines.append(f"{azimuth},{elevation},0,-40")
    return ("\n".join(lines) + "\n").encode()


class AdvancedModelTests(unittest.TestCase):
    def test_hardware_errors_are_repeatable_and_seeded(self):
        config = SimulationConfig(
            vertical_count=4,
            horizontal_count=4,
            random_seed=1234,
            position_error_rms_wavelength=0.02,
            amplitude_error_rms_db=0.5,
            phase_error_rms_deg=4.0,
            mutual_coupling_db=-30.0,
        )
        first = build_simulation_state(config)
        second = build_simulation_state(config)
        np.testing.assert_allclose(first.coordinates.y, second.coordinates.y)
        np.testing.assert_allclose(first.complex_weights, second.complex_weights)
        self.assertFalse(np.allclose(first.coordinates.y, first.nominal_coordinates.y))
        self.assertGreater(first.hardware_diagnostics.coupled_neighbor_links, 0)

    def test_uploaded_pattern_and_polarization_are_used_for_directivity(self):
        grid = parse_element_pattern_csv(_isotropic_pattern_csv())
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=1,
                horizontal_count=1,
                element_pattern_grid=grid,
                polarization_angle_deg=20.0,
            )
        )
        result = calculate_state_directivity(state)
        self.assertTrue(result.is_approximate)
        self.assertIn("measured-pattern", result.integration_method)
        self.assertAlmostEqual(result.directivity_dbi, 0.0, places=2)

    def test_wideband_channel_mvdr_and_music_are_finite(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=6,
                horizontal_count=6,
                target_azimuth_deg=25.0,
                wideband_bandwidth_percent=12.0,
                enable_channel_analysis=True,
                channel_snapshots=128,
                multipath_count=1,
                adaptive_beamforming_method="mvdr",
                enable_doa_estimation=True,
            )
        )
        analysis = calculate_advanced_analysis(state)
        self.assertIsNotNone(analysis.wideband)
        self.assertGreater(analysis.wideband.maximum_squint_deg, 0.0)
        self.assertTrue(np.isfinite(analysis.channel.conventional_sinr_db))
        self.assertTrue(np.isfinite(analysis.adaptive.output_sinr_db))
        self.assertGreaterEqual(len(analysis.doa.peaks), 1)

    def test_near_field_focus_is_coherent_and_exclusive_with_nulls(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=8,
                horizontal_count=8,
                target_azimuth_deg=20.0,
                near_field_focus_range_m=0.5,
            )
        )
        near = calculate_advanced_analysis(state).near_field
        self.assertGreater(near.focus_coherence_db, -0.1)
        with self.assertRaises(ValueError):
            build_simulation_state(
                SimulationConfig(
                    enable_null_steering=True, near_field_focus_range_m=1.0
                )
            )

    def test_matlab_golden_dataset_cross_validation_passes(self):
        payload = json.dumps(
            {
                "schema_version": 1,
                "name": "single-element MATLAB reference",
                "source": "matlab",
                "normalization": "peak_db",
                "tolerance_db": 1e-9,
                "samples": [
                    {"azimuth_deg": -60, "elevation_deg": 0, "gain_db": 0},
                    {"azimuth_deg": 0, "elevation_deg": 0, "gain_db": 0},
                    {"azimuth_deg": 60, "elevation_deg": 0, "gain_db": 0},
                ],
            }
        ).encode()
        dataset = parse_golden_dataset(payload, filename="reference.json")
        state = build_simulation_state(
            SimulationConfig(vertical_count=1, horizontal_count=1)
        )
        validation = validate_golden_dataset(state, dataset)
        self.assertTrue(validation.passed)
        self.assertAlmostEqual(validation.maximum_error_db, 0.0)


if __name__ == "__main__":
    unittest.main()
