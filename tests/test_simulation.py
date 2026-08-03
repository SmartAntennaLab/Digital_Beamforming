import unittest

import numpy as np

from simulation import (
    SURFACE_PATTERN_SCHEMA_VERSION,
    SimulationConfig,
    build_simulation_state,
    calculate_pattern_cuts,
    calculate_surface_pattern,
    scan_direction,
    summarize_array_layout,
    surface_local_sample_count,
    surface_resolution,
)


class SimulationStateTests(unittest.TestCase):
    def test_state_applies_independent_upa_spacings(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=3,
                horizontal_count=4,
                geometry="UPA",
                horizontal_spacing_wavelength=0.4,
                vertical_spacing_wavelength=0.7,
            )
        )

        np.testing.assert_allclose(
            np.diff(state.coordinates.y[0]) / state.wavelength_m,
            0.4,
        )
        np.testing.assert_allclose(
            np.diff(state.coordinates.z[:, 0]) / state.wavelength_m,
            0.7,
        )
        self.assertAlmostEqual(
            state.horizontal_spacing_m / state.wavelength_m,
            0.4,
        )
        self.assertAlmostEqual(
            state.vertical_spacing_m / state.wavelength_m,
            0.7,
        )

    def test_state_combines_geometry_failures_and_weights_deterministically(self):
        config = SimulationConfig(
            vertical_count=4,
            horizontal_count=8,
            failure_rate_percent=25.0,
            taper_option="Hamming",
            target_azimuth_deg=12.0,
            target_elevation_deg=-7.0,
        )
        first = build_simulation_state(config)
        second = build_simulation_state(config)

        self.assertEqual(first.coordinates.y.shape, (4, 8))
        self.assertEqual(first.gain_metrics.active_elements, 24)
        np.testing.assert_array_equal(first.active_mask, second.active_mask)
        np.testing.assert_allclose(first.complex_weights, second.complex_weights)

    def test_uha_state_counts_only_physical_elements(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=2,
                horizontal_count=4,
                geometry="UHA",
                horizontal_spacing_wavelength=0.5,
                failure_rate_percent=25.0,
            )
        )

        self.assertEqual(state.coordinates.row_lengths, (2, 3, 4, 3, 2))
        self.assertEqual(state.coordinates.element_count, 14)
        self.assertEqual(state.gain_metrics.total_elements, 14)
        self.assertEqual(state.gain_metrics.active_elements, 10)
        self.assertFalse(np.any(state.active_mask & ~state.coordinates.element_mask))
        self.assertTrue(
            np.all(state.complex_weights[~state.coordinates.element_mask] == 0.0)
        )
        self.assertAlmostEqual(
            state.vertical_spacing_m / state.wavelength_m,
            0.5 * np.sin(np.pi / 3.0),
        )

    def test_layout_summary_reports_spacing_aperture_and_counts(self):
        state = build_simulation_state(
            SimulationConfig(
                frequency_ghz=4.0,
                vertical_count=16,
                horizontal_count=32,
                geometry="UPA",
                horizontal_spacing_wavelength=0.5,
                vertical_spacing_wavelength=0.6,
                failure_rate_percent=25.0,
            )
        )
        summary = summarize_array_layout(state)

        self.assertAlmostEqual(summary.horizontal_spacing_wavelength, 0.5)
        self.assertAlmostEqual(summary.horizontal_spacing_cm, 3.75)
        self.assertAlmostEqual(summary.vertical_spacing_wavelength, 0.6)
        self.assertAlmostEqual(summary.vertical_spacing_cm, 4.5)
        self.assertAlmostEqual(summary.horizontal_extent_wavelength, 15.5)
        self.assertAlmostEqual(summary.horizontal_extent_cm, 116.25)
        self.assertAlmostEqual(summary.vertical_extent_wavelength, 9.0)
        self.assertAlmostEqual(summary.vertical_extent_cm, 67.5)
        self.assertEqual(summary.total_elements, 512)
        self.assertEqual(summary.active_elements, 384)
        self.assertEqual(summary.failed_elements, 128)

    def test_uha_layout_uses_derived_row_spacing_and_physical_extent(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=2,
                horizontal_count=4,
                geometry="UHA",
                horizontal_spacing_wavelength=0.5,
            )
        )
        summary = summarize_array_layout(state)

        self.assertAlmostEqual(
            summary.vertical_spacing_wavelength,
            0.5 * np.sin(np.pi / 3.0),
        )
        self.assertAlmostEqual(summary.horizontal_extent_wavelength, 1.5)
        self.assertAlmostEqual(
            summary.vertical_extent_wavelength,
            4.0 * 0.5 * np.sin(np.pi / 3.0),
        )
        self.assertEqual(summary.total_elements, 14)

    def test_ula_and_uca_do_not_report_an_independent_vertical_spacing(self):
        ula = summarize_array_layout(
            build_simulation_state(
                SimulationConfig(
                    vertical_count=1,
                    horizontal_count=8,
                    geometry="ULA",
                )
            )
        )
        uca = summarize_array_layout(
            build_simulation_state(
                SimulationConfig(
                    vertical_count=1,
                    horizontal_count=8,
                    geometry="UCA",
                )
            )
        )

        self.assertIsNone(ula.vertical_spacing_wavelength)
        self.assertIsNone(uca.vertical_spacing_wavelength)
        self.assertEqual(ula.vertical_extent_wavelength, 0.0)
        self.assertGreater(uca.vertical_extent_wavelength, 0.0)

    def test_pattern_views_are_finite_and_have_requested_shapes(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=3,
                horizontal_count=5,
                geometry="UPA",
                target_azimuth_deg=15.0,
                target_elevation_deg=5.0,
            )
        )
        cuts = calculate_pattern_cuts(
            state, sample_count=37, max_chunk_entries=11
        )
        surface = calculate_surface_pattern(
            state, resolution=9, max_chunk_entries=13
        )

        self.assertEqual(cuts.azimuth_pattern.shape, (37,))
        self.assertEqual(cuts.elevation_pattern.shape, (37,))
        self.assertGreater(surface.pattern.shape[0], 9)
        self.assertGreater(surface.pattern.shape[1], 9)
        self.assertEqual(surface.schema_version, SURFACE_PATTERN_SCHEMA_VERSION)
        self.assertTrue(np.all(np.isfinite(cuts.azimuth_pattern_db)))
        self.assertTrue(np.all(np.isfinite(surface.pattern_db)))

    def test_surface_grid_contains_exact_steering_direction(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=16,
                horizontal_count=16,
                target_azimuth_deg=23.0,
                target_elevation_deg=-11.0,
            )
        )
        surface = calculate_surface_pattern(state, resolution=20)
        target_azimuth = np.radians(23.0)
        target_polar = np.pi / 2.0 - np.radians(-11.0)

        self.assertTrue(
            np.any(np.isclose(surface.azimuth_angle_rad[:, 0], target_azimuth))
        )
        self.assertTrue(
            np.any(np.isclose(surface.polar_angle_rad[0, :], target_polar))
        )
        self.assertGreaterEqual(
            surface.sampled_peak_magnitude + 1e-10,
            surface.target_response_magnitude,
        )

    def test_large_surface_captures_broadside_peak_without_grid_loss(self):
        state = build_simulation_state(
            SimulationConfig(vertical_count=128, horizontal_count=128)
        )
        surface = calculate_surface_pattern(state)
        expected_peak = float(state.complex_weights.size)

        self.assertAlmostEqual(
            surface.target_response_magnitude, expected_peak, places=8
        )
        self.assertAlmostEqual(
            surface.sampled_peak_magnitude, expected_peak, places=8
        )
        self.assertEqual(float(np.max(surface.pattern_db)), 0.0)

    def test_32_by_16_boresight_surface_resolves_the_main_lobe(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=16,
                horizontal_count=32,
                geometry="UPA",
            )
        )
        surface = calculate_surface_pattern(state)
        azimuth = surface.azimuth_angle_rad[:, 0]
        elevation = np.pi / 2.0 - surface.polar_angle_rad[0, :]
        elevation_index = int(np.argmin(np.abs(elevation)))
        front_hemisphere = np.abs(azimuth) < np.pi / 2.0
        horizontal_cut_db = surface.pattern_db[front_hemisphere, elevation_index]

        # Plotly needs enough vertices inside the narrow lobe to avoid drawing
        # the 32-element horizontal beam as a flat triangular prism.
        self.assertGreaterEqual(np.count_nonzero(horizontal_cut_db >= -3.0), 9)

    def test_large_array_pattern_cut_uses_chunked_working_sets(self):
        state = build_simulation_state(
            SimulationConfig(vertical_count=64, horizontal_count=64)
        )
        cuts = calculate_pattern_cuts(
            state, sample_count=181, max_chunk_entries=4_096
        )
        self.assertEqual(cuts.azimuth_pattern.shape, (181,))
        self.assertTrue(np.all(np.isfinite(cuts.azimuth_pattern)))


class ScanTests(unittest.TestCase):
    def test_scan_direction_is_azimuth_first_raster(self):
        expected = (
            (-30.0, -10.0),
            (0.0, -10.0),
            (30.0, -10.0),
            (-30.0, 10.0),
            (0.0, 10.0),
            (30.0, 10.0),
        )
        actual = tuple(
            scan_direction(index, (-30.0, 30.0), (-10.0, 10.0), 3, 2)[:2]
            for index in range(6)
        )
        self.assertEqual(actual, expected)

    def test_scan_direction_validates_bounds(self):
        with self.assertRaises(ValueError):
            scan_direction(4, (-1.0, 1.0), (0.0, 0.0), 4, 1)
        with self.assertRaises(ValueError):
            scan_direction(0, (-1.0, 1.0), (0.0, 0.0), 0, 1)

    def test_surface_resolution_is_bounded_for_large_arrays(self):
        self.assertEqual(surface_resolution(256), 50)
        self.assertEqual(surface_resolution(257), 40)
        self.assertEqual(surface_resolution(1025), 30)
        self.assertEqual(surface_resolution(4097), 20)

    def test_local_surface_detail_balances_mesh_quality_and_work(self):
        self.assertEqual(surface_local_sample_count(64), 33)
        self.assertEqual(surface_local_sample_count(512), 65)
        self.assertEqual(surface_local_sample_count(2048), 49)
        self.assertEqual(surface_local_sample_count(16384), 33)


if __name__ == "__main__":
    unittest.main()
