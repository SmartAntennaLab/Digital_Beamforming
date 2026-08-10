import unittest

import numpy as np

from simulation import (
    LIGHT_SPEED_M_S,
    SURFACE_PATTERN_SCHEMA_VERSION,
    SimulationConfig,
    build_simulation_state,
    calculate_great_circle_cuts,
    calculate_pattern_cuts,
    calculate_surface_pattern,
    estimate_scan_timing,
    pattern_cut_local_sample_count,
    scan_direction,
    scan_surface_sampling,
    summarize_array_layout,
    surface_local_sample_count,
    surface_resolution,
)


class SimulationStateTests(unittest.TestCase):
    def test_wavelength_uses_the_exact_si_speed_of_light(self):
        state = build_simulation_state(SimulationConfig(frequency_ghz=28.0))

        self.assertEqual(LIGHT_SPEED_M_S, 299_792_458.0)
        self.assertEqual(
            state.wavelength_m,
            299_792_458.0 / (28.0 * 1.0e9),
        )

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

    def test_state_passes_multiple_practical_null_constraints_to_solver(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=1,
                horizontal_count=16,
                geometry="ULA",
                enable_null_steering=True,
                null_constraints_deg=(
                    (-25.0, 0.0, 25.0),
                    (32.0, 0.0, 25.0),
                ),
                null_optimization_mode="phase_only",
                maximum_element_amplitude=1.0,
            )
        )

        result = state.weight_result
        self.assertEqual(len(result.null_directions_rad), 2)
        self.assertEqual(result.null_required_suppression_db, (25.0, 25.0))
        self.assertEqual(result.optimization_mode, "phase_only")
        self.assertEqual(result.maximum_element_amplitude, 1.0)
        self.assertEqual(result.null_requirement_met, (True, True))

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
        wavelength_cm = 299_792_458.0 / (4.0 * 1.0e9) * 100.0

        self.assertAlmostEqual(summary.horizontal_spacing_wavelength, 0.5)
        self.assertAlmostEqual(summary.horizontal_spacing_cm, 0.5 * wavelength_cm)
        self.assertAlmostEqual(summary.vertical_spacing_wavelength, 0.6)
        self.assertAlmostEqual(summary.vertical_spacing_cm, 0.6 * wavelength_cm)
        self.assertAlmostEqual(summary.horizontal_extent_wavelength, 15.5)
        self.assertAlmostEqual(
            summary.horizontal_extent_cm,
            15.5 * wavelength_cm,
        )
        self.assertAlmostEqual(summary.vertical_extent_wavelength, 9.0)
        self.assertAlmostEqual(summary.vertical_extent_cm, 9.0 * wavelength_cm)
        self.assertEqual(summary.total_elements, 512)
        self.assertEqual(summary.active_elements, 384)
        self.assertEqual(summary.failed_elements, 128)
        self.assertEqual(summary.requested_failure_rate_percent, 25.0)
        self.assertEqual(summary.actual_failure_rate_percent, 25.0)

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
        cuts = calculate_pattern_cuts(state, sample_count=37, max_chunk_entries=11)
        great_circle_cuts = calculate_great_circle_cuts(
            state, sample_count=73, max_chunk_entries=11
        )
        surface = calculate_surface_pattern(state, resolution=9, max_chunk_entries=13)

        self.assertGreater(cuts.azimuth_pattern.size, 37)
        self.assertGreater(cuts.elevation_pattern.size, 37)
        self.assertEqual(
            cuts.azimuth_pattern.shape,
            cuts.azimuth_angles_rad.shape,
        )
        self.assertEqual(
            cuts.elevation_pattern.shape,
            cuts.elevation_angles_rad.shape,
        )
        self.assertGreater(surface.pattern.shape[0], 9)
        self.assertGreater(surface.pattern.shape[1], 9)
        self.assertEqual(surface.schema_version, SURFACE_PATTERN_SCHEMA_VERSION)
        self.assertTrue(np.all(np.isfinite(cuts.azimuth_pattern_db)))
        self.assertTrue(np.all(np.isfinite(great_circle_cuts.horizontal_pattern_db)))
        self.assertTrue(
            np.any(np.isclose(great_circle_cuts.horizontal_offsets_rad, 0.0))
        )
        self.assertTrue(np.all(np.isfinite(surface.pattern_db)))

    def test_broadside_great_circle_and_coordinate_hpbw_match(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=8,
                horizontal_count=16,
                geometry="UPA",
            )
        )
        coordinate = calculate_pattern_cuts(state, sample_count=721)
        physical = calculate_great_circle_cuts(state, sample_count=721)

        self.assertAlmostEqual(
            coordinate.azimuth_metrics.hpbw_deg,
            physical.horizontal_metrics.hpbw_deg,
            delta=0.02,
        )
        self.assertAlmostEqual(
            coordinate.elevation_metrics.hpbw_deg,
            physical.vertical_metrics.hpbw_deg,
            delta=0.02,
        )

    def test_large_elevation_distinguishes_coordinate_and_physical_hpbw(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=1,
                horizontal_count=32,
                geometry="ULA",
                target_elevation_deg=60.0,
            )
        )
        coordinate = calculate_pattern_cuts(state, sample_count=721)
        physical = calculate_great_circle_cuts(state, sample_count=721)

        self.assertGreater(
            coordinate.azimuth_metrics.hpbw_deg,
            1.8 * physical.horizontal_metrics.hpbw_deg,
        )

    def test_pattern_cuts_include_exact_target_and_refine_large_array_beam(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=64,
                horizontal_count=128,
                target_azimuth_deg=23.0,
                target_elevation_deg=-11.0,
            )
        )
        cuts = calculate_pattern_cuts(
            state,
            sample_count=181,
            max_chunk_entries=4_096,
        )

        target_azimuth = np.radians(23.0)
        target_elevation = np.radians(-11.0)
        global_spacing = np.pi / 180.0
        azimuth_near_target = np.flatnonzero(
            np.abs(cuts.azimuth_angles_rad - target_azimuth)
            <= np.radians(cuts.azimuth_refinement_half_width_deg)
        )
        elevation_near_target = np.flatnonzero(
            np.abs(cuts.elevation_angles_rad - target_elevation)
            <= np.radians(cuts.elevation_refinement_half_width_deg)
        )

        self.assertTrue(np.any(np.isclose(cuts.azimuth_angles_rad, target_azimuth)))
        self.assertTrue(np.any(np.isclose(cuts.elevation_angles_rad, target_elevation)))
        self.assertEqual(cuts.base_sample_count, 181)
        self.assertEqual(cuts.local_sample_count, 129)
        self.assertLess(
            np.min(np.diff(cuts.azimuth_angles_rad[azimuth_near_target])),
            global_spacing / 4.0,
        )
        self.assertLess(
            np.min(np.diff(cuts.elevation_angles_rad[elevation_near_target])),
            global_spacing / 4.0,
        )

    def test_adaptive_cut_metrics_match_dense_128_element_reference(self):
        state = build_simulation_state(
            SimulationConfig(
                vertical_count=1,
                horizontal_count=128,
                geometry="ULA",
                target_azimuth_deg=17.0,
            )
        )
        adaptive = calculate_pattern_cuts(
            state,
            sample_count=181,
            local_sample_count=129,
            max_chunk_entries=4_096,
        )
        dense = calculate_pattern_cuts(
            state,
            sample_count=7_201,
            local_sample_count=3,
            max_chunk_entries=4_096,
        )

        self.assertAlmostEqual(
            adaptive.azimuth_metrics.hpbw_deg,
            dense.azimuth_metrics.hpbw_deg,
            delta=0.05,
        )
        self.assertAlmostEqual(
            adaptive.azimuth_metrics.first_null_beamwidth_deg,
            dense.azimuth_metrics.first_null_beamwidth_deg,
            delta=0.15,
        )
        self.assertAlmostEqual(
            adaptive.azimuth_metrics.sidelobe_level_db,
            dense.azimuth_metrics.sidelobe_level_db,
            delta=0.2,
        )

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
        self.assertTrue(np.any(np.isclose(surface.polar_angle_rad[0, :], target_polar)))
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
        self.assertAlmostEqual(surface.sampled_peak_magnitude, expected_peak, places=8)
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
        cuts = calculate_pattern_cuts(state, sample_count=181, max_chunk_entries=4_096)
        self.assertGreater(cuts.azimuth_pattern.size, 181)
        self.assertTrue(np.all(np.isfinite(cuts.azimuth_pattern)))


class ScanTests(unittest.TestCase):
    def test_scan_modes_select_preview_work_only_while_running(self):
        preview = scan_surface_sampling(4096, "preview_3d", scanning=True)
        full = scan_surface_sampling(4096, "full_3d", scanning=True)
        two_d = scan_surface_sampling(4096, "2d", scanning=True)
        stationary = scan_surface_sampling(4096, "2d", scanning=False)

        self.assertTrue(preview.render_3d)
        self.assertEqual(preview.quality, "preview")
        self.assertLess(preview.resolution, full.resolution)
        self.assertLess(preview.local_sample_count, full.local_sample_count)
        self.assertFalse(two_d.render_3d)
        self.assertIsNone(two_d.resolution)
        self.assertEqual(stationary, full)

    def test_scan_timing_is_calibrated_and_includes_final_full_frame(self):
        full = estimate_scan_timing(4096, 100, "full_3d", 0.1)
        preview = estimate_scan_timing(4096, 100, "preview_3d", 0.1)
        two_d = estimate_scan_timing(4096, 100, "2d", 0.1)

        self.assertAlmostEqual(full.frame_seconds, 0.85)
        self.assertEqual(full.finalization_seconds, 0.0)
        self.assertLess(preview.frame_seconds, full.frame_seconds)
        self.assertLess(two_d.frame_seconds, preview.frame_seconds)
        self.assertAlmostEqual(preview.finalization_seconds, 0.85)
        self.assertAlmostEqual(two_d.finalization_seconds, 0.85)
        self.assertLess(preview.total_seconds, full.total_seconds)

        rate_limited = estimate_scan_timing(
            4096,
            100,
            "preview_3d",
            0.1,
            session_calculations_per_minute=120,
            session_burst=8,
        )
        self.assertGreater(rate_limited.total_seconds, preview.total_seconds)

    def test_scan_mode_and_timing_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            scan_surface_sampling(16, "unknown", scanning=True)
        with self.assertRaises(ValueError):
            estimate_scan_timing(16, 0, "2d", 0.2)
        with self.assertRaises(ValueError):
            estimate_scan_timing(16, 2, "2d", float("nan"))

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

    def test_local_cut_detail_is_bounded_for_large_arrays(self):
        self.assertEqual(pattern_cut_local_sample_count(64), 65)
        self.assertEqual(pattern_cut_local_sample_count(65), 129)
        self.assertEqual(pattern_cut_local_sample_count(16384), 129)
        with self.assertRaises(ValueError):
            pattern_cut_local_sample_count(0)


if __name__ == "__main__":
    unittest.main()
