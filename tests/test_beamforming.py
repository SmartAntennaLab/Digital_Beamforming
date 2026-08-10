import unittest
from unittest.mock import patch

import numpy as np

from beamforming import (
    array_factor,
    assess_grating_lobes,
    calculate_array_gain_metrics,
    calculate_pattern_metrics,
    compute_beamforming_weights,
    create_array_coordinates,
    create_array_taper,
    create_failure_mask,
    direction_cosines,
    element_pattern_factor,
    get_steering_limits,
    get_window_weights,
    great_circle_directions,
    normalize_pattern_db,
    normalize_pattern_linear,
    steering_vector,
)


class GreatCircleDirectionTests(unittest.TestCase):
    def test_offsets_are_exact_spherical_angular_distances(self):
        target_azimuth = np.radians(37.0)
        target_elevation = np.radians(58.0)
        offsets = np.radians(np.array([-70.0, -12.0, 0.0, 19.0, 80.0]))

        for plane in ("horizontal", "vertical"):
            azimuth, elevation = great_circle_directions(
                target_azimuth,
                target_elevation,
                offsets,
                plane=plane,
            )
            target = np.array(
                [
                    np.cos(target_elevation) * np.cos(target_azimuth),
                    np.cos(target_elevation) * np.sin(target_azimuth),
                    np.sin(target_elevation),
                ]
            )
            directions = np.column_stack(direction_cosines(azimuth, elevation))
            # atan2(||u x v||, u dot v) remains well-conditioned at zero
            # separation, unlike arccos(u dot v), whose derivative is singular
            # near one and can turn round-off into a spurious nonzero angle.
            cross_norms = np.linalg.norm(np.cross(directions, target), axis=1)
            dot_products = np.clip(directions @ target, -1.0, 1.0)
            distances = np.arctan2(cross_norms, dot_products)
            np.testing.assert_allclose(distances, np.abs(offsets), atol=1e-12)

    def test_zero_offset_returns_the_target_for_both_planes(self):
        for plane in ("horizontal", "vertical"):
            azimuth, elevation = great_circle_directions(
                np.radians(-23.0),
                np.radians(41.0),
                np.array([0.0]),
                plane=plane,
            )
            self.assertAlmostEqual(float(azimuth[0]), np.radians(-23.0))
            self.assertAlmostEqual(float(elevation[0]), np.radians(41.0))


class ArrayCoordinateTests(unittest.TestCase):
    def test_upa_coordinates_are_centered(self):
        coordinates = create_array_coordinates(2, 3, 0.5, "UPA (사각형 평면형)")
        self.assertEqual((coordinates.rows, coordinates.columns), (2, 3))
        self.assertEqual(coordinates.y.shape, (2, 3))
        np.testing.assert_allclose(coordinates.y.mean(), 0.0, atol=1e-15)
        np.testing.assert_allclose(coordinates.z.mean(), 0.0, atol=1e-15)
        np.testing.assert_allclose(coordinates.y[0], [-0.5, 0.0, 0.5])
        np.testing.assert_allclose(coordinates.z[:, 0], [-0.25, 0.25])

    def test_upa_uses_independent_horizontal_and_vertical_spacings(self):
        coordinates = create_array_coordinates(
            3,
            4,
            0.4,
            "UPA",
            vertical_spacing_m=0.7,
        )
        np.testing.assert_allclose(np.diff(coordinates.y[0]), 0.4)
        np.testing.assert_allclose(np.diff(coordinates.z[:, 0]), 0.7)

    def test_ula_forces_one_vertical_row(self):
        coordinates = create_array_coordinates(8, 4, 0.5, "ULA (수평 선형)")
        self.assertEqual((coordinates.rows, coordinates.columns), (1, 4))
        np.testing.assert_allclose(coordinates.z, 0.0)
        np.testing.assert_allclose(coordinates.y[0], [-0.75, -0.25, 0.25, 0.75])

    def test_uca_coordinates_use_adjacent_chord_spacing(self):
        coordinates = create_array_coordinates(8, 4, 0.5, "UCA (수평 원형)")
        expected_radius = 0.5 / (2.0 * np.sin(np.pi / 4.0))
        radius = np.sqrt(coordinates.y**2 + coordinates.z**2)
        self.assertEqual((coordinates.rows, coordinates.columns), (1, 4))
        np.testing.assert_allclose(radius, expected_radius)
        points = np.column_stack((coordinates.y.ravel(), coordinates.z.ravel()))
        adjacent_chords = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        np.testing.assert_allclose(adjacent_chords, 0.5)

    def test_uca_chord_spacing_is_exact_for_multiple_element_counts(self):
        for element_count in (2, 3, 8, 32):
            with self.subTest(element_count=element_count):
                coordinates = create_array_coordinates(1, element_count, 0.4, "UCA")
                points = np.column_stack((coordinates.y.ravel(), coordinates.z.ravel()))
                adjacent_chords = np.linalg.norm(
                    points - np.roll(points, -1, axis=0), axis=1
                )
                np.testing.assert_allclose(adjacent_chords, 0.4, atol=1e-12)

    def test_single_uca_element_is_at_origin(self):
        coordinates = create_array_coordinates(1, 1, 0.5, "UCA")
        np.testing.assert_allclose(coordinates.y, 0.0)
        np.testing.assert_allclose(coordinates.z, 0.0)

    def test_uha_matches_mathworks_uniform_hexagonal_row_construction(self):
        coordinates = create_array_coordinates(2, 4, 0.5, "UHA")

        self.assertEqual((coordinates.rows, coordinates.columns), (5, 4))
        self.assertEqual(coordinates.row_lengths, (2, 3, 4, 3, 2))
        self.assertEqual(coordinates.element_count, 14)
        np.testing.assert_allclose(
            coordinates.y[0, coordinates.element_mask[0]],
            [-0.25, 0.25],
        )
        np.testing.assert_allclose(
            coordinates.y[1, coordinates.element_mask[1]],
            [-0.5, 0.0, 0.5],
        )
        np.testing.assert_allclose(
            coordinates.y[2, coordinates.element_mask[2]],
            [-0.75, -0.25, 0.25, 0.75],
        )
        expected_z = np.arange(-2, 3) * 0.5 * np.sin(np.pi / 3.0)
        actual_z = np.array(
            [coordinates.z[row, coordinates.element_mask[row]][0] for row in range(5)]
        )
        np.testing.assert_allclose(actual_z, expected_z)

    def test_uha_nearest_neighbors_use_triangular_lattice_spacing(self):
        coordinates = create_array_coordinates(1, 3, 0.4, "UHA")
        points = np.column_stack(
            (
                coordinates.y[coordinates.element_mask],
                coordinates.z[coordinates.element_mask],
            )
        )
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        nearest = np.min(np.where(distances > 1e-12, distances, np.inf), axis=1)
        np.testing.assert_allclose(nearest, 0.4, atol=1e-12)

    def test_uha_rejects_nmin_greater_than_nmax(self):
        with self.assertRaisesRegex(ValueError, "Nmin"):
            create_array_coordinates(4, 3, 0.5, "UHA")


class GeometryConsistencyTests(unittest.TestCase):
    def test_steering_axes_follow_effective_geometry_dimensions(self):
        cases = (
            ("ULA", 4, 1, False, False),
            ("ULA", 4, 8, True, False),
            ("UPA", 1, 1, False, False),
            ("UPA", 1, 8, True, False),
            ("UPA", 8, 1, False, True),
            ("UPA", 4, 4, True, True),
            ("UCA", 1, 1, False, False),
            ("UCA", 1, 2, True, False),
            ("UCA", 1, 3, True, False),
            ("UHA", 1, 1, False, False),
            ("UHA", 1, 4, True, False),
            ("UHA", 5, 4, True, True),
        )
        for geometry, rows, columns, azimuth, elevation in cases:
            with self.subTest(geometry=geometry, rows=rows, columns=columns):
                limits = get_steering_limits(rows, columns, geometry)
                self.assertEqual(limits.azimuth_controllable, azimuth)
                self.assertEqual(limits.elevation_controllable, elevation)

    def test_direction_cosines_form_a_unit_vector(self):
        azimuth = np.radians(np.array([-70.0, 0.0, 35.0]))
        elevation = np.radians(np.array([-20.0, 10.0, 65.0]))
        u_x, u_y, u_z = direction_cosines(azimuth, elevation)
        np.testing.assert_allclose(u_x**2 + u_y**2 + u_z**2, 1.0, atol=1e-12)

    def test_element_patterns_use_one_direction_cosine_model(self):
        azimuth = np.radians(np.array([-60.0, 0.0, 60.0]))
        elevation = np.radians(np.array([-30.0, 0.0, 30.0]))
        theta = np.pi / 2.0 - elevation
        for option in (
            "Isotropic (등방성)",
            "Cosine (코사인)",
            "Cosine² (코사인 제곱)",
            "Dipole (다이폴)",
        ):
            with self.subTest(option=option):
                two_dimensional_cut = element_pattern_factor(option, azimuth, elevation)
                three_dimensional_slice = element_pattern_factor(
                    option, azimuth, np.pi / 2.0 - theta
                )
                np.testing.assert_allclose(
                    two_dimensional_cut,
                    three_dimensional_slice,
                    atol=1e-12,
                )

    def test_element_pattern_known_values_and_dipole_endpoints(self):
        self.assertAlmostEqual(element_pattern_factor("Cosine", 0.0, 0.0).item(), 1.0)
        self.assertAlmostEqual(element_pattern_factor("Cosine", np.pi, 0.0).item(), 0.0)
        self.assertAlmostEqual(
            element_pattern_factor("Cosine²", np.radians(60.0), 0.0).item(),
            0.25,
        )
        dipole_endpoints = element_pattern_factor(
            "Dipole",
            np.zeros(3),
            np.radians(np.array([-90.0, 0.0, 90.0])),
        )
        np.testing.assert_allclose(dipole_endpoints, [0.0, 1.0, 0.0], atol=1e-12)
        self.assertTrue(np.all(np.isfinite(dipole_endpoints)))

    def test_grating_lobe_conditions_are_geometry_specific(self):
        ula_safe = assess_grating_lobes(
            "ULA", 0.5, 0.0, 0.0, vertical_count=1, horizontal_count=8
        )
        ula_aliased = assess_grating_lobes(
            "ULA", 1.0, 0.0, 0.0, vertical_count=1, horizontal_count=8
        )
        self.assertFalse(ula_safe.has_aliasing_risk)
        self.assertTrue(ula_aliased.has_aliasing_risk)
        self.assertTrue(
            all(direction.order_z == 0 for direction in ula_aliased.directions)
        )

        upa_aliased = assess_grating_lobes(
            "UPA",
            0.75,
            np.radians(30.0),
            0.0,
            vertical_count=4,
            horizontal_count=4,
        )
        self.assertTrue(upa_aliased.has_aliasing_risk)
        self.assertFalse(upa_aliased.risk_only)
        self.assertTrue(
            any(direction.order_y == -1 for direction in upa_aliased.directions)
        )

        uca_safe = assess_grating_lobes(
            "UCA", 0.5, 0.0, 0.0, vertical_count=1, horizontal_count=8
        )
        uca_risk = assess_grating_lobes(
            "UCA", 0.55, 0.0, 0.0, vertical_count=1, horizontal_count=8
        )
        self.assertFalse(uca_safe.has_aliasing_risk)
        self.assertTrue(uca_risk.has_aliasing_risk)
        self.assertTrue(uca_risk.risk_only)
        self.assertEqual(uca_risk.directions, ())

        uha_safe = assess_grating_lobes(
            "UHA",
            0.5,
            0.0,
            0.0,
            vertical_count=5,
            horizontal_count=4,
            vertical_spacing_over_wavelength=0.5 * np.sin(np.pi / 3.0),
        )
        uha_aliased = assess_grating_lobes(
            "UHA",
            1.2,
            0.0,
            0.0,
            vertical_count=5,
            horizontal_count=4,
            vertical_spacing_over_wavelength=1.2 * np.sin(np.pi / 3.0),
        )
        self.assertFalse(uha_safe.has_aliasing_risk)
        self.assertTrue(uha_aliased.has_aliasing_risk)
        self.assertIn("triangular UHA", uha_aliased.criterion)

    def test_upa_grating_lobes_use_independent_axis_spacings(self):
        horizontal_alias = assess_grating_lobes(
            "UPA",
            1.0,
            0.0,
            0.0,
            vertical_count=4,
            horizontal_count=4,
            vertical_spacing_over_wavelength=0.5,
        )
        vertical_alias = assess_grating_lobes(
            "UPA",
            0.5,
            0.0,
            0.0,
            vertical_count=4,
            horizontal_count=4,
            vertical_spacing_over_wavelength=1.0,
        )

        self.assertTrue(horizontal_alias.has_aliasing_risk)
        self.assertTrue(all(item.order_z == 0 for item in horizontal_alias.directions))
        self.assertTrue(vertical_alias.has_aliasing_risk)
        self.assertTrue(all(item.order_y == 0 for item in vertical_alias.directions))


class WindowAndFailureTests(unittest.TestCase):
    def test_all_windows_are_finite_for_one_and_two_elements(self):
        options = ["Uniform (균일)", "Hamming", "Hanning", "Blackman", "Bartlett"]
        for length in (1, 2):
            for option in options:
                with self.subTest(length=length, option=option):
                    weights = get_window_weights(length, option)
                    self.assertEqual(weights.shape, (length,))
                    self.assertTrue(np.all(np.isfinite(weights)))
                    self.assertAlmostEqual(float(np.max(np.abs(weights))), 1.0)

    def test_failure_mask_is_deterministic_and_has_expected_count(self):
        first = create_failure_mask(4, 4, 25, seed=42)
        second = create_failure_mask(4, 4, 25, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(np.count_nonzero(~first)), 4)
        self.assertTrue(np.all(create_failure_mask(2, 2, 0)))
        self.assertFalse(np.any(create_failure_mask(2, 2, 100)))

    def test_failure_count_uses_explicit_round_half_up_policy(self):
        mask = create_failure_mask(1, 10, 5.0, seed=42)

        self.assertEqual(int(np.count_nonzero(~mask)), 1)

    def test_failure_mask_excludes_uha_padding_and_counts_only_real_elements(self):
        coordinates = create_array_coordinates(2, 4, 0.5, "UHA")
        active = create_failure_mask(
            coordinates.rows,
            coordinates.columns,
            25.0,
            seed=42,
            element_mask=coordinates.element_mask,
        )
        self.assertFalse(np.any(active & ~coordinates.element_mask))
        self.assertEqual(np.count_nonzero(active), 10)

    def test_uha_taper_is_symmetric_and_zero_on_padding(self):
        coordinates = create_array_coordinates(2, 4, 0.5, "UHA")
        taper = create_array_taper(coordinates, "Hamming")
        np.testing.assert_allclose(taper[0, :2], taper[-1, :2])
        np.testing.assert_allclose(taper[1, :3], taper[-2, :3])
        np.testing.assert_allclose(taper[2], taper[2, ::-1])
        self.assertTrue(np.all(taper[~coordinates.element_mask] == 0.0))


class SteeringAndArrayFactorTests(unittest.TestCase):
    def test_array_factor_checks_cooperative_cancellation_between_chunks(self):
        checks = 0

        def cancel_check():
            nonlocal checks
            checks += 1
            if checks >= 3:
                raise RuntimeError("cancelled")

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            array_factor(
                np.arange(4, dtype=float),
                np.zeros(4),
                np.ones(4, dtype=complex),
                1.0,
                np.linspace(-0.5, 0.5, 4),
                np.zeros(4),
                angle_chunk_size=1,
                element_chunk_size=1,
                cancel_check=cancel_check,
            )

    wavelength = 1.0

    def _uniform_weights(self, geometry, rows, columns, azimuth, elevation, bits=None):
        coordinates = create_array_coordinates(rows, columns, 0.5, geometry)
        amplitudes = coordinates.element_mask.astype(float)
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            self.wavelength,
            azimuth,
            elevation,
            amplitudes,
            phase_bits=bits,
        )
        return coordinates, result

    def test_uha_broadside_and_two_axis_steering_are_coherent(self):
        coordinates, broadside = self._uniform_weights("UHA", 2, 4, 0.0, 0.0)
        response = array_factor(
            coordinates.y,
            coordinates.z,
            broadside.weights,
            self.wavelength,
            0.0,
            0.0,
        )
        self.assertAlmostEqual(
            abs(response.item()), coordinates.element_count, places=10
        )

        target_azimuth = np.radians(21.0)
        target_elevation = np.radians(-13.0)
        _, steered = self._uniform_weights(
            "UHA", 2, 4, target_azimuth, target_elevation
        )
        response = array_factor(
            coordinates.y,
            coordinates.z,
            steered.weights,
            self.wavelength,
            target_azimuth,
            target_elevation,
        )
        self.assertAlmostEqual(
            abs(response.item()), coordinates.element_count, places=10
        )

    def test_broadside_steering_is_coherent_for_all_geometries(self):
        for geometry, rows, columns in (("ULA", 1, 8), ("UPA", 4, 4), ("UCA", 1, 8)):
            with self.subTest(geometry=geometry):
                coordinates, result = self._uniform_weights(
                    geometry, rows, columns, 0.0, 0.0
                )
                response = array_factor(
                    coordinates.y,
                    coordinates.z,
                    result.weights,
                    self.wavelength,
                    0.0,
                    0.0,
                )
                self.assertAlmostEqual(abs(response.item()), rows * columns, places=10)
                np.testing.assert_allclose(
                    steering_vector(
                        coordinates.y,
                        coordinates.z,
                        self.wavelength,
                        0.0,
                        0.0,
                    ),
                    1.0,
                )

    def test_endfire_steering_is_coherent_for_all_geometries(self):
        for geometry, rows, columns in (("ULA", 1, 8), ("UPA", 2, 4), ("UCA", 1, 8)):
            with self.subTest(geometry=geometry):
                coordinates, result = self._uniform_weights(
                    geometry, rows, columns, np.pi / 2.0, 0.0
                )
                endfire = array_factor(
                    coordinates.y,
                    coordinates.z,
                    result.weights,
                    self.wavelength,
                    np.pi / 2.0,
                    0.0,
                )
                self.assertAlmostEqual(
                    abs(endfire.item()), coordinates.y.size, places=10
                )

                if geometry == "ULA":
                    broadside = array_factor(
                        coordinates.y,
                        coordinates.z,
                        result.weights,
                        self.wavelength,
                        0.0,
                        0.0,
                    )
                    self.assertLess(abs(broadside.item()), 1e-10)

    def test_one_and_two_element_calculations_are_finite_for_all_geometries(self):
        for geometry in ("ULA", "UPA", "UCA"):
            for element_count in (1, 2):
                with self.subTest(geometry=geometry, element_count=element_count):
                    coordinates, result = self._uniform_weights(
                        geometry,
                        1,
                        element_count,
                        np.radians(15.0),
                        np.radians(5.0),
                    )
                    response = array_factor(
                        coordinates.y,
                        coordinates.z,
                        result.weights,
                        self.wavelength,
                        np.radians(15.0),
                        np.radians(5.0),
                    )
                    self.assertEqual(coordinates.y.size, element_count)
                    self.assertTrue(np.all(np.isfinite(result.weights)))
                    self.assertTrue(np.isfinite(response.item()))

    def test_every_supported_phase_bit_uses_quantized_states(self):
        target_azimuth = np.radians(23.0)
        target_elevation = np.radians(11.0)
        for bits in (2, 3, 4, 5, 6):
            with self.subTest(bits=bits):
                _, result = self._uniform_weights(
                    "UPA", 4, 4, target_azimuth, target_elevation, bits
                )
                step = 2.0 * np.pi / (2**bits)
                phase_states = np.angle(result.weights).ravel() / step
                np.testing.assert_allclose(
                    phase_states, np.round(phase_states), atol=1e-12
                )

    def test_array_factor_supports_vector_angle_inputs(self):
        coordinates, result = self._uniform_weights("ULA", 1, 4, 0.0, 0.0)
        azimuth = np.radians(np.array([-30.0, 0.0, 30.0]))
        response = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            self.wavelength,
            azimuth,
            np.zeros_like(azimuth),
        )
        self.assertEqual(response.shape, (3,))
        self.assertEqual(int(np.argmax(np.abs(response))), 1)

    def test_array_factor_chunking_matches_full_matrix_for_angle_grids(self):
        coordinates, result = self._uniform_weights(
            "UPA", 4, 5, np.radians(13.0), np.radians(-7.0)
        )
        azimuth = np.radians(np.linspace(-70.0, 70.0, 17))[:, None]
        elevation = np.radians(np.linspace(-35.0, 35.0, 11))[None, :]
        expected = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            self.wavelength,
            azimuth,
            elevation,
            max_chunk_entries=100_000,
        )

        for options in (
            {"angle_chunk_size": 7},
            {"element_chunk_size": 3},
            {"angle_chunk_size": 5, "element_chunk_size": 4},
            {"max_chunk_entries": 13},
        ):
            with self.subTest(options=options):
                actual = array_factor(
                    coordinates.y,
                    coordinates.z,
                    result.weights,
                    self.wavelength,
                    azimuth,
                    elevation,
                    **options,
                )
                self.assertEqual(actual.shape, (17, 11))
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_array_factor_rejects_invalid_chunk_limits(self):
        coordinates, result = self._uniform_weights("ULA", 1, 4, 0.0, 0.0)
        for options in (
            {"angle_chunk_size": 0},
            {"element_chunk_size": 0},
            {"max_chunk_entries": 0},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    array_factor(
                        coordinates.y,
                        coordinates.z,
                        result.weights,
                        self.wavelength,
                        0.0,
                        0.0,
                        **options,
                    )

    def test_array_factor_uses_the_public_steering_vector_definition(self):
        coordinates, result = self._uniform_weights(
            "UPA", 2, 3, np.radians(17.0), np.radians(-8.0)
        )
        observation_azimuth = np.radians(31.0)
        observation_elevation = np.radians(12.0)
        response_vector = steering_vector(
            coordinates.y,
            coordinates.z,
            self.wavelength,
            observation_azimuth,
            observation_elevation,
        )
        expected = np.sum(result.weights * response_vector)
        actual = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            self.wavelength,
            observation_azimuth,
            observation_elevation,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)


class PatternMetricAndNullTests(unittest.TestCase):
    def test_metrics_can_anchor_an_equal_height_target_lobe(self):
        angles = np.radians(np.array([-180.0, -20.0, -10.0, 0.0, 10.0, 20.0]))
        pattern = np.array([1.0, 0.1, 0.5, 1.0, 0.5, 0.1])

        metrics = calculate_pattern_metrics(pattern, angles, peak_index=3)

        self.assertEqual(metrics.peak_index, 3)
        self.assertGreater(metrics.hpbw_left_angle_deg, -20.0)
        self.assertLess(metrics.hpbw_right_angle_deg, 20.0)

    def test_hpbw_uses_linear_interpolation_between_samples(self):
        angles_deg = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        pattern = np.array([0.5, 0.8, 1.0, 0.8, 0.5])
        metrics = calculate_pattern_metrics(pattern, np.radians(angles_deg))

        half_power = 1.0 / np.sqrt(2.0)
        right_crossing = 1.0 + (0.8 - half_power) / (0.8 - 0.5)
        expected_hpbw = 2.0 * right_crossing
        self.assertAlmostEqual(metrics.hpbw_deg, expected_hpbw, places=12)
        self.assertAlmostEqual(metrics.hpbw_left_angle_deg, -right_crossing, places=12)
        self.assertAlmostEqual(metrics.hpbw_right_angle_deg, right_crossing, places=12)

    def test_fnbw_one_sided_formulas_use_distance_from_peak(self):
        right_only = calculate_pattern_metrics(
            np.array([1.0, 0.0, 0.25]),
            np.radians(np.array([10.0, 20.0, 30.0])),
        )
        left_only = calculate_pattern_metrics(
            np.array([0.25, 0.0, 1.0]),
            np.radians(np.array([-30.0, -20.0, -10.0])),
        )
        self.assertAlmostEqual(right_only.first_null_beamwidth_deg, 20.0)
        self.assertAlmostEqual(left_only.first_null_beamwidth_deg, 20.0)

    def test_uniform_eight_element_ula_regression_metrics(self):
        coordinates = create_array_coordinates(1, 8, 0.5, "ULA")
        weights = np.ones_like(coordinates.y, dtype=complex)
        angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 4001)
        pattern = array_factor(
            coordinates.y,
            coordinates.z,
            weights,
            1.0,
            angles,
            np.zeros_like(angles),
        )
        metrics = calculate_pattern_metrics(pattern, angles)
        self.assertAlmostEqual(np.degrees(angles[metrics.peak_index]), 0.0, delta=0.1)
        self.assertGreater(metrics.hpbw_deg, 12.0)
        self.assertLess(metrics.hpbw_deg, 14.0)
        self.assertGreater(metrics.first_null_beamwidth_deg, 28.0)
        self.assertLess(metrics.first_null_beamwidth_deg, 30.0)
        self.assertGreater(metrics.sidelobe_level_db, -14.0)
        self.assertLess(metrics.sidelobe_level_db, -12.0)

    def test_undetected_single_element_metrics_are_explicitly_unavailable(self):
        angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181)
        metrics = calculate_pattern_metrics(np.ones_like(angles), angles)
        self.assertIsNone(metrics.hpbw_deg)
        self.assertIsNone(metrics.hpbw_left_index)
        self.assertIsNone(metrics.hpbw_right_index)
        self.assertIsNone(metrics.first_null_beamwidth_deg)
        self.assertIsNone(metrics.first_null_left_index)
        self.assertIsNone(metrics.first_null_right_index)
        self.assertIsNone(metrics.sidelobe_level_db)
        self.assertIsNone(metrics.sidelobe_angle_deg)

    def test_null_steering_suppresses_requested_direction(self):
        coordinates = create_array_coordinates(1, 8, 0.5, "ULA")
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_direction_rad=(np.radians(30.0), 0.0),
        )
        target = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            1.0,
            0.0,
            0.0,
        )
        interferer = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            1.0,
            np.radians(30.0),
            0.0,
        )
        self.assertTrue(result.null_applied)
        self.assertGreater(abs(target.item()), 0.1)
        self.assertLess(abs(interferer.item()), 1e-10)
        self.assertGreaterEqual(result.null_depths_db[0], 250.0)

    def test_null_solver_uses_direct_svd_and_reports_weight_diagnostics(self):
        coordinates = create_array_coordinates(1, 16, 0.5, "ULA")
        amplitudes = np.hamming(16).reshape(coordinates.y.shape)
        null_directions = [
            (np.radians(-18.0), 0.0),
            (np.radians(27.0), 0.0),
        ]
        with patch(
            "numpy.linalg.solve",
            side_effect=AssertionError("normal-equation solve must not be used"),
        ):
            result = compute_beamforming_weights(
                coordinates.y,
                coordinates.z,
                1.0,
                np.radians(5.0),
                0.0,
                amplitudes,
                null_directions_rad=null_directions,
            )

        diagnostics = result.continuous_diagnostics
        self.assertTrue(result.null_applied)
        self.assertEqual(result.solver_method, "svd_minimum_norm")
        self.assertIsNone(result.determinant)
        self.assertLess(diagnostics.target_relative_error, 1e-12)
        self.assertTrue(
            all(value < 1e-12 for value in diagnostics.null_relative_residuals)
        )
        self.assertLess(diagnostics.constraint_relative_residual_norm, 1e-12)
        self.assertAlmostEqual(
            diagnostics.max_amplitude,
            float(np.max(np.abs(result.continuous_weights))),
            places=12,
        )
        self.assertAlmostEqual(
            diagnostics.total_weight_power,
            float(np.sum(np.abs(result.continuous_weights) ** 2)),
            places=12,
        )
        self.assertEqual(result.continuous_diagnostics, result.final_diagnostics)
        self.assertAlmostEqual(result.quantization_constraint_degradation_db, 0.0)

    def test_svd_solver_remains_accurate_for_nearby_null_constraints(self):
        coordinates = create_array_coordinates(1, 32, 0.5, "ULA")
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_directions_rad=[
                (np.radians(20.0), 0.0),
                (np.radians(20.0001), 0.0),
            ],
            singular_tolerance=1e-8,
        )

        self.assertTrue(result.null_applied)
        self.assertGreater(result.condition_number, 1e4)
        self.assertLess(
            result.continuous_diagnostics.constraint_relative_residual_norm,
            1e-12,
        )
        self.assertTrue(
            all(
                residual < 1e-12
                for residual in result.continuous_diagnostics.null_relative_residuals
            )
        )

    def test_final_phase_quantization_reports_actual_null_depth_and_amplitude(self):
        coordinates = create_array_coordinates(1, 8, 0.5, "ULA")
        target_azimuth = np.radians(10.0)
        null_azimuth = np.radians(23.0)
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            target_azimuth,
            0.0,
            np.ones_like(coordinates.y),
            phase_bits=2,
            null_direction_rad=(null_azimuth, 0.0),
        )

        phase_step = 2.0 * np.pi / 4.0
        np.testing.assert_allclose(
            result.final_phases / phase_step,
            np.round(result.final_phases / phase_step),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.abs(result.weights),
            np.abs(result.continuous_weights),
            atol=1e-12,
        )
        self.assertFalse(np.allclose(np.abs(result.weights), 1.0))
        self.assertGreaterEqual(result.continuous_null_depths_db[0], 250.0)
        self.assertGreater(result.null_depths_db[0], 10.0)
        self.assertLess(result.null_depths_db[0], 30.0)

        target_response = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            1.0,
            target_azimuth,
            0.0,
        )
        null_response = array_factor(
            coordinates.y,
            coordinates.z,
            result.weights,
            1.0,
            null_azimuth,
            0.0,
        )
        measured_depth = -20.0 * np.log10(
            abs(null_response.item()) / abs(target_response.item())
        )
        self.assertAlmostEqual(result.null_depths_db[0], measured_depth, places=12)
        desired_target_magnitude = float(np.sum(np.ones_like(coordinates.y)))
        self.assertAlmostEqual(
            result.final_diagnostics.target_response_error,
            abs(target_response.item() - desired_target_magnitude),
            places=12,
        )
        self.assertAlmostEqual(
            result.final_diagnostics.null_constraint_residuals[0],
            abs(null_response.item()),
            places=12,
        )
        self.assertGreater(
            result.final_diagnostics.target_relative_error,
            result.continuous_diagnostics.target_relative_error,
        )
        self.assertGreater(
            result.final_diagnostics.null_relative_residuals[0],
            result.continuous_diagnostics.null_relative_residuals[0],
        )
        self.assertGreater(result.quantization_target_degradation_db, 0.0)
        self.assertGreater(result.quantization_null_degradation_db[0], 0.0)
        self.assertGreater(result.quantization_constraint_degradation_db, 0.0)
        self.assertAlmostEqual(
            result.final_diagnostics.max_amplitude,
            result.continuous_diagnostics.max_amplitude,
            places=12,
        )
        self.assertAlmostEqual(
            result.final_diagnostics.total_weight_power,
            result.continuous_diagnostics.total_weight_power,
            places=12,
        )

    def test_multiple_null_directions_use_one_expandable_constraint_matrix(self):
        coordinates = create_array_coordinates(1, 8, 0.5, "ULA")
        null_directions = [
            (np.radians(-25.0), 0.0),
            (np.radians(32.0), 0.0),
        ]
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_directions_rad=null_directions,
        )
        self.assertTrue(result.null_applied)
        self.assertEqual(result.constraint_count, 3)
        self.assertEqual(result.constraint_rank, 3)
        self.assertEqual(len(result.null_depths_db), 2)
        self.assertEqual(
            len(result.continuous_diagnostics.null_constraint_residuals),
            2,
        )
        self.assertEqual(len(result.quantization_null_degradation_db), 2)
        self.assertTrue(all(depth >= 250.0 for depth in result.null_depths_db))

    def test_phase_only_optimizer_keeps_amplitudes_and_meets_requirements(self):
        coordinates = create_array_coordinates(1, 16, 0.5, "ULA")
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_directions_rad=[
                (np.radians(-25.0), 0.0),
                (np.radians(32.0), 0.0),
            ],
            null_required_suppression_db=[25.0, 25.0],
            optimization_mode="phase_only",
        )

        self.assertTrue(result.null_applied)
        self.assertEqual(result.solver_method, "phase_only_projected_gradient")
        self.assertGreater(result.optimizer_iterations, 0)
        self.assertGreaterEqual(
            result.optimizer_total_iterations,
            result.optimizer_iterations,
        )
        self.assertIn(result.optimizer_selected_restart, range(1, 5))
        self.assertEqual(result.optimizer_restart_count, 4)
        self.assertNotEqual(result.optimizer_convergence_reason, "not_run")
        self.assertIsNotNone(result.optimizer_final_objective)
        selected_trace = [
            point
            for point in result.optimizer_trace
            if point.restart_index == result.optimizer_selected_restart
        ]
        self.assertEqual(len(selected_trace), result.optimizer_iterations + 1)
        self.assertTrue(
            all(
                later.objective <= earlier.objective
                for earlier, later in zip(
                    selected_trace, selected_trace[1:], strict=False
                )
            )
        )
        self.assertTrue(
            all(point.worst_null_residual_db is not None for point in selected_trace)
        )
        self.assertTrue(
            all(point.target_loss_db is not None for point in selected_trace)
        )
        np.testing.assert_allclose(np.abs(result.continuous_weights), 1.0)
        self.assertEqual(result.null_required_suppression_db, (25.0, 25.0))
        self.assertEqual(result.null_requirement_met, (True, True))
        self.assertLess(result.final_diagnostics.target_relative_error, 0.02)

    def test_amplitude_limit_reoptimizes_after_projection(self):
        coordinates = create_array_coordinates(1, 16, 0.5, "ULA")
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_directions_rad=[
                (np.radians(-25.0), 0.0),
                (np.radians(32.0), 0.0),
            ],
            null_required_suppression_db=[40.0, 40.0],
            maximum_element_amplitude=1.0,
        )

        self.assertLessEqual(np.max(np.abs(result.weights)), 1.0 + 1.0e-12)
        self.assertEqual(result.solver_method, "bounded_projected_gradient")
        self.assertGreater(result.saturated_element_count, 0)
        self.assertEqual(
            result.saturated_element_count,
            int(np.count_nonzero(result.saturated_element_mask)),
        )
        self.assertEqual(result.null_requirement_met, (True, True))
        self.assertGreater(result.optimizer_iterations, 0)
        self.assertGreater(len(result.optimizer_trace), 1)
        self.assertLess(
            result.optimizer_trace[-1].objective,
            result.optimizer_trace[0].objective,
        )
        self.assertLess(
            result.optimizer_trace[-1].worst_null_residual_db,
            result.optimizer_trace[0].worst_null_residual_db,
        )

    def test_phase_only_restarts_are_deterministic_and_never_worse_than_single_start(
        self,
    ):
        coordinates = create_array_coordinates(1, 16, 0.5, "ULA")
        arguments = dict(
            y=coordinates.y,
            z=coordinates.z,
            wavelength_m=1.0,
            target_azimuth_rad=0.0,
            target_elevation_rad=0.0,
            amplitude_weights=np.ones_like(coordinates.y),
            null_directions_rad=[
                (np.radians(-25.0), 0.0),
                (np.radians(32.0), 0.0),
            ],
            null_required_suppression_db=[35.0, 35.0],
            optimization_mode="phase_only",
            optimizer_max_iterations=100,
        )
        single = compute_beamforming_weights(
            **arguments,
            optimizer_restart_count=1,
        )
        first = compute_beamforming_weights(
            **arguments,
            optimizer_restart_count=4,
        )
        second = compute_beamforming_weights(
            **arguments,
            optimizer_restart_count=4,
        )

        self.assertLessEqual(
            first.optimizer_final_objective,
            single.optimizer_final_objective + 1e-15,
        )
        np.testing.assert_allclose(first.continuous_weights, second.continuous_weights)
        self.assertEqual(first.optimizer_trace, second.optimizer_trace)
        self.assertEqual(
            first.optimizer_selected_restart,
            second.optimizer_selected_restart,
        )

    def test_iterative_null_optimizer_checks_cancellation_inside_loop(self):
        coordinates = create_array_coordinates(1, 16, 0.5, "ULA")
        check_count = 0

        def cancel_check():
            nonlocal check_count
            check_count += 1
            if check_count >= 5:
                raise RuntimeError("cancelled in optimizer")

        with self.assertRaisesRegex(RuntimeError, "cancelled in optimizer"):
            compute_beamforming_weights(
                coordinates.y,
                coordinates.z,
                1.0,
                0.0,
                0.0,
                np.ones_like(coordinates.y),
                null_directions_rad=[
                    (np.radians(-25.0), 0.0),
                    (np.radians(32.0), 0.0),
                ],
                optimization_mode="phase_only",
                optimizer_tolerance=1e-12,
                optimizer_max_iterations=400,
                cancel_check=cancel_check,
            )
        self.assertGreaterEqual(check_count, 5)

    def test_optimizer_settings_are_validated_and_iteration_limit_is_reported(self):
        coordinates = create_array_coordinates(1, 16, 0.5, "ULA")
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_direction_rad=(np.radians(23.0), 0.0),
            optimization_mode="phase_only",
            optimizer_tolerance=1e-15,
            optimizer_max_iterations=1,
            optimizer_restart_count=1,
        )
        self.assertEqual(result.optimizer_max_iterations, 1)
        self.assertEqual(result.optimizer_tolerance, 1e-15)
        self.assertEqual(result.optimizer_convergence_reason, "max_iterations")

        with self.assertRaisesRegex(ValueError, "maximum iterations"):
            compute_beamforming_weights(
                coordinates.y,
                coordinates.z,
                1.0,
                0.0,
                0.0,
                np.ones_like(coordinates.y),
                optimizer_max_iterations=0,
            )
        with self.assertRaisesRegex(ValueError, "restart count"):
            compute_beamforming_weights(
                coordinates.y,
                coordinates.z,
                1.0,
                0.0,
                0.0,
                np.ones_like(coordinates.y),
                optimizer_restart_count=0,
            )

    def test_each_null_direction_requires_one_suppression_value(self):
        coordinates = create_array_coordinates(1, 8, 0.5, "ULA")
        with self.assertRaisesRegex(ValueError, "one suppression"):
            compute_beamforming_weights(
                coordinates.y,
                coordinates.z,
                1.0,
                0.0,
                0.0,
                np.ones_like(coordinates.y),
                null_directions_rad=[
                    (np.radians(-20.0), 0.0),
                    (np.radians(30.0), 0.0),
                ],
                null_required_suppression_db=[40.0],
            )

    def test_coincident_null_falls_back_without_non_finite_weights(self):
        coordinates = create_array_coordinates(1, 4, 0.5, "ULA")
        result = compute_beamforming_weights(
            coordinates.y,
            coordinates.z,
            1.0,
            0.0,
            0.0,
            np.ones_like(coordinates.y),
            null_direction_rad=(0.0, 0.0),
        )
        self.assertFalse(result.null_applied)
        self.assertTrue(np.all(np.isfinite(result.weights)))
        self.assertEqual(result.constraint_rank, 1)
        self.assertEqual(result.constraint_count, 2)
        self.assertTrue(np.isinf(result.condition_number))
        self.assertIsNotNone(result.diagnostic_message)
        self.assertAlmostEqual(result.null_depths_db[0], 0.0)
        self.assertEqual(result.solver_method, "svd_rejected")
        self.assertAlmostEqual(
            result.final_diagnostics.null_relative_residuals[0],
            1.0,
        )

    def test_zero_pattern_db_normalization_is_finite(self):
        normalized = normalize_pattern_db(np.zeros(4, dtype=complex))
        np.testing.assert_allclose(normalized, -120.0)

    def test_linear_normalization_checks_zero_and_tiny_denominators(self):
        np.testing.assert_allclose(
            normalize_pattern_linear(np.zeros((2, 2), dtype=complex)),
            0.0,
        )
        np.testing.assert_allclose(
            normalize_pattern_linear(np.array([0.0, 1e-300, 5e-301])),
            [0.0, 1.0, 0.5],
        )


class ArrayGainMetricTests(unittest.TestCase):
    def setUp(self):
        coordinates = create_array_coordinates(1, 4, 0.5, "ULA")
        self.y = coordinates.y
        self.z = coordinates.z

    def _metrics(self, weights, mask):
        return calculate_array_gain_metrics(
            self.y,
            self.z,
            np.asarray(weights, dtype=complex).reshape(1, 4),
            np.asarray(mask, dtype=bool).reshape(1, 4),
            1.0,
            0.0,
            0.0,
        )

    def test_uniform_relative_gain_uses_actual_active_element_count(self):
        metrics = self._metrics([1.0, 1.0, 0.0, 0.0], [True, True, False, False])
        self.assertEqual(metrics.total_elements, 4)
        self.assertEqual(metrics.active_elements, 2)
        self.assertAlmostEqual(metrics.taper_efficiency, 1.0)
        self.assertAlmostEqual(metrics.phase_efficiency, 1.0)
        self.assertAlmostEqual(metrics.effective_element_count, 2.0)
        self.assertAlmostEqual(
            metrics.relative_array_gain_db,
            10.0 * np.log10(2.0),
        )

    def test_taper_efficiency_reduces_relative_gain(self):
        metrics = self._metrics([1.0, 0.5, 0.5, 1.0], [True] * 4)
        self.assertAlmostEqual(metrics.taper_efficiency, 0.9)
        self.assertAlmostEqual(metrics.phase_efficiency, 1.0)
        self.assertAlmostEqual(metrics.effective_element_count, 3.6)
        self.assertAlmostEqual(
            metrics.relative_array_gain_db,
            10.0 * np.log10(3.6),
        )

    def test_relative_gain_is_invariant_for_tiny_nonzero_weights(self):
        metrics = self._metrics([1e-300] * 4, [True] * 4)
        self.assertAlmostEqual(metrics.taper_efficiency, 1.0)
        self.assertAlmostEqual(metrics.phase_efficiency, 1.0)
        self.assertAlmostEqual(metrics.effective_element_count, 4.0)
        self.assertAlmostEqual(
            metrics.relative_array_gain_db,
            10.0 * np.log10(4.0),
        )

    def test_zero_weights_return_unavailable_relative_gain(self):
        metrics = self._metrics([0.0] * 4, [False] * 4)
        self.assertEqual(metrics.active_elements, 0)
        self.assertEqual(metrics.taper_efficiency, 0.0)
        self.assertEqual(metrics.phase_efficiency, 0.0)
        self.assertEqual(metrics.effective_element_count, 0.0)
        self.assertIsNone(metrics.relative_array_gain_db)

    def test_inactive_elements_cannot_have_nonzero_weights(self):
        with self.assertRaises(ValueError):
            self._metrics([1.0, 0.0, 0.0, 0.0], [False] * 4)


if __name__ == "__main__":
    unittest.main()
