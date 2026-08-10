import unittest

import numpy as np

from beamforming import array_factor, element_pattern_factor
from directivity import DIRECTIVITY_SCHEMA_VERSION, calculate_directivity


class DirectivityTests(unittest.TestCase):
    def _single_element(self, element_option: str):
        return calculate_directivity(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0 + 0.0j]),
            1.0,
            0.0,
            0.0,
            element_option,
        )

    def test_single_isotropic_element_is_zero_dbi(self):
        result = self._single_element("isotropic")

        self.assertEqual(result.schema_version, DIRECTIVITY_SCHEMA_VERSION)
        self.assertAlmostEqual(result.directivity_linear, 1.0, places=12)
        self.assertAlmostEqual(result.directivity_dbi, 0.0, places=12)
        self.assertAlmostEqual(result.radiated_power_integral, 4.0 * np.pi)

    def test_single_cosine_models_use_full_sphere_power(self):
        cosine = self._single_element("cosine")
        cosine_squared = self._single_element("cosine_squared")

        self.assertAlmostEqual(cosine.directivity_linear, 6.0, places=11)
        self.assertAlmostEqual(
            cosine.directivity_dbi,
            10.0 * np.log10(6.0),
            places=11,
        )
        self.assertAlmostEqual(
            cosine_squared.directivity_linear,
            10.0,
            places=10,
        )
        self.assertAlmostEqual(cosine_squared.directivity_dbi, 10.0, places=10)

    def test_two_half_wavelength_isotropic_elements_have_three_dbi(self):
        result = calculate_directivity(
            np.array([-0.25, 0.25]),
            np.zeros(2),
            np.ones(2, dtype=complex),
            1.0,
            0.0,
            0.0,
            "isotropic",
            max_chunk_entries=2,
        )

        self.assertAlmostEqual(result.directivity_linear, 2.0, places=11)
        self.assertAlmostEqual(
            result.directivity_dbi,
            10.0 * np.log10(2.0),
            places=11,
        )

    def test_pairwise_kernels_match_independent_spherical_quadrature(self):
        y = np.array([-0.31, 0.04, 0.47])
        z = np.array([0.17, -0.22, 0.08])
        weights = np.array([1.0 + 0.2j, -0.3 + 0.7j, 0.5 - 0.4j])
        weights = weights / np.max(np.abs(weights))
        mu, mu_weights = np.polynomial.legendre.leggauss(120)
        azimuth = np.linspace(-np.pi, np.pi, 240, endpoint=False)
        azimuth_grid, elevation_grid = np.meshgrid(
            azimuth,
            np.arcsin(mu),
            indexing="ij",
        )
        array_response = array_factor(
            y,
            z,
            weights,
            1.0,
            azimuth_grid,
            elevation_grid,
        )

        for pattern_id in ("isotropic", "cosine", "cosine_squared"):
            with self.subTest(pattern_id=pattern_id):
                expected = (2.0 * np.pi / azimuth.size) * np.sum(
                    np.abs(array_response) ** 2
                    * element_pattern_factor(
                        pattern_id,
                        azimuth_grid,
                        elevation_grid,
                    )
                    ** 2
                    * mu_weights[None, :]
                )
                result = calculate_directivity(
                    y,
                    z,
                    weights,
                    1.0,
                    0.0,
                    0.0,
                    pattern_id,
                )
                self.assertAlmostEqual(
                    result.radiated_power_integral,
                    expected,
                    delta=2.0e-5 * expected,
                )

    def test_half_wave_dipole_quadrature_matches_known_directivity(self):
        result = self._single_element("dipole")

        self.assertIn("Gauss-Legendre", result.integration_method)
        self.assertEqual(result.azimuth_samples, 144)
        self.assertEqual(result.elevation_samples, 96)
        self.assertAlmostEqual(result.directivity_linear, 1.6409, delta=0.001)
        self.assertAlmostEqual(result.directivity_dbi, 2.15, delta=0.01)

    def test_all_zero_weights_report_unavailable(self):
        result = calculate_directivity(
            np.zeros(2),
            np.zeros(2),
            np.zeros(2, dtype=complex),
            1.0,
            0.0,
            0.0,
            "isotropic",
        )

        self.assertIsNone(result.directivity_linear)
        self.assertIsNone(result.directivity_dbi)
        self.assertEqual(result.integration_method, "unavailable")

    def test_pairwise_integration_checks_for_cancellation_between_chunks(self):
        checks = []
        calculate_directivity(
            np.linspace(-1.0, 1.0, 8),
            np.zeros(8),
            np.ones(8, dtype=complex),
            1.0,
            0.0,
            0.0,
            "isotropic",
            max_chunk_entries=8,
            cancel_check=lambda: checks.append(True),
        )

        self.assertGreaterEqual(len(checks), 10)


if __name__ == "__main__":
    unittest.main()
