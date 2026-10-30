"""Verify Gamow-window coverage and percentiles against analytic integrals."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from post_processing import calc_gamow_window_cdf as gamow


class GamowWindowNumericsTest(unittest.TestCase):
    def test_log_interpolation_reproduces_an_exponential_without_extrapolation(self):
        samples = np.array([0.2, 0.5, 0.8])
        energy, cross_section = gamow.talys_on_fine_grid(
            samples, np.exp(3 * samples), np.linspace(0.0, 1.0, 101))
        self.assertEqual((energy[0], energy[-1]), (0.2, 0.8))
        np.testing.assert_allclose(cross_section, np.exp(3 * energy), rtol=1e-12)

    def test_refined_median_matches_an_analytic_gamow_integral(self):
        # sigma=exp(3E), kT=1 MeV gives E exp(2E), whose antiderivative is
        # exp(2E) (E/2 - 1/4). This independently checks the CDF percentile.
        coarse = np.array([0.1, 1.0, 2.0])
        energy, cross_section = gamow.talys_on_fine_grid(
            coarse, np.exp(3 * coarse), np.linspace(0.1, 2.0, 5001))
        temperature = 1.0 / gamow.kB_MeV_per_GK
        calculated = gamow.percentile_energy(
            0.5, gamow.cdf(energy, gamow.integrand(energy, cross_section, temperature)), energy)
        def primitive(value):
            return np.exp(2 * value) * (value / 2 - 0.25)
        total = primitive(2.0) - primitive(0.1)
        expected = brentq(lambda value: (primitive(value) - primitive(0.1)) / total - 0.5,
                          0.1, 2.0)
        self.assertAlmostEqual(calculated, expected, delta=1e-6)
        unrefined = gamow.percentile_energy(
            0.5, gamow.cdf(coarse, gamow.integrand(coarse, np.exp(3 * coarse), temperature)), coarse)
        self.assertGreater(abs(unrefined - expected), 0.005)

    def test_coverage_includes_partial_boundary_cells(self):
        energy = np.array([0.0, 1.0, 2.0])
        below, above = gamow.fraction_outside(energy, np.ones(3), 0.25, 1.75)
        self.assertAlmostEqual(below, 0.125)
        self.assertAlmostEqual(above, 0.125)
        below, above = gamow.fraction_outside(energy, energy, 0.5, 1.5)
        self.assertAlmostEqual(below, 0.0625)
        self.assertAlmostEqual(above, 0.4375)

    def test_nonpositive_samples_and_missing_overlap_have_no_interpolated_curve(self):
        for samples, cross_section, reference in [
            ([1, 2], [0, 0], [1, 2]), ([1, 2], [1, 2], [3, 4]),
        ]:
            with self.subTest(samples=samples, reference=reference):
                energy, values = gamow.talys_on_fine_grid(samples, cross_section, reference)
                self.assertEqual(energy.size, 0)
                self.assertEqual(values.size, 0)

    def test_invalid_normalizations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            gamow.cdf(np.array([1, 2]), np.zeros(2))
        with self.assertRaisesRegex(ValueError, 'positive'):
            gamow.cdf_normalized_to_ratesmc(np.array([1, 2]), np.ones(2), 0)


class GamowWindowInputsTest(unittest.TestCase):
    def prepare_run(self, root):
        run = root / 'talys_opt/22Mg_ap_25Al/binwidth_0.2/RUN_0'
        run.mkdir(parents=True)
        cross_sections = root / 'cross_sections/22Mg_ap_25Al'
        cross_sections.mkdir(parents=True)
        np.savetxt(cross_sections / '22Mg(a,p)25Al_xs_unintegrated_parallel_run0.txt',
                   [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]], delimiter=',')
        return run

    def test_multiple_saved_results_for_a_run_are_reported_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.prepare_run(root)
            (run / 'first.npz').touch()
            (run / 'second.npz').touch()
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                gamow.discover_runs(str(root), '22Mg(a,p)25Al', '0.2')

    def test_ratesmc_curve_survives_empty_talys_energy_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.prepare_run(root)
            np.savez(run / 'fit.npz', x_xs=[2.0, 3.0], y_xs=[1.0, 2.0])
            output = root / 'cdf.png'
            arguments = ['calc_gamow_window_cdf.py', '--base-dir', str(root),
                         '--reaction', '22Mg(a,p)25Al', '--t9', '1', '--e-max', '1',
                         '--no-band', '--output', str(output)]
            with patch('sys.argv', arguments), patch.object(gamow.plt, 'show'):
                gamow.main()
            self.assertTrue(output.is_file())
            gamow.plt.close('all')
