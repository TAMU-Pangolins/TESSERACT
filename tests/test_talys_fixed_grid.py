"""Verify the optional fixed TALYS energy grid (talys_grid_step) and its bin averages."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import talys_opt_with_unc as topt


def gamow_like(E):
    """Sub-barrier alpha-induced cross section (mb) with a smooth S-factor wiggle."""
    E = np.asarray(E, dtype=float)
    return 1e3 / E * np.exp(-43.7 / np.sqrt(E)) * (1.0 + 0.3 * np.sin(E))


class FixedGridLayoutTest(unittest.TestCase):
    def test_grid_covers_every_bin_node_within_the_step(self):
        centres = np.arange(0.205, 10.0, 0.01)
        cfg = {}
        topt.configure_talys_energies(cfg, centres, 0.01, 3, 0.05)
        grid, nodes = cfg['_talys_grid'], cfg['_bin_nodes']
        self.assertAlmostEqual(grid[0], nodes.min())
        self.assertAlmostEqual(grid[-1], nodes.max())
        self.assertLessEqual(np.diff(grid).max(), 0.05 + 1e-12)
        self.assertIs(cfg['_talys_E_cm'], grid)

    def test_fine_bins_need_far_fewer_talys_energies(self):
        centres = np.arange(0.205, 10.0, 0.01)
        cfg = {}
        topt.configure_talys_energies(cfg, centres, 0.01, 3, 0.05)
        self.assertLess(len(cfg['_talys_E_cm']), 250)
        self.assertEqual(len(cfg['_bin_nodes']), 3 * len(centres))
        bins = topt.talys_to_bins(gamow_like(cfg['_talys_E_cm']), cfg, len(centres))
        self.assertEqual(len(bins), len(centres))

    def test_unset_step_keeps_the_per_bin_layout(self):
        centres = np.array([1.0, 1.2, 1.4])
        cfg = {}
        topt.configure_talys_energies(cfg, centres, 0.2, 3, None)
        nodes, weights = topt.talys_bin_grid(centres, 0.2, 3)
        np.testing.assert_array_equal(cfg['_talys_E_cm'], nodes)
        np.testing.assert_array_equal(cfg['_bin_weights'], weights)
        self.assertNotIn('_talys_grid', cfg)
        y = gamow_like(nodes)
        np.testing.assert_array_equal(topt.talys_to_bins(y, cfg, 3), topt.talys_bin_average(y, weights))
        centres_only = {}
        topt.configure_talys_energies(centres_only, centres, 0.2, 1, None)
        self.assertEqual(centres_only, {})
        np.testing.assert_array_equal(topt.talys_to_bins(y[:3], centres_only, 3), y[:3])


class FixedGridAccuracyTest(unittest.TestCase):
    def test_grid_bin_averages_match_direct_per_bin_evaluation(self):
        for width in (0.01, 0.05, 0.2, 1.0):
            with self.subTest(width=width):
                centres = np.arange(0.2 + width / 2, 10.0, width)
                grid_cfg, direct_cfg = {}, {}
                topt.configure_talys_energies(grid_cfg, centres, width, 3, 0.05)
                topt.configure_talys_energies(direct_cfg, centres, width, 3, None)
                via_grid = topt.talys_to_bins(gamow_like(grid_cfg['_talys_E_cm']), grid_cfg, len(centres))
                direct = topt.talys_to_bins(gamow_like(direct_cfg['_talys_E_cm']), direct_cfg, len(centres))
                rel = np.abs(via_grid / direct - 1.0)
                self.assertLess(rel[centres >= 0.5].max(), 1e-3)
                self.assertLess(rel.max(), 5e-3)

    def test_interpolation_is_exact_on_grid_points_and_handles_zeros(self):
        grid = np.array([1.0, 1.5, 2.0, 2.5])
        y = np.array([0.0, 0.0, 2.0, 8.0])
        np.testing.assert_allclose(topt.interp_cross_section(grid, y, grid), y)
        mid = topt.interp_cross_section(grid, y, np.array([1.25, 1.75]))
        self.assertEqual(mid[0], 0.0)                 # between two zeros
        self.assertTrue(0.0 < mid[1] < 2.0)           # across the threshold: linear, non-negative
        self.assertTrue(np.all(np.isfinite(mid)))

    def test_wrong_talys_output_length_is_rejected(self):
        cfg = {}
        topt.configure_talys_energies(cfg, np.array([1.0, 1.2]), 0.2, 3, 0.05)
        self.assertIsNone(topt.talys_to_bins(np.ones(len(cfg['_talys_grid']) + 1), cfg, 2))


class FixedGridObjectiveTest(unittest.TestCase):
    def test_objective_compares_grid_bin_averages_with_the_data(self):
        centres = np.arange(1.025, 3.0, 0.05)
        cfg = {'opt_params': [SimpleNamespace(name='p', x0=1.0, lo=0.0, hi=2.0)],
               'script': {'exp_rel_err': '0.10', 'prior_rel_std': '0', 'debug_every': '0'}}
        topt.configure_talys_energies(cfg, centres, 0.05, 3, 0.05)
        grid = cfg['_talys_E_cm']
        sigma = 1e25 * gamow_like(grid)   # well above the fit's small-cross-section floor
        data = topt.talys_to_bins(sigma, cfg, len(centres))
        mask = np.ones(len(centres), dtype=bool)
        with patch.object(topt, 'run_talys_get_xs', return_value=(grid, sigma)):
            objective = topt.make_objective(cfg, centres, data, 0.1 * data, mask)
            self.assertAlmostEqual(objective(np.array([1.0])), 0.0)
        statistics = objective.statistics(np.array([1.0]), 2.0 * sigma)
        self.assertAlmostEqual(statistics['chi2'], len(centres) * (np.log(2.0) / 0.1) ** 2, places=6)
        with self.assertRaisesRegex(ValueError, 'bin grid'):
            objective.statistics(np.array([1.0]), np.ones(3))


if __name__ == '__main__':
    unittest.main()
