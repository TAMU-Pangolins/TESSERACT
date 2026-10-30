"""Verify parameter selection, fit filters, and rate-ratio output names."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

class PlotScriptFixesTest(unittest.TestCase):
    def test_missing_width_is_not_in_either_saved_plot_name(self):
        from post_processing import plot_talys_results_new as results
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / 'talys_opt/22Mg_ap_25Al/binwidth_0.2'
            available.mkdir(parents=True)
            (available / 'talys_optimization.out').write_text('fixture')
            values = (np.array([0.5, 1.0, 1.5]), np.array([[1.0, 1.1, 1.2]]), ['run 0'])
            arguments = ['plot_talys_results_new.py', '--target', '22Mg',
                         '--projectile', 'a', '--ejectile', 'p', '--residual', '25Al',
                         '--base-dir', str(root), '--bin-width', '0.1', '0.2',
                         '--no-band', '--no-axes']
            with patch('sys.argv', arguments), \
                 patch.object(results, 'parse_talys_opt_out', return_value=[{}]), \
                 patch.object(results, 'compute_rate_ratios', return_value=values), \
                 patch.object(results.plt, 'show'):
                results.main()
            self.assertTrue((root / 'plots/22Mg_ap_25Al_bw_0.2_no_band.png').is_file())
            self.assertTrue((root / 'plots/22Mg_ap_25Al_bw_0.2_no_band_no_axes.png').is_file())
            results.plt.close('all')

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    def test_auto_pairs_respect_max_chi2(self):
        import plot_talys_param_grid as grid
        # a/b correlate only in the high-chi2 records, a/c only in the low ones.
        low = np.array([1.0, 2.0, 3.0, 4.0])
        high = np.tile(low, 2)
        chi2 = np.r_[np.full(4, 1.0), np.full(8, 9.0)]
        a = np.r_[low, high]
        b = np.r_[[3.0, 1.0, 4.0, 2.0], 10.0 * high]          # r(a, b) = 0 at low chi2
        c = np.r_[2.0 * low, np.tile([3.0, 1.0, 4.0, 2.0], 2)]  # r(a, c) = 1 at low chi2
        table = {"chi2_red": chi2, "a": a, "b": b, "c": c}
        top_all = grid.strongest_parameter_pairs(table, ["a", "b", "c"], count=1)
        top_low = grid.strongest_parameter_pairs(table, ["a", "b", "c"], count=1, max_chi2=2.0)
        self.assertEqual(top_all[0][:2], ("a", "b"))
        self.assertEqual(top_low[0][:2], ("a", "c"))
        self.assertAlmostEqual(top_low[0][2], 1.0)

    def test_pair_keeps_dropped_parameter_and_rejects_unknown(self):
        import plot_talys_param_grid as grid
        params, restored = grid.focused_pair_params(
            [("a", "flat")], ["a", "b"], ["a", "b", "flat"])
        self.assertEqual(params, ["a", "b", "flat"])
        self.assertEqual(restored, ["flat"])
        with self.assertRaises(SystemExit) as ctx:
            grid.focused_pair_params([("a", "typo")], ["a", "b"], ["a", "b", "flat"])
        self.assertIn("typo", str(ctx.exception))

    def test_ratio_plot_name_lists_every_bin_width(self):
        from post_processing import plot_talys_results_new as results
        self.assertEqual(results.ratio_plot_name("22Mg_ap_25Al", ["0.2"]),
                         "22Mg_ap_25Al_bw_0.2.png")
        self.assertEqual(results.ratio_plot_name("22Mg_ap_25Al", ["0.2", "0.5"]),
                         "22Mg_ap_25Al_bw_0.2_0.5.png")
