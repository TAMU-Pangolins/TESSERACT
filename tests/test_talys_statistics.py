import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import talys_opt_with_unc as topt


class _OptParam:
    def __init__(self, name="p1", x0=1.0, lo=0.0, hi=2.0):
        self.name, self.x0, self.lo, self.hi = name, x0, lo, hi


class BinAveragingHelpersTest(unittest.TestCase):
    def test_constant_cross_section_averages_to_itself(self):
        centres = np.array([1.0, 2.0, 3.0])
        for npb in (1, 3, 5):
            E, w = topt.talys_bin_grid(centres, 0.2, npb)
            avg = topt.talys_bin_average(np.full_like(E, 7.0), w)
            np.testing.assert_allclose(avg, 7.0)

    def test_linear_cross_section_averages_to_value_at_centre(self):
        centres = np.array([1.0, 2.0, 3.0])
        E, w = topt.talys_bin_grid(centres, 0.2, 5)
        avg = topt.talys_bin_average(3.0 * E, w)
        np.testing.assert_allclose(avg, 3.0 * centres, atol=1e-10)

    def test_n_per_bin_one_is_bin_centres(self):
        centres = np.array([1.0, 2.0, 3.0])
        E, w = topt.talys_bin_grid(centres, 0.2, 1)
        np.testing.assert_allclose(E, centres)
        np.testing.assert_allclose(w, 1.0)


class ChiSquarePriorScalingTest(unittest.TestCase):
    """
    chi2 must NOT be divided by dof in the objective; only chi2_red
    (reported separately) is. Dividing chi2 but not the prior would make
    the prior dof times stronger than its configured width implies.
    """

    def _cfg(self, **script_overrides):
        script = {'exp_rel_err': '0.10', 'prior_rel_std': '0.20',
                  'prior_abs_floor': '0.01', 'debug_every': '0'}
        script.update(script_overrides)
        return {'opt_params': [_OptParam()], 'script': script}

    def test_prior_is_not_scaled_by_dof(self):
        x_exp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_exp = np.array([10.0, 20.0, 15.0, 8.0, 5.0])
        y_err = 0.1 * y_exp
        mask = np.ones(len(x_exp), dtype=bool)
        cfg = self._cfg()

        with patch.object(topt, 'run_talys_get_xs', return_value=(x_exp, y_exp.copy())):
            obj = topt.make_objective(cfg, x_exp, y_exp, y_err, mask)
            # At X0 with TALYS == y_exp, chi2 = 0 and prior = 0.
            self.assertAlmostEqual(obj(np.array([1.0])), 0.0)

            # Shift away from X0: the loss should be exactly the (undivided)
            # prior term, since chi2 is still 0.
            delta = 0.5
            prior_sigma = 0.20 * max(abs(1.0), 0.01)
            expected_prior = (delta / prior_sigma) ** 2
            total = obj(np.array([1.0 + delta]))
            self.assertAlmostEqual(total, expected_prior, places=9)

    def test_objective_equals_unnormalized_chi2_not_reduced(self):
        x_exp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_exp = np.array([10.0, 20.0, 15.0, 8.0, 5.0])
        y_err = 0.1 * y_exp
        mask = np.ones(len(x_exp), dtype=bool)
        cfg = self._cfg()
        y_t_wrong = y_exp * 1.1  # 10% off everywhere -> nonzero chi2

        with patch.object(topt, 'run_talys_get_xs', return_value=(x_exp, y_t_wrong)):
            obj = topt.make_objective(cfg, x_exp, y_exp, y_err, mask)
            total = obj(np.array([1.0]))  # at X0, so prior = 0

        log_res = (np.log(y_exp) - np.log(y_t_wrong)) / 0.10
        dof = max(len(x_exp) - 1, 1)
        chi2 = float(np.sum(log_res ** 2))
        self.assertAlmostEqual(total, chi2, places=6)
        self.assertGreater(abs(total - chi2 / dof), 1e-3)


class PerPointErrorColumnTest(unittest.TestCase):
    """
    The data file's third (uncertainty) column must set each point's
    sigma(ln y), not just the scalar exp_rel_err -- a point with a larger
    reported uncertainty should be downweighted relative to one with a
    smaller uncertainty, even when both differ from TALYS by the same
    relative amount.
    """

    def _cfg(self, **script_overrides):
        script = {'exp_rel_err': '0.10', 'prior_rel_std': '0.0',
                  'prior_abs_floor': '0.01', 'debug_every': '0'}
        script.update(script_overrides)
        return {'opt_params': [_OptParam()], 'script': script}

    def test_chi2_uses_per_point_sigma_not_scalar_exp_rel_err(self):
        x_exp = np.array([1.0, 2.0])
        y_exp = np.array([10.0, 10.0])
        y_err = np.array([1.0, 5.0])          # 10% and 50% relative error
        mask = np.ones(len(x_exp), dtype=bool)
        cfg = self._cfg()
        y_t = y_exp * 1.1                      # both points 10% off TALYS

        with patch.object(topt, 'run_talys_get_xs', return_value=(x_exp, y_t)):
            obj = topt.make_objective(cfg, x_exp, y_exp, y_err, mask)
            total = obj(np.array([1.0]))

        sigma_ln = y_err / y_exp               # [0.10, 0.50]
        log_res = (np.log(y_exp) - np.log(y_t)) / sigma_ln
        expected = float(np.sum(log_res ** 2))
        self.assertAlmostEqual(total, expected, places=6)

        # Using the scalar exp_rel_err for every point would give a
        # different (here, larger) chi2 -- confirming the per-point column
        # actually changes the result instead of being silently ignored.
        scalar_log_res = (np.log(y_exp) - np.log(y_t)) / 0.10
        scalar_chi2 = float(np.sum(scalar_log_res ** 2))
        self.assertGreater(abs(total - scalar_chi2), 1e-6)

    def test_nonpositive_error_column_falls_back_to_exp_rel_err(self):
        x_exp = np.array([1.0, 2.0])
        y_exp = np.array([10.0, 10.0])
        y_err = np.array([1.0, -1.0])           # second point: no usable error
        mask = np.ones(len(x_exp), dtype=bool)
        cfg = self._cfg(exp_rel_err='0.20')
        y_t = y_exp * 1.1

        with patch.object(topt, 'run_talys_get_xs', return_value=(x_exp, y_t)):
            obj = topt.make_objective(cfg, x_exp, y_exp, y_err, mask)
            total = obj(np.array([1.0]))

        sigma_ln = np.array([1.0 / 10.0, 0.20])  # point 1 from the column, point 2 the fallback
        log_res = (np.log(y_exp) - np.log(y_t)) / sigma_ln
        expected = float(np.sum(log_res ** 2))
        self.assertAlmostEqual(total, expected, places=6)


class BinAveragedComparisonTest(unittest.TestCase):
    """TALYS must be averaged over each bin before comparison, not read at the centre."""

    def test_talys_is_compared_against_its_own_bin_average(self):
        x_exp = np.array([1.0, 2.0])
        mask = np.ones(2, dtype=bool)
        width = 0.2
        E_sub, w_sub = topt.talys_bin_grid(x_exp, width, 3)

        # A steep (sub-Coulomb-like) fake cross section: centre value and
        # bin average genuinely differ for this.
        y_sub = np.exp(5.0 * E_sub)
        bin_avg = topt.talys_bin_average(y_sub, w_sub)
        centre_values = np.exp(5.0 * x_exp)
        self.assertFalse(np.allclose(bin_avg, centre_values))

        cfg = {
            'opt_params': [_OptParam()],
            'script': {'exp_rel_err': '0.10', 'prior_rel_std': '0', 'debug_every': '0'},
            '_bin_weights': w_sub,
        }

        with patch.object(topt, 'run_talys_get_xs', return_value=(E_sub, y_sub.copy())):
            # Data matching the bin average -> chi2 ~ 0.
            obj_avg = topt.make_objective(cfg, x_exp, bin_avg.copy(),
                                          0.1 * bin_avg, mask)
            self.assertLess(obj_avg(np.array([1.0])), 1e-6)

        with patch.object(topt, 'run_talys_get_xs', return_value=(E_sub, y_sub.copy())):
            # Data matching the bin CENTRE instead -> chi2 clearly nonzero,
            # proving the comparison isn't silently using centre values.
            obj_centre = topt.make_objective(cfg, x_exp, centre_values.copy(),
                                             0.1 * centre_values, mask)
            self.assertGreater(obj_centre(np.array([1.0])), 1e-3)


class ExpTableHeaderParsingTest(unittest.TestCase):
    """
    pd.read_csv(path, comment='#') with no explicit header= defaults to
    header=0, so for a file with only a '#'-prefixed header line (step 2's
    integrated-cross-section output), it silently took the first DATA row
    as column names and dropped it.
    """

    def test_hash_only_header_keeps_every_data_row(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fh:
            fh.write("# E (MeV),sigma (mb)| dE=0.2\n"
                     "1.0,2.0\n2.0,3.0\n3.0,4.0\n")
            path = fh.name
        try:
            table = topt.read_exp_table(path)
        finally:
            os.remove(path)
        self.assertEqual(len(table), 3)
        np.testing.assert_allclose(table[0], [1.0, 2.0])

    def test_real_integrated_xs_file_is_read_in_full(self):
        repo = Path(__file__).resolve().parent.parent
        path = repo / "22Mg_ap_25Al_integrated_xs_dE_0.2.csv"
        if not path.exists():
            self.skipTest("sample integrated-xs file not present in this checkout")
        with open(path) as fh:
            n_data_lines = sum(1 for ln in fh if not ln.startswith("#"))
        table = topt.read_exp_table(str(path))
        self.assertEqual(len(table), n_data_lines)

    def test_plain_text_header_without_hash_is_still_skipped(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fh:
            fh.write("E,sigma\n1.0,2.0\n2.0,3.0\n")
            path = fh.name
        try:
            table = topt.read_exp_table(path)
        finally:
            os.remove(path)
        self.assertEqual(len(table), 2)

    def test_file_with_no_header_row_keeps_all_rows(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fh:
            fh.write("1.0,2.0\n2.0,3.0\n")
            path = fh.name
        try:
            table = topt.read_exp_table(path)
        finally:
            os.remove(path)
        self.assertEqual(len(table), 2)


class CheckpointTest(unittest.TestCase):
    def test_entries_per_fit_in_one_file(self):
        import json

        names = ["rvadjust_a"]
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "checkpoint.json")
            entry = lambda x: {"params": [x], "param_names": names, "loss": 1.0, "nfev": 3, "timestamp": "t"}
            topt.checkpoint_save(path, "fit_dE_0.2", entry(1.1))
            topt.checkpoint_save(path, "fit_dE_0.5", entry(1.2))
            self.assertEqual(topt.checkpoint_load(path, "fit_dE_0.2", names)["params"], [1.1])
            self.assertEqual(topt.checkpoint_load(path, "fit_dE_0.5", names)["params"], [1.2])
            self.assertIsNone(topt.checkpoint_load(path, "fit_dE_0.5", ["other"]))  # \opt changed
            topt.checkpoint_clear(path, "fit_dE_0.2")
            self.assertIsNone(topt.checkpoint_load(path, "fit_dE_0.2", names))
            self.assertEqual(sorted(json.loads(Path(path).read_text())), ["fit_dE_0.5"])
            topt.checkpoint_clear(path, "fit_dE_0.5")
            self.assertFalse(Path(path).exists())                 # removed when empty

    def test_old_single_fit_checkpoint_is_not_resumed(self):
        import json

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.json"
            path.write_text(json.dumps({"params": [1.4], "loss": 2.0, "nfev": 5, "timestamp": "t"}))
            self.assertIsNone(topt.checkpoint_load(str(path), "fit_dE_0.2", ["rvadjust_a"]))

    def test_failed_fit_does_not_poison_the_next_fit(self):
        """A fit whose entry is never saved (e.g. the run crashed before a
        best was found) must not leave anything another fit in the same
        run directory could accidentally resume from."""
        names = ["rvadjust_a"]
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "checkpoint.json")
            topt.checkpoint_save(path, "fit_dE_0.2",
                                 {"params": [9.9], "param_names": names,
                                  "loss": 1.0, "nfev": 1, "timestamp": "t"})
            # fit_dE_0.5 never wrote an entry (e.g. its best-fit TALYS run
            # failed before any checkpoint was saved).
            self.assertIsNone(topt.checkpoint_load(path, "fit_dE_0.5", names))
            # fit_dE_0.2's own entry is unaffected.
            self.assertEqual(topt.checkpoint_load(path, "fit_dE_0.2", names)["params"], [9.9])


def _opt_block(input_file, run, dE, chi2, param):
    return (
        "=" * 72 + "\n"
        f"Date/Time   : 2026-10-01 00:00:00\nInput file  : {input_file}\n"
        f"Reaction    : 22Mg(α,p)25Al\nBin width   : dE = {dE} MeV\nRun index   : {run}\n"
        + "-" * 72 + "\nOptimizer   : Powell  (converged=True, nfev=10)\n"
        f"Min reduced chi-square: {chi2}\n" + "-" * 72 + "\nOptimized parameters:\n"
        f"  rvadjust_a = {param:+.6g}\n" + "-" * 72 + "\nCross sections  E[MeV]  sigma[mb]:\n"
        "  2.0e+00  1.0e-03\n" + "-" * 72 + "\nReaction rates  T9[GK]  rate[cm3/s/mol]:\n"
        + "".join(f"  {t:.6e}  {1e-5 * t ** 3:.6e}\n" for t in (0.5, 1.0, 1.5, 2.0, 3.0))
        + "=" * 72 + "\n\n"
    )


class FitRecordDedupTest(unittest.TestCase):
    """
    talys_optimization.out is append-only: a re-run or a resumed job appends
    a further block for the same fit instead of replacing the old one. The
    ratio, parameter-grid, and results-plotting scripts must count that fit
    once (its newest block), not once per appended block.
    """

    def test_reruns_are_counted_once(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import plot_talys_param_grid as grid
        import plot_rate_ratio_wide as ratio
        import plot_talys_results_new as results

        a = "out/22Mg_ap_25Al_integrated_xs_dE_0.2_run0.csv"
        b = "out/22Mg_ap_25Al_integrated_xs_dE_0.5_run0.csv"
        text = (_opt_block(a, 0, 0.2, 5.0, 1.1) + _opt_block(b, 0, 0.5, 4.0, 1.2)
                + _opt_block(a, 0, 0.2, 3.0, 1.3))      # re-run of fit a
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "talys_optimization.out"
            path.write_text(text)
            g = grid.parse_talys_optimization(path)
            self.assertEqual(len(g), 2)
            self.assertEqual([r["params"]["rvadjust_a"] for r in g], [1.3, 1.2])  # newest kept
            self.assertEqual(len(ratio.parse_talys_rate_records(path)), 2)
            self.assertEqual(len(results.parse_talys_opt_out(str(path))), 2)


if __name__ == "__main__":
    unittest.main()
