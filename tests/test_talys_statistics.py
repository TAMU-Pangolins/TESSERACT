import unittest
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


if __name__ == "__main__":
    unittest.main()
