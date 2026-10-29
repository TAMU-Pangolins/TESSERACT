"""Verify fit records describe the selected parameters and binned predictions."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import talys_opt_with_unc as optimizer


class TalysFitRecordTest(unittest.TestCase):
    def configuration(self, prior='0.20'):
        return {'opt_params': [SimpleNamespace(name='parameter', x0=1.0, lo=0.0, hi=2.0)],
                'script': {'exp_rel_err': '0.10', 'prior_rel_std': prior,
                           'debug_every': '0', 'exp_file': 'experiment.csv'}}

    def test_statistics_follow_selected_parameters_instead_of_a_better_probe(self):
        configuration = self.configuration()
        energies = np.linspace(2.0, 2.6, 4)
        observations = np.full(4, 10.0)
        predictions = np.full(4, 5.0)
        with patch.object(optimizer, 'run_talys_get_xs',
                          side_effect=[(energies, observations), (energies, predictions)]) as native:
            objective = optimizer.make_objective(configuration, energies, observations,
                                                  np.ones(4), np.ones(4, dtype=bool))
            self.assertAlmostEqual(objective(np.array([1.0])), 0.0)
            selected_total = objective(np.array([1.5]))
            statistics = objective.statistics(np.array([1.5]), predictions)
        self.assertEqual(native.call_count, 2)
        self.assertEqual(objective.state['best'][0], 0.0)
        self.assertAlmostEqual(statistics['chi2'], 4 * (np.log(2) / 0.1) ** 2)
        self.assertAlmostEqual(statistics['prior'], (0.5 / 0.2) ** 2)
        self.assertAlmostEqual(statistics['objective'], selected_total)
        self.assertAlmostEqual(statistics['chi2_red'], statistics['chi2'] / 3)

    def test_statistics_average_the_same_bin_samples_as_the_objective(self):
        configuration = self.configuration(prior='0')
        configuration['_bin_weights'] = np.array([0.5, 0.5])
        objective = optimizer.make_objective(configuration, np.array([1.0, 2.0]),
                                              np.array([2.0, 6.0]), np.ones(2),
                                              np.ones(2, dtype=bool))
        statistics = objective.statistics(np.array([1.0]), np.array([1.0, 3.0, 5.0, 7.0]))
        self.assertAlmostEqual(statistics['objective'], 0.0)
        with self.assertRaisesRegex(ValueError, 'bin grid'):
            objective.statistics(np.array([1.0]), np.array([1.0]))

    def test_saved_results_preserve_predictions_and_diagnostics_for_replotting(self):
        configuration = self.configuration()
        energy = np.array([1.0, 2.0])
        predictions = np.array([3.0, 4.0])
        info = {'objective': 7.0, 'chi2': 6.0, 'prior': 1.0, 'chi2_red': 6.0,
                'dof': 1, 'e_fit_min': 1.5, 'bin_width': 0.2, 'talys_points_per_bin': 3}
        with tempfile.TemporaryDirectory() as directory:
            optimizer.save_results(configuration, energy, predictions, np.ones(2),
                                   energy, predictions, energy, predictions, np.array([1.0]),
                                   out_dir=directory, y_bin=predictions, fit_info=info)
            path = Path(directory) / 'talys_results_experiment.npz'
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['y_xs_binned'], predictions)
                self.assertEqual(float(saved['objective']), 7.0)
                self.assertEqual(float(saved['bin_width']), 0.2)
            with patch.object(optimizer, 'load_config', return_value=configuration), \
                 patch.object(optimizer, 'plot_best_fit_xs') as plot, \
                 patch.object(optimizer, 'plot_reaction_rate'), \
                 patch.object(optimizer, 'read_ratesmc_file', return_value=None):
                optimizer.replot(str(path))
            np.testing.assert_array_equal(plot.call_args.kwargs['y_bin'], predictions)
            self.assertEqual(plot.call_args.kwargs['e_fit_min'], 1.5)
