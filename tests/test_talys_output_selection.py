"""Verify that the optimizer reads outputs for the configured TALYS channel."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import talys_opt_with_unc as optimizer


class TalysOutputSelectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_table(self, filename, values=(10.0, 20.0)):
        np.savetxt(self.root / filename, np.column_stack(([2.0, 4.0], values)))

    def configuration(self, projectile='a', ejectile='p', **settings):
        return {'proj': projectile, 'ejec': ejectile, 'script': settings,
                'tar': 'Mg', 'A_tar': 22, 'talys_fixed': {}, 'opt_params': [],
                '_x_exp': np.array([1.0, 2.0]), '_exp_to_lab': 2.0}

    def test_candidates_follow_particle_identity(self):
        for projectile, ejectile, expected in [
            ('a', 'p', ['ap.tot', 'xs010000.tot']),
            ('p', 'g', ['pg.tot', 'xs000000.tot']),
            ('n', 'p', ['np.tot', 'xs010000.tot']),
            ('p', 'a', ['pa.tot', 'xs000001.tot']),
            ('a', 'h', ['ah.tot', 'xs000010.tot']),
        ]:
            with self.subTest(projectile=projectile, ejectile=ejectile):
                cfg = self.configuration(projectile, ejectile)
                self.assertEqual(optimizer.talys_cross_section_candidates(cfg), expected)

    def test_explicit_filename_precedes_standard_outputs(self):
        cfg = self.configuration(talys_xs_file='selected.tot')
        self.write_table('selected.tot', [30.0, 40.0])
        self.write_table('ap.tot')
        _, cross_section = optimizer.read_ap_tot(str(self.root), cfg)
        np.testing.assert_array_equal(cross_section, [30.0, 40.0])

    def test_reaction_filename_precedes_exclusive_fallback(self):
        self.write_table('pg.tot', [30.0, 40.0])
        self.write_table('xs000000.tot')
        _, cross_section = optimizer.read_ap_tot(str(self.root), self.configuration('p', 'g'))
        np.testing.assert_array_equal(cross_section, [30.0, 40.0])

    def test_missing_override_reports_the_channel_fallback(self):
        self.write_table('ap.tot')
        with contextlib.redirect_stdout(io.StringIO()) as diagnostic:
            _, cross_section = optimizer.read_ap_tot(
                str(self.root), self.configuration(talys_xs_file='absent.tot'))
        np.testing.assert_array_equal(cross_section, [10.0, 20.0])
        self.assertIn('absent.tot', diagnostic.getvalue())
        self.assertIn('ap.tot', diagnostic.getvalue())

    def test_unknown_ejectile_is_not_treated_as_radiative_capture(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported TALYS ejectile'):
            optimizer.talys_cross_section_candidates(self.configuration(ejectile='C'))

    def test_exclusive_only_outputs_are_read_for_each_channel(self):
        for projectile, ejectile, filename in [
            ('a', 'p', 'xs010000.tot'), ('p', 'g', 'xs000000.tot'),
            ('p', 'a', 'xs000001.tot'),
        ]:
            with self.subTest(ejectile=ejectile):
                self.write_table(filename)
                energy, cross_section = optimizer.read_ap_tot(
                    str(self.root), self.configuration(projectile, ejectile))
                np.testing.assert_array_equal(energy, [2.0, 4.0])
                np.testing.assert_array_equal(cross_section, [10.0, 20.0])
                (self.root / filename).unlink()

    def test_missing_channel_does_not_read_unrelated_alpha_proton_output(self):
        self.write_table('ap.tot')
        with contextlib.redirect_stdout(io.StringIO()) as diagnostic:
            result = optimizer.read_ap_tot(str(self.root), self.configuration('p', 'g'))
        self.assertIsNone(result)
        self.assertIn('pg.tot', diagnostic.getvalue())
        self.assertIn('xs000000.tot', diagnostic.getvalue())

    def test_invalid_preferred_output_does_not_silently_use_fallback(self):
        (self.root / 'pg.tot').write_text('not a numeric table\n')
        self.write_table('xs000000.tot')
        with contextlib.redirect_stdout(io.StringIO()) as diagnostic:
            result = optimizer.read_ap_tot(str(self.root), self.configuration('p', 'g'))
        self.assertIsNone(result)
        self.assertIn('pg.tot', diagnostic.getvalue())

    def test_rates_follow_ejectile_when_proton_output_also_exists(self):
        self.write_table('astrorate.p', [999.0, 999.0])
        for ejectile in ['g', 'a']:
            with self.subTest(ejectile=ejectile):
                self.write_table(f'astrorate.{ejectile}')
                temperature, rate = optimizer.read_astrorate(
                    str(self.root), self.configuration('p', ejectile))
                np.testing.assert_array_equal(temperature, [2.0, 4.0])
                np.testing.assert_array_equal(rate, [10.0, 20.0])

    def test_missing_rate_does_not_read_proton_output(self):
        self.write_table('astrorate.p')
        with contextlib.redirect_stdout(io.StringIO()) as diagnostic:
            result = optimizer.read_astrorate(str(self.root), self.configuration('p', 'g'))
        self.assertIsNone(result)
        self.assertIn('astrorate.g', diagnostic.getvalue())

    def test_nonfinite_values_are_rejected_by_both_readers(self):
        for value in [np.nan, np.inf]:
            with self.subTest(value=value), contextlib.redirect_stdout(io.StringIO()):
                self.write_table('pg.tot', [value, 1.0])
                self.write_table('astrorate.g', [value, 1.0])
                cfg = self.configuration('p', 'g')
                self.assertIsNone(optimizer.read_ap_tot(str(self.root), cfg))
                self.assertIsNone(optimizer.read_astrorate(str(self.root), cfg))

    def test_no_configuration_preserves_alpha_proton_readers(self):
        self.write_table('ap.tot')
        self.write_table('astrorate.p')
        for reader in [optimizer.read_ap_tot, optimizer.read_astrorate]:
            with self.subTest(reader=reader.__name__):
                _, values = reader(str(self.root))
                np.testing.assert_array_equal(values, [10.0, 20.0])

    def test_config_recognizes_gamma_and_helium_three_and_consumes_override(self):
        for ejectile_charge, ejectile_mass, expected in [(0, 0, 'g'), (2, 3, 'h')]:
            with self.subTest(ejectile=expected):
                configuration = self.root / 'tesseract.in'
                configuration.write_text(
                    '[basics]\nprojectile_Z = 1\nprojectile_A = 1\n'
                    'target_Z = 12\ntarget_A = 22\n'
                    f'ejectile_Z = {ejectile_charge}\nejectile_A = {ejectile_mass}\n'
                    'residual_Z = 13\nresidual_A = 23\n[talys]\n'
                    'talys_xs_file = selected.tot\n')
                cfg = optimizer.load_config(str(configuration))
                self.assertEqual(cfg['ejec'], expected)
                self.assertEqual(cfg['script']['talys_xs_file'], 'selected.tot')
                self.assertNotIn('talys_xs_file', cfg['talys_fixed'])

    def test_cross_section_execution_selects_channel_and_preserves_energy_frame(self):
        def execute(workdir, input_path):
            energy = np.loadtxt(Path(workdir) / 'energies.txt')
            np.testing.assert_array_equal(energy, [2.0, 4.0])
            np.savetxt(Path(workdir) / 'xs000000.tot', np.column_stack((energy, [10.0, 20.0])))
            return True

        with patch.object(optimizer, 'run_talys', side_effect=execute):
            energy, cross_section = optimizer.run_talys_get_xs(
                np.array([]), self.configuration('p', 'g'))
        np.testing.assert_array_equal(energy, [1.0, 2.0])
        np.testing.assert_array_equal(cross_section, [10.0, 20.0])

    def test_rate_execution_selects_channel_and_converts_only_cross_section_energy(self):
        def execute(workdir, input_path):
            table = np.column_stack(([2.0, 4.0], [10.0, 20.0]))
            np.savetxt(Path(workdir) / 'astrorate.g', table)
            np.savetxt(Path(workdir) / 'xs000000.tot', table)
            np.savetxt(Path(workdir) / 'astrorate.p', table * 100.0)
            return True

        with patch.object(optimizer, 'run_talys', side_effect=execute):
            rate, cross_section = optimizer.run_talys_get_rate(
                np.array([]), self.configuration('p', 'g'))
        np.testing.assert_array_equal(rate[0], [2.0, 4.0])
        np.testing.assert_array_equal(rate[1], [10.0, 20.0])
        np.testing.assert_array_equal(cross_section[0], [1.0, 2.0])


if __name__ == '__main__':
    unittest.main()
