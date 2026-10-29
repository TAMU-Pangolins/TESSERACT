"""Verify template conversion preserves physical values and rejects ambiguity."""

from pathlib import Path
import tempfile
import unittest

from scripts.convert_ratesmc_templates import convert


class RatesMCTemplateConversionTest(unittest.TestCase):
    def template(self, directory, row, name='example.txt'):
        path = Path(directory) / name
        path.write_text(
            'Example reaction\n0 ! rate output at all temperatures\n'
            'Resonant Contribution\n'
            'Ecm DEcm wg Dwg J G1 DG1 L1 G2 DG2 L2 G3 DG3 L3 Exf Int\n'
            + row + '\n***************************************************\n')
        return path

    def test_conversion_adds_correlation_without_changing_resonance_values(self):
        row = '100 1 0 0 0.5 2 0.1 0 3 0.2 1 0 0 0 0 1'
        with tempfile.TemporaryDirectory() as directory:
            path = self.template(directory, row)
            original = path.read_bytes()
            lines, messages = convert(path)
            self.assertIsNotNone(lines, messages)
            self.assertEqual(path.read_bytes(), original)
            self.assertIn(row + ' 0', lines)
            self.assertTrue(any('Corr/Frac' in line for line in lines))
            path.write_text('\n'.join(lines) + '\n')
            self.assertEqual(convert(path), (None, ['already in the current format']))

    def test_unambiguous_exponent_split_is_repaired(self):
        row = '100 1 0 0 0.5 2 e-3 0.1 0 3 0.2 1 0 0 0 0 1'
        with tempfile.TemporaryDirectory() as directory:
            lines, messages = convert(self.template(directory, row))
        self.assertIsNotNone(lines, messages)
        self.assertTrue(any('2e-3' in line for line in lines))

    def test_ambiguous_extra_integer_is_rejected_without_modifying_input(self):
        row = '100 1 0 0 0.5 2 0.1 0 3 0.2 1 0 0 0 13 69 1'
        with tempfile.TemporaryDirectory() as directory:
            path = self.template(directory, row)
            original = path.read_bytes()
            lines, messages = convert(path)
            self.assertIsNone(lines)
            self.assertTrue(any('UNREPAIRED' in message for message in messages))
            self.assertEqual(path.read_bytes(), original)

    def test_documented_magnesium_final_state_repair_preserves_1369_kev(self):
        row = '2653.1 1.0 0.0 0.0 0 2.4e+3 500.0 2 0.13 0.02 2 0.046 0.010 0 136 9.0 1'
        with tempfile.TemporaryDirectory() as directory:
            lines, messages = convert(self.template(directory, row, '20Ne(a,g)24Mg.txt'))
        self.assertIsNotNone(lines, messages)
        values = next(line.split() for line in lines if line.startswith('2653.1 '))
        self.assertEqual(float(values[14]), 1369.0)
        self.assertEqual(values[15:], ['1', '0'])

    def test_all_repository_templates_are_current_and_idempotent(self):
        directory = Path(__file__).resolve().parent.parent / 'input'
        paths = sorted(directory.glob('*.txt'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(template=path.name):
                self.assertEqual(convert(path), (None, ['already in the current format']))
