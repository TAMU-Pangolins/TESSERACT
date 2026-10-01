import unittest
from pathlib import Path

import numpy as np

from common.densities_retrieval import (
    apply_hfb_corrections,
    read_hfb_cor,
    read_hfb_tab,
)
from nucres.hfb_adapter import _corrected, build_density_grid

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "densities" / "level-densities-hfb"
TAB_PATH = DATA_ROOT / "z009.tab"
COR_PATH = DATA_ROOT / "z009.cor"


class ReadHfbCorTest(unittest.TestCase):
    def test_parses_real_ripl3_file(self):
        if not COR_PATH.exists():
            self.skipTest("HFB .cor table not available.")
        cor = read_hfb_cor(str(COR_PATH))
        # z009.cor: "9  17   2  18     0.00000    -0.03483 ... 17F"
        self.assertIn((9, 17), cor)
        c, delta = cor[(9, 17)]
        self.assertAlmostEqual(c, 0.0)
        self.assertAlmostEqual(delta, -0.03483)

    def test_missing_file_gives_empty_dict(self):
        self.assertEqual(read_hfb_cor("/no/such/file.cor"), {})


class ApplyHfbCorrectionsTest(unittest.TestCase):
    def test_raises_for_isotope_with_no_entry(self):
        if not (TAB_PATH.exists() and COR_PATH.exists()):
            self.skipTest("HFB tables not available.")
        rec = read_hfb_tab(str(TAB_PATH))  # single-isotope file: z=9, a=17
        cor = {(9, 999): (0.1, 0.1)}  # no entry for the actual (Z, A)
        with self.assertRaises(KeyError):
            apply_hfb_corrections(rec, cor)

    def test_renormalises_density_for_real_isotope(self):
        if not (TAB_PATH.exists() and COR_PATH.exists()):
            self.skipTest("HFB tables not available.")
        rec = read_hfb_tab(str(TAB_PATH))
        cor = read_hfb_cor(str(COR_PATH))
        self.assertIn((rec.header.Z, rec.header.A), cor)
        corrected = apply_hfb_corrections(rec, cor)
        # ptable != 0 here, so the renormalised density must differ from the
        # raw table (shifted in U and/or rescaled), not just be a no-op copy.
        self.assertFalse(np.allclose(corrected.positive.Rho_level, rec.positive.Rho_level))
        # Header and parity bookkeeping are untouched by the correction.
        self.assertEqual(corrected.header, rec.header)
        self.assertEqual(corrected.positive.parity, rec.positive.parity)


class CorrectedHelperTest(unittest.TestCase):
    def test_falls_back_when_isotope_has_no_entry(self):
        if not TAB_PATH.exists():
            self.skipTest("HFB tables not available.")
        rec = read_hfb_tab(str(TAB_PATH))
        # No .cor file at all: should return the record unchanged, not raise.
        out = _corrected(rec, None)
        self.assertIs(out, rec)


class BuildDensityGridCorrectionsTest(unittest.TestCase):
    def test_use_corrections_changes_the_grid_for_a_real_isotope(self):
        if not (TAB_PATH.exists() and COR_PATH.exists()):
            self.skipTest("HFB tables not available.")
        kwargs = dict(Z=9, A=17, J_phys=0.5, pi=1, data_root=DATA_ROOT,
                      E_min_mev=0.5, E_max_mev=3.0, n_points=50)
        _, rho_raw = build_density_grid(**kwargs, use_corrections=False)
        _, rho_corrected = build_density_grid(**kwargs, use_corrections=True)
        self.assertGreater(np.max(np.abs(rho_corrected - rho_raw)), 0.0)

    def test_use_corrections_is_a_no_op_without_a_cor_file(self):
        if not TAB_PATH.exists():
            self.skipTest("HFB tables not available.")
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # Copy only the .tab file: no sibling .cor exists in this directory,
            # so build_density_grid's tab_path override must find cor_p=None
            # and fall through to the uncorrected table even with
            # use_corrections=True (via _corrected's cor_path=None path).
            tab_copy = Path(d) / "z009.tab"
            shutil.copyfile(TAB_PATH, tab_copy)
            kwargs = dict(tab_path=tab_copy, A=17, J_phys=0.5, pi=1,
                          E_min_mev=0.5, E_max_mev=3.0, n_points=50)
            _, rho_raw = build_density_grid(**kwargs, use_corrections=False)
            _, rho_same = build_density_grid(**kwargs, use_corrections=True)
            np.testing.assert_array_equal(rho_same, rho_raw)


if __name__ == "__main__":
    unittest.main()
