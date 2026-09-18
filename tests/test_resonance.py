import unittest

import numpy as np

from nucres.physics import M_TO_BARNS, MASS_PROTON, PI, spin_stat_factor, wavenumber_from_E_eV
from nucres.resonance import (
    Resonance,
    effective_alpha_potential_mev,
    finite_size_coulomb_mev,
    jwkb_log_transmission_mev,
    make_jwkb_log_transmission_interp,
    make_penetrability_interp,
    sigma_bw_constant,
    sigma_bw_energy_dep,
)


class ResonanceModelTest(unittest.TestCase):
    def test_sigma_bw_constant_golden_values(self):
        r = Resonance(
            E_r=1000.0,
            J=0.5,
            s1=0.5,
            s2=0.5,
            m1=MASS_PROTON,
            m2=MASS_PROTON,
            Gamma_i=1.0,
            Gamma_o=1.0,
        )
        energies = np.array([900.0, 1000.0, 1100.0])
        expected = np.array([7.24232633e-02, 6.51874551e02, 5.92553973e-02])
        sigma = sigma_bw_constant(energies, r)
        np.testing.assert_allclose(sigma, expected, rtol=1e-9, atol=0.0)

    def test_sigma_bw_constant_peaks_near_er(self):
        r = Resonance(
            E_r=1000.0,
            J=0.5,
            s1=0.5,
            s2=0.5,
            m1=MASS_PROTON,
            m2=MASS_PROTON,
            Gamma_i=1.0,
            Gamma_o=1.0,
        )
        e_center = np.array([r.E_r])
        e_off = np.array([r.E_r + 10000.0])
        sigma_center = sigma_bw_constant(e_center, r)[0]
        sigma_off = sigma_bw_constant(e_off, r)[0]
        self.assertTrue(np.isfinite(sigma_center))
        self.assertTrue(np.isfinite(sigma_off))
        self.assertGreater(sigma_center, sigma_off)

    def test_sigma_bw_energy_dep_nonnegative(self):
        r = Resonance(
            E_r=2.0e5,  # 0.2 MeV
            J=0.5,
            s1=0.5,
            s2=0.5,
            m1=MASS_PROTON,
            m2=MASS_PROTON,
            Gamma_i=0.0,
            Gamma_o=1.0,
        )
        P_interp = make_penetrability_interp(
            l=0, Z1=1, Z2=1, A1=1, A2=1, Emin_mev=0.05, Emax_mev=0.5, npts=50
        )
        energies = np.array([1.0e5, 2.0e5, 3.0e5])
        sigma = sigma_bw_energy_dep(
            energies,
            r,
            Z1=1,
            Z2=1,
            A1=1,
            A2=1,
            l=0,
            Gamma_i_Er_eV=1e-3,
            P_interp=P_interp,
        )
        self.assertTrue(np.all(np.isfinite(sigma)))
        self.assertTrue(np.all(sigma >= 0.0))

    def test_finite_size_coulomb_matches_at_radius(self):
        Rc = 5.0
        value = finite_size_coulomb_mev(np.array([Rc]), Z1=2, Z2=10, Rc_fm=Rc)[0]
        expected = 2 * 10 * 1.43996448 / Rc
        self.assertAlmostEqual(value, expected)

    def test_effective_alpha_potential_is_finite(self):
        r = np.linspace(0.1, 40.0, 200)
        v = effective_alpha_potential_mev(
            r, E_mev=1.0, l=0, Z1=2, Z2=10, A1=4, A2=20
        )
        self.assertTrue(np.all(np.isfinite(v)))

    def test_jwkb_log_transmission_bounds(self):
        logT = jwkb_log_transmission_mev(
            E_mev=1.0, l=0, Z1=2, Z2=10, A1=4, A2=20, npts=600
        )
        self.assertTrue(np.isfinite(logT))
        self.assertLessEqual(logT, 0.0)

    def test_jwkb_energy_dependent_width_preserves_er_normalization(self):
        r = Resonance(
            E_r=1.0e6,
            J=0.5,
            s1=0.0,
            s2=0.0,
            m1=4.0 * MASS_PROTON,
            m2=20.0 * MASS_PROTON,
            Gamma_i=2.0e-3,
            Gamma_o=1.0,
        )
        logT_interp = make_jwkb_log_transmission_interp(
            l=0,
            Z1=2,
            Z2=10,
            A1=4,
            A2=20,
            Emin_mev=0.8,
            Emax_mev=1.2,
            npts=12,
            radial_npts=500,
        )
        energy = np.array([r.E_r])
        sigma_jwkb = sigma_bw_energy_dep(
            energy,
            r,
            Z1=2,
            Z2=10,
            A1=4,
            A2=20,
            l=0,
            Gamma_i_Er_eV=r.Gamma_i,
            penetrability_model="jwkb_real_omp",
            logT_interp=logT_interp,
        )
        sigma_const = sigma_bw_constant(energy, r)
        np.testing.assert_allclose(sigma_jwkb, sigma_const, rtol=1e-12)

    def test_sigma_bw_constant_integrated_area_matches_narrow_resonance_limit(self):
        r = Resonance(
            E_r=1.0e6,
            J=1.0,
            s1=0.0,
            s2=0.0,
            m1=MASS_PROTON,
            m2=MASS_PROTON,
            Gamma_i=4.0,
            Gamma_o=6.0,
        )
        energies = np.linspace(r.E_r - 5000.0, r.E_r + 5000.0, 200001)
        sigma = sigma_bw_constant(energies, r)
        numerical_area = np.trapezoid(sigma, energies)

        gamma_total = r.Gamma_i + r.Gamma_o
        k_r = wavenumber_from_E_eV(r.E_r, MASS_PROTON / 2.0)
        analytic_area = (
            spin_stat_factor(r.J, r.s1, r.s2)
            * (PI / k_r**2)
            * M_TO_BARNS
            * (2.0 * PI * r.Gamma_i * r.Gamma_o / gamma_total)
        )
        np.testing.assert_allclose(numerical_area, analytic_area, rtol=3e-3)


if __name__ == "__main__":
    unittest.main()
