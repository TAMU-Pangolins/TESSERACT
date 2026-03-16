from __future__ import annotations

import numpy as np

# Physical constants and simple unit conversions plus kinematic helpers.

HBAR = 1.054571817e-34  # J*s
EV_TO_J = 1.602176634e-19  # J / eV
M_TO_BARNS = 1e28  # barns / m^2
BARNS_TO_M2 = 1e-28  # m^2 / barn
MASS_PROTON = 1.672621898e-27  # kg
PI = 3.141592653589793
BOLTZMANN = 1.380649e-23  # J / K
AVOGADRO = 6.02214076e23  # 1 / mol


def ev_to_j(x):
    """Convert energies from eV to J."""
    return np.asarray(x, dtype=float) * EV_TO_J


def reduced_mass(m1, m2):
    r"""Return the reduced mass \(\mu = m_1 m_2 / (m_1 + m_2)\) in kg."""
    return (m1 * m2) / (m1 + m2)


def wavenumber_from_E_eV(E_eV, mu):
    r"""Return the wave number \(k = \sqrt{2 \mu E} / \hbar\) for energies in eV."""
    E_J = ev_to_j(E_eV)
    return (2.0 * mu * E_J) ** 0.5 / HBAR


def spin_stat_factor(J, s1, s2):
    r"""Return the statistical factor \((2J + 1) / [(2s_1 + 1)(2s_2 + 1)]\)."""
    return (2 * J + 1) / ((2 * s1 + 1) * (2 * s2 + 1))


def energy_grid(center_eV, half_width_eV=25.0, n=200):
    """Build a uniformly spaced energy grid centered on `center_eV`."""
    return np.linspace(center_eV - half_width_eV, center_eV + half_width_eV, n)
