import numpy as np
from .constants import HBAR
from .units import ev_to_j

def reduced_mass(m1, m2):
    return (m1*m2) / (m1 + m2)

def wavenumber_from_E_eV(E_eV, mu):
    E_J = ev_to_j(E_eV)
    return (2.0*mu*E_J)**0.5 / HBAR

def spin_stat_factor(J, s1, s2):
    return (2*J + 1) / ((2*s1 + 1)*(2*s2 + 1))

def energy_grid(center_eV, half_width_eV=25.0, n=200):
    return np.linspace(center_eV - half_width_eV, center_eV + half_width_eV, n)
