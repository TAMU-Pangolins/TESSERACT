from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import mpmath as mp
import numpy as np

from .physics import (
    M_TO_BARNS,
    PI,
    ev_to_j,
    reduced_mass,
    spin_stat_factor,
    wavenumber_from_E_eV,
)


@dataclass(frozen=True)
class Resonance:
    """Single resonance parameterization used by the Breit-Wigner helpers."""

    E_r: float  # eV
    J: float
    s1: float
    s2: float
    m1: float  # kg
    m2: float  # kg
    Gamma_i: float  # eV at E_r (set to >0 if known; else set 0 and use gamma2 in energy-dependent call)
    Gamma_o: float  # eV
    L1: Optional[int] = None
    L2: Optional[int] = None
    L3: Optional[int] = None


def channel_radius_fm(A1, A2, r0=1.25):
    r"""Return the channel radius in fm using \(a = r_0 (A_1^{1/3} + A_2^{1/3})\)."""
    return r0 * (A1 ** (1 / 3) + A2 ** (1 / 3))


def penetrability_P_l_mev(l, Z1, Z2, A1, A2, E_mev, r0=1.25):
    r"""Evaluate the Coulomb penetrability \(P_\ell(E)\) at energy `E_mev`."""
    if E_mev <= 0.0:
        return 0.0
    a_fm = channel_radius_fm(A1, A2, r0)
    mu_amu = (A1 * A2) / (A1 + A2)
    rho = 0.218735 * a_fm * (mu_amu * E_mev) ** 0.5
    eta = 0.157489 * (Z1 * Z2) * (mu_amu / E_mev) ** 0.5
    F = float(mp.coulombf(l, eta, rho))
    G = float(mp.coulombg(l, eta, rho))
    d = F * F + G * G
    return float(rho / d) if d != 0.0 else 0.0


@lru_cache(maxsize=128)
def _P_grid_cached(l, Z1, Z2, A1, A2, r0, Emin_mev, Emax_mev, npts):
    Es = np.linspace(Emin_mev, Emax_mev, int(npts))
    Ps = np.array(
        [penetrability_P_l_mev(l, Z1, Z2, A1, A2, float(e), r0) for e in Es],
        dtype=float,
    )
    return Es, Ps


def make_penetrability_interp(
    l, Z1, Z2, A1, A2, r0=1.25, Emin_mev=1e-6, Emax_mev=5.0, npts=600
):
    r"""
    Precompute \(P_\ell(E)\) on a grid \([E_{\min}, E_{\max}]\) in MeV and
    return a fast interpolator \(P(E)\).

    First creation is the only slow step; subsequent uses are very fast.
    """
    Es, Ps = _P_grid_cached(
        l, Z1, Z2, A1, A2, float(r0), float(Emin_mev), float(Emax_mev), int(npts)
    )

    def P(E_mev):
        e = np.asarray(E_mev, dtype=float)
        return np.interp(e, Es, Ps, left=Ps[0], right=Ps[-1])

    return P


def sigma_bw_constant(E_eV, r):
    r"""
    Evaluate a single-level Breit-Wigner cross section with constant widths.

    The model is

    \[
    \sigma(E) = \omega \frac{\pi}{k^2}
    \frac{\Gamma_i \Gamma_o}{(E - E_r)^2 + (\Gamma_t/2)^2}
    \]

    where \(\Gamma_t = \Gamma_i + \Gamma_o\). Returns barns. `E_eV` may be a
    scalar or array.
    """
    E_eV = np.asarray(E_eV, dtype=float)
    mu = reduced_mass(r.m1, r.m2)
    k = wavenumber_from_E_eV(E_eV, mu)
    S = spin_stat_factor(r.J, r.s1, r.s2)

    EiJ = ev_to_j(r.Gamma_i)
    EoJ = ev_to_j(r.Gamma_o)
    EtJ = EiJ + EoJ
    EJ = ev_to_j(E_eV)
    ErJ = ev_to_j(r.E_r)

    BW = (EiJ * EoJ) / ((EJ - ErJ) ** 2 + (EtJ / 2.0) ** 2)
    return S * (PI / k**2) * BW * M_TO_BARNS


def sigma_bw_energy_dep(E_eV, r, Z1, Z2, A1, A2, l, gamma2, r0=1.25, P_interp=None):
    r"""
    Evaluate a Breit-Wigner cross section with energy-dependent entrance width.

    The entrance width is scaled as

    \[
    \Gamma_i(E) = \Gamma_i(E_r)\frac{P_\ell(E)}{P_\ell(E_r)}
    \]

    If `r.Gamma_i <= 0`, the resonance-energy width is inferred from

    \[
    \Gamma_i(E_r) = 2 \gamma^2 P_\ell(E_r)
    \]

    and `gamma2` is interpreted in eV. Returns barns.
    """
    E_eV = np.asarray(E_eV, dtype=float)
    mu = reduced_mass(r.m1, r.m2)
    k = wavenumber_from_E_eV(E_eV, mu)
    S = spin_stat_factor(r.J, r.s1, r.s2)

    E_mev = np.maximum(E_eV, 0.0) * 1e-6
    Er_mev = max(r.E_r, 0.0) * 1e-6

    # fast path with interpolator
    if P_interp is None:
        P_E = np.array(
            [penetrability_P_l_mev(l, Z1, Z2, A1, A2, ee, r0) for ee in E_mev],
            dtype=float,
        )
        P_Er = penetrability_P_l_mev(l, Z1, Z2, A1, A2, Er_mev, r0)
    else:
        P_E = P_interp(E_mev)
        P_Er = float(P_interp(Er_mev))

    # if (r.Gamma_i is not None) and (r.Gamma_i > 0.0):
    #    Gamma_i_Er_eV = r.Gamma_i
    # else:
    Gamma_i_Er_eV = 2.0 * gamma2 * P_Er  # eV

    Gamma_i_E_eV = Gamma_i_Er_eV * (P_E / P_Er) if P_Er != 0.0 else np.zeros_like(P_E)

    EiJ = ev_to_j(Gamma_i_E_eV)
    EoJ = ev_to_j(r.Gamma_o)
    EtJ = EiJ + EoJ
    EJ = ev_to_j(E_eV)
    ErJ = ev_to_j(r.E_r)

    BW = (EiJ * EoJ) / ((EJ - ErJ) ** 2 + (EtJ / 2.0) ** 2)
    return S * (PI / k**2) * BW * M_TO_BARNS
