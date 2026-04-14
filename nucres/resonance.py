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
    r"""
    Single-resonance parameterization used by the Breit-Wigner helpers.

    Attributes
    ----------
    E_r : float
        Resonance energy in eV.
    J : float
        Resonance spin.
    s1, s2 : float
        Projectile and target spins used in the statistical factor.
    m1, m2 : float
        Projectile and target masses in kg.
    Gamma_i : float
        Entrance partial width in eV, referenced at `E_r`.
        For `sigma_bw_energy_dep`, a nonpositive value means the entrance width
        should be inferred from `gamma2` and penetrability.
    Gamma_o : float
        Exit partial width in eV.
    L1, L2, L3 : int or None
        Optional orbital angular momenta for downstream serialization or analysis.
    """

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
    r"""
    Return the channel radius in fm.

    Parameters
    ----------
    A1, A2 : float
        Projectile and target mass numbers.
    r0 : float, default=1.25
        Radius coefficient in fm.

    Returns
    -------
    float
        Channel radius

        \[
        a = r_0 \left(A_1^{1/3} + A_2^{1/3}\right).
        \]
    """
    return r0 * (A1 ** (1 / 3) + A2 ** (1 / 3))


def penetrability_P_l_mev(l, Z1, Z2, A1, A2, E_mev, r0=1.25):
    r"""
    Evaluate the Coulomb penetrability \(P_\ell(E)\).

    Parameters
    ----------
    l : int
        Orbital angular momentum.
    Z1, Z2 : int
        Projectile and target proton numbers.
    A1, A2 : float
        Projectile and target mass numbers.
    E_mev : float
        Center-of-mass energy in MeV.
    r0 : float, default=1.25
        Radius coefficient in fm.

    Returns
    -------
    float
        Penetrability evaluated at `E_mev`. Nonpositive energies return `0.0`.
    """
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

    Parameters
    ----------
    l : int
        Orbital angular momentum.
    Z1, Z2 : int
        Projectile and target proton numbers.
    A1, A2 : float
        Projectile and target mass numbers.
    r0 : float, default=1.25
        Radius coefficient in fm.
    Emin_mev, Emax_mev : float
        Interpolation range in MeV.
    npts : int, default=600
        Number of cached samples.

    Returns
    -------
    callable
        Function `P(E_mev)` that interpolates the cached penetrability grid.

    Notes
    -----
    The first construction is the expensive step. Repeated calls with identical
    arguments reuse the cached grid.
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

    where \(\Gamma_t = \Gamma_i + \Gamma_o\).

    Parameters
    ----------
    E_eV : float or array-like
        Center-of-mass energy grid in eV.
    r : Resonance
        Resonance parameters. Widths are interpreted in eV.

    Returns
    -------
    numpy.ndarray
        Cross section in barns, broadcast over `E_eV`.

    Notes
    -----
    This routine assumes constant entrance and exit widths and does not guard
    against nonphysical resonance parameters such as zero reduced mass.
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


def sigma_bw_energy_dep(E_eV, r, Z1, Z2, A1, A2, l, Gamma_i_Er_eV, r0=1.25, P_interp=None):
    r"""
    Evaluate a Breit-Wigner cross section with energy-dependent entrance width.

    The entrance width is scaled as

    \[
    \Gamma_i(E) = \Gamma_i(E_r)\frac{P_\ell(E)}{P_\ell(E_r)}
    \]

    If `r.Gamma_i <= 0`, the resonance-energy width is inferred from

    \[
    \Gamma_i(E_r) = 2 \gamma^2 P_\ell(E_r) (precomputed from build_ratesmc_input.py)
    \]

    and `gamma' is sampled from the Porter-Thomas distribution.

    Parameters
    ----------
    E_eV : float or array-like
        Center-of-mass energy grid in eV.
    r : Resonance
        Resonance parameters. `Gamma_o` is always read from `r`.
    Z1, Z2 : int
        Projectile and target proton numbers.
    A1, A2 : float
        Projectile and target mass numbers.
    l : int
        Entrance-channel orbital angular momentum.
    Gamma_i_Er_eV : float
        Partial width parameter for the entrance channel.
    r0 : float, default=1.25
        Radius coefficient in fm.
    P_interp : callable, optional
        Precomputed interpolator from `make_penetrability_interp`.

    Returns
    -------
    numpy.ndarray
        Cross section in barns, broadcast over `E_eV`.

    Notes
    -----
    If `P_\ell(E_r) = 0`, the entrance width is forced to zero across the grid.
    This function assumes center-of-mass energies and does not validate the
    resonance masses beyond what the algebra requires.
    """
    E_eV = np.asarray(E_eV, dtype=float)
    mu = reduced_mass(r.m1, r.m2)
    k = wavenumber_from_E_eV(E_eV, mu)
    S = spin_stat_factor(r.J, r.s1, r.s2)

    E_mev = np.maximum(E_eV, 0.0) * 1e-6
    Er_mev = max(r.E_r, 0.0) * 1e-6

    # For array inputs auto-build a cached interpolator to avoid per-point mpmath
    # calls; scalar inputs fall back to exact mpmath; caller may supply P_interp.
    if P_interp is not None:
        P_E  = P_interp(E_mev)
        P_Er = float(P_interp(Er_mev))
    elif E_eV.size > 1:
        _Emin = max(float(E_mev.min()), 1e-6)
        _Emax = max(float(E_mev.max()), Er_mev * 1.05, _Emin + 1e-4)
        _auto = make_penetrability_interp(
            l, Z1, Z2, A1, A2, r0=r0, Emin_mev=_Emin, Emax_mev=_Emax, npts=600
        )
        P_E  = _auto(E_mev)
        P_Er = float(_auto(Er_mev))
    else:
        P_E  = np.array(
            [penetrability_P_l_mev(l, Z1, Z2, A1, A2, ee, r0) for ee in E_mev],
            dtype=float,
        )
        P_Er = penetrability_P_l_mev(l, Z1, Z2, A1, A2, Er_mev, r0)

    # if (r.Gamma_i is not None) and (r.Gamma_i > 0.0):
    #    Gamma_i_Er_eV = r.Gamma_i
    # else:
    #Gamma_i_Er_eV = 2.0 * gamma2 * P_Er  # eV

    Gamma_i_E_eV = Gamma_i_Er_eV * (P_E / P_Er) if P_Er != 0.0 else np.zeros_like(P_E)

    EiJ = ev_to_j(Gamma_i_E_eV)
    EoJ = ev_to_j(r.Gamma_o)
    EtJ = EiJ + EoJ
    EJ = ev_to_j(E_eV)
    ErJ = ev_to_j(r.E_r)

    BW = (EiJ * EoJ) / ((EJ - ErJ) ** 2 + (EtJ / 2.0) ** 2)
    return S * (PI / k**2) * BW * M_TO_BARNS
