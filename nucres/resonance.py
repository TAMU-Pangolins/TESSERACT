from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import mpmath as mp
import numpy as np
from scipy.integrate import trapezoid

from .physics import (
    M_TO_BARNS,
    PI,
    ev_to_j,
    reduced_mass,
    spin_stat_factor,
    wavenumber_from_E_eV,
)

HBAR_C_MEV_FM = 197.3269804
AMU_MEV = 931.49410242
E2_MEV_FM = 1.43996448
JWKB_T_FLOOR = 1e-300


@dataclass(frozen=True)
class AlphaOMP:
    r"""
    Real alpha optical-model potential parameters for JWKB tunneling.

    McFadden-Satchler is represented by a real Woods-Saxon well:

    \[
    V_N(r) = -V_0 / (1 + \exp[(r - R_R)/a_R])
    \]

    with \(R_R = r_R A_T^{1/3}\). Its imaginary part is deliberately excluded
    from the tunneling integral.
    """

    name: str = "mcfadden_satchler"
    V0_mev: float = 185.0
    r_real_fm: float = 1.40
    a_real_fm: float = 0.52
    r_coulomb_fm: float = 1.40
    imaginary_ignored: bool = True


def alpha_omp_model(name: str = "mcfadden_satchler") -> AlphaOMP:
    key = name.lower().replace("-", "_")
    if key in {"mcfadden_satchler", "mcfadden", "ms"}:
        return AlphaOMP()
    raise ValueError(f"Unsupported alpha OMP model: {name}")


def alpha_omp_metadata(
    omp_model: str = "mcfadden_satchler",
    penetrability_model: str = "jwkb_real_omp",
    turning_point_fallback: str = "T=1",
    T_floor: float = JWKB_T_FLOOR,
) -> dict:
    omp = alpha_omp_model(omp_model)
    return {
        "penetrability_model": penetrability_model,
        "omp_model": omp.name,
        "coulomb_geometry": "uniformly_charged_sphere",
        "coulomb_radius": "r_coulomb_fm * A_target^(1/3)",
        "real_nuclear_radius": "r_real_fm * A_target^(1/3)",
        "diffuseness": omp.a_real_fm,
        "imaginary_omp_ignored": omp.imaginary_ignored,
        "turning_point_fallback_behavior": turning_point_fallback,
        "T_floor": T_floor,
        **asdict(omp),
    }


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


def finite_size_coulomb_mev(r_fm, Z1, Z2, Rc_fm):
    r"""
    Uniformly charged sphere Coulomb potential in MeV.

    The inside expression is matched to the point-Coulomb field at `Rc_fm`.
    """
    r = np.asarray(r_fm, dtype=float)
    safe_r = np.maximum(r, 1e-12)
    prefactor = Z1 * Z2 * E2_MEV_FM
    inside = prefactor / (2.0 * Rc_fm) * (3.0 - (safe_r / Rc_fm) ** 2)
    outside = prefactor / safe_r
    return np.where(safe_r < Rc_fm, inside, outside)


def real_nuclear_alpha_omp_mev(r_fm, A_target, omp_model="mcfadden_satchler"):
    r"""
    Real nuclear alpha-OMP potential in MeV.

    The imaginary optical-model part is not included in this barrier.
    """
    omp = alpha_omp_model(omp_model)
    r = np.asarray(r_fm, dtype=float)
    R_real = omp.r_real_fm * A_target ** (1.0 / 3.0)
    exponent = np.clip((r - R_real) / omp.a_real_fm, -700.0, 700.0)
    return -omp.V0_mev / (1.0 + np.exp(exponent))


def centrifugal_potential_mev(r_fm, l, A1, A2):
    r"""
    Centrifugal potential in MeV for radius in fm.
    """
    r = np.asarray(r_fm, dtype=float)
    if int(l) <= 0:
        return np.zeros_like(r, dtype=float)
    safe_r = np.maximum(r, 1e-12)
    mu_mev = (A1 * A2) / (A1 + A2) * AMU_MEV
    return HBAR_C_MEV_FM**2 * l * (l + 1.0) / (2.0 * mu_mev * safe_r**2)


def effective_alpha_potential_mev(
    r_fm, E_mev, l, Z1, Z2, A1, A2, omp_model="mcfadden_satchler"
):
    r"""
    Real alpha+target effective potential used for JWKB tunneling.
    """
    omp = alpha_omp_model(omp_model)
    Rc = omp.r_coulomb_fm * A2 ** (1.0 / 3.0)
    return (
        finite_size_coulomb_mev(r_fm, Z1, Z2, Rc)
        + real_nuclear_alpha_omp_mev(r_fm, A2, omp_model)
        + centrifugal_potential_mev(r_fm, l, A1, A2)
    )


def _outer_forbidden_segment(r_fm, barrier_minus_E):
    forbidden = np.asarray(barrier_minus_E, dtype=float) > 0.0
    if not np.any(forbidden):
        return None

    segments = []
    start = None
    for idx, is_forbidden in enumerate(forbidden):
        if is_forbidden and start is None:
            start = idx
        elif not is_forbidden and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(forbidden) - 1))

    start, end = max(segments, key=lambda item: r_fm[item[1]])
    if start == 0:
        r1 = float(r_fm[0])
    else:
        x0, x1 = r_fm[start - 1], r_fm[start]
        y0, y1 = barrier_minus_E[start - 1], barrier_minus_E[start]
        r1 = float(x0 - y0 * (x1 - x0) / (y1 - y0))

    if end == len(r_fm) - 1:
        r2 = float(r_fm[-1])
    else:
        x0, x1 = r_fm[end], r_fm[end + 1]
        y0, y1 = barrier_minus_E[end], barrier_minus_E[end + 1]
        r2 = float(x0 - y0 * (x1 - x0) / (y1 - y0))

    return r1, r2


def jwkb_log_transmission_mev(
    E_mev,
    l,
    Z1,
    Z2,
    A1,
    A2,
    omp_model="mcfadden_satchler",
    r_min_fm=1e-4,
    r_max_fm=None,
    npts=2400,
):
    r"""
    Return \(\log T_\ell(E)\) from the real alpha-OMP JWKB action.

    No forbidden region or invalid turning points return `0.0`, i.e. `T=1`.
    Nonpositive energies return the configured numerical transmission floor.
    """
    E = float(E_mev)
    if E <= 0.0 or not np.isfinite(E):
        return float(np.log(JWKB_T_FLOOR))

    if r_max_fm is None:
        outer_coulomb = (Z1 * Z2 * E2_MEV_FM) / E if Z1 * Z2 > 0 else 30.0
        r_max_fm = max(60.0, outer_coulomb + 40.0)

    r = np.linspace(float(r_min_fm), float(r_max_fm), int(npts))
    v_minus_e = effective_alpha_potential_mev(r, E, l, Z1, Z2, A1, A2, omp_model) - E
    segment = _outer_forbidden_segment(r, v_minus_e)
    if segment is None:
        return 0.0

    r1, r2 = segment
    if not np.isfinite(r1 + r2) or r2 <= r1:
        return 0.0

    mask = (r >= r1) & (r <= r2)
    r_int = np.concatenate(([r1], r[mask], [r2]))
    v_int = effective_alpha_potential_mev(r_int, E, l, Z1, Z2, A1, A2, omp_model) - E
    v_int = np.maximum(v_int, 0.0)

    mu_mev = (A1 * A2) / (A1 + A2) * AMU_MEV
    integrand = np.sqrt((2.0 * mu_mev / HBAR_C_MEV_FM**2) * v_int)
    G = float(trapezoid(integrand, r_int))
    return min(0.0, -2.0 * G)


@lru_cache(maxsize=128)
def _jwkb_logT_grid_cached(
    l, Z1, Z2, A1, A2, omp_model, Emin_mev, Emax_mev, npts, radial_npts
):
    Es = np.linspace(Emin_mev, Emax_mev, int(npts))
    logTs = np.array(
        [
            jwkb_log_transmission_mev(
                float(e),
                l,
                Z1,
                Z2,
                A1,
                A2,
                omp_model=omp_model,
                npts=int(radial_npts),
            )
            for e in Es
        ],
        dtype=float,
    )
    return Es, logTs


def make_jwkb_log_transmission_interp(
    l,
    Z1,
    Z2,
    A1,
    A2,
    omp_model="mcfadden_satchler",
    Emin_mev=1e-6,
    Emax_mev=5.0,
    npts=300,
    radial_npts=2400,
):
    r"""
    Precompute \(\log T_\ell^{JWKB}(E)\) for the real alpha-OMP barrier.
    """
    Es, logTs = _jwkb_logT_grid_cached(
        l,
        Z1,
        Z2,
        A1,
        A2,
        str(omp_model),
        float(Emin_mev),
        float(Emax_mev),
        int(npts),
        int(radial_npts),
    )

    def logT(E_mev):
        e = np.asarray(E_mev, dtype=float)
        values = np.interp(e, Es, logTs, left=logTs[0], right=logTs[-1])
        return np.minimum(values, 0.0)

    return logT


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


def sigma_bw_energy_dep(
    E_eV,
    r,
    Z1,
    Z2,
    A1,
    A2,
    l,
    Gamma_i_Er_eV,
    r0=1.25,
    P_interp=None,
    penetrability_model="coulomb",
    omp_model="mcfadden_satchler",
    logT_interp=None,
    T_floor=JWKB_T_FLOOR,
):
    r"""
    Evaluate a Breit-Wigner cross section with energy-dependent entrance width.

    The default entrance width is scaled as

    \[
    \Gamma_i(E) = \Gamma_i(E_r)\frac{P_\ell(E)}{P_\ell(E_r)}
    \]

    With `penetrability_model="jwkb_real_omp"`, the Coulomb penetrability
    energy dependence is replaced by a real-OMP JWKB transmission ratio:

    \[
    \Gamma_i(E) = \Gamma_i(E_r)
    \frac{T_\ell^{JWKB}(E)}{T_\ell^{JWKB}(E_r)}.
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
    penetrability_model : {"coulomb", "jwkb_real_omp"}, default="coulomb"
        Energy-dependence backend for the entrance width.
    omp_model : str, default="mcfadden_satchler"
        Alpha OMP parameterization for the JWKB backend.
    logT_interp : callable, optional
        Precomputed log-transmission interpolator from
        `make_jwkb_log_transmission_interp`.
    T_floor : float, default=1e-300
        Transmission floor used when forming log-space transmission ratios.

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

    model = penetrability_model.lower()
    if model in {"coulomb", "coulomb_centrifugal", "simple"}:
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

        Gamma_i_E_eV = (
            Gamma_i_Er_eV * (P_E / P_Er) if P_Er != 0.0 else np.zeros_like(P_E)
        )
    elif model == "jwkb_real_omp":
        if logT_interp is None:
            logT_E = np.array(
                [
                    jwkb_log_transmission_mev(
                        ee, l, Z1, Z2, A1, A2, omp_model=omp_model
                    )
                    for ee in E_mev
                ],
                dtype=float,
            )
            logT_Er = jwkb_log_transmission_mev(
                Er_mev, l, Z1, Z2, A1, A2, omp_model=omp_model
            )
        else:
            logT_E = np.asarray(logT_interp(E_mev), dtype=float)
            logT_Er = float(logT_interp(Er_mev))

        log_floor = float(np.log(T_floor))
        log_ratio = np.maximum(logT_E, log_floor) - max(logT_Er, log_floor)
        Gamma_i_E_eV = Gamma_i_Er_eV * np.exp(np.clip(log_ratio, -745.0, 700.0))
    else:
        raise ValueError(f"Unsupported penetrability_model: {penetrability_model}")

    EiJ = ev_to_j(Gamma_i_E_eV)
    EoJ = ev_to_j(r.Gamma_o)
    EtJ = EiJ + EoJ
    EJ = ev_to_j(E_eV)
    ErJ = ev_to_j(r.E_r)

    BW = (EiJ * EoJ) / ((EJ - ErJ) ** 2 + (EtJ / 2.0) ** 2)
    return S * (PI / k**2) * BW * M_TO_BARNS
