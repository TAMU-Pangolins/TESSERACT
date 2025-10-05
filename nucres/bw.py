import numpy as np
from .constants import M_TO_BARNS, PI
from .units import ev_to_j
from .kinematics import reduced_mass, wavenumber_from_E_eV, spin_stat_factor
from .penetrability import penetrability_P_l_mev

def sigma_bw_constant(E_eV, r):
    """
    Classic single-level Breit–Wigner with constant partial widths.
    Returns barns. E_eV can be scalar or array.
    """
    E_eV = np.asarray(E_eV, dtype=float)
    mu = reduced_mass(r.m1, r.m2)
    k  = wavenumber_from_E_eV(E_eV, mu)
    S  = spin_stat_factor(r.J, r.s1, r.s2)

    EiJ  = ev_to_j(r.Gamma_i)
    EoJ  = ev_to_j(r.Gamma_o)
    EtJ  = EiJ + EoJ
    EJ   = ev_to_j(E_eV)
    ErJ  = ev_to_j(r.E_r)

    BW = (EiJ * EoJ) / ((EJ - ErJ)**2 + (EtJ/2.0)**2)
    return S * (PI / k**2) * BW * M_TO_BARNS

def sigma_bw_energy_dep(E_eV, r, Z1, Z2, A1, A2, l, gamma2_mev, r0=1.25, P_interp=None):
    """
    Same Breit–Wigner, but Gamma_i(E) scales with penetrability:
      Gamma_i(E) = Gamma_i(E_r) * P_l(E)/P_l(E_r).
    If r.Gamma_i <= 0, compute Gamma_i(E_r) = 2 * gamma2_mev * P_l(E_r) [MeV], then convert to eV.
    Returns barns.
    """
    E_eV = np.asarray(E_eV, dtype=float)
    mu = reduced_mass(r.m1, r.m2)
    k  = wavenumber_from_E_eV(E_eV, mu)
    S  = spin_stat_factor(r.J, r.s1, r.s2)

    E_mev  = np.maximum(E_eV, 0.0) * 1e-6
    Er_mev = max(r.E_r, 0.0) * 1e-6

    # --- fast path with interpolator ---
    if P_interp is None:
        # fallback: direct, slow mpmath calls
        P_E  = np.array([penetrability_P_l_mev(l, Z1, Z2, A1, A2, ee, r0) for ee in E_mev], dtype=float)
        P_Er = penetrability_P_l_mev(l, Z1, Z2, A1, A2, Er_mev, r0)
    else:
        P_E  = P_interp(E_mev)
        P_Er = float(P_interp(Er_mev))

    # Γ_i at E_r (eV)
    if (r.Gamma_i is not None) and (r.Gamma_i > 0.0):
        Gamma_i_Er_eV = r.Gamma_i
    else:
        Gamma_i_Er_eV = 2.0 * gamma2_mev * P_Er * 1e6  # MeV → eV

    # Γ_i(E)
    Gamma_i_E_eV = Gamma_i_Er_eV * (P_E / P_Er) if P_Er != 0.0 else np.zeros_like(P_E)

    EiJ  = ev_to_j(Gamma_i_E_eV)
    EoJ  = ev_to_j(r.Gamma_o)
    EtJ  = EiJ + EoJ
    EJ   = ev_to_j(E_eV)
    ErJ  = ev_to_j(r.E_r)

    BW = (EiJ * EoJ) / ((EJ - ErJ)**2 + (EtJ/2.0)**2)
    return S * (PI / k**2) * BW * M_TO_BARNS
