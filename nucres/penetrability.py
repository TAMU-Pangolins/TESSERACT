import mpmath as mp
import numpy as np
from functools import lru_cache

def channel_radius_fm(A1, A2, r0=1.25):
    return r0*(A1**(1/3) + A2**(1/3))

def penetrability_P_l_mev(l, Z1, Z2, A1, A2, E_mev, r0=1.25):
    if E_mev <= 0.0:
        return 0.0
    a_fm   = channel_radius_fm(A1, A2, r0)
    mu_amu = (A1*A2)/(A1+A2)
    rho = 0.218735 * a_fm * (mu_amu * E_mev)**0.5
    eta = 0.157489 * (Z1*Z2) * (mu_amu / E_mev)**0.5
    F = float(mp.coulombf(l, eta, rho))
    G = float(mp.coulombg(l, eta, rho))
    d = F*F + G*G
    return float(rho/d) if d != 0.0 else 0.0

@lru_cache(maxsize=128)
def _P_grid_cached(l, Z1, Z2, A1, A2, r0, Emin_mev, Emax_mev, npts):
    Es = np.linspace(Emin_mev, Emax_mev, int(npts))
    Ps = np.array([penetrability_P_l_mev(l, Z1, Z2, A1, A2, float(e), r0) for e in Es], dtype=float)
    return Es, Ps

def make_penetrability_interp(l, Z1, Z2, A1, A2, r0=1.25, Emin_mev=1e-6, Emax_mev=5.0, npts=600):
    """
    Precompute P_l(E) on a grid [Emin,Emax] (MeV) and return a fast interpolator P(E).
    First creation is the only slow step; subsequent uses are very fast.
    """
    Es, Ps = _P_grid_cached(l, Z1, Z2, A1, A2, float(r0), float(Emin_mev), float(Emax_mev), int(npts))
    def P(E_mev):
        e = np.asarray(E_mev, dtype=float)
        return np.interp(e, Es, Ps, left=Ps[0], right=Ps[-1])
    return P
