import numpy as np
from .penetrability import penetrability_P_l_mev

def mean_particle_width_from_strength(S_l_mev, D_mev):
    """
    Mean partial width (MeV) from a strength function S_l and mean spacing D:
      <Gamma> = S_l * D
    """
    return S_l_mev * D_mev

def mean_particle_width_from_penetrability(Er_eV, Z1, Z2, A1, A2, l, gamma2_mev, r0=1.25):
    """
    Mean partial width (eV) from reduced width <gamma^2> (MeV) and P_l(E_r):
      <Gamma> = 2 * P_l(E_r) * <gamma^2>   [MeV], then convert to eV.
    """
    Er_mev = max(Er_eV, 0.0) * 1e-6
    P_r = penetrability_P_l_mev(l, Z1, Z2, A1, A2, Er_mev, r0)
    Gamma_mev = 2.0 * gamma2_mev * P_r
    return Gamma_mev * 1e6  # eV

def fluctuate_widths(mean_vals, df=1, rng=None):
    """
    Apply Porter–Thomas fluctuation factors to an array of means.
    Returns fluctuated widths with same shape as mean_vals.
    """
    rng = np.random.default_rng() if rng is None else rng
    means = np.asarray(mean_vals, dtype=float)
    factors = rng.chisquare(df, size=means.shape) / float(df)
    return means * factors
