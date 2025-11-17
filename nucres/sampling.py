import numpy as np

from .resonance import penetrability_P_l_mev


def sample_er_sqrt_uniform(n, e_min, e_max, rng=None):
    """
    Draw n samples on [e_min, e_max] with pdf f(E) proportional sqrt(E - e_min).
    Inverse-CDF sampler: if U~U(0,1), X = e_min + (e_max - e_min) * U^{2/3}.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    return e_min + (e_max - e_min) * u ** (2.0 / 3.0)


def sample_er_increasing_pdf(n, e_min, e_max, slope=1.0, rng=None):
    """
    Draw n samples with a linear pdf on [e_min, e_max]:
      f(E) proportional (1 + slope * t),  t = (E - e_min)/(e_max - e_min) in [0,1].
    Uses inverse transform; slope >= 0 recommended.
    For slope=0, reduces to uniform.
    """
    rng = np.random.default_rng() if rng is None else rng
    L = e_max - e_min
    s = float(slope)
    u = rng.random(n)
    if abs(s) < 1e-12:  # uniform
        return e_min + L * u
    denom = 1.0 + 0.5 * s
    y = u * denom
    if s > 0:
        t = (-1.0 + np.sqrt(1.0 + 2.0 * s * y)) / s
    else:
        t = (-1.0 - np.sqrt(1.0 + 2.0 * s * y)) / s
    return e_min + L * t


def porter_thomas_factors(n, df=1, rng=None):
    """
    Porter–Thomas factors: x ~ (1/df) * chi2(df).
    For df=1, mean(x)=1, variance=2.
    Return an array of size n.
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.chisquare(df, size=n) / float(df)


def nonhomogeneous_poisson_placements(rho_func, e_min, e_max, dE, rng=None):
    """
    Place levels on [e_min, e_max] using a non-homogeneous Poisson process
    with local rate lambda(E) = rho_func(E) [same units as 1/E: e.g. levels/eV].
    In each small bin [E_i, E_i+dE], draw N ~ Poisson(rho(E_i)*dE) and
    put them uniformly inside the bin.

    Returns:
      positions: np.ndarray of placed energies (possibly empty)
    """
    rng = np.random.default_rng() if rng is None else rng
    edges = np.arange(e_min, e_max, dE, dtype=float)
    if edges.size == 0:
        return np.empty(0, dtype=float)
    lam = np.array([max(rho_func(Ei), 0.0) for Ei in edges], dtype=float) * dE
    N = rng.poisson(lam)
    total = int(N.sum())
    if total == 0:
        return np.empty(0, dtype=float)
    offsets = rng.random(total) * dE
    bin_lefts = np.repeat(edges, N)
    return bin_lefts + offsets


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
