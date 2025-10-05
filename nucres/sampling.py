import numpy as np

def sample_er_sqrt_uniform(n, e_min, e_max, rng=None):
    """
    Draw n samples on [e_min, e_max] with pdf f(E) proportional sqrt(E - e_min).
    Inverse-CDF sampler: if U~U(0,1), X = e_min + (e_max - e_min) * U^{2/3}.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    return e_min + (e_max - e_min) * u**(2.0/3.0)

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
    # CDF F(t) = (t + 0.5*s*t^2) / (1 + 0.5*s), invert quadratic for t
    denom = (1.0 + 0.5*s)
    y = u * denom
    # solve 0.5*s*t^2 + t - y = 0
    if s > 0:
        t = (-1.0 + np.sqrt(1.0 + 2.0*s*y)) / s
    else:
        # s<0: use the other root to keep t in [0,1]
        t = (-1.0 - np.sqrt(1.0 + 2.0*s*y)) / s
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
    if edges.size == 0:  # nothing to do
        return np.empty(0, dtype=float)
    # left Riemann sum points
    lam = np.array([max(rho_func(Ei), 0.0) for Ei in edges], dtype=float) * dE
    N = rng.poisson(lam)
    total = int(N.sum())
    if total == 0:
        return np.empty(0, dtype=float)
    # assign each count to its bin and scatter uniformly within that bin
    offsets = rng.random(total) * dE
    # repeat bin left edges according to N
    bin_lefts = np.repeat(edges, N)
    return bin_lefts + offsets
