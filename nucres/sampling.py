import numpy as np

from .resonance import penetrability_P_l_mev


def sample_er_sqrt_uniform(n, e_min, e_max, rng=None):
    r"""
    Draw `n` samples on `[e_min, e_max]` with
    \(f(E) \propto \sqrt{E - e_{\min}}\).

    Uses the inverse-CDF relation
    \(X = e_{\min} + (e_{\max} - e_{\min}) U^{2/3}\) for \(U \sim U(0,1)\).
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    return e_min + (e_max - e_min) * u ** (2.0 / 3.0)


def sample_er_increasing_pdf(n, e_min, e_max, slope=1.0, rng=None):
    r"""
    Draw `n` samples from a linear density on `[e_min, e_max]`.

    The density is
    \(f(E) \propto 1 + \text{slope} \cdot t\) with
    \(t = (E - e_{\min}) / (e_{\max} - e_{\min})\).
    `slope = 0` reduces to a uniform distribution.
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


def porter_thomas_factors(n, mu, df=1, rng=None):
    r"""
    Sample Porter-Thomas factors with
    \(x \sim \chi^2(\nu) / \nu\), where `df` is \(\nu\).

    For `df=1`, the distribution has mean 1 and variance 2.
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.chisquare(df, size=n) * mu/ float(df)


def nonhomogeneous_poisson_placements(rho_func, e_min, e_max, dE, rng=None):
    r"""
    Place levels on `[e_min, e_max]` using a non-homogeneous Poisson process.

    The local rate is
    \(\lambda(E) = \rho(E)\), with units such as levels/eV. In each small bin
    \([E_i, E_i + dE]\), the algorithm draws
    \(N \sim \mathrm{Poisson}(\rho(E_i)\, dE)\) and distributes those samples
    uniformly inside the bin.

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
    r"""
    Return the mean partial width in MeV from a strength function.

    \[
    \langle \Gamma \rangle = S_\ell D
    \]
    """
    return S_l_mev * D_mev


def mean_particle_width_from_penetrability(
    Er_eV, Z1, Z2, A1, A2, l, gamma2_mev, r0=1.25
):
    r"""
    Return the mean partial width in eV from a reduced width and penetrability.

    \[
    \langle \Gamma \rangle = 2 P_\ell(E_r)\langle \gamma^2 \rangle
    \]

    The intermediate width is computed in MeV and converted to eV.
    """
    Er_mev = max(Er_eV, 0.0) * 1e-6
    P_r = penetrability_P_l_mev(l, Z1, Z2, A1, A2, Er_mev, r0)
    Gamma_mev = 2.0 * gamma2_mev * P_r
    return Gamma_mev * 1e6  # eV


def fluctuate_widths(mean_vals, df=1, rng=None):
    """
    Apply Porter-Thomas fluctuation factors to an array of mean widths.
    """
    rng = np.random.default_rng() if rng is None else rng
    means = np.asarray(mean_vals, dtype=float)
    factors = rng.chisquare(df, size=means.shape) / float(df)
    return means * factors
