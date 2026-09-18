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
    try:
        return rng.chisquare(df, size=n) * mu/ float(df)
    except ValueError:
        print(f"Invalid parameters: n={n}, mu={mu}, df={df}")
        return 0.0


def wigner_surmise_spacings(n, rng=None):
    r"""
    Draw `n` unit-mean nearest-neighbour spacings from the GOE Wigner surmise.

    The sampled density is

    \[
    p(s) = \frac{\pi}{2}s\exp\!\left(-\frac{\pi s^2}{4}\right),
    \qquad s \ge 0,
    \]

    whose vanishing probability density at `s=0` produces level repulsion.
    This is a nearest-neighbour renewal model, not a full GOE eigenspectrum.
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(int(n))
    return np.sqrt((-4.0 / np.pi) * np.log1p(-u))


def stationary_wigner_placements(x_min, x_max, rng=None):
    r"""
    Place a stationary Wigner-surmise renewal sequence on `[x_min, x_max]`.

    The coordinate `x` is expected to be an unfolded level coordinate, so a
    unit interval contains one level on average. The boundary gap is sampled
    from the length-biased Wigner distribution; this avoids pinning a level to
    the lower boundary or suppressing the expected count near that boundary.

    Notes
    -----
    Adjacent spacings follow the Wigner surmise, but longer-range GOE
    correlations are not represented.
    """
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Placement bounds must be finite.")
    if x_max < x_min:
        raise ValueError("x_max must be greater than or equal to x_min.")
    if x_max == x_min:
        return np.empty(0, dtype=float)

    rng = np.random.default_rng() if rng is None else rng

    # For p(s)=(pi/2)s exp(-pi*s^2/4), the gap containing a random
    # boundary point has the length-biased distribution s*p(s). It can be
    # drawn via Y~Gamma(3/2, 1), s=sqrt(4Y/pi).
    boundary_gap = np.sqrt((4.0 / np.pi) * rng.gamma(shape=1.5))
    position = x_min + rng.random() * boundary_gap
    placements = []
    while position <= x_max:
        placements.append(position)
        position += float(wigner_surmise_spacings(1, rng=rng)[0])
    return np.asarray(placements, dtype=float)


def unfolded_wigner_placements(energy, density, rng=None):
    r"""
    Generate Wigner-repelled levels for an energy-dependent level density.

    `density` is integrated over `energy` to form the unfolded coordinate
    `x(E)`. A stationary Wigner-surmise sequence is sampled in `x`, then
    mapped back to physical energy. The returned positions are sorted and lie
    within the supplied energy interval.
    """
    energy = np.asarray(energy, dtype=float)
    density = np.asarray(density, dtype=float)
    if energy.ndim != 1 or density.ndim != 1 or energy.shape != density.shape:
        raise ValueError("energy and density must be one-dimensional arrays of equal shape.")
    if energy.size < 2:
        raise ValueError("Provide at least two energy-density samples.")
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(density)):
        raise ValueError("energy and density must contain only finite values.")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("energy must be strictly increasing.")
    if np.any(density < 0.0):
        raise ValueError("density must be nonnegative.")

    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(energy))]
    )
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.empty(0, dtype=float)

    unfolded = stationary_wigner_placements(0.0, total, rng=rng)
    if unfolded.size == 0:
        return np.empty(0, dtype=float)

    # `side="right"` selects the far edge of any zero-density plateau, so the
    # inverse does not place levels inside a region carrying no probability.
    upper = np.searchsorted(cumulative, unfolded, side="right")
    upper = np.clip(upper, 1, cumulative.size - 1)
    lower = upper - 1
    width = cumulative[upper] - cumulative[lower]
    fraction = np.divide(
        unfolded - cumulative[lower],
        width,
        out=np.zeros_like(unfolded),
        where=width > 0.0,
    )
    return energy[lower] + fraction * (energy[upper] - energy[lower])


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
