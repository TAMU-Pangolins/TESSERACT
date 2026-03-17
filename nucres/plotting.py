import matplotlib.pyplot as plt

from .api import energy_grid, sigma_bw_constant, sigma_bw_energy_dep


def plot_bw_constant(r, half_width_eV=25.0, n=200, ax=None, **kw):
    """
    Plot the constant-width Breit-Wigner cross section for one resonance.

    Parameters
    ----------
    r : Resonance
        Resonance to evaluate.
    half_width_eV : float, default=25.0
        Half-width of the sampled plotting window in eV.
    n : int, default=200
        Number of grid samples.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on. Defaults to `plt.gca()`.
    **kw
        Additional Matplotlib keyword arguments passed to `Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    E = energy_grid(r.E_r, half_width_eV, n)
    sig = sigma_bw_constant(E, r)
    ax = ax or plt.gca()
    ax.plot(E, sig, **kw)
    ax.set(xlabel="Energy (eV)", ylabel="Cross Section (b)")
    return ax


def plot_bw_energy_dep(
    r, Z1, Z2, A1, A2, l, gamma2_mev, half_width_eV=25.0, n=200, ax=None, **kw
):
    """
    Plot the energy-dependent Breit-Wigner cross section for one resonance.

    Parameters
    ----------
    r : Resonance
        Resonance to evaluate.
    Z1, Z2 : int
        Projectile and target proton numbers.
    A1, A2 : float
        Projectile and target mass numbers.
    l : int
        Entrance-channel orbital angular momentum.
    gamma2_mev : float
        Reduced-width parameter used by `sigma_bw_energy_dep`.
    half_width_eV : float, default=25.0
        Half-width of the sampled plotting window in eV.
    n : int, default=200
        Number of grid samples.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on. Defaults to `plt.gca()`.
    **kw
        Additional Matplotlib keyword arguments passed to `Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    E = energy_grid(r.E_r, half_width_eV, n)
    sig = sigma_bw_energy_dep(E, r, Z1, Z2, A1, A2, l, gamma2_mev)
    ax = ax or plt.gca()
    ax.plot(E, sig, **kw)
    ax.set(xlabel="Energy (eV)", ylabel="Cross Section (b)")
    return ax
