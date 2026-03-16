import matplotlib.pyplot as plt

from .api import energy_grid, sigma_bw_constant, sigma_bw_energy_dep


def plot_bw_constant(r, half_width_eV=25.0, n=200, ax=None, **kw):
    """Plot the constant-width Breit-Wigner cross section for one resonance."""
    E = energy_grid(r.E_r, half_width_eV, n)
    sig = sigma_bw_constant(E, r)
    ax = ax or plt.gca()
    ax.plot(E, sig, **kw)
    ax.set(xlabel="Energy (eV)", ylabel="Cross Section (b)")
    return ax


def plot_bw_energy_dep(
    r, Z1, Z2, A1, A2, l, gamma2_mev, half_width_eV=25.0, n=200, ax=None, **kw
):
    """Plot the energy-dependent Breit-Wigner cross section for one resonance."""
    E = energy_grid(r.E_r, half_width_eV, n)
    sig = sigma_bw_energy_dep(E, r, Z1, Z2, A1, A2, l, gamma2_mev)
    ax = ax or plt.gca()
    ax.plot(E, sig, **kw)
    ax.set(xlabel="Energy (eV)", ylabel="Cross Section (b)")
    return ax
