from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from nucres.types import Resonance
from nucres.kinematics import energy_grid
from nucres.bw import sigma_bw_constant as bw
from nucres.constants import MASS_PROTON


def plot_random_resonances(
    n_res: int = 5,
    E_min: float = 900.0,
    E_max: float = 1200.0,
    half_width: float = 25.0,
    n_points: int = 200,
    J: float = 1.0,
    s1: float = 0.5,
    s2: float = 0.5,
    m1: float = MASS_PROTON,
    m2: float = MASS_PROTON,
    Gamma_i: float = 1.0,
    Gamma_o: float = 1.0,
    seed: Optional[int] = None,
):
    """
    Draw n_res resonance energies uniformly on [E_min, E_max], build Resonance objects,
    evaluate Breit–Wigner cross section around each resonance, and plot.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    random_E_r = rng.uniform(E_min, E_max, n_res)

    fig, ax = plt.subplots()

    for i, E_r_val in enumerate(random_E_r, start=1):
        r = Resonance(
            E_r=E_r_val,
            J=J,
            s1=s1,
            s2=s2,
            m1=m1,
            m2=m2,
            Gamma_i=Gamma_i,
            Gamma_o=Gamma_o,
        )

        E_vals = energy_grid(E_r_val, half_width, n_points)
        sigma_vals = bw(E_vals, r)

        ax.plot(E_vals, sigma_vals, label=f"Res {i}: {E_r_val:.1f} eV")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross Section (b)")
    ax.set_title(f"{n_res} Random Resonances (uniform in [{E_min:.1f}, {E_max:.1f}] eV)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_binned_total_cs_from_hfb(
    # Density → binning
    delta_E_mev: float = 0.05,
    E_min_mev: float | None = None,
    E_max_mev: float | None = None,
    # Cross section evaluation grid (on eV axis)
    n_plot_points: int = 4000,
    # Width model (Porter–Thomas with given means)
    Gamma_i_mean_eV: float = 1.0,
    Gamma_o_mean_eV: float = 1.0,
    # Spin/particles
    J: float = 1.0,
    s1: float = 0.5,
    s2: float = 0.5,
    m1: float = MASS_PROTON,
    m2: float = MASS_PROTON,
    # RNG
    seed: int | None = None,
    # Adapter params (strictly via Z)
    *,
    Z: int,
    data_root: str | None = None,
    A: int = 24,
    pi: int = 1,
    n_points: int = 2001,
    U_offset_mev: float = 8.0,   # <<< NEW: approximate Sn shift; tweak per target
):
    """
    Build Sigma(E) using HFB Rho(U) with U ~= U_offset_mev + E_lab.
    Prints quick diagnostics so you can see if any levels were actually drawn.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from nucres.hfb_adapter import build_density_grid

    rng = np.random.default_rng(seed)

    # Map lab energy (eV) -> excitation U (MeV): U = U_offset + E_lab(MeV)
    U_of_E_mev = lambda E_eV: (np.asarray(E_eV, dtype=float) * 1e-6) + float(U_offset_mev)

    # 1) Build density grid (MeV axis, levels/MeV)
    E0 = 0.1 if E_min_mev is None else float(E_min_mev)
    E1 = 2.0 if E_max_mev is None else float(E_max_mev)
    if not (E0 < E1):
        raise ValueError("E_min_mev must be < E_max_mev")

    E_mev, rho = build_density_grid(
        Z=Z, data_root=data_root,
        A=A, J_phys=J, pi=pi,
        E_min_mev=E0, E_max_mev=E1, n_points=n_points,
        U_of_E_mev=U_of_E_mev,
    )

    # --- quick diagnostics ---
    total_levels = float(np.trapz(rho, E_mev))          # expected count in [E0,E1]
    print(f"[hfb] window [{E0:.3f},{E1:.3f}] MeV: expected levels ~= {total_levels:.3g}")

    # 2) Bin Lambda_k = Integral over bin Rho(E)dE
    n_bins = max(1, int(np.ceil((E1 - E0) / delta_E_mev)))
    edges_mev = np.linspace(E0, E1, n_bins + 1)
    lambdas = np.zeros(n_bins)
    for k in range(n_bins):
        a, b = edges_mev[k], edges_mev[k + 1]
        mask = (E_mev >= a) & (E_mev <= b)
        if np.count_nonzero(mask) >= 2:
            lambdas[k] = np.trapz(rho[mask], E_mev[mask])
        else:
            ra = np.interp(a, E_mev, rho)
            rb = np.interp(b, E_mev, rho)
            lambdas[k] = 0.5 * (ra + rb) * (b - a)

    # 3) Draw counts per bin
    counts = rng.poisson(lambdas)
    N_drawn = int(counts.sum())
    nz_bins = int((lambdas > 0).sum())
    print(f"[hfb] bins: {n_bins}, nonzero-Lambda bins: {nz_bins}, total drawn levels: {N_drawn}")

    # If nothing was drawn, surface a helpful hint and bail gracefully
    if N_drawn == 0:
        print("[hfb] drew zero levels — try increasing --U-offset-mev (~= S_n), widening the E window, "
              "or checking that your (J,pi) slice isn't too restrictive.")
        # Plot a faint zero line so the script still produces a figure
        fig, ax = plt.subplots(figsize=(8, 4))
        E_plot_eV = np.linspace(E0 * 1e6, E1 * 1e6, n_plot_points)
        ax.plot(E_plot_eV, np.zeros_like(E_plot_eV))
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Cross Section (b)")
        ax.set_title(f"HFB binned total cross section  (ΔE = {delta_E_mev:g} MeV)")
        fig.tight_layout()
        return fig, ax

    # 4) Global evaluation grid
    E_plot_MeV = np.linspace(E0, E1, n_plot_points)
    E_plot_eV  = E_plot_MeV * 1e6 
    sigma_tot  = np.zeros_like(E_plot_MeV)

    # PT sampler
    def pt_width(mean_eV: float) -> float:
        return float(mean_eV * rng.chisquare(df=1))

    # Sample E within a bin proportional Rho(E)
    def sample_E_in_bin(a_mev: float, b_mev: float, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0)
        mask = (E_mev >= a_mev) & (E_mev <= b_mev)
        Ex = E_mev[mask]; rhx = rho[mask]
        if Ex.size == 0:
            Ex = np.array([a_mev, b_mev])
            rhx = np.array([np.interp(a_mev, E_mev, rho), np.interp(b_mev, E_mev, rho)])
        else:
            if Ex[0] > a_mev:
                Ex = np.concatenate([[a_mev], Ex])
                rhx = np.concatenate([[np.interp(a_mev, E_mev, rho)], rhx])
            if Ex[-1] < b_mev:
                Ex = np.concatenate([Ex, [b_mev]])
                rhx = np.concatenate([rhx, [np.interp(b_mev, E_mev, rho)]])
        dE = np.diff(Ex)
        accum = np.concatenate([[0.0], np.cumsum(0.5 * (rhx[:-1] + rhx[1:]) * dE)])
        total = accum[-1]
        if total <= 0:
            return rng.uniform(a_mev, b_mev, n)
        u = rng.random(n) * total
        return np.interp(u, accum, Ex)

    # 5) Build spectrum & sum BW
    for k, Nk in enumerate(counts):
        if Nk == 0:
            continue
        a_mev, b_mev = edges_mev[k], edges_mev[k + 1]
        Er_mev = sample_E_in_bin(a_mev, b_mev, Nk)
        for Er in Er_mev:
            Er_eV = float(Er * 1e6)
            r = Resonance(
                E_r=Er_eV, J=J, s1=s1, s2=s2, m1=m1, m2=m2,
                Gamma_i=pt_width(Gamma_i_mean_eV),
                Gamma_o=pt_width(Gamma_o_mean_eV),
            )
            sigma_tot += bw(E_plot_eV, r)

    # 6) Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(E_plot_MeV, sigma_tot, label="Total σ(E) from HFB-driven spectrum")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Cross Section (b)")
    ax.set_title(f"HFB binned total cross section  (ΔE = {delta_E_mev:g} MeV)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax



def main():
    p = argparse.ArgumentParser(description="Sanity plot: random Breit–Wigner resonances or HFB-binned total σ(E).")

    # --- mode toggle ---
    p.add_argument(
        "--hfb-binned",
        action="store_true",
        help="If set, run the HFB-driven binned total cross section instead of random resonances.",
    )

    # --- existing random-resonance params (back-compat defaults) ---
    p.add_argument("--n-res", type=int, default=5)
    p.add_argument("--emin", type=float, default=900.0)
    p.add_argument("--emax", type=float, default=1200.0)
    p.add_argument("--half-width", type=float, default=25.0)
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--s1", type=float, default=0.5)
    p.add_argument("--s2", type=float, default=0.5)
    p.add_argument("--Gamma-i", type=float, default=1.0, dest="Gamma_i")
    p.add_argument("--Gamma-o", type=float, default=1.0, dest="Gamma_o")

    # --- shared / general ---
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed (omit for non-deterministic draws).")
    p.add_argument("--save", type=str, default=None, help="If set, save the figure to this path.")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful in headless environments).")

    # --- HFB-binned specific params (ignored unless --hfb-binned) ---
    p.add_argument("--delta-E-mev", type=float, default=0.05, help="Bin width in MeV for HFB-binned mode.")
    p.add_argument("--emin-mev", type=float, default=None, help="Lower energy bound in MeV for HFB data.")
    p.add_argument("--emax-mev", type=float, default=None, help="Upper energy bound in MeV for HFB data.")
    p.add_argument("--n-plot-points", type=int, default=4000, help="Resolution of σ(E) curve (HFB-binned).")
    p.add_argument("--Gamma-i-mean-ev", type=float, default=1.0, help="Mean entrance width (eV) for PT sampling.")
    p.add_argument("--Gamma-o-mean-ev", type=float, default=1.0, help="Mean exit width (eV) for PT sampling.")
    p.add_argument("--Z", type=int, default=None, help="Proton number (auto-resolve zXXX.tab / zXXX.cor).")
    p.add_argument("--data-root", type=str, default=None, help="Override the default data directory (optional).")
    p.add_argument("--A", type=int, default=24)
    p.add_argument("--pi", type=int, choices=(-1, 1), default=1)
    p.add_argument("--U-offset-mev", type=float, default=8.0, help="Excitation offset U≈S_n added to E_lab when querying ρ(U).")


    args = p.parse_args()

    if args.hfb_binned:
        # inside: if args.hfb_binned:
        if args.Z is None:
            raise SystemExit("error: --hfb-binned expects --Z so the adapter can resolve zXXX.tab")

        fig, _ = plot_binned_total_cs_from_hfb(
            delta_E_mev=args.delta_E_mev,
            E_min_mev=args.emin_mev,
            E_max_mev=args.emax_mev,
            n_plot_points=args.n_plot_points,
            Gamma_i_mean_eV=args.Gamma_i_mean_ev,
            Gamma_o_mean_eV=args.Gamma_o_mean_ev,
            J=args.J, s1=args.s1, s2=args.s2,
            m1=MASS_PROTON, m2=MASS_PROTON,
            seed=args.seed,
            Z=args.Z, data_root=args.data_root, A=args.A, pi=args.pi,
            U_offset_mev=args.U_offset_mev,  # <<< pass-through
        )


    else:
        # --- original random-resonance plot ---
        fig, _ = plot_random_resonances(
            n_res=args.n_res,
            E_min=args.emin,
            E_max=args.emax,
            half_width=args.half_width,
            n_points=args.n_points,
            J=args.J,
            s1=args.s1,
            s2=args.s2,
            m1=MASS_PROTON,
            m2=MASS_PROTON,
            Gamma_i=args.Gamma_i,
            Gamma_o=args.Gamma_o,
            seed=args.seed,
        )

    # --- save/show (shared) ---
    if args.save:
        out_path = Path(args.save)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".png")
        os.makedirs(out_path.parent, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[ok] saved figure to {out_path}")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()