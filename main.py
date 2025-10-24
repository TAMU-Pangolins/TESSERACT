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
from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb


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
    import matplotlib.pyplot as plt

    E0 = 0.1 if E_min_mev is None else float(E_min_mev)
    E1 = 2.0 if E_max_mev is None else float(E_max_mev)
    if not (E0 < E1):
        raise ValueError("E_min_mev must be < E_max_mev")

    cfg = HFBSamplerConfig(
        Z=Z,
        data_root=data_root,
        A=A,
        J=J,
        pi=pi,
        s1=s1,
        s2=s2,
        m1=m1,
        m2=m2,
        Gamma_i_mean_eV=Gamma_i_mean_eV,
        Gamma_o_mean_eV=Gamma_o_mean_eV,
        delta_E_mev=delta_E_mev,
        E_min_mev=E0,
        E_max_mev=E1,
        n_density_points=n_points,
        n_sigma_points=n_plot_points,
        U_offset_mev=U_offset_mev,
        seed=seed,
    )
    generated = synthesize_sigma_from_hfb(cfg)
    meta = generated.metadata

    print(f"[hfb] window [{E0:.3f},{E1:.3f}] MeV: expected levels ~= {meta['expected_levels']:.3g}")
    print(f"[hfb] bins: {meta['n_bins']}, nonzero-Lambda bins: {meta['nonzero_lambda_bins']}, total drawn levels: {meta['n_drawn']}")

    if meta["n_drawn"] == 0:
        print("[hfb] drew zero levels — try increasing --U-offset-mev (~= S_n), widening the E window, "
              "or checking that your (J,pi) slice isn't too restrictive.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(generated.energy_MeV, generated.sigma_barns, label="Total σ(E) from HFB-driven spectrum")
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
