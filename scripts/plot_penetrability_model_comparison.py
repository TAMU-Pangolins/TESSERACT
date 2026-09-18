#!/usr/bin/env python
"""Plot Coulomb-penetrability and JWKB/alpha-OMP comparison diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nucres.physics import MASS_PROTON
from nucres.rates import na_sigma_v_from_sigma
from nucres.resonance import (
    Resonance,
    alpha_omp_metadata,
    centrifugal_potential_mev,
    effective_alpha_potential_mev,
    finite_size_coulomb_mev,
    jwkb_log_transmission_mev,
    make_jwkb_log_transmission_interp,
    penetrability_P_l_mev,
    sigma_bw_energy_dep,
)


def _positive_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def _normalized_coulomb(E_mev, Er_mev, l, Z1, Z2, A1, A2, r0):
    P = np.array(
        [penetrability_P_l_mev(l, Z1, Z2, A1, A2, float(e), r0) for e in E_mev],
        dtype=float,
    )
    P_er = penetrability_P_l_mev(l, Z1, Z2, A1, A2, float(Er_mev), r0)
    return _positive_ratio(P, np.full_like(P, P_er))


def _normalized_jwkb(E_mev, Er_mev, l, Z1, Z2, A1, A2, omp_model, radial_npts):
    logT = np.array(
        [
            jwkb_log_transmission_mev(
                float(e), l, Z1, Z2, A1, A2, omp_model=omp_model, npts=radial_npts
            )
            for e in E_mev
        ],
        dtype=float,
    )
    logT_er = jwkb_log_transmission_mev(
        float(Er_mev), l, Z1, Z2, A1, A2, omp_model=omp_model, npts=radial_npts
    )
    return np.exp(np.clip(logT - logT_er, -745.0, 700.0))


def _single_resonance_sigma(E_mev, args, model):
    r = Resonance(
        E_r=args.Er_mev * 1e6,
        J=args.J,
        s1=args.s1,
        s2=args.s2,
        m1=args.A1 * MASS_PROTON,
        m2=args.A2 * MASS_PROTON,
        Gamma_i=args.Gamma_i_eV,
        Gamma_o=args.Gamma_o_eV,
    )
    logT_interp = None
    if model == "jwkb_real_omp":
        logT_interp = make_jwkb_log_transmission_interp(
            args.l,
            args.Z1,
            args.Z2,
            args.A1,
            args.A2,
            omp_model=args.omp_model,
            Emin_mev=float(np.min(E_mev)),
            Emax_mev=float(np.max(E_mev)),
            npts=min(int(args.npts), 300),
            radial_npts=args.radial_npts,
        )
    return sigma_bw_energy_dep(
        E_mev * 1e6,
        r,
        args.Z1,
        args.Z2,
        args.A1,
        args.A2,
        args.l,
        Gamma_i_Er_eV=args.Gamma_i_eV,
        r0=args.r0,
        penetrability_model=model,
        omp_model=args.omp_model,
        logT_interp=logT_interp,
    )


def build_plot(args) -> Path:
    E = np.linspace(args.Emin_mev, args.Emax_mev, args.npts)
    E = E[E > 0.0]
    l_values = [int(x) for x in args.l_values.split(",") if x.strip()]

    fig, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
    ax_scale, ax_scale_ratio, ax_pot, ax_sigma, ax_rate, ax_meta = axes.ravel()

    for l_val in l_values:
        P_norm = _normalized_coulomb(E, args.Er_mev, l_val, args.Z1, args.Z2, args.A1, args.A2, args.r0)
        T_norm = _normalized_jwkb(
            E,
            args.Er_mev,
            l_val,
            args.Z1,
            args.Z2,
            args.A1,
            args.A2,
            args.omp_model,
            args.radial_npts,
        )
        ax_scale.semilogy(E, P_norm, "--", lw=1.4, label=f"Coulomb l={l_val}")
        ax_scale.semilogy(E, T_norm, "-", lw=1.6, label=f"JWKB l={l_val}")
        ax_scale_ratio.semilogy(E, _positive_ratio(T_norm, P_norm), lw=1.6, label=f"l={l_val}")

    ax_scale.axvline(args.Er_mev, color="0.3", lw=1.0, alpha=0.7)
    ax_scale.set_title("Normalized Width Scaling")
    ax_scale.set_xlabel("E_cm [MeV]")
    ax_scale.set_ylabel("P(E)/P(Er) or T(E)/T(Er)")
    ax_scale.legend(fontsize=8, ncol=2)
    ax_scale.grid(True, which="both", alpha=0.25)

    ax_scale_ratio.axhline(1.0, color="0.3", lw=1.0, alpha=0.7)
    ax_scale_ratio.axvline(args.Er_mev, color="0.3", lw=1.0, alpha=0.7)
    ax_scale_ratio.set_title("JWKB/Coulomb Width-Scaling Ratio")
    ax_scale_ratio.set_xlabel("E_cm [MeV]")
    ax_scale_ratio.set_ylabel("[T/T(Er)] / [P/P(Er)]")
    ax_scale_ratio.legend(fontsize=8)
    ax_scale_ratio.grid(True, which="both", alpha=0.25)

    r_fm = np.linspace(0.2, args.rmax_fm, args.radial_npts)
    omp = alpha_omp_metadata(args.omp_model)
    Rc = omp["r_coulomb_fm"] * args.A2 ** (1.0 / 3.0)
    v_c = finite_size_coulomb_mev(r_fm, args.Z1, args.Z2, Rc)
    v_l = centrifugal_potential_mev(r_fm, args.l, args.A1, args.A2)
    v_simple = v_c + v_l
    v_omp = effective_alpha_potential_mev(
        r_fm, args.Er_mev, args.l, args.Z1, args.Z2, args.A1, args.A2, args.omp_model
    )
    ax_pot.plot(r_fm, v_simple, "--", lw=1.5, label="Coulomb sphere + centrifugal")
    ax_pot.plot(r_fm, v_omp, "-", lw=1.7, label="Coulomb sphere + centrifugal + real OMP")
    for label, e_line in [
        ("Emin", args.Emin_mev),
        ("Er", args.Er_mev),
        ("Emax", args.Emax_mev),
    ]:
        ax_pot.axhline(e_line, color="0.35", lw=0.9, alpha=0.55, label=f"{label}={e_line:g} MeV")
    ax_pot.set_ylim(bottom=-20.0, top=max(np.nanmax(v_simple), args.Emax_mev) * 1.05)
    ax_pot.set_title(f"Effective Barrier, l={args.l}")
    ax_pot.set_xlabel("r [fm]")
    ax_pot.set_ylabel("V_eff [MeV]")
    ax_pot.legend(fontsize=8)
    ax_pot.grid(True, alpha=0.25)

    sigma_c = _single_resonance_sigma(E, args, "coulomb")
    sigma_j = _single_resonance_sigma(E, args, "jwkb_real_omp")
    ax_sigma.semilogy(E, np.maximum(sigma_c * 1e3, 1e-300), "--", lw=1.4, label="Coulomb")
    ax_sigma.semilogy(E, np.maximum(sigma_j * 1e3, 1e-300), "-", lw=1.6, label="JWKB real OMP")
    ax_sigma_ratio = ax_sigma.twinx()
    ax_sigma_ratio.plot(E, _positive_ratio(sigma_j, sigma_c), color="tab:green", lw=1.0, alpha=0.8, label="JWKB/Coulomb")
    ax_sigma.axvline(args.Er_mev, color="0.3", lw=1.0, alpha=0.7)
    ax_sigma.set_title("Single-Resonance Cross Section")
    ax_sigma.set_xlabel("E_cm [MeV]")
    ax_sigma.set_ylabel("sigma [mb]")
    ax_sigma_ratio.set_ylabel("sigma ratio")
    ax_sigma.legend(loc="upper left", fontsize=8)
    ax_sigma_ratio.legend(loc="upper right", fontsize=8)
    ax_sigma.grid(True, which="both", alpha=0.25)

    T9 = np.geomspace(args.T9_min, args.T9_max, args.T9_npts)
    rate_c = na_sigma_v_from_sigma(
        E, sigma_c, T9, m1=args.A1 * MASS_PROTON, m2=args.A2 * MASS_PROTON,
        energy_unit="MeV", sigma_unit="barn", temperature_unit="T9",
    ).na_sigma_v
    rate_j = na_sigma_v_from_sigma(
        E, sigma_j, T9, m1=args.A1 * MASS_PROTON, m2=args.A2 * MASS_PROTON,
        energy_unit="MeV", sigma_unit="barn", temperature_unit="T9",
    ).na_sigma_v
    ax_rate.loglog(T9, np.maximum(rate_c, 1e-300), "--", lw=1.4, label="Coulomb")
    ax_rate.loglog(T9, np.maximum(rate_j, 1e-300), "-", lw=1.6, label="JWKB real OMP")
    ax_rate_ratio = ax_rate.twinx()
    ax_rate_ratio.semilogx(T9, _positive_ratio(rate_j, rate_c), color="tab:green", lw=1.2, label="JWKB/Coulomb")
    ax_rate.set_title("Rate From Same Single Resonance")
    ax_rate.set_xlabel("T9 [GK]")
    ax_rate.set_ylabel("N_A <sigma v> [cm^3 mol^-1 s^-1]")
    ax_rate_ratio.set_ylabel("rate ratio")
    ax_rate.legend(loc="upper left", fontsize=8)
    ax_rate_ratio.legend(loc="upper right", fontsize=8)
    ax_rate.grid(True, which="both", alpha=0.25)

    ax_meta.axis("off")
    meta_lines = [
        "Diagnostic setup",
        f"reaction: {args.reaction_label}",
        f"projectile: Z={args.Z1}, A={args.A1}",
        f"target: Z={args.Z2}, A={args.A2}",
        f"Er={args.Er_mev:g} MeV, l={args.l}, l scan={args.l_values}",
        f"Gamma_i(Er)={args.Gamma_i_eV:g} eV, Gamma_o={args.Gamma_o_eV:g} eV",
        f"OMP={args.omp_model}, imaginary ignored=True",
        "JWKB fallback: T=1 when no forbidden region",
        f"Coulomb geometry: uniformly charged sphere, Rc={Rc:.3f} fm",
    ]
    ax_meta.text(0.0, 1.0, "\n".join(meta_lines), va="top", family="monospace", fontsize=10)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Coulomb penetrability and JWKB real alpha-OMP scaling."
    )
    parser.add_argument("--reaction-label", default="22Mg(a,p)25Al")
    parser.add_argument("--output", default="outputs/diagnostics/penetrability_model_comparison.png")
    parser.add_argument("--Z1", type=int, default=2)
    parser.add_argument("--A1", type=int, default=4)
    parser.add_argument("--Z2", type=int, default=12)
    parser.add_argument("--A2", type=int, default=22)
    parser.add_argument("--Er-mev", dest="Er_mev", type=float, default=1.0)
    parser.add_argument("--Emin-mev", dest="Emin_mev", type=float, default=0.2)
    parser.add_argument("--Emax-mev", dest="Emax_mev", type=float, default=3.0)
    parser.add_argument("--npts", type=int, default=400)
    parser.add_argument("--l", type=int, default=0)
    parser.add_argument("--l-values", default="0,1,2,3")
    parser.add_argument("--Gamma-i-eV", dest="Gamma_i_eV", type=float, default=1e-3)
    parser.add_argument("--Gamma-o-eV", dest="Gamma_o_eV", type=float, default=1.0)
    parser.add_argument("--J", type=float, default=0.5)
    parser.add_argument("--s1", type=float, default=0.0)
    parser.add_argument("--s2", type=float, default=0.0)
    parser.add_argument("--r0", type=float, default=1.25)
    parser.add_argument("--omp-model", default="mcfadden_satchler")
    parser.add_argument("--radial-npts", type=int, default=1600)
    parser.add_argument("--rmax-fm", type=float, default=80.0)
    parser.add_argument("--T9-min", dest="T9_min", type=float, default=0.1)
    parser.add_argument("--T9-max", dest="T9_max", type=float, default=3.0)
    parser.add_argument("--T9-npts", dest="T9_npts", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    output = build_plot(parse_args())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
