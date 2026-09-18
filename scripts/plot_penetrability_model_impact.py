#!/usr/bin/env python
"""Plot observable impact of replacing Coulomb penetrability with JWKB real OMP."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_resonance_data import extract_data, load_nuclear_params
from nucres.physics import M_TO_BARNS, MASS_PROTON, PI, ev_to_j, reduced_mass, spin_stat_factor, wavenumber_from_E_eV
from nucres.rates import na_sigma_v_from_sigma
from nucres.resonance import (
    JWKB_T_FLOOR,
    Resonance,
    alpha_omp_metadata,
    centrifugal_potential_mev,
    effective_alpha_potential_mev,
    finite_size_coulomb_mev,
    jwkb_log_transmission_mev,
    make_jwkb_log_transmission_interp,
    make_penetrability_interp,
    penetrability_P_l_mev,
    sigma_bw_energy_dep,
)


WEIGHTED_SIGMA_FLOOR_MB = 1e-30


def positive_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def reaction_file_stem(reaction: str) -> str:
    return reaction.replace("(a,p)", "_ap_").replace("(", "_").replace(")", "").replace(",", "_")


def default_input_for_run(run_idx: int, reaction: str, ratesmc_dir: Path) -> Path:
    return ratesmc_dir / f"RUN_{run_idx}" / f"{reaction}.in"


def load_ensemble(path: Path):
    table = extract_data(path).sort_values("Ecm")
    nuc = load_nuclear_params(path)
    data = {
        "E_cm": np.asarray(table["Ecm"], dtype=float) * 1e-3,
        "Jr": np.asarray(table["Jr"], dtype=float),
        "G1": np.asarray(table["G1"], dtype=float),
        "G2": np.asarray(table["G2"], dtype=float),
        "L1": np.asarray(table["L1"], dtype=int),
    }
    return nuc, data


def build_interps(args, l_values):
    p_interps = {}
    logt_interps = {}
    for l_val in l_values:
        p_interps[l_val] = make_penetrability_interp(
            l_val,
            args.Z1,
            args.Z2,
            args.A1,
            args.A2,
            r0=args.r0,
            Emin_mev=args.Emin_mev,
            Emax_mev=args.Emax_mev,
            npts=args.penetrability_npts,
        )
        logt_interps[l_val] = make_jwkb_log_transmission_interp(
            l_val,
            args.Z1,
            args.Z2,
            args.A1,
            args.A2,
            omp_model=args.omp_model,
            Emin_mev=args.Emin_mev,
            Emax_mev=args.Emax_mev,
            npts=args.jwkb_npts,
            radial_npts=args.jwkb_radial_npts,
        )
    return p_interps, logt_interps


def build_direct_tables(args, E, data, l_values):
    p_grid = {}
    logt_grid = {}
    p_er = {}
    logt_er = {}
    er_values_by_l = {
        l_val: sorted(
            {
                round(float(er), 12)
                for er, l in zip(data["E_cm"], data["L1"])
                if np.isfinite(er) and np.isfinite(l) and int(l) == l_val
            }
        )
        for l_val in l_values
    }
    scale_ers = {round(float(args.scale_Er_mev), 12)}

    for l_val in l_values:
        p_grid[l_val] = np.array(
            [
                penetrability_P_l_mev(l_val, args.Z1, args.Z2, args.A1, args.A2, float(e), args.r0)
                for e in E
            ],
            dtype=float,
        )
        logt_grid[l_val] = np.array(
            [
                jwkb_log_transmission_mev(
                    float(e),
                    l_val,
                    args.Z1,
                    args.Z2,
                    args.A1,
                    args.A2,
                    omp_model=args.omp_model,
                    npts=args.jwkb_radial_npts,
                )
                for e in E
            ],
            dtype=float,
        )
        for er_key in sorted(set(er_values_by_l[l_val]) | scale_ers):
            er = float(er_key)
            p_er[(l_val, er_key)] = penetrability_P_l_mev(
                l_val, args.Z1, args.Z2, args.A1, args.A2, er, args.r0
            )
            logt_er[(l_val, er_key)] = jwkb_log_transmission_mev(
                er,
                l_val,
                args.Z1,
                args.Z2,
                args.A1,
                args.A2,
                omp_model=args.omp_model,
                npts=args.jwkb_radial_npts,
            )

    return {"P_grid": p_grid, "logT_grid": logt_grid, "P_er": p_er, "logT_er": logt_er}


def normalized_scaling(E, Er, p_interp, logt_interp, direct_tables=None, l_val=None):
    if direct_tables is not None:
        er_key = round(float(Er), 12)
        P = direct_tables["P_grid"][l_val]
        P_er = float(direct_tables["P_er"][(l_val, er_key)])
        coul = positive_ratio(P, np.full_like(P, P_er))
        logT = direct_tables["logT_grid"][l_val]
        logT_er = float(direct_tables["logT_er"][(l_val, er_key)])
        log_floor = float(np.log(JWKB_T_FLOOR))
        jwkb = np.exp(np.clip(np.maximum(logT, log_floor) - max(logT_er, log_floor), -745.0, 700.0))
        return coul, jwkb

    P = np.asarray(p_interp(E), dtype=float)
    P_er = float(p_interp(Er))
    coul = positive_ratio(P, np.full_like(P, P_er))
    logT = np.asarray(logt_interp(E), dtype=float)
    logT_er = float(logt_interp(Er))
    jwkb = np.exp(np.clip(logT - logT_er, -745.0, 700.0))
    return coul, jwkb


def _sigma_from_gamma_grid(E_eV, Er_eV, J, s1, s2, m1, m2, Gamma_i_E_eV, Gamma_o_eV):
    mu = reduced_mass(m1, m2)
    k = wavenumber_from_E_eV(E_eV, mu)
    S = spin_stat_factor(J, s1, s2)
    EiJ = ev_to_j(Gamma_i_E_eV)
    EoJ = ev_to_j(Gamma_o_eV)
    EtJ = EiJ + EoJ
    EJ = ev_to_j(E_eV)
    ErJ = ev_to_j(Er_eV)
    BW = (EiJ * EoJ) / ((EJ - ErJ) ** 2 + (EtJ / 2.0) ** 2)
    return S * (PI / k**2) * BW * M_TO_BARNS


def reconstructed_sigma(E, data, args, p_interps, logt_interps, model):
    E_eV = E * 1e6
    sigma = np.zeros_like(E, dtype=float)
    m1 = args.A1 * MASS_PROTON
    m2 = args.A2 * MASS_PROTON
    for Er, J, G1, G2, L1 in zip(data["E_cm"], data["Jr"], data["G1"], data["G2"], data["L1"]):
        if not np.isfinite(Er + J + G1 + G2 + L1) or G1 <= 0.0 or G2 <= 0.0:
            continue
        l_val = int(L1)
        resonance = Resonance(
            E_r=float(Er) * 1e6,
            J=float(J),
            s1=args.s1,
            s2=args.s2,
            m1=m1,
            m2=m2,
            Gamma_i=float(G1),
            Gamma_o=float(G2),
        )
        sigma += sigma_bw_energy_dep(
            E_eV,
            resonance,
            args.Z1,
            args.Z2,
            args.A1,
            args.A2,
            l_val,
            Gamma_i_Er_eV=float(G1),
            r0=args.r0,
            P_interp=p_interps.get(l_val),
            penetrability_model=model,
            omp_model=args.omp_model,
            logT_interp=logt_interps.get(l_val),
        )
    return sigma


def reconstructed_sigma_direct(E, data, args, direct_tables, model):
    E_eV = E * 1e6
    sigma = np.zeros_like(E, dtype=float)
    m1 = args.A1 * MASS_PROTON
    m2 = args.A2 * MASS_PROTON
    log_floor = float(np.log(JWKB_T_FLOOR))
    for Er, J, G1, G2, L1 in zip(data["E_cm"], data["Jr"], data["G1"], data["G2"], data["L1"]):
        if not np.isfinite(Er + J + G1 + G2 + L1) or G1 <= 0.0 or G2 <= 0.0:
            continue
        l_val = int(L1)
        er_key = round(float(Er), 12)
        if model == "coulomb":
            P_er = float(direct_tables["P_er"][(l_val, er_key)])
            Gamma_i_E = G1 * positive_ratio(direct_tables["P_grid"][l_val], np.full_like(E, P_er))
            Gamma_i_E = np.where(np.isfinite(Gamma_i_E), Gamma_i_E, 0.0)
        elif model == "jwkb_real_omp":
            logT = direct_tables["logT_grid"][l_val]
            logT_er = float(direct_tables["logT_er"][(l_val, er_key)])
            log_ratio = np.maximum(logT, log_floor) - max(logT_er, log_floor)
            Gamma_i_E = G1 * np.exp(np.clip(log_ratio, -745.0, 700.0))
        else:
            raise ValueError(f"Unsupported penetrability model: {model}")
        sigma += _sigma_from_gamma_grid(
            E_eV,
            float(Er) * 1e6,
            float(J),
            args.s1,
            args.s2,
            m1,
            m2,
            Gamma_i_E,
            float(G2),
        )
    return sigma


def selected_resonances(data, l_values, center_mev):
    selected = []
    for l_val in l_values:
        mask = (
            np.isfinite(data["E_cm"])
            & np.isfinite(data["G1"])
            & (data["G1"] > 0.0)
            & (data["L1"] == l_val)
        )
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        idx = idxs[np.argmin(np.abs(data["E_cm"][idxs] - center_mev))]
        selected.append(idx)
    return selected


def plot_weighted(ax, E, sigma_b, t9, label, color, linestyle):
    weighted = 1e3 * np.clip(sigma_b, 0.0, None) * np.exp(-11.60451812 * E / t9)
    mask = np.isfinite(E) & np.isfinite(weighted) & (weighted >= WEIGHTED_SIGMA_FLOOR_MB)
    if np.any(mask):
        ax.semilogy(E[mask], weighted[mask], color=color, linestyle=linestyle, lw=1.1, label=label)


def build_plot(args):
    nuc, data = load_ensemble(args.input)
    args.Z1 = nuc["Z_proj"]
    args.A1 = nuc["A_proj"]
    args.Z2 = nuc["Z_tar"]
    args.A2 = nuc["A_tar"]

    E = np.linspace(args.Emin_mev, args.Emax_mev, args.npts)
    l_values = sorted({int(l) for l in data["L1"] if np.isfinite(l)})
    p_interps = logt_interps = direct_tables = None
    if args.no_interp:
        direct_tables = build_direct_tables(args, E, data, l_values)
    else:
        p_interps, logt_interps = build_interps(args, l_values)

    warnings.filterwarnings("ignore", message="overflow encountered in square", category=RuntimeWarning)
    if args.no_interp:
        sigma_c = reconstructed_sigma_direct(E, data, args, direct_tables, "coulomb")
        sigma_j = reconstructed_sigma_direct(E, data, args, direct_tables, "jwkb_real_omp")
    else:
        sigma_c = reconstructed_sigma(E, data, args, p_interps, logt_interps, "coulomb")
        sigma_j = reconstructed_sigma(E, data, args, p_interps, logt_interps, "jwkb_real_omp")

    fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)
    (
        ax_scale,
        ax_scale_ratio,
        ax_barrier,
        ax_gamma,
        ax_sigma,
        ax_sigma_ratio,
        ax_gamow_05,
        ax_gamow_10,
        ax_gamow_20,
        ax_rate,
        ax_rate_ratio,
        ax_meta,
    ) = axes.ravel()

    scan_l = [l for l in l_values if l in args.l_scan]
    if not scan_l:
        scan_l = l_values[:4]
    for l_val in scan_l:
        if args.no_interp:
            c, j = normalized_scaling(E, args.scale_Er_mev, None, None, direct_tables, l_val)
        else:
            c, j = normalized_scaling(E, args.scale_Er_mev, p_interps[l_val], logt_interps[l_val])
        ax_scale.semilogy(E, c, "--", lw=1.1, label=f"Coulomb l={l_val}")
        ax_scale.semilogy(E, j, "-", lw=1.2, label=f"JWKB l={l_val}")
        ax_scale_ratio.semilogy(E, positive_ratio(j, c), lw=1.2, label=f"l={l_val}")
    for ax in (ax_scale, ax_scale_ratio):
        ax.axvline(args.scale_Er_mev, color="0.25", lw=0.9, alpha=0.65)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax_scale.set_title(r"Normalized $\Gamma_\alpha(E)$ Scaling")
    ax_scale.set_ylabel(r"$P(E)/P(E_r)$ or $T(E)/T(E_r)$")
    ax_scale.legend(fontsize=7, ncol=2)
    ax_scale_ratio.axhline(1.0, color="0.25", lw=0.9, alpha=0.65)
    ax_scale_ratio.set_title("JWKB/Coulomb Scaling Ratio")
    ax_scale_ratio.set_ylabel(r"$[T/T_r]/[P/P_r]$")
    ax_scale_ratio.legend(fontsize=7)

    r_fm = np.linspace(0.2, args.rmax_fm, args.radial_npts)
    omp = alpha_omp_metadata(args.omp_model)
    Rc = omp["r_coulomb_fm"] * args.A2 ** (1.0 / 3.0)
    v_simple = finite_size_coulomb_mev(r_fm, args.Z1, args.Z2, Rc) + centrifugal_potential_mev(
        r_fm, args.barrier_l, args.A1, args.A2
    )
    v_omp = effective_alpha_potential_mev(
        r_fm, args.scale_Er_mev, args.barrier_l, args.Z1, args.Z2, args.A1, args.A2, args.omp_model
    )
    ax_barrier.plot(r_fm, v_simple, "--", lw=1.3, label="Coulomb sphere + centrifugal")
    ax_barrier.plot(r_fm, v_omp, "-", lw=1.4, label="+ real McFadden-Satchler")
    for e_line in (args.Emin_mev, args.scale_Er_mev, min(args.Emax_mev, 3.0)):
        ax_barrier.axhline(e_line, color="0.45", lw=0.8, alpha=0.55)
    ax_barrier.set_ylim(-20.0, max(np.nanmax(v_simple), min(args.Emax_mev, 3.0)) * 1.08)
    ax_barrier.set_title(f"Effective Barrier, l={args.barrier_l}")
    ax_barrier.set_xlabel("r [fm]")
    ax_barrier.set_ylabel(r"$V_\mathrm{eff}$ [MeV]")
    ax_barrier.legend(fontsize=7)
    ax_barrier.grid(True, alpha=0.25)

    selected = selected_resonances(data, scan_l[:4], args.scale_Er_mev)
    for idx in selected:
        l_val = int(data["L1"][idx])
        Er = float(data["E_cm"][idx])
        G1 = float(data["G1"][idx])
        if args.no_interp:
            c, j = normalized_scaling(E, Er, None, None, direct_tables, l_val)
        else:
            c, j = normalized_scaling(E, Er, p_interps[l_val], logt_interps[l_val])
        ax_gamma.semilogy(E, np.maximum(G1 * c, 1e-300), "--", lw=1.0, label=f"Coul l={l_val}, Er={Er:.2f}")
        ax_gamma.semilogy(E, np.maximum(G1 * j, 1e-300), "-", lw=1.1, label=f"JWKB l={l_val}, Er={Er:.2f}")
    ax_gamma.set_title(r"Selected Resonance $\Gamma_\alpha(E)$")
    ax_gamma.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax_gamma.set_ylabel(r"$\Gamma_\alpha(E)$ [eV]")
    ax_gamma.legend(fontsize=6, ncol=2)
    ax_gamma.grid(True, which="both", alpha=0.25)

    ax_sigma.semilogy(E, np.maximum(1e3 * sigma_c, 1e-300), "-", lw=1.1, label="Coulomb")
    ax_sigma.semilogy(E, np.maximum(1e3 * sigma_j, 1e-300), "--", lw=1.1, label="JWKB real OMP")
    ax_sigma.set_title("Full Ensemble Cross Section")
    ax_sigma.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax_sigma.set_ylabel(r"$\sigma$ [mb]")
    ax_sigma.legend(fontsize=7)
    ax_sigma.grid(True, which="both", alpha=0.25)

    ax_sigma_ratio.plot(E, positive_ratio(sigma_j, sigma_c), color="C2", lw=1.1)
    ax_sigma_ratio.axhline(1.0, color="0.3", lw=0.9)
    ax_sigma_ratio.set_title("Cross-Section Ratio")
    ax_sigma_ratio.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax_sigma_ratio.set_ylabel(r"$\sigma_\mathrm{JWKB}/\sigma_\mathrm{Coul}$")
    ax_sigma_ratio.grid(True, alpha=0.25)

    for ax, t9 in ((ax_gamow_05, 0.5), (ax_gamow_10, 1.0), (ax_gamow_20, 2.0)):
        plot_weighted(ax, E, sigma_c, t9, "Coulomb", "C0", "-")
        plot_weighted(ax, E, sigma_j, t9, "JWKB real OMP", "C1", "--")
        ax.set_title(rf"Gamow Integrand, $T_9={t9:g}$")
        ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
        ax.set_ylabel(r"$\sigma e^{-11.6045E/T_9}$ [mb]")
        ax.set_ylim(bottom=WEIGHTED_SIGMA_FLOOR_MB)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    T9 = np.geomspace(args.T9_min, args.T9_max, args.T9_npts)
    rate_c = na_sigma_v_from_sigma(
        E,
        sigma_c,
        T9,
        m1=args.A1 * MASS_PROTON,
        m2=args.A2 * MASS_PROTON,
        energy_unit="MeV",
        sigma_unit="barn",
        temperature_unit="T9",
    ).na_sigma_v
    rate_j = na_sigma_v_from_sigma(
        E,
        sigma_j,
        T9,
        m1=args.A1 * MASS_PROTON,
        m2=args.A2 * MASS_PROTON,
        energy_unit="MeV",
        sigma_unit="barn",
        temperature_unit="T9",
    ).na_sigma_v
    ax_rate.loglog(T9, np.maximum(rate_c, 1e-300), "-", lw=1.2, label="Coulomb")
    ax_rate.loglog(T9, np.maximum(rate_j, 1e-300), "--", lw=1.2, label="JWKB real OMP")
    ax_rate.set_title("Rate From Reconstructed Ensemble")
    ax_rate.set_xlabel(r"$T_9$ [GK]")
    ax_rate.set_ylabel(r"$N_A\langle\sigma v\rangle$ [cm$^3$ mol$^{-1}$ s$^{-1}$]")
    ax_rate.legend(fontsize=7)
    ax_rate.grid(True, which="both", alpha=0.25)

    ax_rate_ratio.semilogx(T9, positive_ratio(rate_j, rate_c), color="C2", lw=1.2)
    ax_rate_ratio.axhline(1.0, color="0.3", lw=0.9)
    ax_rate_ratio.set_title("Rate Ratio")
    ax_rate_ratio.set_xlabel(r"$T_9$ [GK]")
    ax_rate_ratio.set_ylabel(r"rate$_\mathrm{JWKB}$/rate$_\mathrm{Coul}$")
    ax_rate_ratio.grid(True, which="both", alpha=0.25)

    ax_meta.axis("off")
    meta = [
        "Impact plot",
        f"reaction: {args.reaction}",
        f"input: {args.input}",
        f"resonances: {len(data['E_cm'])}",
        f"E grid: {args.Emin_mev:g}-{args.Emax_mev:g} MeV, n={args.npts}",
        f"l values: {l_values}",
        f"OMP: {args.omp_model}; imaginary ignored",
        "width normalization: sampled Gamma_alpha(Er) preserved",
        (
            f"evaluation: exact grid/no interpolation; radial n={args.jwkb_radial_npts}"
            if args.no_interp
            else f"Coulomb interp n={args.penetrability_npts}; JWKB n={args.jwkb_npts}, radial n={args.jwkb_radial_npts}"
        ),
        f"Rc={Rc:.3f} fm, fallback T=1 for no forbidden region",
    ]
    ax_meta.text(0.0, 1.0, "\n".join(meta), va="top", family="monospace", fontsize=9)

    fig.suptitle(f"{args.reaction}: Coulomb vs JWKB real OMP impact", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote penetrability impact plot: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot penetrability-model impact for one resonance ensemble.")
    parser.add_argument("--reaction", default="22Mg(a,p)25Al")
    parser.add_argument("--run-idx", type=int, default=956)
    parser.add_argument("--ratesmc-dir", type=Path, default=Path("outputs/reference_ratesmc"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/penetrability_impact/22Mg_ap_25Al_penetrability_impact.png"),
    )
    parser.add_argument("--Emin-mev", dest="Emin_mev", type=float, default=0.3)
    parser.add_argument("--Emax-mev", dest="Emax_mev", type=float, default=5.0)
    parser.add_argument("--npts", type=int, default=650)
    parser.add_argument("--scale-Er-mev", dest="scale_Er_mev", type=float, default=1.0)
    parser.add_argument("--l-scan", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--barrier-l", type=int, default=0)
    parser.add_argument("--r0", type=float, default=1.25)
    parser.add_argument("--s1", type=float, default=0.0)
    parser.add_argument("--s2", type=float, default=0.0)
    parser.add_argument("--omp-model", default="mcfadden_satchler")
    parser.add_argument(
        "--no-interp",
        action="store_true",
        help="Evaluate penetrability and JWKB transmission directly on the plotted grid and at each Er.",
    )
    parser.add_argument("--penetrability-npts", type=int, default=50)
    parser.add_argument("--jwkb-npts", type=int, default=80)
    parser.add_argument("--jwkb-radial-npts", type=int, default=700)
    parser.add_argument("--radial-npts", type=int, default=1000)
    parser.add_argument("--rmax-fm", type=float, default=80.0)
    parser.add_argument("--T9-min", dest="T9_min", type=float, default=0.1)
    parser.add_argument("--T9-max", dest="T9_max", type=float, default=3.0)
    parser.add_argument("--T9-npts", dest="T9_npts", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.input is None:
        args.input = default_input_for_run(args.run_idx, args.reaction, args.ratesmc_dir)
    return args


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    build_plot(args)


if __name__ == "__main__":
    main()
