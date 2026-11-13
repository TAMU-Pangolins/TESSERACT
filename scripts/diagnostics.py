#!/usr/bin/env python3
"""
diagnostics.py — pre-flight validation of a [resonance]-stage ensemble,
before committing to the full RatesMC/TALYS pipeline.

Runs checks #1, #3, #4, #5, #6, #7 from the proposed list (skips #2 and #8,
which need external reference data: known experimental levels, STARLIB).
Operates purely on the generated RUN_j/<reaction>.in files plus the HFB
density table of the compound nucleus — no RatesMC/TALYS run needed.

Usage:
    python scripts/diagnostics.py [data_dir] --target T --projectile P \\
        --ejectile E --residual R [--out-dir DIR]

    --target/--projectile/--ejectile/--residual label the reaction (e.g.
    14O a p 17F for 14O(a,p)17F) and are mandatory: they name the output
    plots and, if data_dir is omitted, also fix the default data_dir to
    the cluster's own layout for that reaction (see DEFAULT_DATA_ROOT
    below), e.g. .../resonances/14O_ap_17F/14O(a,p)17F/. Pass data_dir
    explicitly to point at RUN_0/, RUN_1/, ... anywhere else (e.g. a local
    copy of a subset of runs) instead of that cluster path. Plots are
    written to diagnostic_plots/ (one level above scripts/), or --out-dir
    if given, as <target>_<projectile><ejectile>_<residual>_<check>.png.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.densities_retrieval import J_index, spin_grid  # noqa: E402
from nucres.generator import _allowed_L_values  # noqa: E402
from nucres.hfb_adapter import load_hfb_record, resolve_density_paths  # noqa: E402
from nucres.physics import spin_stat_factor  # noqa: E402
from nucres.read_qvals import ground_state_jpi  # noqa: E402

kB_MeV_per_GK = 0.08617
_trapezoid = getattr(np, "trapezoid", np.trapz)  # numpy<2.0 compat for this standalone script

# Cluster layout for a reaction's resonance-ensemble output, used as the
# default data_dir when one isn't given explicitly. Reaction labels on the
# cluster contain literal "(", ")", "," (e.g. "14O(a,p)17F") -- fine here
# since this is a plain pathlib.Path built by f-string, never passed
# through a shell or a regex, so none of those characters need escaping.
DEFAULT_DATA_ROOT = Path("/data/pagroup/pa02/THICC/outputs/resonances")

HEADER_KEYS = [
    "Zproj", "Ztarget", "Zexit", "Aproj", "Atarget", "Aexit",
    "Jproj", "Jtarget", "Jexit", "S_proj_keV", "S_exit_keV", "R0_fm", "gamma_channel",
]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_ratesmc_in(path: Path) -> tuple[dict, list[dict]]:
    """Parse a RatesMC .in file: header dict + list of Resonant Contribution rows."""
    lines = path.read_text(errors="replace").splitlines()

    header = {}
    hi = 2
    for k in HEADER_KEYS:
        header[k] = float(lines[hi].split("!")[0].strip())
        hi += 1

    start = next(i for i, l in enumerate(lines) if l.strip() == "Resonant Contribution")
    i = start + 1
    while not lines[i].strip().startswith("Ecm"):
        i += 1
    i += 1
    rows = []
    while not lines[i].strip().startswith("***"):
        parts = lines[i].split()
        if parts:
            rows.append({
                "Ecm_keV": float(parts[0]), "DEcm_keV": float(parts[1]),
                "wg_eV": float(parts[2]), "Dwg_eV": float(parts[3]),
                "Jr": float(parts[4]),
                "G1_eV": float(parts[5]), "DG1_eV": float(parts[6]), "L1": int(float(parts[7])),
                "G2_eV": float(parts[8]), "DG2_eV": float(parts[9]), "L2": int(float(parts[10])),
                "G3_eV": float(parts[11]), "DG3_eV": float(parts[12]), "L3": int(float(parts[13])),
                "Exf_keV": float(parts[14]), "Int": int(float(parts[15])),
            })
        i += 1
    return header, rows


_RUN_RE = re.compile(r"^RUN_(\d+)$")


def discover_runs(data_dir: Path) -> list[tuple[int, Path]]:
    runs = []
    for d in sorted(data_dir.glob("RUN_*")):
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        in_files = list(d.glob("*.in"))
        if in_files:
            runs.append((int(m.group(1)), in_files[0]))
    runs.sort(key=lambda r: r[0])
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Shared setup: compound nucleus, HFB density, resonance weighting
# ─────────────────────────────────────────────────────────────────────────────
def compound_ZA(header: dict) -> tuple[int, int]:
    Z = int(round(header["Zproj"] + header["Ztarget"]))
    A = int(round(header["Aproj"])) + int(round(header["Atarget"]))
    return Z, A


def resonance_omega_gamma(row: dict, g: float) -> float:
    """Narrow-resonance strength omega*gamma [eV] = g * G1*G2/(G1+G2+G3)."""
    gt = row["G1_eV"] + row["G2_eV"] + row["G3_eV"]
    if gt <= 0.0:
        return 0.0
    return g * (row["G1_eV"] * row["G2_eV"]) / gt


def rate_weight(row: dict, g: float, T9: float) -> float:
    """Relative narrow-resonance rate contribution at T9 (common prefactor omitted)."""
    wg = resonance_omega_gamma(row, g)
    return wg * np.exp(-row["Ecm_keV"] / 1000.0 / (kB_MeV_per_GK * T9))


def rho_query(record, A: int, U_of_Ecm_keV):
    """
    Build rho(E, J, pi) [levels/MeV] -> callable(Ecm_keV, J, pi).
    Mirrors nucres/generator.py's own usage of the HFB record.
    """
    def rho(Ecm_keV: float, J: float, pi: int) -> float:
        block = record.positive if pi == +1 else record.negative
        jcol = J_index(J, A)
        U = U_of_Ecm_keV(Ecm_keV)
        return float(np.interp(U, block.U, block.rho_J[:, jcol],
                               left=block.rho_J[0, jcol], right=block.rho_J[-1, jcol]))
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────
def check1_sampled_density_vs_hfb(all_runs, header, record, Z, A, j1, j2, pi_intrinsic,
                                  U_of_Ecm_keV, out_dir, reaction_slug):
    """
    Empirical sampled-level density (per MeV) vs the HFB table, filtered to
    only the J^pi sequences the entrance channel can actually form -- the
    same formability filter nucres/generator.py applies (drop_forbidden)
    before ever sampling a level, so this is an apples-to-apples comparison
    rather than comparing against the raw, unfiltered HFB total.
    """
    all_Ecm_keV = np.concatenate([[r["Ecm_keV"] for r in rows] for _, rows in all_runs])
    n_runs = len(all_runs)
    E_max_keV = header["Ecm_max_keV"]
    bins = np.linspace(0, E_max_keV, 60)
    counts, edges = np.histogram(all_Ecm_keV, bins=bins)
    centers_keV = 0.5 * (edges[:-1] + edges[1:])
    empirical_density = counts / n_runs / (edges[1] - edges[0]) * 1000.0  # levels/run/MeV

    # theoretical density, but ONLY over J^pi sequences the entrance channel can form
    J_grid = spin_grid(A)
    theory = np.zeros_like(centers_keV)
    n_formable, n_total = 0, 0
    for pi in (+1, -1):
        block = record.positive if pi == +1 else record.negative
        for jcol in range(block.rho_J.shape[1]):
            J_val = float(J_grid[jcol])
            U = U_of_Ecm_keV(centers_keV)
            n_total += 1
            if _allowed_L_values(J_val, j1, j2, int(pi), pi_intrinsic):
                contrib = np.interp(U, block.U, block.rho_J[:, jcol],
                                    left=block.rho_J[0, jcol], right=block.rho_J[-1, jcol])
                theory += contrib
                n_formable += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(centers_keV / 1000.0, empirical_density, where="mid", label="sampled (per run)", color="C0")
    ax.plot(centers_keV / 1000.0, theory, color="C1", lw=1.5,
           label="HFB table (formable $J^\\pi$ only)")
    ax.set_xlabel("$E_{cm}$ (MeV)")
    ax.set_ylabel("levels / MeV")
    ax.set_yscale("log")
    ax.set_title(f"Check 1: sampled vs. HFB level density  (Z={Z}, A={A})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check1_density_vs_hfb.png", dpi=150)
    plt.close(fig)

    ratio = np.divide(empirical_density, theory, out=np.full_like(empirical_density, np.nan),
                      where=theory > 0)
    print(f"[check 1] {n_formable}/{n_total} (J,pi) sequences are entrance-channel-formable")
    print(f"[check 1] median(sampled/theory) = {np.nanmedian(ratio):.3f} "
          f"(formable-only comparison; should be ~1 if sampling matches the table it was drawn from)")


def check3_effective_n_resonances(all_runs, header, g_table, out_dir, reaction_slug):
    """Per T9: N50/N90/N95 resonances needed to reach that fraction of the rate, median+95% band across runs."""
    T9_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    fracs = [0.5, 0.9, 0.95]
    # shape: (n_runs, n_T9, n_fracs)
    N_needed = np.full((len(all_runs), len(T9_values), len(fracs)), np.nan)

    for ridx, (run_idx, rows) in enumerate(all_runs):
        for tidx, T9 in enumerate(T9_values):
            weights = np.array([rate_weight(r, g_table[r["Jr"]], T9) for r in rows])
            order = np.argsort(weights)[::-1]
            w_sorted = weights[order]
            total = w_sorted.sum()
            if total <= 0:
                continue
            cum = np.cumsum(w_sorted) / total
            for fidx, frac in enumerate(fracs):
                hit = np.searchsorted(cum, frac) + 1
                N_needed[ridx, tidx, fidx] = hit

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["C0", "C1", "C2"]
    for fidx, frac in enumerate(fracs):
        med = np.nanmedian(N_needed[:, :, fidx], axis=0)
        lo = np.nanpercentile(N_needed[:, :, fidx], 2.5, axis=0)
        hi = np.nanpercentile(N_needed[:, :, fidx], 97.5, axis=0)
        ax.plot(T9_values, med, color=colors[fidx], marker="o", label=f"N{int(frac*100)}")
        ax.fill_between(T9_values, lo, hi, color=colors[fidx], alpha=0.2)
    ax.set_xlabel("$T_9$ (GK)")
    ax.set_ylabel("number of resonances")
    ax.set_yscale("log")
    ax.set_title("Check 3: effective number of contributing resonances\n(median + 95% interval across runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check3_effective_n_resonances.png", dpi=150)
    plt.close(fig)

    med_N90 = np.nanmedian(N_needed[:, :, 1], axis=0)
    print("[check 3] median N90 (resonances needed for 90% of rate) vs T9:")
    for t9, n in zip(T9_values, med_N90):
        print(f"           T9={t9:>4}: N90={n:.0f}")
    return T9_values, N_needed


def check4_width_distribution(all_runs, out_dir, reaction_slug):
    """Histogram sampled G1/G2 across the whole ensemble; compare shape to chi^2_1 (Porter-Thomas)."""
    all_G1 = np.concatenate([[r["G1_eV"] for r in rows] for _, rows in all_runs])
    all_G2 = np.concatenate([[r["G2_eV"] for r in rows] for _, rows in all_runs])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, label in zip(axes, [all_G1, all_G2], ["$\\Gamma_1$ (entrance)", "$\\Gamma_2$ (exit)"]):
        vals = vals[vals > 0]
        norm = vals / np.median(vals)
        ax.hist(norm, bins=np.logspace(np.log10(norm.min()), np.log10(norm.max()), 60),
               density=True, alpha=0.6, label="sampled (normalized to median)")
        x = np.logspace(np.log10(norm.min()), np.log10(norm.max()), 200)
        from scipy.stats import chi2
        # chi2_1 has mean 1; our values are normalized to median, median(chi2_1)=0.455
        pt = chi2.pdf(x * 0.4549, df=1) * 0.4549
        ax.plot(x, pt, color="C1", lw=1.5, label=r"Porter-Thomas ($\chi^2_1$, dof=1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{label} / median({label})")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Check 4: sampled width distribution vs. Porter-Thomas expectation")
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check4_width_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[check 4] G1: median={np.median(all_G1):.3e} eV, G2: median={np.median(all_G2):.3e} eV "
          f"(total {len(all_G1)} resonances across {len(all_runs)} runs)")


def check5_narrow_resonance_validity(all_runs, record, A, U_of_Ecm_keV, pi_intrinsic, out_dir,
                                     reaction_slug):
    """Fraction of sampled levels where Gamma_total is NOT << level spacing D=1/rho."""
    all_ratios = []
    all_Ecm_keV = []
    for run_idx, rows in all_runs:
        for r in rows:
            pi = pi_intrinsic if (r["L1"] % 2 == 0) else -pi_intrinsic
            block = record.positive if pi == +1 else record.negative
            jcol = J_index(r["Jr"], A)
            U = U_of_Ecm_keV(r["Ecm_keV"])
            rho_mev = float(np.interp(U, block.U, block.rho_J[:, jcol],
                                      left=block.rho_J[0, jcol], right=block.rho_J[-1, jcol]))
            if rho_mev <= 0:
                continue
            D_eV = (1.0 / rho_mev) * 1e6   # MeV spacing -> eV
            Gamma_total_eV = r["G1_eV"] + r["G2_eV"] + r["G3_eV"]
            ratio = Gamma_total_eV / D_eV
            all_ratios.append(ratio)
            all_Ecm_keV.append(r["Ecm_keV"])

    all_ratios = np.array(all_ratios)
    all_Ecm_keV = np.array(all_Ecm_keV)
    violating = all_ratios > 0.1
    frac_violating = violating.mean() if len(all_ratios) else float("nan")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.log10(all_ratios[all_ratios > 0]), bins=60, color="C0", alpha=0.7)
    ax.axvline(np.log10(0.1), color="C3", ls="--", label=r"$\Gamma/D=0.1$ (validity threshold)")
    ax.set_xlabel(r"$\log_{10}(\Gamma_{total}/D)$")
    ax.set_ylabel("count")
    ax.set_title("Check 5: narrow-resonance validity ($\\Gamma \\ll D$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check5_narrow_resonance_validity.png", dpi=150)
    plt.close(fig)
    print(f"[check 5] {frac_violating:.1%} of sampled levels have Gamma_total/D > 0.1 "
          f"(isolated-resonance treatment questionable for these)")

    # ── Check 5b: where in energy do the violations concentrate? ───────────
    bins_keV = np.linspace(0, all_Ecm_keV.max(), 40)
    centers_MeV = 0.5 * (bins_keV[:-1] + bins_keV[1:]) / 1000.0
    bin_idx = np.digitize(all_Ecm_keV, bins_keV) - 1
    frac_per_bin = np.full(len(centers_MeV), np.nan)
    median_ratio_per_bin = np.full(len(centers_MeV), np.nan)
    n_per_bin = np.zeros(len(centers_MeV), dtype=int)
    for b in range(len(centers_MeV)):
        mask = bin_idx == b
        n_per_bin[b] = mask.sum()
        if mask.sum() >= 5:   # skip bins with too few samples to be meaningful
            frac_per_bin[b] = violating[mask].mean()
            median_ratio_per_bin[b] = np.median(all_ratios[mask])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(centers_MeV, frac_per_bin, color="C0", marker="o", ms=3)
    ax1.axhline(0.5, color="0.6", ls=":", lw=1)
    ax1.set_ylabel("fraction with $\\Gamma/D > 0.1$")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Check 5b: where narrow-resonance validity breaks down")

    ax2.plot(centers_MeV, median_ratio_per_bin, color="C1", marker="o", ms=3)
    ax2.axhline(0.1, color="C3", ls="--", lw=1, label=r"$\Gamma/D=0.1$ threshold")
    ax2.set_yscale("log")
    ax2.set_xlabel("$E_{cm}$ (MeV)")
    ax2.set_ylabel("median $\\Gamma_{total}/D$ in bin")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check5b_violation_vs_energy.png", dpi=150)
    plt.close(fig)

    # Report the approximate energy where the violation fraction crosses 50%
    valid_bins = ~np.isnan(frac_per_bin)
    above_half = valid_bins & (frac_per_bin >= 0.5)
    if above_half.any():
        E_cross_MeV = centers_MeV[above_half][0]
        print(f"[check 5b] violation fraction first exceeds 50% around Ecm ~ {E_cross_MeV:.2f} MeV")
    else:
        print("[check 5b] violation fraction never reaches 50% in any well-sampled bin")


def check6_spin_population(all_runs, header, record, A, j1, j2, pi_intrinsic, U_of_Ecm_keV, out_dir,
                           reaction_slug):
    """
    Sampled Jr histogram vs. the theoretical expected count per run: the HFB
    density, formable (J,pi) only (same filter as check 1), integrated over
    the full sampled energy range (0..E_max) for each J.
    """
    all_Jr = np.concatenate([[r["Jr"] for r in rows] for _, rows in all_runs])
    n_runs = len(all_runs)
    E_max_keV = header["Ecm_max_keV"]

    j_vals, j_counts = np.unique(all_Jr, return_counts=True)
    sampled_per_run = j_counts / n_runs

    J_grid = spin_grid(A)
    E_fine_keV = np.linspace(0, E_max_keV, 2000)
    theory_J, theory_vals = [], []
    for J_val in J_grid:
        if J_val > j_vals.max() + 3:
            break
        total = 0.0
        for pi in (+1, -1):
            if not _allowed_L_values(float(J_val), j1, j2, int(pi), pi_intrinsic):
                continue
            block = record.positive if pi == +1 else record.negative
            jcol = J_index(float(J_val), A)
            U = U_of_Ecm_keV(E_fine_keV)
            rho = np.interp(U, block.U, block.rho_J[:, jcol],
                            left=block.rho_J[0, jcol], right=block.rho_J[-1, jcol])
            total += _trapezoid(rho, E_fine_keV / 1000.0)   # levels/MeV * MeV = levels
        theory_J.append(J_val)
        theory_vals.append(total)
    theory_J = np.array(theory_J)
    theory_vals = np.array(theory_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.4
    ax.bar(j_vals - width / 2, sampled_per_run, width=width, color="C0", alpha=0.8,
          label="sampled (per run)")
    ax.bar(theory_J + width / 2, theory_vals, width=width, color="C1", alpha=0.8,
          label="HFB theory (formable, integrated over sampled range)")
    ax.set_xlabel("$J_r$")
    ax.set_ylabel("count / run")
    ax.set_title("Check 6: sampled vs. theoretical compound-nucleus spin population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check6_spin_population.png", dpi=150)
    plt.close(fig)

    theory_at_sampled_J = dict(zip(theory_J.tolist(), theory_vals.tolist()))
    ratios = [sampled_per_run[i] / theory_at_sampled_J[j_vals[i]]
             for i in range(len(j_vals)) if j_vals[i] in theory_at_sampled_J
             and theory_at_sampled_J[j_vals[i]] > 0]
    print(f"[check 6] J_r range sampled: {j_vals.min():.0f} - {j_vals.max():.0f}, "
          f"mode = {j_vals[np.argmax(j_counts)]:.0f}")
    print(f"[check 6] median(sampled/theory) across J values = {np.median(ratios):.3f} "
          f"(should be ~1 if the spin population matches the formable HFB density)")


def check7_topn_sensitivity(all_runs, g_table, T9_values, out_dir, reaction_slug):
    """Rate using only the top-N resonances vs. the full set, as a fraction, vs T9."""
    top_ns = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_top in top_ns:
        frac_vs_T9 = []
        for T9 in T9_values:
            fracs_this_T9 = []
            for run_idx, rows in all_runs:
                weights = np.array([rate_weight(r, g_table[r["Jr"]], T9) for r in rows])
                total = weights.sum()
                if total <= 0:
                    continue
                top = np.sort(weights)[::-1][:n_top].sum()
                fracs_this_T9.append(top / total)
            frac_vs_T9.append(np.median(fracs_this_T9) if fracs_this_T9 else np.nan)
        ax.plot(T9_values, frac_vs_T9, marker="o", label=f"top {n_top}")
    ax.axhline(0.9, color="0.5", ls="--", lw=1)
    ax.set_xlabel("$T_9$ (GK)")
    ax.set_ylabel("fraction of total rate captured")
    ax.set_ylim(0, 1.05)
    ax.set_title("Check 7: top-N truncation sensitivity (median across runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{reaction_slug}_check7_topn_sensitivity.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, nargs="?", default=None,
                  help="default: DEFAULT_DATA_ROOT/<slug>/<target>(<projectile>,<ejectile>)<residual>/")
    p.add_argument("--target", required=True, help="target nuclide label, e.g. 14O")
    p.add_argument("--projectile", required=True, help="projectile label, e.g. a (alpha)")
    p.add_argument("--ejectile", required=True, help="ejectile (exit particle) label, e.g. p (proton)")
    p.add_argument("--residual", required=True, help="residual nuclide label, e.g. 17F")
    p.add_argument("--out-dir", type=Path, default=None,
                  help="default: diagnostic_plots/ next to scripts/ (one level above this file's directory)")
    args = p.parse_args()

    reaction_slug = f"{args.target}_{args.projectile}{args.ejectile}_{args.residual}"
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        reaction_label = f"{args.target}({args.projectile},{args.ejectile}){args.residual}"
        data_dir = DEFAULT_DATA_ROOT / reaction_slug / reaction_label
        print(f"No data_dir given; defaulting to cluster path: {data_dir}")

    out_dir = args.out_dir or (HERE.parent / "diagnostic_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(data_dir)
    if not runs:
        sys.exit(f"No RUN_*/*.in files found under {data_dir}")
    print(f"Found {len(runs)} run(s)")

    all_runs = []
    header = None
    for run_idx, path in runs:
        h, rows = parse_ratesmc_in(path)
        if header is None:
            header = h
            e_max_keV = max(r["Ecm_keV"] for r in rows)
            header["Ecm_max_keV"] = e_max_keV
        all_runs.append((run_idx, rows))

    Z, A = compound_ZA(header)
    print(f"Compound nucleus: Z={Z}, A={A}")

    tab_p, cor_p = resolve_density_paths(Z)
    record = load_hfb_record(str(tab_p), str(cor_p) if cor_p else None, A=A, warn_if_ignored=False)

    S_proj_MeV = header["S_proj_keV"] / 1000.0
    def U_of_Ecm_keV(Ecm_keV):
        return np.asarray(Ecm_keV, dtype=float) / 1000.0 + S_proj_MeV

    _, pi_proj = ground_state_jpi(int(header["Zproj"]), int(round(header["Aproj"])))
    _, pi_targ = ground_state_jpi(int(header["Ztarget"]), int(round(header["Atarget"])))
    pi_intrinsic = int(pi_proj * pi_targ)
    print(f"Entrance intrinsic parity (NUBASE): projectile={pi_proj:+d}, target={pi_targ:+d}, "
          f"product={pi_intrinsic:+d}")

    # statistical factor g(Jr) for each distinct Jr seen (reuse across checks)
    j1, j2 = header["Jproj"], header["Jtarget"]
    g_table = {}
    for _, rows in all_runs:
        for r in rows:
            if r["Jr"] not in g_table:
                g_table[r["Jr"]] = spin_stat_factor(r["Jr"], j1, j2)

    print()
    check1_sampled_density_vs_hfb(all_runs, header, record, Z, A, j1, j2, pi_intrinsic,
                                 U_of_Ecm_keV, out_dir, reaction_slug)
    print()
    T9_values, _ = check3_effective_n_resonances(all_runs, header, g_table, out_dir, reaction_slug)
    print()
    check4_width_distribution(all_runs, out_dir, reaction_slug)
    print()
    check5_narrow_resonance_validity(all_runs, record, A, U_of_Ecm_keV, pi_intrinsic, out_dir,
                                     reaction_slug)
    print()
    check6_spin_population(all_runs, header, record, A, j1, j2, pi_intrinsic, U_of_Ecm_keV, out_dir,
                           reaction_slug)
    print()
    check7_topn_sensitivity(all_runs, g_table, T9_values, out_dir, reaction_slug)

    print(f"\nPlots written to {out_dir}")


if __name__ == "__main__":
    main()
