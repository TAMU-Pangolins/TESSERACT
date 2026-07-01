#!/usr/bin/env python

import numpy as np
import argparse
import os
import re
from pathlib import Path

from nucres.resonance import Resonance, sigma_bw_energy_dep, make_penetrability_interp
from nucres.physics import MASS_PROTON
from extract_resonance_data import load_resonance_data, load_nuclear_params
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


# ============================================================
def calc_cross_sections(E_test, Z1, A1_proj, Z2, A1_target, res_data):
    """
    Compute σ(E) for every energy in E_test by summing single-level
    Breit-Wigner contributions from all resonances.

    Vectorized over energy: one call to sigma_bw_energy_dep per resonance
    passes the full E_test array.  One penetrability interpolator is
    pre-built per unique entrance-channel l1 value and reused.
    """
    m_alpha  = A1_proj   * MASS_PROTON
    m_target = A1_target * MASS_PROTON

    E_eV  = E_test * 1e6          # MeV → eV
    E_mev = E_test                 # already MeV

    # Pre-build one P_interp per unique l1 value
    Emin_mev = max(float(E_mev.min()), 1e-6)
    Emax_mev = float(E_mev.max())
    unique_l1 = set(int(x) for x in res_data["l1"])
    P_interps = {
        l_val: make_penetrability_interp(
            l_val, Z1, Z2, A1_proj, A1_target,
            Emin_mev=Emin_mev, Emax_mev=Emax_mev, npts=600
        )
        for l_val in unique_l1
    }

    n = len(res_data["E_cm"])
    xs_total = np.zeros(len(E_test))

    for j in tqdm(range(n), desc="Summing resonances", leave=False):
        l1_j = int(res_data["l1"][j])
        r_j = Resonance(
            res_data["E_cm"][j] * 1e6,   # MeV → eV
            res_data["Jr"][j],
            0, 0,
            m_target,
            m_alpha,
            res_data["g1"][j],
            res_data["g2"][j],
        )
        xs_j = sigma_bw_energy_dep(
            E_eV, r_j,
            Z1, Z2,
            A1_proj, A1_target,
            l1_j,
            Gamma_i_Er_eV=res_data["g1"][j],
            P_interp=P_interps[l1_j],
        )
        xs_total += np.asarray(xs_j).ravel()

    return xs_total


# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--reaction", required=True,
                        help="Reaction name e.g. 22Mg(a,p)25Al")
    parser.add_argument("--E-min-mev", dest="E_min", type=float, default=0.1,
                        help="Minimum energy [MeV]")
    parser.add_argument("--E-max-mev", dest="E_max", type=float, default=10.0,
                        help="Maximum energy [MeV]")
    parser.add_argument("--dE", type=str, required=True,
                        help="Comma-separated bin widths (e.g. 0.1,0.2,0.5)")
    parser.add_argument("--n-grid-points", dest="n_grid_points", type=int, default=10000,
                        help="Number of points in the unintegrated energy grid "
                             "(sets integration resolution; default 10000).")
    parser.add_argument("--skip-unintegrated", dest="skip_unintegrated",
                        action="store_true", default=False,
                        help="Skip computing unintegrated cross sections and load from "
                             "the existing file instead.")
    parser.add_argument("--skip-integrated", dest="skip_integrated",
                        action="store_true", default=False,
                        help="Skip computing integrated cross sections.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix appended to output filenames.")
    parser.add_argument("--run-idx", dest="run_idx", type=int, default=0,
                        help="RUN_N index to read the RatesMC .in file from (default 0).")
    parser.add_argument("--resonance-output-dir", dest="resonance_output_dir",
                        type=str, default="outputs",
                        help="Root directory containing RUN_N/.in files (default: outputs).")
    parser.add_argument("--output-dir", dest="output_dir",
                        type=str, default=".",
                        help="Directory to write integrated CSV files (default: .).")
    parser.add_argument("--unint-dir", dest="unint_dir",
                        type=str, default=None,
                        help="Directory for the unintegrated cross section file. "
                             "Defaults to <output_dir>/../<reaction>/ if not set.")
    args = parser.parse_args()

    if args.unint_dir is None:
        args.unint_dir = str(Path(args.output_dir).parent)

    reaction  = args.reaction
    res_dir   = os.path.join(args.resonance_output_dir, reaction, f"RUN_{args.run_idx}")
    infile    = os.path.join(res_dir, f"{reaction}.in")

    if not os.path.exists(infile):
        raise FileNotFoundError(f"{infile} not found.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.unint_dir, exist_ok=True)

    rxn_match = re.match(r'([^(]+)\(a,p\)(.+)', reaction)
    if not rxn_match:
        raise ValueError(f"Cannot parse target/residual from reaction: {reaction}")
    target    = rxn_match.group(1)
    residual  = rxn_match.group(2)
    tag_clean  = args.tag.lstrip('_')
    tag_suffix = f"_{tag_clean}" if tag_clean else ""

    print(f"Processing {reaction}")
    print(f"Reading resonance file: {infile}")

    # ------------------------------------------------
    # Load resonance data and nuclear parameters
    # ------------------------------------------------
    E_cm, g1, g2, g3, Jr, l1, l2, L, G = load_resonance_data(infile)

    nuc = load_nuclear_params(infile)
    Z1, A1_proj   = nuc["Z_proj"], nuc["A_proj"]
    Z2, A1_target = nuc["Z_tar"],  nuc["A_tar"]

    print(f"Projectile: Z={Z1}, A={A1_proj}")
    print(f"Target:     Z={Z2}, A={A1_target}")
    print(f"Resonances: {len(E_cm)}")

    res_data = {
        "E_cm": E_cm,
        "Jr":   Jr,
        "g1":   g1,
        "g2":   g2,
        "l1":   l1,
    }

    # ============================================================
    # Unintegrated cross section
    # ============================================================
    unint_file = os.path.join(args.unint_dir, f"{reaction}_xs_unintegrated_parallel{tag_suffix}.txt")

    if args.skip_unintegrated:
        print(f"Loading existing unintegrated cross sections from {unint_file} ...")
        data     = np.loadtxt(unint_file, delimiter=",")
        E_test   = data[:, 0]
        xs_unint = data[:, 1] * 1e-3   # mb → barns
    else:
        print("Computing unintegrated cross sections...")
        E_test = np.linspace(args.E_min, args.E_max, args.n_grid_points)

        xs_unint = calc_cross_sections(
            E_test,
            Z1, A1_proj,
            Z2, A1_target,
            res_data,
        )

        np.savetxt(
            unint_file,
            np.column_stack((E_test, xs_unint * 1e3)),
            delimiter=",",
            header="E (MeV),sigma (mb)",
            fmt="%.4e"
        )
        print(f"Saved unintegrated cross sections to {unint_file}")

    # ============================================================
    # Integrated cross sections
    # ============================================================
    dE_lst = [float(x.strip()) for x in args.dE.split(",")]

    if args.skip_integrated:
        print("Skipping integrated cross sections.")
        print("Done.")
        return

    deltaE = E_test[1] - E_test[0]

    for dE in dE_lst:
        print(f"\nIntegrating with ΔE = {dE} MeV")

        points_per_bin = int(round(dE / deltaE))

        if points_per_bin < 2:
            raise ValueError(
                f"\n  Bin width dE={dE} MeV is finer than the grid resolution "
                f"({deltaE:.4e} MeV/point, n_grid_points={args.n_grid_points}).\n"
                f"  Fix: increase n_grid_points or use a larger dE."
            )
        if points_per_bin < 10:
            print(f"  Warning: only {points_per_bin} grid points per bin — "
                  f"integration accuracy may be low.")

        xs_bin = []
        E_bins = []

        for start in range(0, len(E_test) - points_per_bin, points_per_bin):
            end      = start + points_per_bin + 1
            E_slice  = E_test[start:end]
            xs_slice = xs_unint[start:end]
            xs_int   = np.trapezoid(xs_slice, E_slice)
            xs_bin.append(xs_int * 1e3 / dE)
            E_bins.append(0.5 * (E_slice[0] + E_slice[-1]))

        xs_bin = np.array(xs_bin)
        E_bins = np.array(E_bins)

        outfile = os.path.join(args.output_dir, f"{target}_ap_{residual}_integrated_xs_dE_{dE}{tag_suffix}.csv")
        np.savetxt(
            outfile,
            np.column_stack((E_bins, xs_bin)),
            delimiter=",",
            header=f"E (MeV),sigma (mb)| dE={dE}",
            fmt="%.4e"
        )
        print(f"Saved integrated cross sections to {outfile}")

    print("Done.")


# ============================================================
if __name__ == "__main__":
    main()
