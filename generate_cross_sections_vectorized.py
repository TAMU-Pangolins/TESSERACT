#!/usr/bin/env python
"""
Step 2 of the TESSERACT pipeline: the "true" cross section and its
thick-target (bin-averaged) version.

The cross section is the sum of Breit-Wigner resonances from a RatesMC input
with RatesMC's own energy dependence, masses and spins
(nucres.resonance_sum.ResonanceSum), so the binned cross section and the
RatesMC reference rate describe the same physics. Bin averages are exact
integrals of each resonance over each bin; the "unintegrated" file is the
cross section sampled on a uniform grid, for plotting only (resonances much
narrower than its spacing are not resolved there).
"""

import numpy as np
import argparse
import json
import os
import re
from pathlib import Path

from nucres.resonance import alpha_omp_metadata
from nucres.resonance_sum import ResonanceSum, rows_from_dataframe
from extract_resonance_data import extract_data, load_reaction_params


# ============================================================
def bin_edges(E_min, E_max, dE):
    """Edges E_min, E_min + dE, ... of every complete bin of width dE in [E_min, E_max]."""
    n_bins = int(np.floor((E_max - E_min) / dE + 1e-9))
    if n_bins < 1:
        raise ValueError(f"dE={dE} MeV is wider than the range [{E_min}, {E_max}] MeV.")
    return E_min + dE * np.arange(n_bins + 1)


def binned_cross_section(res_sum, E_min, E_max, dE):
    """(bin centres [MeV], bin-averaged cross section [mb]) for bins of width dE."""
    edges = bin_edges(E_min, E_max, dE)
    integrals = res_sum.bin_integrals(edges)          # b MeV
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, integrals / np.diff(edges) * 1e3  # mb


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
                        help="Number of points in the unintegrated (plotting) energy grid "
                             "(default 10000). Bin averages are integrated exactly "
                             "and do not depend on it.")
    parser.add_argument("--skip-unintegrated", dest="skip_unintegrated",
                        action="store_true", default=False,
                        help="Do not recompute or overwrite the unintegrated cross-section file.")
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
    parser.add_argument("--penetrability-model", dest="penetrability_model",
                        choices=["coulomb", "jwkb_real_omp"], default="coulomb",
                        help="Entrance-width energy-dependence model.")
    parser.add_argument("--omp-model", dest="omp_model", default="mcfadden_satchler",
                        help="Alpha OMP used with --penetrability-model jwkb_real_omp.")
    parser.add_argument("--jwkb-npts", dest="jwkb_npts", type=int, default=300,
                        help="Energy-grid samples per l for JWKB logT interpolation.")
    parser.add_argument("--jwkb-radial-npts", dest="jwkb_radial_npts",
                        type=int, default=2400,
                        help="Radial samples for each JWKB action integral.")
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
    # Load resonances and the reaction header
    # ------------------------------------------------
    df   = extract_data(infile)
    rxn  = load_reaction_params(infile)
    rows = rows_from_dataframe(df)
    res_sum = ResonanceSum(
        rxn, rows, args.E_min, args.E_max,
        penetrability_model=args.penetrability_model,
        omp_model=args.omp_model,
        jwkb_npts=args.jwkb_npts,
        jwkb_radial_npts=args.jwkb_radial_npts,
    )

    print(f"Projectile: Z={rxn.Z_proj}, M={rxn.M_proj} u, J={rxn.J_proj}")
    print(f"Target:     Z={rxn.Z_targ}, M={rxn.M_targ} u, J={rxn.J_targ}")
    print(f"Resonances: {len(res_sum.rows)} used of {len(rows)}")
    if len(res_sum.rows) < len(rows):
        print(f"  Note: {len(rows) - len(res_sum.rows)} rows skipped (Er <= 0, or no "
              "entrance/exit width, e.g. strength-only rows).")
    print(f"Penetrability model: {args.penetrability_model}")
    if args.penetrability_model == "jwkb_real_omp":
        print(f"OMP model: {args.omp_model} (real part only)")

    # ============================================================
    # Unintegrated cross section (plotting grid)
    # ============================================================
    unint_file = os.path.join(args.unint_dir, f"{reaction}_xs_unintegrated_parallel{tag_suffix}.txt")

    if args.skip_unintegrated:
        print(f"Keeping existing unintegrated cross sections in {unint_file}.")
    else:
        print("Computing unintegrated cross sections (plotting grid)...")
        E_test = np.linspace(args.E_min, args.E_max, args.n_grid_points)
        xs_unint = res_sum.sigma(E_test)

        metadata = {
            "reaction": reaction,
            "projectile": {"Z": int(rxn.Z_proj), "M_u": float(rxn.M_proj), "J": float(rxn.J_proj)},
            "target": {"Z": int(rxn.Z_targ), "M_u": float(rxn.M_targ), "J": float(rxn.J_targ)},
            "n_resonances": int(len(res_sum.rows)),
            "E_min_mev": float(args.E_min),
            "E_max_mev": float(args.E_max),
            "n_grid_points": int(args.n_grid_points),
            "jwkb_npts": int(args.jwkb_npts),
            "jwkb_radial_npts": int(args.jwkb_radial_npts),
            "cross_section_model": "RatesMC Breit-Wigner (entrance and exit widths energy-dependent)",
            "integrated_xs_method": "exact per-resonance bin integrals",
        }
        if args.penetrability_model == "jwkb_real_omp":
            metadata.update(alpha_omp_metadata(args.omp_model))
        else:
            metadata.update({
                "penetrability_model": "coulomb",
                "omp_model": None,
                "coulomb_geometry": "Coulomb functions at channel radius",
                "imaginary_omp_ignored": None,
            })

        np.savetxt(
            unint_file,
            np.column_stack((E_test, xs_unint * 1e3)),
            delimiter=",",
            header=(
                "E (MeV),sigma (mb)"
                f" | penetrability_model={args.penetrability_model}"
                f" | omp_model={args.omp_model if args.penetrability_model == 'jwkb_real_omp' else 'none'}"
                " | plotting grid only: resonances narrower than the spacing are not resolved"
            ),
            fmt="%.4e"
        )
        metadata_file = unint_file.replace(".txt", "_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        print(f"Saved unintegrated cross sections to {unint_file}")
        print(f"Saved metadata to {metadata_file}")

    # ============================================================
    # Integrated (bin-averaged) cross sections
    # ============================================================
    dE_lst = [float(x.strip()) for x in args.dE.split(",")]

    if args.skip_integrated:
        print("Skipping integrated cross sections.")
        print("Done.")
        return

    for dE in dE_lst:
        print(f"\nIntegrating with ΔE = {dE} MeV")
        E_bins, xs_bin = binned_cross_section(res_sum, args.E_min, args.E_max, dE)

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
