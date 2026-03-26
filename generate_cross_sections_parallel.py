#!/usr/bin/env python

# from multiprocessing import Pool, cpu_count
# import matplotlib.pyplot as plt
# import numpy as np
# from io import StringIO
# import pandas as pd
# from nucres.resonance import Resonance, sigma_bw_energy_dep
# from nucres.physics import *
# from tqdm import tqdm, trange
# from extract_resonance_data import load_resonance_data
# import argparse
# import os

# # ============================================================
# # Worker function (REQUIRED for multiprocessing)
# # ============================================================
# def _sigma_worker(args):
#     E, Z1, A1, Z2, A2, res = args

#     E_arr = np.array([E])
#     xs_tot = 0.0

#     E_cm = res["E_cm"]
#     Jr   = res["Jr"]
#     g1   = res["g1"]
#     g2   = res["g2"]
#     L    = res["L"]

#     for j in range(len(E_cm)):
#         r_j = Resonance(
#             E_cm[j]*1e6,
#             Jr[j],
#             0, 0,
#             3.65e-26,
#             6.64e-27,
#             g1[j],
#             g2[j],
#         )

#         xs_j = sigma_bw_energy_dep(
#             E_arr*1e6,
#             r_j,
#             Z1, Z2, A1, A2,
#             L[j],
#             gamma2=g1[j]
#         )

#         xs_tot += xs_j

#     return float(xs_tot)


# # ============================================================
# # Parallelized cross section calculator
# # ============================================================
# def calc_cross_sections(E_test,Z1, A1, Z2, A2,res_data,show_progress=False,nproc=None):

#     if nproc is None:
#         nproc = cpu_count()

#     tasks = [(E, Z1, A1, Z2, A2, res_data) for E in E_test]

#     with Pool(processes=nproc) as pool:
#         if show_progress:
#             xs = list(
#                 tqdm(
#                     pool.imap(_sigma_worker, tasks, chunksize=50),
#                     total=len(E_test),
#                     desc="Calculating σ(E)",
#                     leave=False
#                 )
#             )
#         else:
#             xs = pool.map(_sigma_worker, tasks)

#     return np.array(xs)


# # ============================================================
# # Main driver
# # ============================================================
# def main():

#     # ------------------------------------------------
#     # Load resonance data ONCE (safe on macOS)
#     # ------------------------------------------------
#     E_cm, g1, g2, g3, Jr, l1, l2, L, G = load_resonance_data()

#     res_data = {
#         "E_cm": E_cm,
#         "Jr": Jr,
#         "g1": g1,
#         "g2": g2,
#         "L": L,
#     }

#     # ------------------------------------------------
#     # Define reaction parameters
#     # ------------------------------------------------
#     Z1, A1 = 2, 4      # projectile
#     Z2, A2 = 12, 22    # target



#     xs_unint_file = "22Mg_ap_25Al_xs_unintegrated_parallel.txt"

#     if os.path.exists(xs_unint_file):
#         print("Found existing unintegrated cross sections. Loading from file...")
#         data = np.loadtxt(xs_unint_file, delimiter=",")
#         E_test = data[:, 0]
#         xs_unint = data[:, 1] / 1e3   # convert back from mb → base units
#     else:
#         print("Computing unintegrated cross sections...")
#         E_test = np.linspace(0.1, 10.0, 10000)

#         xs_unint = calc_cross_sections(
#             E_test,
#             Z1, A1, Z2, A2,
#             res_data,
#             show_progress=True,
#             nproc=8)

#         np.savetxt(
#             xs_unint_file,
#             np.column_stack((E_test, xs_unint * 1e3)),
#             delimiter=",",
#             header="E (MeV),sigma (mb)",
#             fmt="%.4e")


#     # ------------------------------------------------
#     # Integrated cross sections
#     # ------------------------------------------------

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--Emin", type=float, default=0.1,
#                         help="Minimum energy [MeV]")
#     parser.add_argument("--Emax", type=float, default=10.0,
#                         help="Maximum energy [MeV]")
#     parser.add_argument("--N_bins", type=int, default=1000,
#                         help="Number of bins per integration interval")
#     parser.add_argument("--dE", type=str, required=True,
#                         help="Bin widths [MeV], comma-separated")
    

#     args = parser.parse_args()

#     E_min = args.Emin if args.Emin is not None else np.min(E_cm)
#     E_max = args.Emax if args.Emax is not None else np.max(E_cm)
#     N_bins = args.N_bins

#     E_min = np.round(E_min, 2)
#     E_max = np.round(E_max, 2)

#     dE_lst = [float(x.strip()) for x in args.dE.split(",")]

#     print(f"E_min = {E_min}")
#     print(f"E_max = {E_max}")
#     print(f"dE list = {dE_lst}")

#     m = 0

#     for dE in dE_lst:
#         xs_bin = []
#         E_bins = np.arange(E_min, E_max + dE, dE)

#         for E0 in tqdm(E_bins, desc=f"Integrating bins (ΔE={dE} MeV)"):
#             E1 = E0 + dE
#             E_int = np.linspace(E0, E1, N_bins)

#             xs_total = calc_cross_sections(
#                 E_int,
#                 Z1, A1, Z2, A2,
#                 res_data,
#                 show_progress=False,
#                 nproc=8
#             )

#             xs_int = np.trapezoid(xs_total, E_int)
#             xs_bin.append(xs_int / dE)

#         xs_bin = np.array(xs_bin)

#         np.savetxt(
#             f"22Mg_ap_25Al_integrated_xs_parallel_{m}.txt",
#             np.column_stack((E_bins, xs_bin)),
#             delimiter=",",
#             header="E (MeV),sigma (mb)| dE=%.2f MeV" % dE,
#             fmt="%.4e"
#         )

#         m += 1

#         plt.plot(E_bins,xs_bin*1e3,label=rf"$\Delta$E = {dE} MeV")
    

#     plt.plot(E_test,xs_unint*1e3,color='k',label = 'Before integration')
#     plt.yscale('log')
#     plt.title('22Mg(a,p)25Al')
#     plt.legend()
#     plt.xlabel('E (MeV)',fontsize=12)
#     plt.ylabel(r'$\sigma$ (mb)',fonsize=12)

#     plt.savefig('22Mg_ap_25Al.png')

#     plt.show()


# # ============================================================
# # REQUIRED for multiprocessing + nohup
# # ============================================================
# if __name__ == "__main__":
#     main()










from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

from nucres.resonance import Resonance, sigma_bw_energy_dep
from nucres.physics import HBAR, MASS_PROTON, reduced_mass
from nucres.sampling import porter_thomas_factors
from extract_resonance_data import load_resonance_data, load_nuclear_params
from tqdm import tqdm


def sample_widths(A_tar, n, mean_i, mean_o):
    """
    Draw per-resonance partial widths from a Porter-Thomas distribution
    centred on mean_i/mean_o * Wigner single-particle limit.

    Parameters
    ----------
    A_tar  : target mass number
    n      : number of resonances  (= len(E_cm))
    mean_i : Wigner-limit fraction for the entrance (alpha) channel
    mean_o : Wigner-limit fraction for the exit (proton) channel

    Returns
    -------
    g1, g2 : ndarrays of length n  [eV]
    """
    A_alpha  = 4
    A_proton = 1
    m_alpha  = A_alpha  * MASS_PROTON
    m_target = A_tar    * MASS_PROTON

    mu_i   = reduced_mass(m_target, m_alpha)
    R_sq_i = (1.25e-15)**2 * (A_tar**(1/3) + A_alpha**(1/3))**2
    wig_i  = 3 * HBAR**2 / (2 * mu_i * R_sq_i) * 6.242e18
    g1     = wig_i * porter_thomas_factors(n, mean_i, df=1)

    mu_o   = reduced_mass(m_target, A_proton * MASS_PROTON)
    R_sq_o = (1.25e-15)**2 * (A_tar**(1/3) + A_proton**(1/3))**2
    wig_o  = 3 * HBAR**2 / (2 * mu_o * R_sq_o) * 6.242e18
    g2     = wig_o * porter_thomas_factors(n, mean_o, df=1)

    return g1, g2


def _sigma_worker(args):
    E, Z1, A1_proj, Z2, A1_target, res = args

    E_arr = np.array([E])
    xs_tot = 0.0

    # Correct masses
    m_alpha = A1_proj * MASS_PROTON
    m_target = A1_target * MASS_PROTON

    for j in range(len(res["E_cm"])):

        r_j = Resonance(
            res["E_cm"][j] * 1e6,
            res["Jr"][j],
            0, 0,
            m_target,
            m_alpha,
            res["g1"][j],
            res["g2"][j],
        )

        xs_j = sigma_bw_energy_dep(
            E_arr * 1e6,
            r_j,
            Z1, Z2,
            A1_proj, A1_target,
            res["L"][j],
            gamma2=res["g2"][j]
        )

        xs_tot += xs_j
#    print(np.shape(xs_tot))
    return np.squeeze(xs_tot).item()


# ============================================================
def calc_cross_sections(E_test, Z1, A1_proj, Z2, A1_target, res_data, nproc=None):

    if nproc is None:
        nproc = 1

    nproc = min(nproc,cpu_count())

    tasks = [(E, Z1, A1_proj, Z2, A1_target, res_data) for E in E_test]

    with Pool(processes=nproc) as pool:
        xs = list(
            tqdm(
                pool.imap(_sigma_worker, tasks, chunksize=50),
                total=len(E_test),
                desc="Calculating σ(E)",
                leave=False
            )
        )

    return np.array(xs)


# ============================================================
# Reaction parser — kept for reference but no longer used.
# Z, A of target and projectile are now read directly from the
# RatesMC .in file via load_nuclear_params(), so there is no
# need for a hardcoded Z_map or element lookup.
# ============================================================
# def parse_reaction(reaction_str):
#     """
#     Example:
#     22Mg(a,p)25Al
#     """
#     Z_map = {
#         "O": 8, "Ne": 10, "Mg": 12, "Si": 14,
#         "S": 16, "Ar": 18, "Ca": 20,
#     }
#     match = re.match(r"(\d+)([A-Za-z]+)\(a,p\)", reaction_str)
#     if not match:
#         raise ValueError(f"Reaction format not recognized: {reaction_str}")
#     A1_target = int(match.group(1))
#     element = match.group(2)
#     if element not in Z_map:
#         raise ValueError(f"Element {element} not supported in Z_map.")
#     Z2 = Z_map[element]
#     return A1_target, Z2


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

    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--mean-i", dest="mean_i", type=float, default=0.010,
                        help="Wigner-limit fraction for entrance (alpha) channel")
    parser.add_argument("--mean-o", dest="mean_o", type=float, default=0.0045,
                        help="Wigner-limit fraction for exit (proton) channel")
    parser.add_argument("--skip-unintegrated", dest="skip_unintegrated",
                        action="store_true", default=False,
                        help="Skip computing unintegrated cross sections and load from "
                             "the existing file instead.")
    parser.add_argument("--skip-integrated", dest="skip_integrated",
                        action="store_true", default=False,
                        help="Skip computing integrated cross sections.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix appended to output filenames before .txt "
                             "(e.g. --tag _test produces 22Mg(a,p)25Al_xs_unintegrated_parallel_test.txt)")
    args = parser.parse_args()

    reaction = args.reaction

    output_dir = f"outputs/{reaction}/RUN_0/"
    infile = f"{output_dir}{reaction}.in"

    if not os.path.exists(infile):
        raise FileNotFoundError(f"{infile} not found.")

    # Parse target and residual for output filenames: "22Mg(a,p)25Al" → "22Mg", "25Al"
    rxn_match = re.match(r'([^(]+)\(a,p\)(.+)', reaction)
    if not rxn_match:
        raise ValueError(f"Cannot parse target/residual from reaction: {reaction}")
    target   = rxn_match.group(1)
    residual = rxn_match.group(2)
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

    # ------------------------------------------------
    # Sample per-resonance widths from Porter-Thomas
    # ------------------------------------------------
    n = len(E_cm)
    g1, g2 = sample_widths(A1_target, n, args.mean_i, args.mean_o)
    print(f"Sampled {n} resonance widths  (mean_i={args.mean_i}, mean_o={args.mean_o})")

    res_data = {
        "E_cm": E_cm,
        "Jr":   Jr,
        "g1":   g1,
        "g2":   g2,
        "L":    L,
    }

    # ============================================================
    # Unintegrated cross section
    # ============================================================
    unint_file = f"{reaction}_xs_unintegrated_parallel{args.tag}.txt"

    if args.skip_unintegrated:
        print(f"Loading existing unintegrated cross sections from {unint_file} ...")
        data = np.loadtxt(unint_file, delimiter=",")
        E_test   = data[:, 0]
        xs_unint = data[:, 1] * 1e-3   # mb → base units to match in-memory convention
    else:
        print("Computing unintegrated cross sections...")
        E_test = np.linspace(args.E_min, args.E_max, args.n_grid_points)

        xs_unint = calc_cross_sections(
            E_test,
            Z1, A1_proj,
            Z2, A1_target,
            res_data, nproc=args.nproc
        )

        np.savetxt(
            unint_file,
            np.column_stack((E_test, xs_unint * 1e3)),
            delimiter=",",
            header="E (MeV),sigma (mb)",
            fmt="%.4e"
        )

    # ============================================================
    # Integrated cross sections — reuse the unintegrated grid,
    # no recomputation of cross sections.
    # ============================================================
    dE_lst = [float(x.strip()) for x in args.dE.split(",")]

    if args.skip_integrated:
        print("Skipping integrated cross sections.")
        print("Done.")
        return

    # Grid spacing of the unintegrated array
    deltaE = E_test[1] - E_test[0]

    for dE in dE_lst:

        print(f"\nIntegrating with ΔE = {dE} MeV")

        points_per_bin = int(round(dE / deltaE))

        if points_per_bin < 2:
            raise ValueError(
                f"\n  Bin width dE={dE} MeV is finer than the grid resolution "
                f"({deltaE:.4e} MeV/point, n_grid_points={args.n_grid_points}).\n"
                f"  Only {points_per_bin} point(s) fall inside a single bin — "
                f"need at least 2 for integration.\n"
                f"  Fix: increase n_grid_points (currently {args.n_grid_points}) "
                f"or use a larger dE (currently {dE} MeV)."
            )

        if points_per_bin < 10:
            print(f"  Warning: only {points_per_bin} grid points per bin for dE={dE} MeV "
                  f"(grid resolution = {deltaE:.4e} MeV/point, n_grid_points={args.n_grid_points}) — "
                  f"integration accuracy may be low.")

        xs_bin = []
        E_bins = []

        for start in range(0, len(E_test) - points_per_bin, points_per_bin):
            end      = start + points_per_bin + 1
            E_slice  = E_test[start:end]
            xs_slice = xs_unint[start:end]
            xs_int   = np.trapz(xs_slice, E_slice)
            xs_bin.append(xs_int * 1e3 / dE)   # convert to mb and normalise by bin width
            E_bins.append(E_slice[0])

        xs_bin = np.array(xs_bin)
        E_bins = np.array(E_bins)

        np.savetxt(
            f"{target}_ap_{residual}_integrated_xs_dE_{dE}{tag_suffix}.csv",
            np.column_stack((E_bins, xs_bin)),
            delimiter=",",
            header=f"E (MeV),sigma (mb)| dE={dE}",
            fmt="%.4e"
        )

    print("Done.")


# ============================================================
if __name__ == "__main__":
    main()
