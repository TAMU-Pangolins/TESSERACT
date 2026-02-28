#!/usr/bin/env python

import numpy as np
import argparse
import os
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from nucres.resonance import Resonance, sigma_bw_energy_dep
from nucres.physics import MASS_PROTON
from extract_resonance_data import load_resonance_data


# ============================================================
# Atomic number mapping
# ============================================================
Z_map = {
    "O": 8,
    "Ne": 10,
    "Mg": 12,
    "Si": 14,
    "S": 16,
    "Ar": 18,
    "Ca": 20
}


# ============================================================
def _sigma_worker(args):
    E, Z1, A1_proj, Z2, A1_target, res = args

    E_arr = np.array([E])
    xs_tot = 0.0

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
            gamma2=res["g1"][j]
        )

        xs_tot += xs_j

    return np.squeeze(xs_tot).item()


# ============================================================
def calc_cross_sections(E_array, Z1, A1_proj, Z2, A1_target, res_data, nproc):

    if nproc is None or nproc <= 1:
        return np.array([
            _sigma_worker((E, Z1, A1_proj, Z2, A1_target, res_data))
            for E in tqdm(E_array, desc="σ(E)", leave=False)
        ])

    nproc = min(nproc, cpu_count())

    tasks = [(E, Z1, A1_proj, Z2, A1_target, res_data) for E in E_array]

    with Pool(processes=nproc) as pool:
        xs = list(
            tqdm(
                pool.imap(_sigma_worker, tasks, chunksize=50),
                total=len(E_array),
                desc="σ(E)",
                leave=False
            )
        )

    return np.array(xs)


# ============================================================
def parse_reaction(reaction_str):
    import re
    match = re.match(r"(\d+)([A-Za-z]+)\(a,p\)", reaction_str)
    if not match:
        raise ValueError(f"Invalid reaction format: {reaction_str}")

    A1_target = int(match.group(1))
    element = match.group(2)

    if element not in Z_map:
        raise ValueError(f"Element {element} not supported.")

    return A1_target, Z_map[element]


# ============================================================
def build_master_grid(Emin, Emax, E_resol):
    deltaE = E_resol
    N_pts = int(round((Emax - Emin) / deltaE)) + 1
    E_full = np.linspace(Emin, Emax, N_pts)
    return E_full


# ============================================================
def merge_chunks(reaction):

    files = sorted(glob.glob(f"{reaction}_xs_part_*.txt"))
    if not files:
        raise RuntimeError("No chunk files found.")

    E_all = []
    xs_all = []

    for f in files:
        data = np.loadtxt(f, delimiter=",")
        E_all.append(data[:, 0])
        xs_all.append(data[:, 1])

    E_full = np.concatenate(E_all)
    xs_full = np.concatenate(xs_all)

    # Sort
    idx = np.argsort(E_full)
    E_full = E_full[idx]
    xs_full = xs_full[idx]

    # Remove duplicates if any
    E_full, unique_idx = np.unique(E_full, return_index=True)
    xs_full = xs_full[unique_idx]

    return E_full, xs_full


# ============================================================
def integrate_bins(E_full, xs_full, dE_list, reaction):

    deltaE = E_full[1] - E_full[0]

    for dE in dE_list:

        print(f"\nIntegrating with ΔE = {dE} MeV")

        points_per_bin = int(round(dE / deltaE))

        if points_per_bin < 2:
            raise ValueError("Bin width too small relative to resolution.")

        xs_bin = []
        E_bins = []

        for start in range(0, len(E_full) - points_per_bin, points_per_bin):

            end = start + points_per_bin + 1

            E_slice = E_full[start:end]
            xs_slice = xs_full[start:end]

            xs_int = np.trapezoid(xs_slice, E_slice)

            xs_bin.append(xs_int*1e3 / dE) #converting to mb
            E_bins.append(E_slice[0])

        np.savetxt(
            f"{reaction}_integrated_xs_dE_{dE}.txt",
            np.column_stack((E_bins, xs_bin)),
            delimiter=",",
            fmt="%.8e",
            header=f"E (MeV), sigma_bin (mb) | dE={dE}"
        )

        print(f"Saved integrated results for dE={dE}")


# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--reaction", type=str,required=True)
    parser.add_argument("--Emin", type=float, required=True)
    parser.add_argument("--Emax", type=float, required=True)
    parser.add_argument("--E_resol", type=float, required=True)
    parser.add_argument("--dE", type=float, nargs='+',required=True)

    parser.add_argument("--job_id", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--nproc", type=int, default=1)

    args = parser.parse_args()

    reaction = args.reaction

    # ------------------------------------------------------------
    # Merge + Integrate Mode
    # ------------------------------------------------------------
    if args.merge_only:
        print("Merging chunks...")
        E_full, xs_full = merge_chunks(reaction)
        #dE_list = [float(x.strip()) for x in args.dE.split(",")]
        integrate_bins(E_full, xs_full,args.dE,reaction)
        return

    # ------------------------------------------------------------
    # Compute Mode
    # ------------------------------------------------------------
    print(f"Processing reaction: {reaction}")

    output_dir = f"outputs/{reaction}/RUN_0/"
    infile = f"{output_dir}{reaction}.in"

    E_cm, g1, g2, g3, Jr, l1, l2, L, G = load_resonance_data(infile)

    res_data = {
        "E_cm": E_cm,
        "Jr": Jr,
        "g1": g1,
        "g2": g2,
        "L": L,
    }

    Z1, A1_proj = 2, 4
    A1_target, Z2 = parse_reaction(reaction)

    # Build master grid
    E_full = build_master_grid(args.Emin, args.Emax, args.E_resol)

    # Split grid if chunked
    if args.job_id is not None and args.n_jobs is not None:
        E_chunks = np.array_split(E_full, args.n_jobs)
        chunk_id = args.job_id % args.n_jobs
        E_test = E_chunks[chunk_id]
    else:
        E_test = E_full

    print(f"Computing {len(E_test)} energy points...")

    xs = calc_cross_sections(
        E_test,
        Z1, A1_proj,
        Z2, A1_target,
        res_data,
        nproc=args.nproc
    )

    # Save chunk
    if args.job_id is not None:
        outname = f"{reaction}_xs_part_{args.job_id}.txt"
    else:
        outname = f"{reaction}_xs_full.txt"

    np.savetxt(
        outname,
        np.column_stack((E_test, xs*1e3)),
        delimiter=",",
        fmt="%.4e",
        header="E (MeV), sigma (mb)"
    )

    print(f"Saved: {outname}")


# ============================================================
if __name__ == "__main__":
    main()
