from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import pandas as pd
from nucres.resonance import Resonance, sigma_bw_energy_dep
from nucres.physics import *
from tqdm import tqdm, trange
from extract_resonance_data import load_resonance_data
import argparse
import os

# ============================================================
# Worker function (REQUIRED for multiprocessing)
# ============================================================
def _sigma_worker(args):
    E, Z1, A1, Z2, A2, res = args

    E_arr = np.array([E])
    xs_tot = 0.0

    E_cm = res["E_cm"]
    Jr   = res["Jr"]
    g1   = res["g1"]
    g2   = res["g2"]
    L    = res["L"]

    for j in range(len(E_cm)):
        r_j = Resonance(
            E_cm[j]*1e6,
            Jr[j],
            0, 0,
            3.65e-26,
            6.64e-27,
            g1[j],
            g2[j],
        )

        xs_j = sigma_bw_energy_dep(
            E_arr*1e6,
            r_j,
            Z1, Z2, A1, A2,
            L[j],
            gamma2=g1[j]
        )

        xs_tot += xs_j

    return float(xs_tot)


# ============================================================
# Parallelized cross section calculator
# ============================================================
def calc_cross_sections(E_test,Z1, A1, Z2, A2,res_data,show_progress=False,nproc=None):

    if nproc is None:
        nproc = cpu_count()

    tasks = [(E, Z1, A1, Z2, A2, res_data) for E in E_test]

    with Pool(processes=nproc) as pool:
        if show_progress:
            xs = list(
                tqdm(
                    pool.imap(_sigma_worker, tasks, chunksize=50),
                    total=len(E_test),
                    desc="Calculating σ(E)",
                    leave=False
                )
            )
        else:
            xs = pool.map(_sigma_worker, tasks)

    return np.array(xs)


# ============================================================
# Main driver
# ============================================================
def main():

    # ------------------------------------------------
    # Load resonance data ONCE (safe on macOS)
    # ------------------------------------------------
    E_cm, g1, g2, g3, Jr, l1, l2, L, G = load_resonance_data()

    res_data = {
        "E_cm": E_cm,
        "Jr": Jr,
        "g1": g1,
        "g2": g2,
        "L": L,
    }

    # ------------------------------------------------
    # Define reaction parameters
    # ------------------------------------------------
    Z1, A1 = 2, 4      # projectile
    Z2, A2 = 12, 22    # target



    xs_unint_file = "22Mg_ap_25Al_xs_unintegrated_parallel.txt"

    if os.path.exists(xs_unint_file):
        print("Found existing unintegrated cross sections. Loading from file...")
        data = np.loadtxt(xs_unint_file, delimiter=",")
        E_test = data[:, 0]
        xs_unint = data[:, 1] / 1e3   # convert back from mb → base units
    else:
        print("Computing unintegrated cross sections...")
        E_test = np.linspace(0.1, 10.0, 10000)

        xs_unint = calc_cross_sections(
            E_test,
            Z1, A1, Z2, A2,
            res_data,
            show_progress=True,
            nproc=8)

        np.savetxt(
            xs_unint_file,
            np.column_stack((E_test, xs_unint * 1e3)),
            delimiter=",",
            header="E (MeV),sigma (mb)",
            fmt="%.4e")


    # ------------------------------------------------
    # Integrated cross sections
    # ------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("--Emin", type=float, default=0.1,
                        help="Minimum energy [MeV]")
    parser.add_argument("--Emax", type=float, default=10.0,
                        help="Maximum energy [MeV]")
    parser.add_argument("--N_bins", type=int, default=1000,
                        help="Number of bins per integration interval")
    parser.add_argument("--dE", type=str, required=True,
                        help="Bin widths [MeV], comma-separated")
    

    args = parser.parse_args()

    E_min = args.Emin if args.Emin is not None else np.min(E_cm)
    E_max = args.Emax if args.Emax is not None else np.max(E_cm)
    N_bins = args.N_bins

    E_min = np.round(E_min, 2)
    E_max = np.round(E_max, 2)

    dE_lst = [float(x.strip()) for x in args.dE.split(",")]

    print(f"E_min = {E_min}")
    print(f"E_max = {E_max}")
    print(f"dE list = {dE_lst}")

    m = 0

    for dE in dE_lst:
        xs_bin = []
        E_bins = np.arange(E_min, E_max + dE, dE)

        for E0 in tqdm(E_bins, desc=f"Integrating bins (ΔE={dE} MeV)"):
            E1 = E0 + dE
            E_int = np.linspace(E0, E1, N_bins)

            xs_total = calc_cross_sections(
                E_int,
                Z1, A1, Z2, A2,
                res_data,
                show_progress=False,
                nproc=8
            )

            xs_int = np.trapezoid(xs_total, E_int)
            xs_bin.append(xs_int / dE)

        xs_bin = np.array(xs_bin)

        np.savetxt(
            f"22Mg_ap_25Al_integrated_xs_parallel_{m}.txt",
            np.column_stack((E_bins, xs_bin)),
            delimiter=",",
            header="E (MeV),sigma (mb)| dE=%.2f MeV" % dE,
            fmt="%.4e"
        )

        m += 1

        plt.plot(E_bins,xs_bin*1e3,label=rf"$\Delta$E = {dE} MeV")
    

    plt.plot(E_test,xs_unint*1e3,color='k',label = 'Before integration')
    plt.yscale('log')
    plt.title('22Mg(a,p)25Al')
    plt.legend()
    plt.xlabel('E (MeV)',fontsize=12)
    plt.ylabel(r'$\sigma$ (mb)',fonsize=12)

    plt.savefig('22Mg_ap_25Al.png')

    plt.show()


# ============================================================
# REQUIRED for multiprocessing + nohup
# ============================================================
if __name__ == "__main__":
    main()
