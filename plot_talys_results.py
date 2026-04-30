1#!/usr/bin/env python3
"""
plot_talys_results.py — Visualise TALYS optimisation results.

Plots:
  1. Each optimised parameter vs. min reduced chi-square (one panel per param).
  2. Ratio of TALYS best-fit reaction rate to RatesMC rate, per run index.

Usage:
    python plot_talys_results.py \
        --talys-out   /path/to/talys_opt/talys_optimization.out \
        --npz-dir     /path/to/talys_opt \
        --ratesmc-dir /path/to/resonances/22Mg(a,p)25Al \
        --reaction    "22Mg(a,p)25Al" \
        [--output-dir  .]
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# Parse talys_optimization.out
# ─────────────────────────────────────────────────────────────────────────────
def parse_talys_opt_out(path: str) -> list[dict]:
    """
    Parse every === block in talys_optimization.out.

    Returns a list of dicts, each with:
        run_idx   : int or None   (None when the file has no _run{N} suffix)
        chi2_red  : float
        params    : {name: value}
        x_rate    : np.ndarray | None   (T9 values)
        y_rate    : np.ndarray | None   (rate values)
        input_file: str
    """
    records = []

    with open(path) as fh:
        content = fh.read()

    # Split on === block boundaries
    blocks = re.split(r"={10,}\n", content)

    for block in blocks:
        if "Run index" not in block:
            continue

        rec = {"params": {}, "x_rate": None, "y_rate": None, "input_file": ""}

        # Input file path (used as fallback for NPZ lookup)
        mf = re.search(r"Input file\s*:\s*(.+)", block)
        rec["input_file"] = mf.group(1).strip() if mf else ""

        # Run index — may be a digit or "?" when exp_file has no _run{N} suffix
        m = re.search(r"Run index\s*:\s*(\d+)", block)
        rec["run_idx"] = int(m.group(1)) if m else None

        # Min reduced chi-square
        m = re.search(r"Min reduced chi-square:\s*([\d.eE+\-]+)", block)
        rec["chi2_red"] = float(m.group(1)) if m else np.nan

        # Optimised parameters block
        in_params = False
        in_rates  = False
        rate_t9   = []
        rate_val  = []

        for line in block.splitlines():
            if "Optimized parameters:" in line:
                in_params = True
                in_rates  = False
                continue
            if "Cross sections" in line:
                in_params = False
                in_rates  = False
                continue
            if "Reaction rates" in line:
                in_params = False
                in_rates  = True
                continue

            if in_params:
                # Format:   rvadjust_a = +1.472000
                pm = re.match(r"\s+([\w]+)\s*=\s*([\d.eE+\-]+)", line)
                if pm:
                    rec["params"][pm.group(1)] = float(pm.group(2))

            if in_rates:
                nums = re.findall(r"[\d.eE+\-]+", line)
                if len(nums) >= 2:
                    rate_t9.append(float(nums[0]))
                    rate_val.append(float(nums[1]))

        if rate_t9:
            rec["x_rate"] = np.array(rate_t9)
            rec["y_rate"] = np.array(rate_val)

        records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Read TALYS rate from .npz  (fallback if not in .out)
# ─────────────────────────────────────────────────────────────────────────────
def load_talys_rate_from_npz(npz_dir: str, run_idx, input_file: str = "") -> tuple | None:
    """
    Find talys_results_*.npz for a given run.

    Search order:
      1. If run_idx is an int, look for talys_results_*_run{run_idx}.npz
      2. If input_file is set, derive the stem from its basename and look for
         talys_results_{stem}.npz  (handles test files without _runN suffix)
      3. Return None if nothing found.
    """
    # Try by run index first
    if run_idx is not None:
        pattern = re.compile(rf"talys_results_.*_run{run_idx}\.npz$")
        for fname in sorted(os.listdir(npz_dir)):
            if pattern.match(fname):
                return _load_npz_rate(os.path.join(npz_dir, fname))

    # Fallback: derive stem from input_file (e.g. test files without _runN)
    if input_file:
        stem = os.path.splitext(os.path.basename(input_file))[0]
        candidate = os.path.join(npz_dir, f"talys_results_{stem}.npz")
        if os.path.exists(candidate):
            return _load_npz_rate(candidate)

    return None


def _load_npz_rate(path: str) -> tuple | None:
    data = np.load(path, allow_pickle=True)
    x = data.get("x_rate")
    y = data.get("y_best_rate")
    if x is not None and y is not None and len(x):
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Read RatesMC rate from {reaction}.out in RUN_N folder
# ─────────────────────────────────────────────────────────────────────────────
def load_ratesmc_rate(ratesmc_dir: str, reaction: str, run_idx: int) -> tuple | None:
    """
    Load (T9, median_rate) from {ratesmc_dir}/RUN_{run_idx}/{reaction}.out.

    RatesMC file format (3 header lines then data):
        line 1 : reaction name
        line 2 : "Calculated with RatesMC ..."
        line 3 : "Samples = N"
        line 4 : "T9   RRate_low   Median Rate   RRate_high   f.u."
        data   : col 0 = T9 (GK),  col 2 = Median Rate
    """
    path = os.path.join(ratesmc_dir, f"RUN_{run_idx}", f"{reaction}.out")
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, skiprows=4)
        if data.ndim < 2 or data.shape[1] < 3:
            return None
        return data[:, 0], data[:, 2]   # T9, Median Rate
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Interpolate rate at a given T9
# ─────────────────────────────────────────────────────────────────────────────
def rate_at_t9(t9_arr, rate_arr, t9_query):
    """
    Interpolate rate_arr (defined on t9_arr) at t9_query.
    Interpolation is done in log space to handle rates spanning many orders of
    magnitude.  Returns NaN if the query is outside the covered T9 range or if
    there are fewer than 2 finite, positive points.
    """
    if t9_arr is None or rate_arr is None:
        return np.nan
    t9_arr   = np.asarray(t9_arr,   dtype=float)
    rate_arr = np.asarray(rate_arr, dtype=float)
    # Keep only finite positive values (zero rates cannot be log-interpolated)
    mask = np.isfinite(rate_arr) & (rate_arr > 0) & np.isfinite(t9_arr)
    if mask.sum() < 2:
        return np.nan
    log_rate = np.log(rate_arr[mask])
    f = interp1d(t9_arr[mask], log_rate, bounds_error=False, fill_value=np.nan)
    val = f(t9_query)
    return float(np.exp(val)) if np.isfinite(val) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot TALYS optimisation results")
    parser.add_argument("--talys-out",   required=True,
                        help="Path to talys_optimization.out")
    #parser.add_argument("--npz-dir",
    #                    help="Directory containing talys_results_*.npz files")
    parser.add_argument("--ratesmc-dir", required=True,
                        help="Root resonance dir, e.g. outputs/resonances/22Mg(a,p)25Al")
    parser.add_argument("--reaction",    required=True,
                        help="Reaction name matching .out filename, e.g. '22Mg(a,p)25Al'")
    parser.add_argument("--output-dir",  default=".",
                        help="Directory to save plots (default: current dir)")
    parser.add_argument("--save-fig", help='Name of the reaction rate ratio plot',default='rate_vs_T9_plot.png')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load optimisation records ─────────────────────────────────────────────
    print(f"Parsing {args.talys_out} ...")
    records = parse_talys_opt_out(args.talys_out)
    if not records:
        sys.exit("No records found in talys_optimization.out")
    print(f"  Found {len(records)} run(s)")

    # ── Plot 1: optimised parameters vs chi2_red ──────────────────────────────
    # Collect all parameter names across all records
    all_params = []
    for rec in records:
        for k in rec["params"]:
            if k not in all_params:
                all_params.append(k)

    chi2_vals = np.array([rec["chi2_red"] for rec in records])

    n_params = len(all_params)
#    if n_params > 0:
#        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), squeeze=False)
#        for ax, pname in zip(axes[0], all_params):
#            pvals = np.array([rec["params"].get(pname, np.nan) for rec in records])
#            mask  = np.isfinite(pvals) & np.isfinite(chi2_vals)
#            sc = ax.scatter(pvals[mask], chi2_vals[mask], c=chi2_vals[mask],
#                            cmap="viridis_r", edgecolors="k", linewidths=0.4, s=40)
#            ax.set_xlabel(pname, fontsize=12)
#            ax.set_ylabel(r"Min $\chi^2_\nu$", fontsize=12)
#            ax.set_title(pname)
#            plt.colorbar(sc, ax=ax, label=r"$\chi^2_\nu$")
#        fig.suptitle(f"{args.reaction} — optimised parameters vs $\\chi^2_\\nu$", fontsize=13)
#        fig.tight_layout()
#        out1 = os.path.join(args.output_dir, "params_vs_chi2.png")
#        fig.savefig(out1, dpi=150, bbox_inches="tight")
#        print(f"Saved: {out1}")
#        plt.show()

    # ── Plot 2: TALYS / RatesMC rate ratio vs full T9 curve ──────────────────
    # Build a common T9 grid from the union of all TALYS T9 arrays.
    # For each run: interpolate both TALYS and RatesMC rates onto this grid,
    # then compute the ratio.  Collect all ratio curves for envelope plotting.

    missing_runs  = []
    ratio_curves  = []   # list of (t9_common, ratio_array) per run
    t9_union      = set()

    for rec in records:
        ridx = rec["run_idx"]

        # TALYS rate arrays — prefer .out block, fallback to .npz
        if rec["x_rate"] is not None and rec["y_rate"] is not None:
            t9_talys   = np.asarray(rec["x_rate"], dtype=float)
            rat_talys  = np.asarray(rec["y_rate"], dtype=float)
        else:
            npz = load_talys_rate_from_npz(args.npz_dir, ridx,
                                           input_file=rec["input_file"])
            if npz is None:
                missing_runs.append(ridx)
                continue
            t9_talys, rat_talys = npz

        # RatesMC median rate for this run
        if ridx is not None:
            rmc = load_ratesmc_rate(args.ratesmc_dir, args.reaction, ridx)
        else:
            rmc = None
        if rmc is None:
            missing_runs.append(ridx)
            continue
        t9_rmc, rat_rmc = rmc

        # Collect TALYS T9 values (positive-rate points only) for common grid
        pos_mask = np.isfinite(rat_talys) & (rat_talys > 0) & np.isfinite(t9_talys)
        t9_union.update(t9_talys[pos_mask].tolist())

        ratio_curves.append((t9_talys, rat_talys, t9_rmc, rat_rmc, ridx))

    if missing_runs:
        print(f"  Skipped {len(missing_runs)} run(s) with missing rate data: "
              f"{missing_runs[:10]}{'...' if len(missing_runs) > 10 else ''}")

    if not ratio_curves:
        print("No valid rate pairs found — rate ratio plot skipped.")
    else:
        # Common T9 grid (sorted)
        t9_common = np.array(sorted(t9_union))

        # Compute ratio curve for each run on the common grid
        all_ratios = []
        fig, ax = plt.subplots(figsize=(10, 5))

        for t9_talys, rat_talys, t9_rmc, rat_rmc, ridx in ratio_curves:
            # Interpolate TALYS rate onto common grid (log space)
            talys_on_grid = np.array([rate_at_t9(t9_talys, rat_talys, t)
                                      for t in t9_common])
            # Interpolate RatesMC median rate onto same common grid (log space)
            rmc_on_grid   = np.array([rate_at_t9(t9_rmc,   rat_rmc,   t)
                                      for t in t9_common])

            valid = (np.isfinite(talys_on_grid) & np.isfinite(rmc_on_grid)
                     & (rmc_on_grid > 0) & (talys_on_grid > 0))
            if valid.sum() < 2:
                continue

            ratio = np.full_like(t9_common, np.nan)
            ratio[valid] = talys_on_grid[valid] / rmc_on_grid[valid]
            all_ratios.append(ratio)

#            ax.plot(t9_common[valid], ratio[valid],
#                    color="steelblue", alpha=0.3, lw=0.8)

        if all_ratios:
            # Median and 16th/84th percentile envelope across all runs
            stack = np.vstack(all_ratios)   # shape (n_runs, n_t9)
            med   = np.nanpercentile(stack, 50, axis=0)
            lo    = np.nanpercentile(stack, 16, axis=0)
            hi    = np.nanpercentile(stack, 84, axis=0)

            valid_env = np.isfinite(med) & (med > 0)
            ax.plot(t9_common[valid_env], med[valid_env],
                    color="navy", lw=2.0, label="Median across runs")
            ax.fill_between(t9_common[valid_env],
                            lo[valid_env], hi[valid_env],
                            color="steelblue", alpha=0.25, label=r"1$\sigma$ band")

        ax.axhline(1.0, color="k", lw=1.5, linestyle="--", label="Ratio = 1")
        ax.set_xlabel("T9 (GK)", fontsize=12)
        ax.set_xlim(0.,2.1)
        ax.set_ylabel("Reaction Rate  (TALYS best-fit / RatesMC median)", fontsize=12)
        ax.set_title(f"{args.reaction} — TALYS vs RatesMC rate ratio", fontsize=13)
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        fig.tight_layout()
        out2 = os.path.join(args.output_dir, args.save_fig)
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved: {out2}")
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
