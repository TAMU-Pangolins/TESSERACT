#!/usr/bin/env python3
"""
plot_talys_results.py — Visualise TALYS optimisation results.

Plots the ratio of TALYS best-fit reaction rate to RatesMC rate (median +
2-sigma band), across one or more bin widths overlaid on the same graph.

All input/output paths are derived from the reaction elements and a base
outputs directory, following the THICC layout:

    {base_dir}/talys_opt/{target}_{projectile}{ejectile}_{residual}/binwidth_{bw}/talys_optimization.out
    {base_dir}/resonances/{target}_{projectile}{ejectile}_{residual}/{target}({projectile},{ejectile}){residual}/RUN_N/...
    {base_dir}/plots/  (default output location)

Usage:
    python plot_talys_results.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al

    # Restrict to specific bin width(s) instead of plotting all of them:
    python plot_talys_results.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al \
        --bin-width 0.2

    # Multiple bin widths overlaid on the same graph:
    python plot_talys_results.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al \
        --bin-width 0.2 0.5
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# Reaction naming
# ─────────────────────────────────────────────────────────────────────────────
def reaction_names(target: str, projectile: str, ejectile: str, residual: str) -> tuple[str, str]:
    """
    Build the reaction slug (used for directory/file names) and the reaction
    string (used to look up RatesMC .out files), e.g.

        target=22Mg, projectile=a, ejectile=p, residual=25Al
        -> slug     = "22Mg_ap_25Al"
        -> reaction = "22Mg(a,p)25Al"
    """
    slug = f"{target}_{projectile}{ejectile}_{residual}"
    reaction = f"{target}({projectile},{ejectile}){residual}"
    return slug, reaction


def discover_bin_widths(talys_opt_dir: Path) -> list[str]:
    """Find every binwidth_* subdirectory that has a talys_optimization.out file."""
    widths = []
    if talys_opt_dir.is_dir():
        for p in sorted(talys_opt_dir.glob("binwidth_*")):
            if p.is_dir() and (p / "talys_optimization.out").exists():
                widths.append(p.name[len("binwidth_"):])

    def sort_key(w):
        try:
            return (0, float(w))
        except ValueError:
            return (1, w)

    widths.sort(key=sort_key)
    return widths


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
# Build the TALYS / RatesMC rate-ratio envelope for one bin width
# ─────────────────────────────────────────────────────────────────────────────
def compute_rate_ratio_envelope(records, npz_dir: str, ratesmc_dir: str, reaction: str):
    """
    For each run: interpolate both TALYS and RatesMC rates onto a common T9
    grid (union of all TALYS T9 points), compute the ratio, then take the
    median and 2-sigma envelope across runs.

    Returns (t9_common, med, lo, hi, valid_env) or None if no valid pairs.
    """
    missing_runs  = []
    ratio_curves  = []   # list of (t9_talys, rat_talys, t9_rmc, rat_rmc)
    t9_union      = set()

    for rec in records:
        ridx = rec["run_idx"]

        # TALYS rate arrays — prefer .out block, fallback to .npz
        if rec["x_rate"] is not None and rec["y_rate"] is not None:
            t9_talys  = np.asarray(rec["x_rate"], dtype=float)
            rat_talys = np.asarray(rec["y_rate"], dtype=float)
        else:
            npz = load_talys_rate_from_npz(npz_dir, ridx, input_file=rec["input_file"])
            if npz is None:
                missing_runs.append(ridx)
                continue
            t9_talys, rat_talys = npz

        # RatesMC median rate for this run
        rmc = load_ratesmc_rate(ratesmc_dir, reaction, ridx) if ridx is not None else None
        if rmc is None:
            missing_runs.append(ridx)
            continue
        t9_rmc, rat_rmc = rmc

        # Collect TALYS T9 values (positive-rate points only) for common grid
        pos_mask = np.isfinite(rat_talys) & (rat_talys > 0) & np.isfinite(t9_talys)
        t9_union.update(t9_talys[pos_mask].tolist())

        ratio_curves.append((t9_talys, rat_talys, t9_rmc, rat_rmc))

    if missing_runs:
        print(f"    Skipped {len(missing_runs)} run(s) with missing rate data: "
              f"{missing_runs[:10]}{'...' if len(missing_runs) > 10 else ''}")

    if not ratio_curves:
        return None

    t9_common = np.array(sorted(t9_union))

    all_ratios = []
    for t9_talys, rat_talys, t9_rmc, rat_rmc in ratio_curves:
        talys_on_grid = np.array([rate_at_t9(t9_talys, rat_talys, t) for t in t9_common])
        rmc_on_grid   = np.array([rate_at_t9(t9_rmc,   rat_rmc,   t) for t in t9_common])

        valid = (np.isfinite(talys_on_grid) & np.isfinite(rmc_on_grid)
                 & (rmc_on_grid > 0) & (talys_on_grid > 0))
        if valid.sum() < 2:
            continue

        ratio = np.full_like(t9_common, np.nan)
        ratio[valid] = talys_on_grid[valid] / rmc_on_grid[valid]
        all_ratios.append(ratio)

    if not all_ratios:
        return None

    stack = np.vstack(all_ratios)   # shape (n_runs, n_t9)
    med   = np.nanpercentile(stack, 50, axis=0)
    lo    = np.nanpercentile(stack, 2.5, axis=0)
    hi    = np.nanpercentile(stack, 97.5, axis=0)
    valid_env = np.isfinite(med) & (med > 0)

    return t9_common, med, lo, hi, valid_env


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot TALYS optimisation results")
    parser.add_argument("--target",     required=True, help="Target nucleus, e.g. 22Mg")
    parser.add_argument("--projectile", required=True, help="Projectile, e.g. a")
    parser.add_argument("--ejectile",   required=True, help="Ejectile, e.g. p")
    parser.add_argument("--residual",   required=True, help="Residual nucleus, e.g. 25Al")
    parser.add_argument("--bin-width", nargs="*", default=None,
                         help="Bin width(s) to plot, e.g. 0.2 or 0.2 0.5. "
                              "Default: plot every bin width found for this reaction.")
    parser.add_argument("--base-dir", default="/data/pagroup/pa02/THICC/outputs",
                         help="Root outputs directory containing talys_opt/, resonances/, and plots/")
    parser.add_argument("--output-dir", default=None,
                         help="Directory to save plots (default: {base-dir}/plots)")
    parser.add_argument("--no-axes", action="store_true",
                         help="Also save an artistic, axes-less version of the plot "
                              "(for public lectures and talks)")

    args = parser.parse_args()

    reaction_slug, reaction = reaction_names(args.target, args.projectile,
                                              args.ejectile, args.residual)

    base_dir      = Path(args.base_dir)
    talys_opt_dir = base_dir / "talys_opt" / reaction_slug
    ratesmc_dir   = base_dir / "resonances" / reaction_slug / reaction
    output_dir    = Path(args.output_dir) if args.output_dir else base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bin_width:
        bin_widths = args.bin_width
    else:
        bin_widths = discover_bin_widths(talys_opt_dir)
        if not bin_widths:
            sys.exit(f"No binwidth_* directories with talys_optimization.out found under {talys_opt_dir}")
        print(f"No --bin-width given; using all {len(bin_widths)} found: {', '.join(bin_widths)}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    multi  = len(bin_widths) > 1

    fig, ax = plt.subplots(figsize=(10, 5))
    envelopes = {}   # bin_width -> (t9_common, med, lo, hi, valid_env)

    for i, bw in enumerate(bin_widths):
        talys_out = talys_opt_dir / f"binwidth_{bw}" / "talys_optimization.out"
        npz_dir   = talys_opt_dir / f"binwidth_{bw}"

        if not talys_out.exists():
            print(f"[binwidth {bw}] talys_optimization.out not found at {talys_out} — skipping")
            continue

        print(f"[binwidth {bw}] Parsing {talys_out} ...")
        records = parse_talys_opt_out(str(talys_out))
        if not records:
            print(f"  No records found — skipping")
            continue
        print(f"  Found {len(records)} run(s)")

        result = compute_rate_ratio_envelope(records, str(npz_dir), str(ratesmc_dir), reaction)
        if result is None:
            print(f"  No valid TALYS/RatesMC rate pairs — skipping")
            continue

        envelopes[bw] = result
        t9_common, med, lo, hi, valid_env = result
        color = colors[i % len(colors)]

        #med_label  = f"Median (bin width {bw})" if multi else "Median across runs"
        band_label = f"2$\\sigma$ band (bin width {bw})" if multi else r"2$\sigma$ band"

        #ax.plot(t9_common[valid_env], med[valid_env], color=color, lw=2.0, label=med_label)
        ax.fill_between(t9_common[valid_env], lo[valid_env], hi[valid_env],
                         color=color, alpha=0.25, label=band_label)

    if not envelopes:
        sys.exit("No valid rate-ratio data found for any bin width — nothing plotted.")

    ax.axhline(1.0, color="k", lw=1.5, linestyle="--")
    ax.set_xlabel("T9 (GK)", fontsize=12)
    ax.set_xlim(0., 2.1)
    ax.set_ylabel(r"$RR_{\rm Talys} / RR_{\rm RMC}$", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    fig.tight_layout()

    bw_label = bin_widths[0] if len(envelopes) == 1 else ""
    out_main = output_dir / f"{reaction_slug}_bw_{bw_label}.png"
    fig.savefig(out_main, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_main}")
    plt.show()

    if args.no_axes:
        fig2 = plt.figure(figsize=(10, 5))
        ax2 = plt.gca()

        for i, bw in enumerate(bin_widths):
            if bw not in envelopes:
                continue
            t9_common, med, lo, hi, valid_env = envelopes[bw]
            color = colors[i % len(colors)] if multi else "darkgreen"
            ax2.fill_between(t9_common[valid_env], lo[valid_env], hi[valid_env],
                              color=color, alpha=0.4)

        ax2.axhline(y=1.0, ls="--", color="purple", lw=3)

        plt.xticks([])
        plt.yticks([1.0, 20.0, 40.0, 60.0, 80.0], fontsize=24)

        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax2.annotate("", xy=(1.05, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction",
                     textcoords="axes fraction",
                     arrowprops=dict(facecolor="black", edgecolor="black", lw=2,
                                      headwidth=15, headlength=18, width=4))
        ax2.text(0.5, -0.05, "Temperature of Reaction", transform=ax2.transAxes,
                  color="black", ha="center", va="top", fontsize=28, fontname="serif")

        ax2.annotate("", xy=(0.0, 1.05), xytext=(0.0, 0.0), xycoords="axes fraction",
                     textcoords="axes fraction",
                     arrowprops=dict(facecolor="black", edgecolor="black", lw=2,
                                      headwidth=15, headlength=18, width=4))
        ax2.text(-0.09, 0.5, "Ratio of Reaction Rates", color="black",
                  transform=ax2.transAxes, ha="right", va="center",
                  rotation="vertical", fontsize=28, fontname="serif")

        out_no_axes = output_dir / f"{reaction_slug}_no_axes.png"
        fig2.savefig(out_no_axes, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_no_axes}")
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()

