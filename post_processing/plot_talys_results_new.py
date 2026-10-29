#!/usr/bin/env python3
"""
plot_talys_results_new.py — Visualise TALYS optimisation results.

Plots the ratio of TALYS best-fit reaction rate to RatesMC rate (median +
95% interval, i.e. the 2.5-97.5 percentile band across runs — not
necessarily a true Gaussian 2-sigma band unless the underlying distribution
happens to be Gaussian), across one or more bin widths overlaid on the same
graph.

All input/output paths are derived from the reaction elements and a base
outputs directory, following the TESSERACT layout:

    {base_dir}/talys_opt/{target}_{projectile}{ejectile}_{residual}/binwidth_{bw}/talys_optimization.out
    {base_dir}/resonances/{target}_{projectile}{ejectile}_{residual}/{target}({projectile},{ejectile}){residual}/RUN_N/...
    {base_dir}/plots/  (default output location)

Usage:
    python plot_talys_results_new.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al

    # Restrict to specific bin width(s) instead of plotting all of them:
    python plot_talys_results_new.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al \
        --bin-width 0.2

    # Multiple bin widths overlaid on the same graph:
    python plot_talys_results_new.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al \
        --bin-width 0.2 0.5

    # Skip the 95% interval and plot every run's ratio curve instead:
    python plot_talys_results_new.py \
        --target 22Mg --projectile a --ejectile p --residual 25Al --no-band
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from nucres.ratesmc_output import read_ratesmc_out
from nucres.talys_output import fit_key, keep_latest

# ─────────────────────────────────────────────────────────────────────────────
# Phys. Rev. C figure sizing
# ─────────────────────────────────────────────────────────────────────────────
FIGSIZE = (8, 6)
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 6.5

# Okabe-Ito colorblind-safe qualitative palette (black held back, since it's
# already used for the ratio=1 reference line and axis spines).
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
]


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

        rec = {"params": {}, "x_rate": None, "y_rate": None, "input_file": "",
               "fit_key": fit_key(block)}

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

    # One record per fit (re-runs append further blocks for the same fit).
    return keep_latest(records)


# ─────────────────────────────────────────────────────────────────────────────
# Read TALYS rate from .npz  (fallback if not in .out)
# ─────────────────────────────────────────────────────────────────────────────
_NPZ_RUN_RE = re.compile(r"talys_results_.*_run(\d+)\.npz$")


def index_npz_dir(npz_dir: str) -> dict:
    """
    One-time scan of npz_dir: run_idx -> filename.

    load_talys_rate_from_npz() used to re-list and re-sort the whole
    directory on every call (once per run on the fallback path) — O(n_runs
    x n_files_in_dir) and expensive on networked cluster storage where each
    os.listdir/stat is a round trip. Scanning once and looking up by run
    index afterwards is O(n_files_in_dir) total instead.
    """
    index = {}
    if not os.path.isdir(npz_dir):
        return index
    for fname in sorted(os.listdir(npz_dir)):
        m = _NPZ_RUN_RE.match(fname)
        if m:
            index[int(m.group(1))] = fname
    return index


def load_talys_rate_from_npz(npz_dir: str, npz_index: dict, run_idx, input_file: str = "") -> tuple:
    """
    Find talys_results_*.npz for a given run.

    Search order:
      1. If run_idx is an int, look it up in npz_index (built once by
         index_npz_dir() over npz_dir).
      2. If input_file is set, derive the stem from its basename and look for
         talys_results_{stem}.npz  (handles test files without _runN suffix)

    Returns ((x, y), None) on success, or (None, reason) on failure.
    """
    # Try by run index first
    if run_idx is not None:
        fname = npz_index.get(run_idx)
        if fname is not None:
            return _load_npz_rate(os.path.join(npz_dir, fname))

    # Fallback: derive stem from input_file (e.g. test files without _runN)
    if input_file:
        stem = os.path.splitext(os.path.basename(input_file))[0]
        candidate = os.path.join(npz_dir, f"talys_results_{stem}.npz")
        if os.path.exists(candidate):
            return _load_npz_rate(candidate)

    return None, "no rate table in talys_optimization.out and no matching talys_results_*.npz"


def _load_npz_rate(path: str) -> tuple:
    name = os.path.basename(path)
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        return None, f"no rate table in talys_optimization.out; {name} unreadable ({exc})"
    x = data.get("x_rate")
    y = data.get("y_best_rate")
    if x is None or y is None:
        return None, f"no rate table in talys_optimization.out; {name} has no x_rate/y_best_rate"
    if not len(x):
        return None, f"no rate table in talys_optimization.out; {name} has empty rate arrays"
    return (np.asarray(x, dtype=float), np.asarray(y, dtype=float)), None


# ─────────────────────────────────────────────────────────────────────────────
# Read RatesMC rate from {reaction}.out in RUN_N folder
# ─────────────────────────────────────────────────────────────────────────────
def load_ratesmc_rate(ratesmc_dir: str, reaction: str, run_idx: int) -> tuple:
    """
    Load (T9, median_rate) for one run from {ratesmc_dir}/RUN_{run_idx}/.

    Reads {reaction}.out (written by tesseract.py [ratesmc]), falling back to
    RatesMC.out. Columns are found by header name, so RatesMC 2.11 output
    (T9, low, Classical, Median, ...) and 2.2+ output (T9, low, Median,
    high, f.u.) are both read correctly.

    Returns ((T9, median), None) on success, or (None, reason) on failure.
    """
    run_dir = os.path.join(ratesmc_dir, f"RUN_{run_idx}")
    if not os.path.isdir(run_dir):
        return None, f"RUN_{run_idx} directory not found"
    for name in (f"{reaction}.out", "RatesMC.out"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            try:
                if os.path.getsize(path) == 0:
                    return None, f"empty file: {name}"
                table = read_ratesmc_out(path)
            except (OSError, ValueError) as exc:
                return None, f"faulty file: {name} ({str(exc).replace(path + ': ', '')})"
            if not _has_usable_rates(table["T9"], table["median"]):
                return None, f"faulty file: {name} (fewer than 2 positive median rates)"
            return (table["T9"], table["median"]), None
    return None, f"no {reaction}.out or RatesMC.out in RUN_{run_idx}"


def _has_usable_rates(t9_arr, rate_arr) -> bool:
    """True if there are at least 2 finite, positive rate points (needed to interpolate)."""
    t9_arr   = np.asarray(t9_arr,   dtype=float)
    rate_arr = np.asarray(rate_arr, dtype=float)
    return int((np.isfinite(t9_arr) & np.isfinite(rate_arr) & (rate_arr > 0)).sum()) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Interpolate rate onto a T9 grid
# ─────────────────────────────────────────────────────────────────────────────
def rate_interpolator(t9_arr, rate_arr):
    """
    Build a log-space rate interpolator from (t9_arr, rate_arr), usable on a
    whole T9 array at once. Interpolation is done in log space to handle
    rates spanning many orders of magnitude. Returns None if there are fewer
    than 2 finite, positive points.
    """
    if t9_arr is None or rate_arr is None:
        return None
    t9_arr   = np.asarray(t9_arr,   dtype=float)
    rate_arr = np.asarray(rate_arr, dtype=float)
    mask = np.isfinite(rate_arr) & (rate_arr > 0) & np.isfinite(t9_arr)
    if mask.sum() < 2:
        return None
    log_rate = np.log(rate_arr[mask])
    return interp1d(t9_arr[mask], log_rate, bounds_error=False, fill_value=np.nan)


def rates_at_t9_grid(f, t9_grid: np.ndarray) -> np.ndarray:
    """Evaluate a rate_interpolator() result on a whole T9 grid at once."""
    if f is None:
        return np.full(len(t9_grid), np.nan)
    with np.errstate(over="ignore"):
        return np.exp(f(t9_grid))


# ─────────────────────────────────────────────────────────────────────────────
# Build the TALYS / RatesMC rate-ratio curves for one bin width
# ─────────────────────────────────────────────────────────────────────────────
def compute_rate_ratios(records, npz_dir: str, ratesmc_dir: str, reaction: str):
    """
    For each run: interpolate both TALYS and RatesMC rates onto a common T9
    grid (union of all TALYS T9 points), compute the ratio.

    Returns (t9_common, all_ratios, labels) where all_ratios is a list of
    per-run ratio arrays (aligned to t9_common, possibly containing NaN where
    a run has no coverage), or None if no valid pairs.
    """
    failures      = []   # list of (run_label, talys_reason, ratesmc_reason)
    ratio_curves  = []   # list of (run_label, t9_talys, rat_talys, t9_rmc, rat_rmc)
    t9_union      = set()
    npz_index     = index_npz_dir(npz_dir)

    for rec in records:
        ridx = rec["run_idx"]
        label = ridx if ridx is not None else (os.path.basename(rec["input_file"]) or "?")

        # TALYS rate arrays — prefer .out block, fallback to .npz
        talys, talys_reason = None, None
        if rec["x_rate"] is not None and rec["y_rate"] is not None:
            talys = (np.asarray(rec["x_rate"], dtype=float),
                     np.asarray(rec["y_rate"], dtype=float))
        else:
            talys, talys_reason = load_talys_rate_from_npz(npz_dir, npz_index, ridx,
                                                           input_file=rec["input_file"])
        if talys is not None and not _has_usable_rates(*talys):
            talys, talys_reason = None, "rates extracted but fewer than 2 positive values"

        # RatesMC median rate for this run (checked even if TALYS failed, so
        # runs where both sides are broken are reported as such)
        if ridx is None:
            rmc, rmc_reason = None, "no run index in talys_optimization.out, cannot locate RUN_N"
        else:
            rmc, rmc_reason = load_ratesmc_rate(ratesmc_dir, reaction, ridx)

        if talys is None or rmc is None:
            failures.append((label, talys_reason, rmc_reason))
            continue

        t9_talys, rat_talys = talys
        t9_rmc, rat_rmc = rmc

        # Collect TALYS T9 values (positive-rate points only) for common grid
        pos_mask = np.isfinite(rat_talys) & (rat_talys > 0) & np.isfinite(t9_talys)
        t9_union.update(t9_talys[pos_mask].tolist())

        ratio_curves.append((label, t9_talys, rat_talys, t9_rmc, rat_rmc))

    t9_common = np.array(sorted(t9_union))

    all_ratios = []
    labels = []
    no_overlap = []
    for label, t9_talys, rat_talys, t9_rmc, rat_rmc in ratio_curves:
        f_talys = rate_interpolator(t9_talys, rat_talys)
        f_rmc   = rate_interpolator(t9_rmc,   rat_rmc)
        talys_on_grid = rates_at_t9_grid(f_talys, t9_common)
        rmc_on_grid   = rates_at_t9_grid(f_rmc,   t9_common)

        valid = (np.isfinite(talys_on_grid) & np.isfinite(rmc_on_grid)
                 & (rmc_on_grid > 0) & (talys_on_grid > 0))
        if valid.sum() < 2:
            no_overlap.append(label)
            continue

        ratio = np.full_like(t9_common, np.nan)
        ratio[valid] = talys_on_grid[valid] / rmc_on_grid[valid]
        all_ratios.append(ratio)
        labels.append(label)

    report_skipped_runs(failures, no_overlap, len(records))

    if not all_ratios:
        return None

    return t9_common, all_ratios, labels


def rate_ratio_envelope(t9_common, all_ratios):
    """95% interval (2.5-97.5 percentile) and median across a list of per-run ratio arrays."""
    stack = np.vstack(all_ratios)   # shape (n_runs, n_t9)
    med   = np.nanpercentile(stack, 50, axis=0)
    lo    = np.nanpercentile(stack, 2.5, axis=0)
    hi    = np.nanpercentile(stack, 97.5, axis=0)
    valid_env = np.isfinite(med) & (med > 0)
    return med, lo, hi, valid_env


def report_skipped_runs(failures, no_overlap, n_total):
    """
    Print every skipped run (no truncation), grouped by what failed, followed
    by the specific reason for each run.
    """
    n_skipped = len(failures) + len(no_overlap)
    if not n_skipped:
        print(f"    All {n_total} run(s) used.")
        return

    def run_sort_key(label):
        return (0, label, "") if isinstance(label, int) else (1, 0, str(label))

    talys_only = sorted((f for f in failures if f[1] and not f[2]), key=lambda f: run_sort_key(f[0]))
    rmc_only   = sorted((f for f in failures if f[2] and not f[1]), key=lambda f: run_sort_key(f[0]))
    both       = sorted((f for f in failures if f[1] and f[2]),     key=lambda f: run_sort_key(f[0]))
    no_overlap = sorted(no_overlap, key=run_sort_key)

    print(f"    Skipped {n_skipped} of {n_total} run(s):")
    groups = [
        ("TALYS optimization failed only", [f[0] for f in talys_only]),
        ("RatesMC failed only",            [f[0] for f in rmc_only]),
        ("Both TALYS and RatesMC failed",  [f[0] for f in both]),
        ("Both OK but < 2 overlapping T9 points", no_overlap),
    ]
    for title, labels in groups:
        if labels:
            print(f"      {title} ({len(labels)}): {', '.join(str(l) for l in labels)}")

    print("    Per-run details:")
    for label, talys_reason, rmc_reason in sorted(failures, key=lambda f: run_sort_key(f[0])):
        parts = []
        if talys_reason:
            parts.append(f"TALYS: {talys_reason}")
        if rmc_reason:
            parts.append(f"RatesMC: {rmc_reason}")
        print(f"      Run {label}: " + " | ".join(parts))
    for label in no_overlap:
        print(f"      Run {label}: TALYS and RatesMC rates have < 2 common positive T9 points")


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
    parser.add_argument("--no-band", action="store_true",
                         help="Skip the 95%% interval calculation and instead plot every "
                              "run's individual ratio curve.")
    parser.add_argument("--no-axes", action="store_true",
                         help="Also save an artistic, axes-less version of the plot "
                              "(for public lectures and talks, not for publication — "
                              "not sized to Phys. Rev. C column width).")

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

    multi = len(bin_widths) > 1

    fig, ax = plt.subplots(figsize=FIGSIZE)
    results = {}   # bin_width -> (t9_common, all_ratios, labels)

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

        result = compute_rate_ratios(records, str(npz_dir), str(ratesmc_dir), reaction)
        if result is None:
            print(f"  No valid TALYS/RatesMC rate pairs — skipping")
            continue

        results[bw] = result
        t9_common, all_ratios, labels = result
        color = OKABE_ITO[i % len(OKABE_ITO)]

        if args.no_band:
            curve_label = f"bin width {bw}" if multi else "individual runs"
            for j, ratio in enumerate(all_ratios):
                valid = np.isfinite(ratio) & (ratio > 0)
                ax.plot(t9_common[valid], ratio[valid], color=color, lw=0.5, alpha=0.25,
                        label=curve_label if j == 0 else None)
        else:
            med, lo, hi, valid_env = rate_ratio_envelope(t9_common, all_ratios)
            band_label = f"95% interval (bin width {bw})" if multi else "95% interval"
            ax.fill_between(t9_common[valid_env], lo[valid_env], hi[valid_env],
                             color=color, alpha=0.3, label=band_label, linewidth=0)

    if not results:
        sys.exit("No valid rate-ratio data found for any bin width — nothing plotted.")

    ax.axhline(1.0, color="k", lw=1.0, linestyle="--")
    ax.set_xlabel(r"$T_9$ (GK)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0., 2.1)
    ax.set_ylabel(r"$RR_{\mathrm{TALYS}} / RR_{\mathrm{RMC}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    fig.tight_layout()

    bw_label = bin_widths[0] if len(results) == 1 else ""
    suffix = "_no_band" if args.no_band else ""
    out_main = output_dir / f"{reaction_slug}_bw_{bw_label}{suffix}.png"
    fig.savefig(out_main, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_main}")
    plt.show()

    if args.no_axes:
        fig2 = plt.figure(figsize=(10, 5))
        ax2 = plt.gca()

        for i, bw in enumerate(bin_widths):
            if bw not in results:
                continue
            t9_common, all_ratios, labels = results[bw]
            med, lo, hi, valid_env = rate_ratio_envelope(t9_common, all_ratios)
            color = OKABE_ITO[i % len(OKABE_ITO)] if multi else "darkgreen"
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
