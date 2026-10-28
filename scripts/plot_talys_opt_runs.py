#!/usr/bin/env python3
"""
Overlay every run's best-fit TALYS reaction rate found in a talys_optimization.out file.

talys_optimization.out is append-only: one "====="-delimited block per fit,
each containing a "Reaction rates  T9[GK]  rate[cm3/s/mol]:" section. This
overlays every run's rate curve on the same axes, so you can see at a glance
whether runs differ from each other.

Usage:
    python scripts/plot_talys_opt_runs.py outputs/talys_opt/14O_ap_17F/binwidth_0.2/talys_optimization.out
    python scripts/plot_talys_opt_runs.py .../talys_optimization.out --output runs.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_talys_results_new import parse_talys_opt_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "talys_out",
        type=Path,
        help="Path to talys_optimization.out, "
             "e.g. outputs/talys_opt/<reaction_slug>/binwidth_<dE>/talys_optimization.out",
    )
    parser.add_argument("--t9-min", type=float, default=None, help="Lower T9 (GK) cutoff.")
    parser.add_argument("--t9-max", type=float, default=None, help="Upper T9 (GK) cutoff.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Per-run line transparency (default: 0.25).")
    parser.add_argument("--output", type=Path, default=None, help="Save the plot here instead of showing it.")
    args = parser.parse_args()

    if not args.talys_out.exists():
        raise SystemExit(f"{args.talys_out} not found.")

    records = parse_talys_opt_out(str(args.talys_out))
    if not records:
        raise SystemExit(f"No fit blocks (with 'Run index') found in {args.talys_out}.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    n_plotted = 0
    for rec in records:
        t9, rate = rec["x_rate"], rec["y_rate"]
        if t9 is None or rate is None:
            continue

        mask = np.isfinite(t9) & np.isfinite(rate) & (rate > 0.0)
        if args.t9_min is not None:
            mask &= t9 >= args.t9_min
        if args.t9_max is not None:
            mask &= t9 <= args.t9_max
        if not np.any(mask):
            continue

        ax.plot(t9[mask], rate[mask], color="C0", linewidth=0.9, alpha=args.alpha)
        n_plotted += 1

    if n_plotted == 0:
        raise SystemExit("Found fit blocks, but no usable rate curves to plot.")

    ax.set_yscale("log")
    ax.set_xlabel(r"$T_9$ [GK]")
    ax.set_ylabel(r"TALYS best-fit rate [cm$^3$ mol$^{-1}$ s$^{-1}$]")
    ax.set_title(f"{args.talys_out.parent.name}: {n_plotted} TALYS fit runs")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    print(f"Plotted {n_plotted} of {len(records)} run(s) found.")

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
