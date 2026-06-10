#!/usr/bin/env python3
"""Plot a wide-format TALYS/reference reaction-rate ratio comparison."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BLOCK_SPLIT_RE = re.compile(r"={10,}\s*\n")


def parse_talys_rate_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[dict] = []
    for block in BLOCK_SPLIT_RE.split(text):
        if "Reaction rates" not in block:
            continue
        run_match = re.search(r"Run index\s*:\s*(\d+)", block)
        if not run_match:
            continue
        run_idx = int(run_match.group(1))
        chi2_match = re.search(r"Min reduced chi-square:\s*([+-]?[\d.eE+-]+)", block)
        chi2 = float(chi2_match.group(1)) if chi2_match else np.nan

        t9_vals: list[float] = []
        rate_vals: list[float] = []
        in_rates = False
        for line in block.splitlines():
            if "Reaction rates" in line:
                in_rates = True
                continue
            if in_rates and line.startswith("="):
                break
            if in_rates and line.startswith("-"):
                continue
            if not in_rates:
                continue
            nums = re.findall(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", line)
            if len(nums) >= 2:
                t9_vals.append(float(nums[0]))
                rate_vals.append(float(nums[1]))

        t9 = np.asarray(t9_vals, dtype=float)
        rate = np.asarray(rate_vals, dtype=float)
        mask = np.isfinite(t9) & np.isfinite(rate) & (rate > 0.0)
        if np.count_nonzero(mask) >= 4:
            records.append({"run_idx": run_idx, "chi2_red": chi2, "t9": t9[mask], "rate": rate[mask]})
    return records


def load_reference_rate(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path, skiprows=4)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 3:
        return None
    t9 = data[:, 0]
    rate = data[:, 2]
    mask = np.isfinite(t9) & np.isfinite(rate) & (rate > 0.0)
    if np.count_nonzero(mask) < 4:
        return None
    return t9[mask], rate[mask]


def log_interp(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    out = np.full_like(grid, np.nan, dtype=float)
    valid = (grid >= x[0]) & (grid <= x[-1])
    out[valid] = np.exp(np.interp(grid[valid], x, np.log(y)))
    return out


def build_ratio_stack(
    talys_records: list[dict],
    *,
    ratesmc_dir: Path,
    reaction: str,
    t9_min: float,
    t9_max: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    t9_grid = np.linspace(t9_min, t9_max, n_grid)
    ratios: list[np.ndarray] = []
    used_runs: list[int] = []

    for rec in talys_records:
        run_idx = rec["run_idx"]
        ref_path = ratesmc_dir / f"RUN_{run_idx}" / f"{reaction}.out"
        ref = load_reference_rate(ref_path)
        if ref is None:
            continue
        talys_on_grid = log_interp(rec["t9"], rec["rate"], t9_grid)
        ref_on_grid = log_interp(ref[0], ref[1], t9_grid)
        valid = np.isfinite(talys_on_grid) & np.isfinite(ref_on_grid) & (ref_on_grid > 0.0)
        if np.count_nonzero(valid) < 4:
            continue
        ratio = np.full_like(t9_grid, np.nan, dtype=float)
        ratio[valid] = talys_on_grid[valid] / ref_on_grid[valid]
        ratios.append(ratio)
        used_runs.append(run_idx)

    if not ratios:
        raise ValueError("No matched TALYS/reference rate curves found.")
    return t9_grid, np.vstack(ratios), used_runs


def plot_rate_ratio(
    output: Path,
    *,
    t9: np.ndarray,
    ratio_stack: np.ndarray,
    reaction: str,
    show_source: bool,
) -> None:
    med = np.nanpercentile(ratio_stack, 50, axis=0)
    lo = np.nanpercentile(ratio_stack, 16, axis=0)
    hi = np.nanpercentile(ratio_stack, 84, axis=0)
    valid = np.isfinite(med) & np.isfinite(lo) & np.isfinite(hi) & (lo > 0.0) & (hi > 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(t9[valid], lo[valid], hi[valid], color="C0", alpha=0.25, linewidth=0, label=r"$1\sigma$ band")
    ax.plot(t9[valid], med[valid], color="C0", linewidth=2.0, label="Median")
    ax.axhline(1.0, color="#8b1a1a", linestyle="--", linewidth=1.2)

    title = reaction.replace("(a,p)", "(α,p)").replace("(a,g)", "(α,γ)")
    ax.set_title(title)
    ax.set_xlabel(r"$T_9$ (GK)")
    ax.set_ylabel(r"$R_\mathrm{TALYS}/R_\mathrm{ref}$")
    ax.set_yscale("log")
    ax.set_xlim(float(np.nanmin(t9)), float(np.nanmax(t9)))
    if np.any(valid):
        ymin = max(0.02, float(np.nanmin(lo[valid])) * 0.75)
        ymax = min(250.0, float(np.nanmax(hi[valid])) * 1.15)
        if ymin < ymax:
            ax.set_ylim(ymin, ymax)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False)
    if show_source:
        ax.text(
            0.98,
            0.04,
            f"n = {ratio_stack.shape[0]} matched runs",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#555555",
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a wide-format TALYS/reference rate-ratio plot.")
    parser.add_argument("--talys-out", type=Path, required=True, help="talys_optimization.out with TALYS rates.")
    parser.add_argument("--ratesmc-dir", type=Path, required=True, help="Directory containing RUN_*/<reaction>.out.")
    parser.add_argument("--reaction", default="22Mg(a,p)25Al", help="RatesMC output filename/reaction label.")
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/rate_ratio/rate_ratio_wide.png"))
    parser.add_argument("--t9-min", type=float, default=0.1)
    parser.add_argument("--t9-max", type=float, default=2.2)
    parser.add_argument("--n-grid", type=int, default=140)
    parser.add_argument("--show-source", action="store_true", help="Annotate the number of matched runs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = parse_talys_rate_records(args.talys_out)
    if not records:
        raise SystemExit(f"No TALYS rate records found in {args.talys_out}")
    t9, ratio_stack, used_runs = build_ratio_stack(
        records,
        ratesmc_dir=args.ratesmc_dir,
        reaction=args.reaction,
        t9_min=args.t9_min,
        t9_max=args.t9_max,
        n_grid=args.n_grid,
    )
    plot_rate_ratio(
        args.output,
        t9=t9,
        ratio_stack=ratio_stack,
        reaction=args.reaction,
        show_source=args.show_source,
    )
    print(f"Wrote rate-ratio plot: {args.output}")
    print(f"Matched runs: {len(used_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
