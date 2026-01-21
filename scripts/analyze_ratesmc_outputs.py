from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def iter_ratesmc_out(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        reaction = path.parts[-3]
        run = path.parts[-2]
        rows.extend(read_ratesmc_out(path, reaction, run))
    if not rows:
        return pd.DataFrame(
            columns=["reaction", "run", "T9", "low", "median", "high", "fu"]
        )
    return pd.DataFrame(
        rows, columns=["reaction", "run", "T9", "low", "median", "high", "fu"]
    )


def read_ratesmc_out(path: Path, reaction: str, run: str) -> Iterable[Tuple]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip().startswith("T9"):
                break
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            t9, low, med, high, fu = map(float, parts[:5])
            rows.append((reaction, run, t9, low, med, high, fu))
    return rows


def parse_quantiles(value: str) -> Tuple[float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "quantiles must be three comma values, e.g. 0.16,0.5,0.84"
        )
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize RatesMC batch outputs and plot quantile bands."
    )
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory containing reaction/Run_* folders (default: outputs).",
    )
    parser.add_argument(
        "--pattern",
        default="*/Run_*/RatesMC.out",
        help="Glob pattern relative to root (default: */Run_*/RatesMC.out).",
    )
    parser.add_argument(
        "--reaction",
        default=None,
        help="Filter to a single reaction name (folder name under root).",
    )
    parser.add_argument(
        "--rate-column",
        default="median",
        choices=["low", "median", "high", "fu"],
        help="Which column from RatesMC.out to analyze (default: median).",
    )
    parser.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=(0.16, 0.5, 0.84),
        help="Quantiles to plot (default: 0.16,0.5,0.84).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write plot to file instead of showing it.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Write summary stats to CSV.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob(args.pattern))
    if args.reaction:
        paths = [p for p in paths if p.parts[-3] == args.reaction]

    data = iter_ratesmc_out(paths)
    if data.empty:
        raise SystemExit("No RatesMC.out files found for the given inputs.")

    q_low, q_mid, q_high = args.quantiles
    grouped = data.groupby(["reaction", "T9"])[args.rate_column]
    stats = grouped.quantile([q_low, q_mid, q_high]).unstack()
    stats.columns = ["q_low", "q_mid", "q_high"]

    if args.summary_csv:
        stats.reset_index().to_csv(args.summary_csv, index=False)

    if args.reaction:
        reactions = [args.reaction]
    else:
        reactions = stats.index.get_level_values(0).unique().tolist()

    for reaction in reactions:
        series = stats.loc[reaction]
        plt.figure(figsize=(7, 4.5))
        plt.fill_between(series.index, series["q_low"], series["q_high"], alpha=0.2)
        plt.plot(series.index, series["q_mid"], label="median")
        plt.yscale("log")
        plt.xlabel("T9")
        plt.ylabel(f"{args.rate_column} (across runs)")
        plt.title(reaction)
        plt.legend()
        plt.tight_layout()

        if args.output:
            out_path = Path(args.output)
            if len(reactions) > 1:
                stem = out_path.stem
                suffix = out_path.suffix or ".png"
                per_reaction = out_path.with_name(f"{stem}_{reaction}{suffix}")
                plt.savefig(per_reaction, dpi=150)
            else:
                plt.savefig(out_path, dpi=150)
        else:
            plt.show()
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
