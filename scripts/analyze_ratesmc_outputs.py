from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nucres.ratesmc_output import (
    find_ratesmc_outputs,
    iter_reaction_dirs,
    read_ratesmc_out as read_ratesmc_table,
)


COLUMNS = ["reaction", "run", "T9", "low", "median", "high", "fu"]


def iter_ratesmc_out(entries: Iterable[Tuple[str, str, Path]]) -> pd.DataFrame:
    """Rows of (reaction, run, T9, low, median, high, fu) from RatesMC tables.

    Columns are located by header name, so RatesMC 2.11 and 2.2+ outputs are
    both read correctly; fu is NaN for 2.11 files, which have no f.u. column.
    """
    frames = []
    for reaction, run, path in entries:
        table = read_ratesmc_table(path)
        n = len(table["T9"])
        frames.append(pd.DataFrame({
            "reaction": reaction,
            "run": run,
            "T9": table["T9"],
            "low": table["low"],
            "median": table["median"],
            "high": table["high"],
            "fu": table.get("fu", np.full(n, np.nan)),
        }))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)[COLUMNS]


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
        help=(
            "Root directory containing <reaction>/RUN_<j>/<reaction>.out "
            "(tesseract.py) or <reaction>/Run_<NN>/RatesMC.out "
            "(run_ratesmc_batches.sh) (default: outputs)."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Optional glob relative to root overriding the run-directory "
            "search, e.g. '*/Run_*/RatesMC.out'; the reaction and run are "
            "taken from the file's grandparent and parent directory names."
        ),
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
    if args.pattern:
        entries = [
            (p.parts[-3], p.parts[-2], p) for p in sorted(root.glob(args.pattern))
        ]
        if args.reaction:
            entries = [e for e in entries if e[0] == args.reaction]
    else:
        entries = [
            (rdir.name, run, path)
            for rdir in iter_reaction_dirs(root, args.reaction)
            for run, path in find_ratesmc_outputs(rdir)
        ]

    data = iter_ratesmc_out(entries)
    if data.empty:
        raise SystemExit(f"No RatesMC rate tables found under {root}.")
    if data[args.rate_column].isna().all():
        raise SystemExit(
            f"Column {args.rate_column!r} is not in these RatesMC outputs "
            "(RatesMC 2.11 writes no f.u. column)."
        )

    q_low, q_mid, q_high = args.quantiles
    grouped = data.groupby(["reaction", "T9"])[args.rate_column]
    # Quantiles one at a time so each label matches the requested level even
    # when the levels are not given in increasing order.
    stats = pd.DataFrame({
        "q_low": grouped.quantile(q_low),
        "q_mid": grouped.quantile(q_mid),
        "q_high": grouped.quantile(q_high),
    })

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
