import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def read_ratesmc_out(path: Path, reaction: str, run: str) -> List[Tuple[str, str, float, float]]:
    rows: List[Tuple[str, str, float, float]] = []
    with path.open() as handle:
        # Skip until the table header begins
        for line in handle:
            if line.strip().startswith("T9"):
                break
        # Read table rows: T9 low median high fu
        for line in handle:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            t9, _low, med, _high, _fu = map(float, parts[:5])
            rows.append((reaction, run, t9, med))
    return rows


def iter_ratesmc(paths: Iterable[Path]) -> pd.DataFrame:
    rows: List[Tuple[str, str, float, float]] = []
    for path in paths:
        # .../<reaction>/<Run_###>/RatesMC.out
        reaction = path.parent.parent.name
        run = path.parent.name
        rows.extend(read_ratesmc_out(path, reaction, run))
    if not rows:
        return pd.DataFrame(columns=["reaction", "run", "T9", "rate_median"])
    return pd.DataFrame(rows, columns=["reaction", "run", "T9", "rate_median"])


def read_astrorate(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as handle:
        for line in handle:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("##"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                t9 = float(parts[0])
                rate = float(parts[1])
            except ValueError:
                continue
            rows.append((t9, rate))
    df = pd.DataFrame(rows, columns=["T9", "rate"]).drop_duplicates(subset=["T9"]).sort_values("T9")
    return df


def parse_quantiles(value: str) -> Tuple[float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("quantiles must be three comma values, e.g. 0.16,0.5,0.84")
    return parts[0], parts[1], parts[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay RatesMC median quantile band with TALYS astrorate curve.")
    ap.add_argument("--ratesmc-reaction-dir", required=True, help="Path to reaction directory containing Run_*/RatesMC.out")
    ap.add_argument("--talys-astrorate", required=True, help="Path to TALYS astrorate file (e.g. astrorate.g)")
    ap.add_argument("--quantiles", type=parse_quantiles, default=(0.16, 0.5, 0.84), help="Band/line quantiles")
    ap.add_argument("--output", default="overlay.png", help="Output image filename")
    args = ap.parse_args()

    reaction_dir = Path(args.ratesmc_reaction_dir).expanduser()
    ratesmc_paths = sorted(reaction_dir.glob("Run_*/RatesMC.out"))
    data = iter_ratesmc(ratesmc_paths)
    if data.empty:
        raise SystemExit(f"No RatesMC.out files found under: {reaction_dir}/Run_*/RatesMC.out")

    q_low, q_mid, q_high = args.quantiles
    stats = (
        data.groupby(["reaction", "T9"])["rate_median"]
        .quantile([q_low, q_mid, q_high])
        .unstack()
    )
    stats.columns = ["q_low", "q_mid", "q_high"]

    reaction_name = reaction_dir.name
    series = stats.loc[reaction_name].sort_index()

    talys = read_astrorate(Path(args.talys_astrorate).expanduser())
    if talys.empty:
        raise SystemExit("No numeric (T9, rate) rows found in the TALYS astrorate file.")

    plt.figure(figsize=(7, 4.5))
    plt.fill_between(series.index, series["q_low"], series["q_high"], alpha=0.2, label="RatesMC band")
    plt.plot(series.index, series["q_mid"], label="RatesMC median")
    plt.plot(talys["T9"], talys["rate"], label="TALYS")

    plt.yscale("log")
    plt.xlabel("T9")
    plt.ylabel("Rate (cm3/mol/s)")
    plt.title(reaction_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

