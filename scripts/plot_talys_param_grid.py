#!/usr/bin/env python3
"""
Plot TALYS optimization parameter-pair maps from talys_optimization.out.

This is intentionally focused on the current comparison plot: parse optimized
parameter records, write a compact CSV, and build a chi-square colored
parameter grid. Red outlines mark the best 15% of records by reduced chi-square.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np


BLOCK_SPLIT_RE = re.compile(r"={10,}\s*\n")
PARAM_RE = re.compile(r"^\s*([A-Za-z]\w*)\s*=\s*([+-]?[\d.eE+-]+)\s*$")


def repo_root() -> Path:
    """Return the repository root when the script lives in scripts/."""
    return Path(__file__).resolve().parents[1]


def normalize_bin_width(value: str | float | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)", text)
    if not match:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or None
    number = float(match.group(1))
    return f"{number:.6g}"


def reaction_slug(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    replacements = {
        "α": "a",
        "alpha": "a",
        "γ": "g",
        "gamma": "g",
        "β": "b",
        "²": "2",
        "³": "3",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(
        r"\(([A-Za-z]+),([A-Za-z]+)\)",
        lambda match: f"_{match.group(1)}{match.group(2)}_",
        text,
    )
    text = text.replace("(", "_").replace(",", "_").replace(")", "_")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or None


def first_record_value(records: list[dict], field: str) -> str | None:
    return next((str(rec[field]) for rec in records if rec.get(field)), None)


def resolve_talys_out(args: argparse.Namespace) -> Path:
    if args.talys_out is not None:
        return args.talys_out
    if args.reaction and args.bin_width:
        return (
            args.input_root
            / args.reaction
            / f"binwidth_{normalize_bin_width(args.bin_width)}"
            / "talys_optimization.out"
        )
    return args.input_root / "talys_optimization.out"


def resolve_output_paths(
    args: argparse.Namespace,
    records: list[dict],
    talys_out: Path,
) -> tuple[Path, Path | None]:
    reaction = args.reaction or reaction_slug(first_record_value(records, "reaction")) or talys_out.stem
    bin_width = normalize_bin_width(args.bin_width) or normalize_bin_width(
        first_record_value(records, "bin_width")
    )

    output_dir = args.analysis_dir / reaction
    if bin_width is not None:
        output_dir = output_dir / f"binwidth_{bin_width}"

    output = args.output
    if output is None:
        best_suffix = f"lowest{args.best_count}" if args.best_count else f"lowest{100 * args.best_fraction:g}pct"
        triangle = "full" if args.triangle == "full" else "lower"
        output = output_dir / f"talys_param_grid_chi2_red_{triangle}_{best_suffix}.png"

    if args.csv_output == "":
        csv_output = None
    elif args.csv_output is None:
        csv_output = output_dir / "talys_param_records.csv"
    else:
        csv_output = Path(args.csv_output)

    return output, csv_output


def parse_talys_optimization(path: Path) -> list[dict]:
    """Parse optimized parameter blocks from a talys_optimization.out file."""
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []

    for block in BLOCK_SPLIT_RE.split(text):
        if "Optimized parameters:" not in block:
            continue

        rec = {
            "run_idx": None,
            "chi2_red": np.nan,
            "nfev": np.nan,
            "reaction": "",
            "bin_width": "",
            "params": {},
        }

        match = re.search(r"Run index\s*:\s*(\d+)", block)
        if match:
            rec["run_idx"] = int(match.group(1))

        match = re.search(r"Min reduced chi-square:\s*([+-]?[\d.eE+-]+)", block)
        if match:
            rec["chi2_red"] = float(match.group(1))

        match = re.search(r"nfev=(\d+)", block)
        if match:
            rec["nfev"] = float(match.group(1))

        for key, field in (
            ("Reaction", "reaction"),
            ("Bin width", "bin_width"),
        ):
            match = re.search(rf"{re.escape(key)}\s*:\s*(.+)", block)
            if match:
                rec[field] = match.group(1).strip()

        in_params = False
        for line in block.splitlines():
            if "Optimized parameters:" in line:
                in_params = True
                continue
            if in_params and line.startswith("-"):
                in_params = False
                continue
            if not in_params:
                continue
            match = PARAM_RE.match(line)
            if match:
                rec["params"][match.group(1)] = float(match.group(2))

        if rec["params"] and np.isfinite(rec["chi2_red"]):
            records.append(rec)

    return records


def parameter_names(records: list[dict]) -> list[str]:
    names: list[str] = []
    for rec in records:
        for name in rec["params"]:
            if name not in names:
                names.append(name)
    return names


def table_from_records(records: list[dict], params: list[str]) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {
        "run_idx": np.array(
            [np.nan if rec["run_idx"] is None else rec["run_idx"] for rec in records],
            dtype=float,
        ),
        "chi2_red": np.array([rec["chi2_red"] for rec in records], dtype=float),
        "nfev": np.array([rec["nfev"] for rec in records], dtype=float),
    }
    for name in params:
        table[name] = np.array(
            [rec["params"].get(name, np.nan) for rec in records],
            dtype=float,
        )
    return table


def select_informative_params(
    table: dict[str, np.ndarray],
    params: list[str],
    *,
    robust_range_tol: float,
) -> tuple[list[str], list[str]]:
    """Drop parameters whose 5-95 percentile range is effectively zero."""
    keep: list[str] = []
    drop: list[str] = []
    for name in params:
        values = table[name]
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            drop.append(name)
            continue
        robust_range = float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5))
        if robust_range <= robust_range_tol:
            drop.append(name)
        else:
            keep.append(name)
    return keep, drop


def write_csv(path: Path, records: list[dict], params: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_idx", "chi2_red", "nfev", "reaction", "bin_width", *params])
        for rec in records:
            writer.writerow(
                [
                    "" if rec["run_idx"] is None else rec["run_idx"],
                    rec["chi2_red"],
                    rec["nfev"],
                    rec["reaction"],
                    rec["bin_width"],
                    *[rec["params"].get(name, "") for name in params],
                ]
            )


def finite_pair(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask]


def has_range(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size >= 2 and float(np.nanmax(finite) - np.nanmin(finite)) > 0.0


def best_fraction_threshold(values: np.ndarray, fraction: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    fraction = max(0.0, min(float(fraction), 1.0))
    return float(np.nanpercentile(finite, 100.0 * fraction))


def best_count_threshold(values: np.ndarray, count: int) -> float:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return np.nan
    count = max(1, min(int(count), finite.size))
    return float(finite[count - 1])


def hist_min_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    hist_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), hist_bins + 1)
    y_edges = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), hist_bins + 1)
    x_idx = np.searchsorted(x_edges, x, side="right") - 1
    y_idx = np.searchsorted(y_edges, y, side="right") - 1
    x_idx = np.clip(x_idx, 0, hist_bins - 1)
    y_idx = np.clip(y_idx, 0, hist_bins - 1)
    grid = np.full((hist_bins, hist_bins), np.nan)
    for iy in range(hist_bins):
        for ix in range(hist_bins):
            vals = z[(x_idx == ix) & (y_idx == iy)]
            if vals.size:
                grid[iy, ix] = float(np.nanmin(vals))
    return x_edges, y_edges, grid


def bin_centers(values: np.ndarray, n_bins: int) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if lo == hi:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def compact_tick_labels(values: np.ndarray) -> list[str]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return [f"{value:.4g}" for value in values]
    unique = np.unique(finite)
    if unique.size < 2:
        decimals = 4
    else:
        min_step = float(np.nanmin(np.diff(unique)))
        decimals = int(np.clip(np.ceil(-np.log10(min_step)) + 1, 4, 8))
    return [f"{value:.{decimals}f}".rstrip("0").rstrip(".") for value in values]


def format_x_tick_labels(ax, *, top: bool = False) -> None:
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_rotation(35)
        label.set_rotation_mode("anchor")
        label.set_ha("left" if top else "right")
        label.set_va("bottom" if top else "top")


def draw_diagonal_hist(ax, values: np.ndarray, *, hist_bins: int, color: str) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if lo == hi:
        pad = abs(lo) * 0.05 or 0.5
        ax.hist(
            finite,
            bins=1,
            range=(lo - pad, hi + pad),
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
    else:
        ax.hist(
            finite,
            bins=hist_bins,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def grid_layout(n_params: int) -> dict[str, float]:
    dense = n_params >= 7
    very_dense = n_params >= 8
    top_margin = min(0.968, 0.902 + 0.008 * n_params)
    right_margin = max(0.80, 0.875 - 0.007 * n_params)
    return {
        "left": 0.08 if not dense else 0.065,
        "right": right_margin,
        "bottom": 0.08 if not dense else 0.055,
        "top": top_margin,
        "wspace": 0.24 if not dense else 0.18,
        "hspace": 0.24 if not dense else 0.18,
        "title_y": min(0.997, top_margin + (0.055 if not dense else 0.035)),
        "colorbar_gap": 0.04 if not dense else 0.045,
        "colorbar_width": 0.03 if not very_dense else 0.025,
        "colorbar_height": 0.56 if dense else 0.62,
    }


def grid_text_sizes(n_params: int) -> dict[str, float]:
    if n_params <= 4:
        return {"title": 14, "axis": 10, "diagonal": 11, "colorbar": 11}
    if n_params <= 6:
        return {"title": 13, "axis": 9, "diagonal": 10, "colorbar": 10}
    return {"title": 12, "axis": 8.5, "diagonal": 9.5, "colorbar": 9.5}


def draw_pair_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    cmap: str,
    norm,
    point_size: float,
    hist_bins: int,
    highlight_cutoff: float,
):
    x, y, z = finite_pair(x, y, z)
    if x.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return None
    if not has_range(x) or not has_range(y):
        ax.text(0.5, 0.5, "constant", transform=ax.transAxes, ha="center", va="center")
        return None

    x_edges, y_edges, grid = hist_min_grid(x, y, z, hist_bins=hist_bins)
    artist = ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, norm=norm, shading="flat")
    ax.scatter(x, y, c="k", s=max(point_size * 0.35, 2.0), alpha=0.28, linewidths=0)

    highlight = z <= highlight_cutoff
    if np.any(highlight):
        ax.scatter(
            x[highlight],
            y[highlight],
            facecolors="none",
            edgecolors="crimson",
            s=max(point_size * 1.15, 12.0),
            linewidths=0.75,
        )
    return artist


def plot_grid(
    records: list[dict],
    params: list[str],
    output: Path,
    *,
    max_chi2: float | None,
    hist_bins: int,
    diag_bins: int | None,
    triangle: str,
    best_fraction: float,
    best_count: int | None,
    cmap: str,
    dpi: int,
    max_size: float,
    show_inner_ticks: bool,
) -> None:
    table = table_from_records(records, params)
    chi2 = table["chi2_red"].copy()
    valid_records = np.isfinite(chi2)
    if max_chi2 is not None:
        valid_records &= chi2 <= max_chi2

    z_plot = np.where(valid_records, chi2, np.nan)
    finite_z = z_plot[np.isfinite(z_plot)]
    if finite_z.size == 0:
        raise ValueError("No finite chi-square values remain after filtering.")
    if best_count is not None and best_count > 0:
        highlight_cutoff = best_count_threshold(z_plot, best_count)
    else:
        highlight_cutoff = best_fraction_threshold(z_plot, best_fraction)

    n = len(params)
    cell = min(max_size / max(n, 1), 2.0)
    fig_size = max(7.0, cell * n)
    fig, axes = plt.subplots(n, n, figsize=(fig_size, fig_size), squeeze=False)
    norm = plt.Normalize(vmin=float(np.nanmin(finite_z)), vmax=float(np.nanmax(finite_z)))
    cmap_obj = plt.get_cmap(cmap)
    point_size = 12 if n >= 7 else 18
    text_sizes = grid_text_sizes(n)
    diag_hist_bins = hist_bins if diag_bins is None else diag_bins

    for row, y_name in enumerate(params):
        for col, x_name in enumerate(params):
            ax = axes[row, col]
            ax.set_box_aspect(1)
            if triangle == "lower" and col > row:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                continue
            if row == col:
                x_diag = np.where(valid_records, table[x_name], np.nan)
                draw_diagonal_hist(ax, x_diag, hist_bins=diag_hist_bins, color=cmap_obj(0.62))
                centers = bin_centers(x_diag, diag_hist_bins)
                if centers is not None:
                    ax.set_xticks(centers)
                    ax.set_xticklabels(compact_tick_labels(centers))
                if row == n - 1:
                    format_x_tick_labels(ax)
                    ax.tick_params(axis="x", labelsize=6, length=2)
                elif not show_inner_ticks:
                    ax.tick_params(axis="x", labelbottom=False)
                if col == 0:
                    ax.tick_params(axis="y", labelsize=6, length=2)
                elif not show_inner_ticks:
                    ax.tick_params(axis="y", labelleft=False)
                ax.tick_params(axis="both", labelsize=6, length=2)
                if row == n - 1:
                    ax.set_xlabel(x_name, fontsize=text_sizes["axis"])
                ax.text(
                    0.06,
                    0.92,
                    x_name,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(text_sizes["diagonal"] - 1.0, 6.5),
                    fontweight="bold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                        "pad": 1.4,
                    },
                )
                continue

            x = np.where(valid_records, table[x_name], np.nan)
            y = np.where(valid_records, table[y_name], np.nan)
            draw_pair_panel(
                ax,
                x,
                y,
                z_plot,
                cmap=cmap,
                norm=norm,
                point_size=point_size,
                hist_bins=hist_bins,
                highlight_cutoff=highlight_cutoff,
            )

            x_centers = bin_centers(x, hist_bins)
            y_centers = bin_centers(y, hist_bins)
            if x_centers is not None:
                ax.set_xticks(x_centers)
                ax.set_xticklabels(compact_tick_labels(x_centers))
            if y_centers is not None:
                ax.set_yticks(y_centers)
                ax.set_yticklabels(compact_tick_labels(y_centers))
            if row == n - 1:
                ax.set_xlabel(x_name, fontsize=text_sizes["axis"])
                format_x_tick_labels(ax)
            elif not show_inner_ticks:
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel(y_name, fontsize=text_sizes["axis"])
            elif not show_inner_ticks:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="both", labelsize=6, length=2)

    reaction = next((rec["reaction"] for rec in records if rec.get("reaction")), "")
    title = "TALYS optimized parameter-pair grid"
    if reaction:
        title += f" - {reaction}"
    subtitle = f"{int(valid_records.sum())} records"
    if max_chi2 is not None:
        subtitle += f", chi2 <= {max_chi2:g}"
    if best_count is not None and best_count > 0:
        subtitle += f", red outlines = lowest {best_count} chi2"
    else:
        subtitle += f", red outlines = lowest {100 * best_fraction:g}% chi2"

    layout = grid_layout(n)
    fig.subplots_adjust(
        left=layout["left"],
        right=layout["right"],
        bottom=layout["bottom"],
        top=layout["top"],
        wspace=layout["wspace"],
        hspace=layout["hspace"],
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=text_sizes["title"], y=layout["title_y"])

    cbar_left = layout["right"] + layout["colorbar_gap"]
    cbar_height = min(layout["colorbar_height"], layout["top"] - layout["bottom"])
    cbar_bottom = layout["bottom"] + 0.5 * (layout["top"] - layout["bottom"] - cbar_height)
    cbar_ax = fig.add_axes(
        [
            cbar_left,
            cbar_bottom,
            layout["colorbar_width"],
            cbar_height,
        ]
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(r"Lowest-bin min $\chi^2_\nu$", fontsize=text_sizes["colorbar"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Create a TALYS optimized parameter-vs-parameter chi-square grid."
    )
    parser.add_argument(
        "--talys-out",
        type=Path,
        default=None,
        help=(
            "Path to talys_optimization.out. If omitted with --reaction and "
            "--bin-width, uses outputs/talys_opt/<reaction>/binwidth_<bin-width>/"
            "talys_optimization.out."
        ),
    )
    parser.add_argument(
        "--reaction",
        default=None,
        help=(
            "Reaction slug used for automatic input/output paths, e.g. "
            "22Mg_ap_25Al. If omitted, output paths use metadata from the .out file."
        ),
    )
    parser.add_argument(
        "--bin-width",
        default=None,
        help=(
            "Bin-width label used for automatic input/output paths, e.g. 0.2. "
            "If omitted, output paths use metadata from the .out file when present."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "outputs" / "talys_opt",
        help="Root directory for raw TALYS/TESSERACT optimization outputs.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=root / "outputs" / "analysis" / "talys_param_grid",
        help="Root directory for generated parameter-grid analysis outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults under outputs/analysis/talys_param_grid/.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help=(
            "Parsed records CSV path. Defaults next to the output image as "
            "talys_param_records.csv. Use an empty string to skip."
        ),
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Optional subset/order of parameter names to plot.",
    )
    parser.add_argument(
        "--triangle",
        choices=("lower", "full"),
        default="full",
        help="Use a lower-triangle pair grid or the full mirrored grid.",
    )
    parser.add_argument(
        "--max-chi2",
        type=float,
        default=None,
        help="Optional upper chi-square filter.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=5,
        help="Number of bins per axis for parameter-pair panels.",
    )
    parser.add_argument(
        "--diag-bins",
        type=int,
        default=10,
        help="Number of bins for the diagonal 1D parameter histograms.",
    )
    parser.add_argument(
        "--show-inner-ticks",
        action="store_true",
        help="Show tick labels on interior parameter-pair panels, not only outer axes.",
    )
    parser.add_argument(
        "--keep-near-constant",
        action="store_true",
        help="Keep near-constant parameters in the grid.",
    )
    parser.add_argument(
        "--constant-range-tol",
        type=float,
        default=1e-3,
        help="Drop auto-selected params with 5-95 percentile range at or below this value.",
    )
    parser.add_argument(
        "--best-fraction",
        type=float,
        default=0.15,
        help="Fraction of records highlighted with red outlines, using lowest chi-square.",
    )
    parser.add_argument(
        "--best-count",
        type=int,
        default=None,
        help="Highlight exactly this many records with the lowest chi-square.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis_r",
        help="Matplotlib colormap name. Default makes lower chi-square lighter.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=16.0,
        help="Maximum figure width/height in inches before bbox trimming.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    talys_out = resolve_talys_out(args)
    records = parse_talys_optimization(talys_out)
    if not records:
        raise SystemExit(f"No optimization records found in {talys_out}")

    params = parameter_names(records)
    if args.params:
        missing = [name for name in args.params if name not in params]
        if missing:
            raise SystemExit("Requested parameter(s) not found: " + ", ".join(missing))
        params = list(args.params)

    table = table_from_records(records, params)
    if not args.params and not args.keep_near_constant:
        params, dropped = select_informative_params(
            table,
            params,
            robust_range_tol=args.constant_range_tol,
        )
        if dropped:
            print("Dropped near-constant parameter(s): " + ", ".join(dropped))

    if len(params) < 2:
        raise SystemExit("Need at least two optimized parameters for a pair grid.")

    output, csv_output = resolve_output_paths(args, records, talys_out)

    if csv_output is not None:
        write_csv(csv_output, records, params)
        print(f"Wrote parsed records: {csv_output}")

    plot_grid(
        records,
        params,
        output,
        max_chi2=args.max_chi2,
        hist_bins=args.hist_bins,
        diag_bins=args.diag_bins,
        triangle=args.triangle,
        best_fraction=args.best_fraction,
        best_count=args.best_count,
        cmap=args.cmap,
        dpi=args.dpi,
        max_size=args.max_size,
        show_inner_ticks=args.show_inner_ticks,
    )
    print(f"Wrote parameter grid: {output}")
    print(f"Records: {len(records)} | Parameters: {len(params)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
