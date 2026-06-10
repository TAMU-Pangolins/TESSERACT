#!/usr/bin/env python3
"""Plot raw, TALYS optimized, and integrated cross sections."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BLOCK_SPLIT_RE = re.compile(r"={10,}\s*\n")


def gamow_peak_and_width(
    t9: np.ndarray,
    *,
    projectile_z: float,
    target_z: float,
    projectile_a: float,
    target_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the charged-particle Gamow peak E0 and full window width in MeV."""
    reduced_mass_amu = projectile_a * target_a / (projectile_a + target_a)
    charge_mass = projectile_z**2 * target_z**2 * reduced_mass_amu
    e0 = 0.12204 * (charge_mass * t9**2) ** (1.0 / 3.0)
    width = 0.23682 * (charge_mass * t9**5) ** (1.0 / 6.0)
    return e0, width


def add_gamow_window(
    ax,
    *,
    t9_min: float,
    t9_max: float,
    projectile_z: float,
    target_z: float,
    projectile_a: float,
    target_a: float,
) -> None:
    t9 = np.linspace(t9_min, t9_max, 160)
    e0, width = gamow_peak_and_width(
        t9,
        projectile_z=projectile_z,
        target_z=target_z,
        projectile_a=projectile_a,
        target_a=target_a,
    )
    lower = e0 - 0.5 * width
    upper = e0 + 0.5 * width
    window_min = float(np.min(lower))
    window_max = float(np.max(upper))

    ax.axvspan(
        window_min,
        window_max,
        color="0.55",
        alpha=0.14,
        linewidth=0,
        zorder=0,
    )
    inset = ax.inset_axes([0.43, 0.37, 0.27, 0.26])
    inset.fill_between(t9, lower, upper, color="0.55", alpha=0.22, linewidth=0)
    inset.plot(t9, e0, color="0.25", linewidth=1.0)
    inset.set_title(rf"Gamow window, $T_9={t9_min:g}-{t9_max:g}$", fontsize=7.5, pad=2)
    inset.set_xlabel(r"$T_9$ [GK]", fontsize=7)
    inset.set_ylabel(r"$E_\mathrm{cm}$ [MeV]", fontsize=7)
    inset.tick_params(axis="both", labelsize=6, length=2)
    inset.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the first two numeric columns from a whitespace or comma table."""
    rows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [part for part in re.split(r"[\s,]+", stripped) if part]
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric x/y rows found in {path}")
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


def parse_optimized_xs_from_out(path: Path, *, run_idx: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read optimized TALYS cross sections from a talys_optimization.out block."""
    text = path.read_text(encoding="utf-8", errors="replace")
    candidates: list[tuple[float, int | None, np.ndarray, np.ndarray, dict]] = []

    for block in BLOCK_SPLIT_RE.split(text):
        if "Cross sections" not in block:
            continue

        idx_match = re.search(r"Run index\s*:\s*(\d+)", block)
        block_run_idx = int(idx_match.group(1)) if idx_match else None
        if run_idx is not None and block_run_idx != run_idx:
            continue

        chi2_match = re.search(r"Min reduced chi-square:\s*([+-]?[\d.eE+-]+)", block)
        chi2 = float(chi2_match.group(1)) if chi2_match else np.inf

        reaction_match = re.search(r"Reaction\s*:\s*(.+)", block)
        bin_width_match = re.search(r"Bin width\s*:\s*(.+)", block)
        metadata = {
            "run_idx": block_run_idx,
            "chi2_red": chi2,
            "reaction": reaction_match.group(1).strip() if reaction_match else "",
            "bin_width": bin_width_match.group(1).strip() if bin_width_match else "",
        }

        energies: list[float] = []
        sigmas: list[float] = []
        in_xs = False
        for line in block.splitlines():
            if "Cross sections" in line:
                in_xs = True
                continue
            if in_xs and (line.startswith("-") or "Reaction rates" in line):
                break
            if not in_xs:
                continue
            nums = re.findall(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", line)
            if len(nums) >= 2:
                energies.append(float(nums[0]))
                sigmas.append(float(nums[1]))

        if energies:
            candidates.append(
                (
                    chi2,
                    block_run_idx,
                    np.asarray(energies, dtype=float),
                    np.asarray(sigmas, dtype=float),
                    metadata,
                )
            )

    if not candidates:
        target = f"run {run_idx}" if run_idx is not None else "any run"
        raise ValueError(f"No optimized cross-section block found for {target} in {path}")

    candidates.sort(key=lambda item: item[0])
    _chi2, _block_run_idx, x, y, metadata = candidates[0]
    return x, y, metadata


def plot_overlay(
    *,
    output: Path,
    optimized_x: np.ndarray,
    optimized_y: np.ndarray,
    metadata: dict,
    raw_xy: tuple[np.ndarray, np.ndarray] | None,
    default_xy: tuple[np.ndarray, np.ndarray] | None,
    integrated_xy: tuple[np.ndarray, np.ndarray] | None,
    title: str | None,
    y_min: float | None,
    y_max: float | None,
    x_min: float | None,
    x_max: float | None,
    gamow_t9_range: tuple[float, float] | None,
    gamow_projectile_z: float,
    gamow_target_z: float,
    gamow_projectile_a: float,
    gamow_target_a: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    if default_xy is not None:
        ax.plot(default_xy[0], default_xy[1], color="C0", linewidth=1.8, label="TALYS Default")

    ax.plot(optimized_x, optimized_y, color="C1", linewidth=2.0, label="TALYS Optimized", zorder=3)

    if integrated_xy is not None:
        ax.plot(
            integrated_xy[0],
            integrated_xy[1],
            color="black",
            linewidth=1.2,
            marker="o",
            markersize=3.8,
            label=r"Integrated cross section ($\Delta E = 0.2$ MeV)",
            zorder=4,
        )

    if raw_xy is not None:
        raw_marker = "o" if len(raw_xy[0]) <= 250 else None
        ax.plot(
            raw_xy[0],
            raw_xy[1],
            color="C0",
            linewidth=1.2,
            marker=raw_marker,
            markersize=3.2,
            label="Raw cross section",
            zorder=5,
        )

    resolved_title = title
    if resolved_title is None:
        reaction = metadata.get("reaction") or "TALYS cross-section comparison"
        resolved_title = reaction
    ax.set_title(resolved_title)
    ax.set_xlabel("Center-of-mass energy [MeV]")
    ax.set_ylabel("Cross section [mb]")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False)

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    if gamow_t9_range is not None:
        add_gamow_window(
            ax,
            t9_min=gamow_t9_range[0],
            t9_max=gamow_t9_range[1],
            projectile_z=gamow_projectile_z,
            target_z=gamow_target_z,
            projectile_a=gamow_projectile_a,
            target_a=gamow_target_a,
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay raw, TALYS optimized, and integrated cross sections."
    )
    parser.add_argument(
        "--optimized-out",
        type=Path,
        required=True,
        help="talys_optimization.out containing optimized cross-section blocks.",
    )
    parser.add_argument(
        "--run-idx",
        type=int,
        default=None,
        help="Specific optimized run to plot. Defaults to the lowest chi-square run.",
    )
    parser.add_argument(
        "--default-xs",
        type=Path,
        default=None,
        help="Optional TALYS default cross-section table with E and sigma columns.",
    )
    parser.add_argument(
        "--raw-xs",
        type=Path,
        default=None,
        help="Optional raw cross-section table with E and sigma columns.",
    )
    parser.add_argument(
        "--integrated-xs",
        type=Path,
        default=None,
        help="Optional integrated cross-section CSV/table with E and sigma columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/talys_xs_overlay/talys_xs_overlay.png"),
        help="Output image path.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument(
        "--gamow-t9-range",
        type=float,
        nargs=2,
        metavar=("T9_MIN", "T9_MAX"),
        default=None,
        help="Add a Gamow-window band and inset for this T9 range in GK.",
    )
    parser.add_argument("--gamow-projectile-z", type=float, default=2.0)
    parser.add_argument("--gamow-target-z", type=float, default=12.0)
    parser.add_argument("--gamow-projectile-a", type=float, default=4.0)
    parser.add_argument("--gamow-target-a", type=float, default=22.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    optimized_x, optimized_y, metadata = parse_optimized_xs_from_out(
        args.optimized_out,
        run_idx=args.run_idx,
    )
    default_xy = load_xy(args.default_xs) if args.default_xs else None
    raw_xy = load_xy(args.raw_xs) if args.raw_xs else None
    integrated_xy = load_xy(args.integrated_xs) if args.integrated_xs else None

    plot_overlay(
        output=args.output,
        optimized_x=optimized_x,
        optimized_y=optimized_y,
        metadata=metadata,
        raw_xy=raw_xy,
        default_xy=default_xy,
        integrated_xy=integrated_xy,
        title=args.title,
        y_min=args.y_min,
        y_max=args.y_max,
        x_min=args.x_min,
        x_max=args.x_max,
        gamow_t9_range=tuple(args.gamow_t9_range) if args.gamow_t9_range else None,
        gamow_projectile_z=args.gamow_projectile_z,
        gamow_target_z=args.gamow_target_z,
        gamow_projectile_a=args.gamow_projectile_a,
        gamow_target_a=args.gamow_target_a,
    )
    print(f"Wrote overlay: {args.output}")
    print(
        "Optimized block: "
        f"run={metadata.get('run_idx')} chi2={metadata.get('chi2_red'):.6g} "
        f"reaction={metadata.get('reaction')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
