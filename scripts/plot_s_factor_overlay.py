#!/usr/bin/env python3
"""Plot S-factor overlays from cross-section curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from diagnostic_plot import _sommerfeld_eta
from plot_talys_xs_overlay import (
    gamow_peak_and_width,
    load_xy,
    parse_optimized_xs_from_out,
)


def s_factor_from_mb(
    energy_mev: np.ndarray,
    sigma_mb: np.ndarray,
    *,
    projectile_z: int,
    target_z: int,
    projectile_a: int,
    target_a: int,
    sigma_unit: str,
) -> np.ndarray:
    """Return S(E) using E in MeV and sigma in either mb or converted barns."""
    eta = _sommerfeld_eta(projectile_z, target_z, projectile_a, target_a, energy_mev)
    sigma_factor = 1.0 if sigma_unit == "mb" else 1e-3
    return energy_mev * sigma_mb * sigma_factor * np.exp(2.0 * np.pi * eta)


def finite_positive_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[mask], y[mask]


def window_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float | None,
    x_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    return x[mask], y[mask]


def add_gamow_band(
    ax,
    *,
    t9_min: float,
    t9_max: float,
    projectile_z: int,
    target_z: int,
    projectile_a: int,
    target_a: int,
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
    ax.axvspan(
        float(np.min(lower)),
        float(np.max(upper)),
        color="0.55",
        alpha=0.14,
        linewidth=0,
        zorder=0,
    )


def plot_s_factor(
    *,
    output: Path,
    optimized_xy: tuple[np.ndarray, np.ndarray],
    metadata: dict,
    raw_xy: tuple[np.ndarray, np.ndarray] | None,
    integrated_xy: tuple[np.ndarray, np.ndarray] | None,
    title: str | None,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
    projectile_z: int,
    target_z: int,
    projectile_a: int,
    target_a: int,
    sigma_unit: str,
    gamow_t9_range: tuple[float, float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    opt_x, opt_s = finite_positive_xy(
        optimized_xy[0],
        s_factor_from_mb(
            optimized_xy[0],
            optimized_xy[1],
            projectile_z=projectile_z,
            target_z=target_z,
            projectile_a=projectile_a,
            target_a=target_a,
            sigma_unit=sigma_unit,
        ),
    )
    opt_x, opt_s = window_xy(opt_x, opt_s, x_min=x_min, x_max=x_max)
    ax.plot(opt_x, opt_s, color="C1", linewidth=2.0, label=r"TALYS optimized $S(E)$", zorder=3)

    if integrated_xy is not None:
        int_x, int_s = finite_positive_xy(
            integrated_xy[0],
            s_factor_from_mb(
                integrated_xy[0],
                integrated_xy[1],
                projectile_z=projectile_z,
                target_z=target_z,
                projectile_a=projectile_a,
                target_a=target_a,
                sigma_unit=sigma_unit,
            ),
        )
        int_x, int_s = window_xy(int_x, int_s, x_min=x_min, x_max=x_max)
        ax.plot(
            int_x,
            int_s,
            color="black",
            linewidth=1.2,
            marker="o",
            markersize=3.8,
            label=r"Integrated $S(E)$ ($\Delta E = 0.2$ MeV)",
            zorder=4,
        )

    if raw_xy is not None:
        raw_x, raw_s = finite_positive_xy(
            raw_xy[0],
            s_factor_from_mb(
                raw_xy[0],
                raw_xy[1],
                projectile_z=projectile_z,
                target_z=target_z,
                projectile_a=projectile_a,
                target_a=target_a,
                sigma_unit=sigma_unit,
            ),
        )
        raw_x, raw_s = window_xy(raw_x, raw_s, x_min=x_min, x_max=x_max)
        raw_marker = "o" if raw_x.size <= 250 else None
        ax.plot(
            raw_x,
            raw_s,
            color="C0",
            linewidth=1.2,
            marker=raw_marker,
            markersize=3.2,
            label=r"Raw $S(E)$",
            zorder=5,
        )

    resolved_title = title
    if resolved_title is None:
        resolved_title = metadata.get("reaction") or "S-factor comparison"
    ax.set_title(resolved_title)
    ax.set_xlabel("Center-of-mass energy [MeV]")
    ax.set_ylabel(rf"$S(E)$ [MeV {sigma_unit}]")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False)

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    if gamow_t9_range is not None:
        add_gamow_band(
            ax,
            t9_min=gamow_t9_range[0],
            t9_max=gamow_t9_range[1],
            projectile_z=projectile_z,
            target_z=target_z,
            projectile_a=projectile_a,
            target_a=target_a,
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot S(E) from raw, TALYS optimized, and integrated cross sections."
    )
    parser.add_argument("--optimized-out", type=Path, required=True)
    parser.add_argument("--run-idx", type=int, default=None)
    parser.add_argument("--raw-xs", type=Path, default=None)
    parser.add_argument("--integrated-xs", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/s_factor/s_factor_overlay.png"),
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--projectile-z", type=int, default=2)
    parser.add_argument("--target-z", type=int, default=12)
    parser.add_argument("--projectile-a", type=int, default=4)
    parser.add_argument("--target-a", type=int, default=22)
    parser.add_argument(
        "--sigma-unit",
        choices=("b", "mb"),
        default="b",
        help="Cross-section unit to keep in the S-factor output; default converts mb to b.",
    )
    parser.add_argument(
        "--gamow-t9-range",
        type=float,
        nargs=2,
        metavar=("T9_MIN", "T9_MAX"),
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    optimized_x, optimized_y, metadata = parse_optimized_xs_from_out(
        args.optimized_out,
        run_idx=args.run_idx,
    )
    raw_xy = load_xy(args.raw_xs) if args.raw_xs else None
    integrated_xy = load_xy(args.integrated_xs) if args.integrated_xs else None

    plot_s_factor(
        output=args.output,
        optimized_xy=(optimized_x, optimized_y),
        metadata=metadata,
        raw_xy=raw_xy,
        integrated_xy=integrated_xy,
        title=args.title,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        projectile_z=args.projectile_z,
        target_z=args.target_z,
        projectile_a=args.projectile_a,
        target_a=args.target_a,
        sigma_unit=args.sigma_unit,
        gamow_t9_range=tuple(args.gamow_t9_range) if args.gamow_t9_range else None,
    )
    print(f"Wrote S-factor overlay: {args.output}")
    print(
        "Optimized block: "
        f"run={metadata.get('run_idx')} chi2={metadata.get('chi2_red'):.6g} "
        f"reaction={metadata.get('reaction')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
