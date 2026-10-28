#!/usr/bin/env python3
"""
plot_talys_param_violins.py

Spread of the TALYS parameters fitted to the ATCS, across all runs: one
violin per reaction for every fitted parameter.

Each run of the ensemble is a different synthetic nucleus, and each run's
ATCS is fitted separately ([talys] step). If the fitted level-density and
optical-model parameters were measuring physics, they would cluster; a wide
or bound-hugging spread means they are absorbing the resonance structure
instead.

Input
-----
The talys_optimization.out file written by talys_opt_with_unc.py, found at

    <talys-opt-root>/<stem>/binwidth_<dE>/talys_optimization.out
    e.g. outputs/talys_opt/14O_ap_17F/binwidth_0.2/talys_optimization.out

It is parsed with parse_talys_optimization() from plot_talys_param_grid.py,
which keeps only the latest block of each fit (the file is append-only).

Parameters are matched across reactions by keyword: level-density entries
such as T_9_17 (14O+α) and T_11_21 (18Ne+α) both go in the "T" panel. The
optical-model multipliers (e.g. rvadjust_a) keep their names.

Optional: pass the tesseract .in file used for each reaction with --config
(same order as --reaction) to mark each parameter's starting value (x0) and
its lower and upper bounds, and to report how many fits ended at a bound.

Example
-------
    python scripts/plot_talys_param_violins.py \\
        --reaction "14O(a,p)17F" --reaction "18Ne(a,p)21Na" \\
        --binwidth 0.2 \\
        --config tesseract_14O.in --config tesseract_18Ne.in \\
        --out talys_param_violins.png
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # nucres, talys_opt_with_unc
sys.path.insert(0, HERE)                                          # plot_talys_param_grid
from nucres.reaction_names import file_stem as reaction_stem  # noqa: E402
from plot_talys_param_grid import parse_talys_optimization  # noqa: E402

COLOURS = ["#2F5D9E", "#B4501F", "#3C8D5A", "#7A4FA0"]

# Display labels for known keywords; anything else is shown as its name.
LABELS = {
    "T": ("T", "MeV", "level density"),
    "E0": (r"$E_0$", "MeV", "level density"),
    "a": (r"$a$", r"MeV$^{-1}$", "level density"),
    "Exmatch": (r"$E_x^\mathrm{match}$", "MeV", "level density"),
    "ctable": ("ctable", "", "level density"),
    "ptable": ("ptable", "", "level density"),
    "rvadjust_a": (r"$r_V$ scale", "", "α optical model"),
    "avadjust_a": (r"$a_V$ scale", "", "α optical model"),
    "v1adjust_a": (r"$V_1$ scale", "", "α optical model"),
    "w1adjust_a": (r"$W_1$ scale", "", "α optical model"),
    "rwadjust_a": (r"$r_W$ scale", "", "α optical model"),
    "awadjust_a": (r"$a_W$ scale", "", "α optical model"),
}

_NUCLEUS_SUFFIX = re.compile(r"^(?P<key>[A-Za-z]\w*?)_(?P<Z>\d+)_(?P<A>\d+)$")


def generic_name(name: str) -> str:
    """'T_13_25' -> 'T'; names without a _Z_A suffix are returned unchanged."""
    m = _NUCLEUS_SUFFIX.match(name)
    return m.group("key") if m else name


# --------------------------------------------------------------------------
# Data for one reaction
# --------------------------------------------------------------------------
@dataclass
class Bounds:
    """Starting value and bounds of one parameter, from the \\opt block."""
    x0: float
    lo: float
    hi: float


@dataclass
class FittedParameters:
    """
    The fitted parameters of every run of one reaction.

    Attributes
    ----------
    reaction : e.g. '14O(a,p)17F'
    values   : generic parameter name -> array of fitted values (one per run)
    chi2_red : reduced chi-square of each fit, same order as the values
    bounds   : generic parameter name -> Bounds (empty without --config)
    """
    reaction: str
    values: dict[str, np.ndarray]
    chi2_red: np.ndarray
    bounds: dict[str, Bounds] = field(default_factory=dict)

    @classmethod
    def from_talys_output(cls, reaction: str, path: str,
                          max_chi2: float | None = None) -> "FittedParameters":
        """Read talys_optimization.out; optionally drop fits with chi2_red > max_chi2."""
        records = parse_talys_optimization(Path(path))
        if not records:
            raise ValueError(f"{path}: no fit blocks with optimized parameters found.")
        chi2 = np.array([r["chi2_red"] for r in records], float)
        keep = np.ones(len(records), bool) if max_chi2 is None else chi2 <= max_chi2
        if max_chi2 is not None:
            print(f"  {reaction}: kept {keep.sum()} of {len(records)} fits with chi2_red <= {max_chi2:g}")
        records = [r for r, k in zip(records, keep) if k]

        names: list[str] = []
        for r in records:
            for n in r["params"]:
                if n not in names:
                    names.append(n)
        values = {generic_name(n): np.array([r["params"].get(n, np.nan) for r in records], float)
                  for n in names}
        return cls(reaction, values, chi2[keep])

    def load_bounds(self, config_path: str) -> None:
        """Starting values and bounds from the \\opt block of a tesseract .in file."""
        from talys_opt_with_unc import _parse_opt_line, parse_tesseract_in
        lines = parse_tesseract_in(config_path).get("talys", {}).get("_opt_lines", [])
        for line in lines:
            op = _parse_opt_line(line)
            self.bounds[generic_name(op.name)] = Bounds(op.x0, op.lo, op.hi)

    @property
    def n_fits(self) -> int:
        return len(self.chi2_red)

    def summary(self) -> str:
        """One line per parameter: median, 16-84% range, and fits at a bound."""
        lines = [f"{self.reaction}: {self.n_fits} fits"]
        for name, v in self.values.items():
            v = v[np.isfinite(v)]
            lo, med, hi = np.percentile(v, [16, 50, 84])
            text = f"  {name:<12} median {med:+.4g}   16–84%: [{lo:+.4g}, {hi:+.4g}]"
            b = self.bounds.get(name)
            if b is not None:
                tol = 1e-3 * (b.hi - b.lo)
                at_lo = np.mean(v <= b.lo + tol)
                at_hi = np.mean(v >= b.hi - tol)
                text += f"   x0 {b.x0:+.4g}   at lower bound {at_lo:.0%}, at upper bound {at_hi:.0%}"
            lines.append(text)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
class ViolinGrid:
    """One panel per parameter, one violin per reaction in each panel."""

    def __init__(self, fits: list[FittedParameters], ncols: int | None = None,
                 full_range: bool = False):
        self.fits = fits
        self.full_range = full_range
        self.params: list[str] = []
        for f in fits:
            for name in f.values:
                if name not in self.params:
                    self.params.append(name)
        # Level-density parameters first, then optical-model ones.
        self.params.sort(key=lambda n: (LABELS.get(n, (n, "", "other"))[2] != "level density"))
        self.ncols = ncols or min(len(self.params), 5)
        self.nrows = math.ceil(len(self.params) / self.ncols)

    def draw(self, out: str) -> None:
        fig, axes = plt.subplots(self.nrows, self.ncols,
                                 figsize=(2.6 * self.ncols + 0.6, 3.6 * self.nrows),
                                 squeeze=False)
        for ax, name in zip(axes.flat, self.params):
            self._panel(ax, name)
        for ax in list(axes.flat)[len(self.params):]:
            ax.set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[i % len(COLOURS)], alpha=0.5)
                   for i in range(len(self.fits))]
        labels = [f"{f.reaction} ({f.n_fits} fits)" for f in self.fits]
        if any(f.bounds for f in self.fits):
            handles += [plt.Line2D([], [], color="0.2", lw=2),
                        plt.Line2D([], [], color="0.4", ls="--", lw=1)]
            labels += ["starting value", "fit bound (shown when reached)"]
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")

    def _panel(self, ax, name: str) -> None:
        symbol, unit, group = LABELS.get(name, (name, "", "other"))
        for i, f in enumerate(self.fits):
            v = f.values.get(name)
            if v is None:
                continue
            v = v[np.isfinite(v)]
            colour = COLOURS[i % len(COLOURS)]
            if np.ptp(v) > 0:
                parts = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False,
                                      showmedians=True, quantiles=[0.16, 0.84])
                for body in parts["bodies"]:
                    body.set_facecolor(colour)
                    body.set_edgecolor(colour)
                    body.set_alpha(0.45)
                for key in ("cmedians", "cquantiles"):
                    if key in parts:
                        parts[key].set_color(colour)
                parts["cmedians"].set_linewidth(2)
            else:  # every fit returned the same value: a violin cannot be drawn
                ax.plot([i - 0.3, i + 0.3], [v[0], v[0]], color=colour, lw=2)

            b = f.bounds.get(name)
            if b is not None:
                ax.hlines(b.x0, i - 0.4, i + 0.4, color="0.2", lw=2)
                ax.hlines([b.lo, b.hi], i - 0.45, i + 0.45, color="0.4", ls="--", lw=1)

        # Zoom to the fitted values (and starting values); a bound is then
        # visible only when the fits come close to it. --full-range shows
        # the whole bounded interval instead.
        shown = [f.values[name][np.isfinite(f.values[name])] for f in self.fits if name in f.values]
        shown += [np.array([f.bounds[name].x0]) for f in self.fits if name in f.bounds]
        if self.full_range:
            shown += [np.array([f.bounds[name].lo, f.bounds[name].hi])
                      for f in self.fits if name in f.bounds]
        allv = np.concatenate(shown)
        pad = 0.08 * (allv.max() - allv.min() or abs(allv.max()) or 1.0)
        ax.set_ylim(allv.min() - pad, allv.max() + pad)

        ax.set_xticks(range(len(self.fits)))
        ax.set_xticklabels([short_label(f.reaction) for f in self.fits], fontsize=9)
        ax.set_xlim(-0.6, len(self.fits) - 0.4)
        ax.set_title(f"{symbol}\n{group}", fontsize=11)
        if unit:
            ax.set_ylabel(unit)


def short_label(reaction: str) -> str:
    """'14O(a,p)17F' -> '14O+α' for compact tick labels."""
    m = re.match(r"^\s*([^(\s]+)\(([^,]+),", reaction)
    if not m:
        return reaction
    proj = {"a": "α", "p": "p", "n": "n"}.get(m.group(2).strip(), m.group(2).strip())
    return f"{m.group(1)}+{proj}"


def default_path(root: str, reaction: str, binwidth: str) -> str:
    return os.path.join(root, reaction_stem(reaction), f"binwidth_{binwidth}",
                        "talys_optimization.out")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reaction", action="append", required=True,
                   help="Reaction, e.g. '14O(a,p)17F'. Repeatable.")
    p.add_argument("--talys-opt-root", default="outputs/talys_opt")
    p.add_argument("--binwidth", default="0.2", help="Bin width as in the folder name binwidth_<dE>.")
    p.add_argument("--talys-out", action="append", default=None,
                   help="Explicit talys_optimization.out path(s), same order as --reaction "
                        "(overrides --talys-opt-root/--binwidth).")
    p.add_argument("--config", action="append", default=None,
                   help="tesseract .in file per reaction (same order) to mark x0 and bounds.")
    p.add_argument("--max-chi2", type=float, default=None,
                   help="Drop fits with reduced chi-square above this value.")
    p.add_argument("--full-range", action="store_true",
                   help="Show each parameter's whole bounded range instead of zooming to the fits.")
    p.add_argument("--ncols", type=int, default=None, help="Panels per row (default: up to 5).")
    p.add_argument("--out", default="talys_param_violins.png")
    args = p.parse_args()

    for opt in ("talys_out", "config"):
        given = getattr(args, opt)
        if given is not None and len(given) != len(args.reaction):
            p.error(f"--{opt.replace('_', '-')} must be given once per --reaction.")

    fits = []
    for i, reaction in enumerate(args.reaction):
        path = args.talys_out[i] if args.talys_out else \
            default_path(args.talys_opt_root, reaction, args.binwidth)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{reaction}: {path} not found.")
        print(f"{reaction}: reading {path}")
        f = FittedParameters.from_talys_output(reaction, path, max_chi2=args.max_chi2)
        if args.config:
            f.load_bounds(args.config[i])
        print(f.summary())
        fits.append(f)

    ViolinGrid(fits, ncols=args.ncols, full_range=args.full_range).draw(args.out)


if __name__ == "__main__":
    main()
