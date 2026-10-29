#!/usr/bin/env python3
"""
plot_resonances_per_bin.py

How many resonances does each measured energy bin actually average over?

For every sampled resonance ladder (one RUN_* directory = one synthetic
nucleus), the resonances are compared with the SAME energy bins that the
[integration] step uses to build the ATCS. Two numbers are computed per bin:

* N      - how many resonances have their peak (E_r) in the bin.
* N_eff  - how many resonances effectively set the bin's averaged cross
           section:

               w_ik  = integral of resonance i's cross section over bin k
               N_eff = (Σ_i w_ik)² / Σ_i w_ik²

           If one resonance dominates the bin, N_eff ≈ 1 even when N is large.
           If k resonances contribute equally, N_eff = k. A bin with no
           resonance of its own can still get its cross section from the
           tail of a neighbouring resonance; N_eff then reports that (≈ 1)
           while N is 0.

The weights w_ik are computed in one of two ways (--method):

* exact  (default) - nucres.ResonanceSum.bin_integrals_per_resonance: the
           same per-resonance bin integrals, with energy-dependent widths,
           that the [integration] step sums to make the ATCS. Tails that
           spill into neighbouring bins are assigned to those bins.
* narrow - the narrow-resonance shortcut, w_i ∝ ωγ_i / E_r,i, placed
           entirely in the bin that holds E_r. Fast, but blind to tails.

Over the ensemble of runs, the median and the 16-84% range of each quantity
are plotted per bin. An approximate Gamow window is shaded for reference.

Example
-------
    python scripts/plot_resonances_per_bin.py \\
        --reaction "14O(a,p)17F" --reaction "18Ne(a,p)21Na" \\
        --runs-root outputs/resonances --emin 0.1 --emax 10.0 --de 0.2 \\
        --xmax 3.0 --t9 1.0 --out resonances_per_bin.png

Runtime: building the penetrability tables takes about a minute per reaction
(once per worker process); after that each run takes a fraction of a second.
With 1,000 runs, --workers 1 or 2 is enough.

Optional check against the ATCS files written by [integration]:

        --check-csv "path/to/{stem}_integrated_xs_dE_0.2_run{run}.csv"

{stem} becomes e.g. 14O_ap_17F, {reaction} e.g. 14O(a,p)17F, {de} the bin
width as given to --de (e.g. 0.2) and {run} the RUN_* index. For every run whose
file exists, the summed per-resonance integrals must reproduce the saved
bin-averaged cross section; the largest mismatch is reported.

Notes
-----
* --emin, --emax and --de must match the [basics]/[integration] settings of
  the ensemble, otherwise the bins differ from the ATCS bins. --xmax only
  zooms the plot.
* --penetrability-model must match the ensemble (default: coulomb).
* Parity is reconstructed as π = π_p π_t (-1)^L1. For α on an even-even
  target both intrinsic parities are +, so π = (-1)^L1.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The parsers and nucres live in the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from extract_resonance_data import (  # noqa: E402
    extract_data,
    load_nuclear_params,
    load_reaction_params,
)
from nucres.reaction_names import file_stem as reaction_stem  # noqa: E402
from nucres.resonance_sum import ResonanceSum, rows_from_dataframe  # noqa: E402


# --------------------------------------------------------------------------
# Bins
# --------------------------------------------------------------------------
def bin_edges(e_min: float, e_max: float, de: float) -> np.ndarray:
    """
    Edges e_min, e_min + de, ... of every complete bin of width de in
    [e_min, e_max]; the same rule as generate_cross_sections_vectorized.py.
    """
    n_bins = int(np.floor((e_max - e_min) / de + 1e-9))
    if n_bins < 1:
        raise ValueError(f"--de {de} MeV is wider than [{e_min}, {e_max}] MeV.")
    return e_min + de * np.arange(n_bins + 1)


# --------------------------------------------------------------------------
# One synthetic nucleus
# --------------------------------------------------------------------------
@dataclass
class ResonanceLadder:
    """
    The resonances of one sampled run, and their weight in each bin.

    Attributes
    ----------
    energy_mev  : resonance energies E_r (centre of mass, MeV), shape (n_res,)
    spin        : resonance spins J, shape (n_res,)
    parity      : resonance parities (+1 or -1), shape (n_res,)
    bin_weights : w[i, k], resonance i's integrated cross section in bin k
                  (b MeV for "exact"; arbitrary units for "narrow"),
                  shape (n_res, n_bins)
    """

    energy_mev: np.ndarray
    spin: np.ndarray
    parity: np.ndarray
    bin_weights: np.ndarray

    @classmethod
    def from_ratesmc_input(cls, path: str, edges: np.ndarray, method: str = "exact",
                           e_min: float = 0.1, e_max: float = 10.0,
                           penetrability_model: str = "coulomb",
                           cache: dict | None = None) -> "ResonanceLadder":
        """
        Build a ladder from a RatesMC .in file written by the [resonance] step.

        Uses the resonances ResonanceSum keeps: E_r > 0 and nonzero entrance
        and exit widths (for sub-threshold rows G1 holds a reduced width, not
        a partial width). e_min and e_max set the range of the penetrability
        tables and should match [basics].
        """
        reaction = load_reaction_params(path)
        rs = ResonanceSum(reaction, rows_from_dataframe(extract_data(path)),
                          e_min, e_max, penetrability_model=penetrability_model,
                          penetrability_cache=cache)
        energy = np.array([r.E_r for r in rs.rows])
        spin = np.array([r.J for r in rs.rows])
        parity = np.array([1 if r.L0 % 2 == 0 else -1 for r in rs.rows])

        if method == "exact":
            weights = rs.bin_integrals_per_resonance(edges)
        elif method == "narrow":
            omega = (2 * spin + 1) / rs.omega_den
            g_in = np.array([r.G0 for r in rs.rows])
            g_out = np.array([r.G1 for r in rs.rows])
            g_tot = g_in + g_out + np.array([r.G2 for r in rs.rows])
            w = omega * g_in * g_out / g_tot / energy
            weights = np.zeros((len(energy), len(edges) - 1))
            k = np.digitize(energy, edges) - 1
            inside = (k >= 0) & (k < len(edges) - 1)
            weights[np.flatnonzero(inside), k[inside]] = w[inside]
        else:
            raise ValueError(f"Unknown method {method!r}; use 'exact' or 'narrow'.")
        return cls(energy, spin, parity, weights)


# --------------------------------------------------------------------------
# Counting resonances in bins
# --------------------------------------------------------------------------
class BinOccupancy:
    """N and N_eff per bin for one ladder."""

    def __init__(self, edges_mev: np.ndarray):
        self.edges = np.asarray(edges_mev, float)

    def count(self, ladder: ResonanceLadder) -> np.ndarray:
        """N: number of resonances whose E_r lies in each bin."""
        return np.histogram(ladder.energy_mev, bins=self.edges)[0].astype(float)

    @staticmethod
    def effective_count(ladder: ResonanceLadder) -> np.ndarray:
        """N_eff = (Σ_i w_ik)² / Σ_i w_ik² in each bin (0 where no resonance contributes)."""
        w = ladder.bin_weights
        total = w.sum(axis=0)
        total_sq = (w ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total_sq > 0, total ** 2 / total_sq, 0.0)



# --------------------------------------------------------------------------
# The ensemble of runs
# --------------------------------------------------------------------------
class EnsembleOccupancy:
    """
    N and N_eff for every run of one reaction.

    `counts` and `effective_counts` are arrays of shape (n_runs, n_bins).
    """

    def __init__(self, reaction: str, counts: np.ndarray,
                 effective_counts: np.ndarray, edges_mev: np.ndarray):
        self.reaction = reaction
        self.edges = np.asarray(edges_mev, float)
        self.counts = counts
        self.effective_counts = effective_counts

    @property
    def n_runs(self) -> int:
        return self.counts.shape[0]

    @staticmethod
    def band(values: np.ndarray, low: float = 0.16, high: float = 0.84):
        """Median and the [low, high] quantiles across runs, per bin."""
        q = np.quantile(values, [low, 0.5, high], axis=0)
        return q[0], q[1], q[2]

    def plot(self, ax, gamow: tuple[float, float] | None = None,
             gamow_label: str = "", xmax: float | None = None) -> None:
        """Draw median and 16-84% band of N and N_eff on one axes."""
        for values, colour, label in (
            (self.counts, "#2F5D9E", "N (resonance peaks in bin)"),
            (self.effective_counts, "#B4501F", r"$N_\mathrm{eff}$ (resonances that set the bin)"),
        ):
            lo, med, hi = self.band(values)
            ax.stairs(hi, self.edges, baseline=lo, fill=True, color=colour, alpha=0.25)
            ax.stairs(med, self.edges, color=colour, linewidth=2, label=label + ", median")

        ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1)
        ax.text(self.edges[0], 1.05, " one resonance", va="bottom", fontsize=9, color="0.2")
        if gamow is not None:
            ax.axvspan(*gamow, color="0.85", zorder=0, label=gamow_label)

        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
        ax.set_xlim(self.edges[0], xmax if xmax is not None else self.edges[-1])
        ax.set_xlabel(r"$E_\mathrm{c.m.}$ (MeV)")
        ax.set_title(f"{self.reaction}  ({self.n_runs} runs, band = 16–84%)")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def gamow_window(z1: int, z2: int, a1: float, a2: float, t9: float) -> tuple[float, float]:
    """
    Approximate Gamow window E0 ± Δ/2 in MeV (non-resonant estimate):

        E0 = 0.1220 (Z1² Z2² μ T9²)^(1/3),   Δ = 0.2368 (Z1² Z2² μ T9⁵)^(1/6)

    with μ the reduced mass in u. For narrow resonances the energies that
    really set the rate can sit elsewhere; calc_gamow_window_cdf.py gives
    the exact picture.
    """
    mu = a1 * a2 / (a1 + a2)
    e0 = 0.1220 * (z1 ** 2 * z2 ** 2 * mu * t9 ** 2) ** (1 / 3)
    width = 0.2368 * (z1 ** 2 * z2 ** 2 * mu * t9 ** 5) ** (1 / 6)
    return e0 - width / 2, e0 + width / 2


def find_runs(runs_root: str, reaction: str) -> list[str]:
    """
    All RatesMC inputs of a reaction, one per RUN_* directory, in run order.

    Looks first in the ensemble layout written by the [resonance] step,

        <runs_root>/<stem>/<reaction>/RUN_*/*.in     e.g.
        outputs/resonances/14O_ap_17F/14O(a,p)17F/RUN_0/14O(a,p)17F.in

    and, if nothing is found there, in the flat layout
    <runs_root>/<reaction>/RUN_*/*.in. Inside a run directory the file named
    <reaction>.in is preferred; otherwise the only .in file is used.
    """
    candidates = [
        os.path.join(runs_root, reaction_stem(reaction), reaction),
        os.path.join(runs_root, reaction),
    ]
    for base in candidates:
        run_dirs = sorted(glob.glob(os.path.join(glob.escape(base), "RUN_*")), key=run_index)
        files = []
        for run_dir in run_dirs:
            preferred = os.path.join(run_dir, reaction + ".in")
            if os.path.isfile(preferred):
                files.append(preferred)
                continue
            found = glob.glob(os.path.join(glob.escape(run_dir), "*.in"))
            if len(found) == 1:
                files.append(found[0])
            elif len(found) > 1:
                raise ValueError(f"{run_dir}: several .in files and none named {reaction}.in")
        if files:
            return files
    raise FileNotFoundError(
        f"No RUN_*/*.in files for {reaction} under: " + ", ".join(candidates))


def run_index(path: str) -> int:
    """The N of the RUN_N directory in a path (-1 if there is none)."""
    m = re.search(r"RUN_(\d+)", path)
    return int(m.group(1)) if m else -1


_worker_cache: dict = {}


def _process_run(task):
    """Worker: N, N_eff and (optionally) the bin-averaged cross section of one run."""
    path, edges, method, e_min, e_max, model = task
    # One penetrability cache per worker process, reused across that worker's runs.
    cache = _worker_cache.setdefault((load_reaction_params(path), e_min, e_max, model), {})
    ladder = ResonanceLadder.from_ratesmc_input(
        path, edges, method=method, e_min=e_min, e_max=e_max,
        penetrability_model=model, cache=cache)
    binner = BinOccupancy(edges)
    avg_mb = ladder.bin_weights.sum(axis=0) / np.diff(edges) * 1e3 if method == "exact" else None
    return binner.count(ladder), binner.effective_count(ladder), avg_mb


def check_against_csv(pattern: str, reaction: str, files: list[str],
                      averages: list, edges: np.ndarray, de: float) -> None:
    """
    Compare recomputed bin averages with the ATCS files of [integration].

    The pattern may use {stem} (e.g. 14O_ap_17F), {reaction} (e.g.
    14O(a,p)17F), {de} (the bin width, e.g. 0.2) and {run} (the RUN_* index).
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    worst, n_checked = 0.0, 0
    for path, avg in zip(files, averages):
        try:
            csv = pattern.format(stem=reaction_stem(reaction), reaction=reaction,
                                 de=f"{de:g}", run=run_index(path))
        except KeyError as exc:
            raise ValueError(f"--check-csv: unknown placeholder {exc}; use {{stem}}, "
                             "{reaction}, {de} or {run}.") from None
        if not os.path.isfile(csv):
            continue
        saved = np.loadtxt(csv, delimiter=",", comments="#")
        if saved.shape[0] != len(centres) or not np.allclose(saved[:, 0], centres, atol=1e-6):
            raise ValueError(f"{csv}: bins differ from --emin/--emax/--de "
                             f"({saved.shape[0]} bins in file, {len(centres)} here).")
        ok = saved[:, 1] > 0
        rel = np.abs(avg[ok] - saved[ok, 1]) / saved[ok, 1]
        worst = max(worst, rel.max(initial=0.0))
        n_checked += 1
    if n_checked == 0:
        print(f"  check: no files matched {pattern!r}")
    else:
        flag = "OK" if worst < 1e-3 else "MISMATCH - check model and bins"
        print(f"  check: {n_checked} runs vs saved ATCS, largest relative difference "
              f"{worst:.2e}  [{flag}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reaction", action="append", required=True,
                   help="Reaction name as used in outputs/, e.g. '14O(a,p)17F'. Repeatable.")
    p.add_argument("--runs-root", default="outputs/resonances",
                   help="Folder holding <stem>/<reaction>/RUN_*/ (default: outputs/resonances).")
    p.add_argument("--emin", type=float, default=0.1, help="E_min_mev of the ensemble (MeV).")
    p.add_argument("--emax", type=float, default=10.0, help="E_max_mev of the ensemble (MeV).")
    p.add_argument("--de", type=float, default=0.2, help="ATCS bin width dE (MeV).")
    p.add_argument("--xmax", type=float, default=None, help="Upper end of the plotted range (MeV).")
    p.add_argument("--method", choices=["exact", "narrow"], default="exact")
    p.add_argument("--penetrability-model", choices=["coulomb", "jwkb_real_omp"],
                   default="coulomb")
    p.add_argument("--t9", type=float, default=1.0, help="Temperature for the shaded Gamow window.")
    p.add_argument("--max-runs", type=int, default=None, help="Use only the first N runs (for testing).")
    p.add_argument("--workers", type=int, default=1, help="Parallel processes.")
    p.add_argument("--check-csv", default=None,
                   help="Pattern of saved ATCS files; may use {stem}, {reaction}, {de} and {run}, "
                        "e.g. 'outputs/cross_sections/{stem}/{reaction}/binwidth_{de}/"
                        "{stem}_integrated_xs_dE_{de}_run{run}.csv'.")
    p.add_argument("--out", default="resonances_per_bin.png")
    args = p.parse_args()

    edges = bin_edges(args.emin, args.emax, args.de)
    fig, axes = plt.subplots(1, len(args.reaction), figsize=(7 * len(args.reaction), 5),
                             sharey=True, squeeze=False)

    for ax, reaction in zip(axes[0], args.reaction):
        files = find_runs(args.runs_root, reaction)[: args.max_runs]
        print(f"{reaction}: {len(files)} runs from {os.path.dirname(os.path.dirname(files[0]))}"
              f"  [method={args.method}, {args.penetrability_model}]")
        tasks = [(f, edges, args.method, args.emin, args.emax, args.penetrability_model)
                 for f in files]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_process_run, tasks, chunksize=8))
        else:
            results = [_process_run(t) for t in tasks]

        counts = np.array([r[0] for r in results])
        neff = np.array([r[1] for r in results])
        if args.check_csv and args.method == "exact":
            check_against_csv(args.check_csv, reaction, files, [r[2] for r in results], edges, args.de)

        ensemble = EnsembleOccupancy(reaction, counts, neff, edges)
        nuc = load_nuclear_params(files[0])
        window = gamow_window(nuc["Z_proj"], nuc["Z_tar"], nuc["A_proj"], nuc["A_tar"], args.t9)
        ensemble.plot(ax, gamow=window, gamow_label=f"Gamow window, T9 = {args.t9:g} (approx.)",
                      xmax=args.xmax)
        print(f"  Gamow window ≈ {window[0]:.2f}–{window[1]:.2f} MeV")

    axes[0][0].set_ylabel("Resonances per bin")
    axes[0][0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
