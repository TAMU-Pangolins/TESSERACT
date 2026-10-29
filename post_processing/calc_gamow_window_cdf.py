#!/usr/bin/env python3
"""
calc_gamow_window_cdf.py

Calculates and plots the CDF of the Gamow window integrand  sigma(E) * E * exp(-E / kT)
for TALYS and/or RatesMC cross sections at one T9 value, across every
available run of a reaction's ensemble, showing the fractional contribution
of each energy to the total reaction-rate integral.

By default, plots the 95% interval (2.5-97.5 percentile band across runs)
of the CDF. Pass --no-band to instead plot every run's individual CDF
curve, optionally restricted to specific run(s) with --run-idx.

Input layout (TESSERACT outputs/ convention):

    {base_dir}/talys_opt/{stem}/binwidth_{bw}/RUN_{N}/talys_results_*.npz
    {base_dir}/cross_sections/{stem}/{reaction}_xs_unintegrated_parallel_run{N}.txt

Usage:
    # 95% interval across every available run, at T9 = 1.0 GK:
    python calc_gamow_window_cdf.py --reaction "14O(a,p)17F" --binwidth 0.2 --t9 1.0

    # Every individual run's curve instead of the band:
    python calc_gamow_window_cdf.py --reaction "14O(a,p)17F" --binwidth 0.2 --t9 1.0 --no-band

    # Just run(s) 3 and 7 (only valid together with --no-band):
    python calc_gamow_window_cdf.py --reaction "14O(a,p)17F" --binwidth 0.2 --t9 1.0 \
        --no-band --run-idx 3 7
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # nucres
from nucres.reaction_names import file_stem as reaction_slug  # noqa: E402

# Boltzmann constant in MeV/GK
kB_MeV_per_GK = 0.08617

STYLE = {"TALYS": ("-", "C0"), "RatesMC": ("--", "C1")}


def integrand(E_mev, sigma_mb, T9):
    """sigma(E) * E * exp(-E / kT)  [mb * MeV]"""
    kT = kB_MeV_per_GK * T9
    return sigma_mb * E_mev * np.exp(-E_mev / kT)


def cdf(E_mev, ig):
    """Normalized cumulative integral of the integrand vs energy (0 -> 1)."""
    cum = cumulative_trapezoid(ig, E_mev, initial=0.0)
    if not np.isfinite(cum[-1]) or cum[-1] <= 0:
        raise ValueError('A CDF requires a finite positive total integral')
    return cum / cum[-1]


def cdf_normalized_to_ratesmc(E_mev, ig, ratesmc_total):
    """
    Cumulative integral of the integrand vs energy, normalized to the
    RatesMC total rate of the same run instead of its own total. RatesMC's
    own curve is unaffected (it ends at 1.0 as before), but any other curve
    now shows what fraction of the RatesMC-computed rate it has reproduced
    by energy E -- it can plateau below 1.0 (under-predicts the total rate)
    or rise above 1.0 (over-predicts it).
    """
    if not np.isfinite(ratesmc_total) or ratesmc_total <= 0:
        raise ValueError('The reference integral must be finite and positive')
    cum = cumulative_trapezoid(ig, E_mev, initial=0.0)
    return cum / ratesmc_total


def percentile_energy(target_frac, c, E_mev):
    """
    Energy at which the cumulative first reaches target_frac, or None if
    the curve never gets there -- only possible when normalized to a
    reference other than its own total (see cdf_normalized_to_ratesmc).
    """
    if target_frac > c[-1]:
        return None
    return np.interp(target_frac, c, E_mev)


def load_talys_xs(npz_path):
    """Load TALYS cross section from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return data["x_xs"], data["y_xs"]   # MeV, mb


def load_ratesmc_xs(txt_path):
    """
    Load RatesMC unintegrated cross section.
    File format: comma-separated, header line starting with #
      E (MeV), sigma (mb)
    """
    data = np.loadtxt(txt_path, delimiter=",", comments="#", ndmin=2)
    return data[:, 0], data[:, 1]   # MeV, mb


def talys_on_fine_grid(energies, cross_sections, reference_energies):
    """Interpolate positive TALYS cross sections in log space without extrapolation.

    All energy arrays are in MeV and cross sections are in mb. TALYS energies
    must be strictly increasing. Fewer than two positive finite samples, or
    no reference-grid overlap, returns empty arrays for the caller to skip.
    """
    energies = np.asarray(energies, dtype=float)
    cross_sections = np.asarray(cross_sections, dtype=float)
    reference_energies = np.asarray(reference_energies, dtype=float)
    reference_energies = reference_energies[np.isfinite(reference_energies)]
    valid = np.isfinite(energies) & np.isfinite(cross_sections) & (cross_sections > 0)
    energies, cross_sections = energies[valid], cross_sections[valid]
    if energies.size < 2 or reference_energies.size == 0:
        return np.array([]), np.array([])
    if np.any(np.diff(energies) <= 0):
        raise ValueError('TALYS energies must be strictly increasing')
    inside = (reference_energies >= energies[0]) & (reference_energies <= energies[-1])
    lower_energy = max(energies[0], reference_energies.min())
    upper_energy = min(energies[-1], reference_energies.max())
    if lower_energy >= upper_energy:
        return np.array([]), np.array([])
    output_energies = np.unique(np.concatenate(
        ([lower_energy], reference_energies[inside], [upper_energy])))
    output_cross_sections = np.exp(np.interp(output_energies, energies, np.log(cross_sections)))
    return output_energies, output_cross_sections


def fraction_outside(energies, integrand_values, lower_energy, upper_energy):
    """Return reference-integral fractions below/above the TALYS energy range.

    Integrate a linearly interpolated reference integrand, including partial
    grid cells at each boundary. The energy grid must be strictly increasing.
    """
    energies = np.asarray(energies, dtype=float)
    integrand_values = np.asarray(integrand_values, dtype=float)
    if energies.size < 2:
        return 0.0, 0.0
    if np.any(np.diff(energies) <= 0):
        raise ValueError('Reference energies must be strictly increasing')
    cumulative = cumulative_trapezoid(integrand_values, energies, initial=0.0)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return 0.0, 0.0

    def integral_below(boundary):
        if boundary <= energies[0]:
            return 0.0
        if boundary >= energies[-1]:
            return total
        index = np.searchsorted(energies, boundary, side='right') - 1
        boundary_value = np.interp(boundary, energies, integrand_values)
        partial_area = 0.5 * (integrand_values[index] + boundary_value) * (boundary - energies[index])
        return cumulative[index] + partial_area

    return integral_below(lower_energy) / total, 1.0 - integral_below(upper_energy) / total


# ─────────────────────────────────────────────────────────────────────────────
# Discover every run that has both a TALYS npz and a RatesMC unintegrated xs
# ─────────────────────────────────────────────────────────────────────────────
_RUN_DIR_RE = re.compile(r"^RUN_(\d+)$")


def discover_runs(base_dir: str, reaction: str, binwidth: str) -> list[tuple[int, Path, Path]]:
    """
    Find every run index with both files present.

    Returns a list of (run_idx, npz_path, xs_path), sorted by run index.
    Run index comes from the talys_opt RUN_N directory name; the matching
    xs file is looked up by substituting that same run index into the
    cross_sections filename pattern.
    """
    slug = reaction_slug(reaction)
    talys_opt_dir = Path(base_dir) / "talys_opt" / slug / f"binwidth_{binwidth}"
    print(talys_opt_dir)
    xs_dir = Path(base_dir) / "cross_sections" / slug

    runs = []
    for run_dir in sorted(talys_opt_dir.glob("RUN_*")):
        m = _RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        run = int(m.group(1))
        npz_matches = sorted(run_dir.glob("*.npz"))
        #print(npz_matches)
        if not npz_matches:
            continue
        if len(npz_matches) > 1:
            raise ValueError(f'Ambiguous TALYS results in {run_dir}: {npz_matches}')
        xs_path = xs_dir / f"{reaction}_xs_unintegrated_parallel_run{run}.txt"
        if not xs_path.exists():
            continue
        runs.append((run, npz_matches[0], xs_path))

    runs.sort(key=lambda r: r[0])
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default="/data/pagroup/pa02/THICC/outputs",
                        help="Root outputs directory containing talys_opt/ and cross_sections/")
    parser.add_argument("--reaction", required=True, help="e.g. '14O(a,p)17F'")
    parser.add_argument("--binwidth", default="0.2",
                        help="Bin width as in the talys_opt folder name binwidth_<dE>.")
    parser.add_argument("--t9", type=float, required=True, help="T9 value (GK) to plot.")
    parser.add_argument("--e-max", type=float, default=10.0,
                        help="Maximum energy (MeV) for x-axis (default: 10)")
    parser.add_argument("--no-talys", action="store_true",
                        help="Skip plotting the TALYS CDF")
    parser.add_argument("--no-ratesmc", action="store_true",
                        help="Skip plotting the RatesMC CDF")
    parser.add_argument("--no-band", action="store_true",
                        help="Skip the 95%% interval and instead plot every run's "
                             "individual CDF curve.")
    parser.add_argument("--run-idx", type=int, nargs="+", default=None,
                        help="Restrict to specific run(s) (only valid together with --no-band).")
    parser.add_argument("--output", default=None,
                        help="Output PNG (default: <reaction>_gamow_window_cdf_T9_<t9>_<band|no_band>.png)")
    parser.add_argument("--normalize", choices=["self", "ratesmc"], default="self",
                        help="Normalize each run's CDF to its own total integral (default), "
                             "or to that same run's RatesMC total rate.")
    args = parser.parse_args()

    if not np.isfinite(args.t9) or args.t9 <= 0:
        parser.error('--t9 must be finite and positive')

    if args.run_idx is not None and not args.no_band:
        parser.error("--run-idx is only meaningful together with --no-band.")

    runs = discover_runs(args.base_dir, args.reaction, args.binwidth)
    if not runs:
        raise SystemExit(
            f"No runs found with both a TALYS .npz and a RatesMC unintegrated xs file "
            f"under {args.base_dir} for {args.reaction!r}, binwidth {args.binwidth}."
        )
    if args.run_idx is not None:
        wanted = set(args.run_idx)
        found = {r[0] for r in runs}
        missing = wanted - found
        if missing:
            print(f"WARNING: requested run(s) not found, skipping: {sorted(missing)}")
        runs = [r for r in runs if r[0] in wanted]
        if not runs:
            raise SystemExit("None of the requested --run-idx run(s) were found.")

    sources = []
    if not args.no_talys:
        sources.append("TALYS")
    if not args.no_ratesmc:
        sources.append("RatesMC")
    if not sources:
        raise SystemExit("--no-talys and --no-ratesmc both given; nothing to plot.")

    print(f"Found {len(runs)} run(s)")

    # ── Per-run CDF curves, keyed by source ─────────────────────────────────
    curves = {label: [] for label in sources}   # label -> list of (run, E_m, c)

    for run, npz_path, xs_path in runs:
        E_talys, xs_talys = load_talys_xs(str(npz_path))
        E_rmc,   xs_rmc   = load_ratesmc_xs(str(xs_path))

        order_t = np.argsort(E_talys)
        order_r = np.argsort(E_rmc)
        E_talys, xs_talys = E_talys[order_t], xs_talys[order_t]
        E_rmc,   xs_rmc   = E_rmc[order_r],   xs_rmc[order_r]

        mask_t = E_talys <= args.e_max
        mask_r = E_rmc   <= args.e_max
        E_talys, xs_talys = E_talys[mask_t], xs_talys[mask_t]
        E_rmc,   xs_rmc   = E_rmc[mask_r],   xs_rmc[mask_r]

        if not args.no_talys:
            E_talys, xs_talys = talys_on_fine_grid(E_talys, xs_talys, E_rmc)
            if E_talys.size < 2:
                print(f'WARNING: run {run}: insufficient positive TALYS coverage; skipping its CDF')

        # This run's RatesMC total, used as the normalization reference when
        # --normalize ratesmc (always computed, since RatesMC's own curve
        # needs it too).
        ig_rmc_ref = integrand(E_rmc, xs_rmc, args.t9)
        mask_ref = np.isfinite(ig_rmc_ref) & (ig_rmc_ref >= 0)
        if mask_ref.sum() < 2:
            print(f"  run {run}: RatesMC cross section unusable at T9={args.t9} GK, skipping")
            continue
        ratesmc_total = cumulative_trapezoid(
            ig_rmc_ref[mask_ref], E_rmc[mask_ref], initial=0.0)[-1]
        if not np.isfinite(ratesmc_total) or ratesmc_total <= 0:
            print(f'WARNING: run {run}: reference integral is not positive; skipping')
            continue
        if not args.no_talys and E_talys.size >= 2:
            below, above = fraction_outside(E_rmc[mask_ref], ig_rmc_ref[mask_ref],
                                            E_talys[0], E_talys[-1])
            if below > 0.01 or above > 0.01:
                print(f'WARNING: run {run}: {below:.1%} of the reference integral is below '
                      f'TALYS coverage and {above:.1%} is above it')

        per_source = {"TALYS": (E_talys, xs_talys), "RatesMC": (E_rmc, xs_rmc)}
        for label in sources:
            E, xs = per_source[label]
            ig = integrand(E, xs, args.t9)
            mask = np.isfinite(ig) & (ig >= 0)
            E_m, ig_m = E[mask], ig[mask]
            if E_m.size < 2:
                continue
            if not np.any(ig_m > 0):
                continue
            if args.normalize == "ratesmc":
                c = cdf_normalized_to_ratesmc(E_m, ig_m, ratesmc_total)
            else:
                c = cdf(E_m, ig_m)
            curves[label].append((run, E_m, c))

            e05 = percentile_energy(0.05, c, E_m)
            e50 = percentile_energy(0.50, c, E_m)
            e95 = percentile_energy(0.95, c, E_m)

            def fmt(v):
                return f"{v:.3f} MeV" if v is not None else "not reached"
            print(f"{label}  run={run}  T9={args.t9} GK  ->  E5%={fmt(e05)}  "
                  f"E50%={fmt(e50)}  E95%={fmt(e95)}")

    if not any(curves.values()):
        raise SystemExit('No usable cross-section curves remain in the requested energy range')

    # ── Common energy grid for band mode (union of every curve's points) ───
    e_union = set()
    for label in sources:
        for run, E_m, c in curves[label]:
            e_union.update(E_m.tolist())
    e_common = np.array(sorted(e_union))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    max_c_seen = 1.0

    for label in sources:
        ls, color = STYLE[label]
        run_curves = curves[label]
        if not run_curves:
            continue

        if args.no_band:
            for j, (run, E_m, c) in enumerate(run_curves):
                show_label = (f"{label} (run {run})" if len(run_curves) <= 6
                             else (label if j == 0 else None))
                ax.plot(E_m, c, lw=1.0, ls=ls, color=color, alpha=0.5, label=show_label)
                max_c_seen = max(max_c_seen, float(c[-1]))
        else:
            stack = []
            for run, E_m, c in run_curves:
                f = interp1d(E_m, c, bounds_error=False, fill_value=np.nan)
                stack.append(f(e_common))
            stack = np.vstack(stack)
            med = np.nanpercentile(stack, 50, axis=0)
            lo  = np.nanpercentile(stack, 2.5, axis=0)
            hi  = np.nanpercentile(stack, 97.5, axis=0)
            valid = np.isfinite(med)
            ax.fill_between(e_common[valid], lo[valid], hi[valid], color=color, alpha=0.3,
                            label=f"{label} 95% interval")
            ax.plot(e_common[valid], med[valid], color=color, lw=1.5, ls=ls)
            max_c_seen = max(max_c_seen, float(np.nanmax(hi)))

    ax.set_xlabel("Energy (MeV)", fontsize=12)
    ylabel = r'$F(\langle\sigma v\rangle)$'
    if args.normalize == "ratesmc":
        ylabel += r"$_{RMC}$"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, args.e_max)
    ax.set_ylim(-0.03, max(1.1, 1.05 * max_c_seen))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    if args.output:
        out_png = args.output
    else:
        slug = reaction_slug(args.reaction)
        mode = "no_band" if args.no_band else "band"
        out_png = f"{slug}_gamow_window_cdf_T9_{args.t9:g}_{mode}.png"
    plt.savefig(out_png, dpi=300)
    print(f"Wrote {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
