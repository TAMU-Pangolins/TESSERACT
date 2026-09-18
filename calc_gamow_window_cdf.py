#!/usr/bin/env python3
"""
calc_gamow_window_cdf.py

Calculates and plots the CDF of the Gamow window integrand  sigma(E) * E * exp(-E / kT)
for both TALYS and RatesMC cross sections at several T9 values, showing the
fractional contribution of each energy to the total reaction-rate integral.

Usage:
    python calc_gamow_window_cdf.py \
        --npz   /data/pagroup/pa02/TESSERACT/outputs/talys_opt/RUN_0/talys_results_*.npz \
        --xs    /data/pagroup/pa02/TESSERACT/outputs/cross_sections/22Mg_ap_25Al_xs_unintegrated_parallel_run0.txt \
        --t9    0.3 0.5 1.0 1.5 2.0 \
        --e-max 10.0
"""

import argparse
import glob
import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Boltzmann constant in MeV/GK
kB_MeV_per_GK = 0.08617

# Fixed horizontal nudge (MeV) applied to percentile labels, away from
# whichever curve of the pair sits closer to it.
LABEL_OFFSET = 0.02


def integrand(E_mev, sigma_mb, T9):
    """sigma(E) * E * exp(-E / kT)  [mb * MeV]"""
    kT = kB_MeV_per_GK * T9
    return sigma_mb * E_mev * np.exp(-E_mev / kT)


def cdf(E_mev, ig):
    """Normalized cumulative integral of the integrand vs energy (0 -> 1)."""
    cum = cumulative_trapezoid(ig, E_mev, initial=0.0)
    return cum / cum[-1]


def cdf_normalized_to_ratesmc(E_mev, ig, ratesmc_total):
    """
    Cumulative integral of the integrand vs energy, normalized to the
    RatesMC total rate instead of its own total. RatesMC's own curve is
    unaffected (it ends at 1.0 as before), but any other curve now shows
    what fraction of the RatesMC-computed rate it has reproduced by energy
    E -- it can plateau below 1.0 (under-predicts the total rate) or rise
    above 1.0 (over-predicts it).
    """
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
    data = np.loadtxt(txt_path, delimiter=",", comments="#")
    return data[:, 0], data[:, 1]   # MeV, mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",   required=True,
                        help="Path to TALYS .npz file (glob patterns accepted)")
    parser.add_argument("--xs",    required=True,
                        help="Path to RatesMC unintegrated cross section .txt file")
    parser.add_argument("--t9",    nargs="+", type=float,
                        default=[0.3, 0.5, 1.0, 1.5, 2.0],
                        help="T9 values (GK) to plot (default: 0.3 0.5 1.0 1.5 2.0)")
    parser.add_argument("--e-max", type=float, default=10.0,
                        help="Maximum energy (MeV) for x-axis (default: 10)")
    parser.add_argument("--no-talys", action="store_true",
                        help="Skip plotting the TALYS CDF")
    parser.add_argument("--no-ratesmc", action="store_true",
                        help="Skip plotting the RatesMC CDF")
    parser.add_argument("--normalize", choices=["self", "ratesmc"], default="self",
                        help="Normalize each curve's CDF to its own total "
                             "integral (default), or to the RatesMC total "
                             "rate so all curves share one reference scale")
    args = parser.parse_args()

    # ── Resolve NPZ path (supports glob) ────────────────────────────────────
    npz_matches = sorted(glob.glob(args.npz))
    if not npz_matches:
        raise FileNotFoundError(f"No NPZ file found matching: {args.npz}")
    npz_path = npz_matches[0]
    print(f"TALYS NPZ : {npz_path}")
    print(f"RatesMC XS: {args.xs}")

    # ── Load cross sections ──────────────────────────────────────────────────
    E_talys, xs_talys   = load_talys_xs(npz_path)
    E_rmc,   xs_rmc     = load_ratesmc_xs(args.xs)

    # Sort by energy (required for cumulative integration)
    order_t = np.argsort(E_talys)
    order_r = np.argsort(E_rmc)
    E_talys, xs_talys = E_talys[order_t], xs_talys[order_t]
    E_rmc,   xs_rmc   = E_rmc[order_r],   xs_rmc[order_r]

    # Apply energy limit
    mask_t = E_talys <= args.e_max
    mask_r = E_rmc   <= args.e_max
    E_talys, xs_talys = E_talys[mask_t], xs_talys[mask_t]
    E_rmc,   xs_rmc   = E_rmc[mask_r],   xs_rmc[mask_r]

    # ── Color map — one color per T9 ────────────────────────────────────────
    t9_vals = sorted(args.t9)
    colors  = cm.plasma(np.linspace(0.15, 0.85, len(t9_vals)))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    sources = []
    if not args.no_talys:
        sources.append((E_talys, xs_talys, "TALYS",   "-"))
    if not args.no_ratesmc:
        sources.append((E_rmc,   xs_rmc,   "RatesMC", "--"))

    percentiles = [(0.05, "5%"), (0.50, "50%"), (0.95, "95%")]

    max_c_seen = 1.0

    for T9, color in zip(t9_vals, colors):
        # RatesMC's own total at this T9 -- always computed (even under
        # --no-ratesmc) since it's needed as the normalization reference.
        ig_rmc_ref = integrand(E_rmc, xs_rmc, T9)
        mask_ref = np.isfinite(ig_rmc_ref) & (ig_rmc_ref >= 0)
        ratesmc_total = cumulative_trapezoid(
            ig_rmc_ref[mask_ref], E_rmc[mask_ref], initial=0.0)[-1]

        # Compute the CDF and its E5%/E50%/E95% energies for each source
        # present at this T9.
        results = {}
        for E, xs, label, ls in sources:
            ig = integrand(E, xs, T9)
            mask = np.isfinite(ig) & (ig >= 0)
            E_m, ig_m = E[mask], ig[mask]
            if E_m.size < 2:
                continue
            if args.normalize == "ratesmc":
                c = cdf_normalized_to_ratesmc(E_m, ig_m, ratesmc_total)
            else:
                c = cdf(E_m, ig_m)
            max_c_seen = max(max_c_seen, c[-1])

            e05 = percentile_energy(0.05, c, E_m)
            e50 = percentile_energy(0.50, c, E_m)
            e95 = percentile_energy(0.95, c, E_m)

            def fmt(v):
                return f"{v:.3f} MeV" if v is not None else "not reached"
            print(f"{label}  T9={T9} GK  ->  E5%={fmt(e05)}  "
                  f"E50%={fmt(e50)}  E95%={fmt(e95)}")

            ax.plot(E_m, c, lw=1.8, ls=ls, color=color,
                    label=f"{label}  T9={T9} GK")
            results[label] = dict(e05=e05, e50=e50, e95=e95)

        # Place the E5%/E50%/E95% labels: at each percentile, whichever
        # curve sits at the smaller energy gets its label shifted left by
        # LABEL_OFFSET, and the other gets shifted right by LABEL_OFFSET.
        # A curve that never reaches a given percentile (only possible
        # under --normalize ratesmc) simply has no label placed for it.
        labels_present = list(results.keys())
        for frac, pct_str in percentiles:
            key = {"5%": "e05", "50%": "e50", "95%": "e95"}[pct_str]
            x_vals = {lbl: results[lbl][key] for lbl in labels_present
                      if results[lbl][key] is not None}
            if not x_vals:
                continue

            ax.scatter(list(x_vals.values()), [frac] * len(x_vals),
                       color="red", s=18, zorder=5)

            if len(x_vals) == 2:
                left_lbl, right_lbl = sorted(x_vals, key=lambda l: x_vals[l])
                ax.text(x_vals[left_lbl] - LABEL_OFFSET, frac,
                        f"{pct_str}: {x_vals[left_lbl]:.2f}",
                        fontsize=6, color="red", ha="right", va="center")
                ax.text(x_vals[right_lbl] + LABEL_OFFSET, frac,
                        f"{pct_str}: {x_vals[right_lbl]:.2f}",
                        fontsize=6, color="red", ha="left", va="center")
            else:
                only_lbl = next(iter(x_vals))
                ax.text(x_vals[only_lbl] + LABEL_OFFSET, frac,
                        f"{pct_str}: {x_vals[only_lbl]:.2f}",
                        fontsize=6, color="red", ha="left", va="center")

    ax.set_xlabel("Energy (MeV)", fontsize=12)
    ylabel = r'$F(\langle\sigma v\rangle)$'
    if args.normalize == "ratesmc":
        ylabel += r"$_{RMC}$"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, args.e_max)
    ax.set_ylim(-0.03, max(1.1, 1.05 * max_c_seen))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    #fig.suptitle("22Mg(a,p)25Al — Gamow window CDF", fontsize=13)
    plt.tight_layout()
    plt.savefig('22Mg_ap_25Al_gamow_window_cdf.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

