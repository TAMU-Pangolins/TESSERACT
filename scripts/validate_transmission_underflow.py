#!/usr/bin/env python
"""Produce reproducible physics checks for #23, without TALYS or HFB data.

Run from the repository root:
    uv run python scripts/validate_transmission_underflow.py

The baseline defaults to pre-fix commit 588d3d5. Use --baseline-ref
to select another pre-fix commit. Resonances are
illustrative, with allowed angular momenta and widths; they are
not a prediction of the level spectrum. mpmath references use 40-digit arithmetic
and direct Coulomb functions, without the production interpolator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from extract_resonance_data import load_reaction_params
from nucres.resonance_sum import ResonanceRow, ResonanceSum
from tesseract import _wigner_limit_ev


def baseline_sum(ref):
    """Load the actual old sources, including their old interpolation floor."""
    with tempfile.TemporaryDirectory() as directory:
        for filename in ("resonance", "resonance_sum"):
            source = subprocess.check_output(
                ["git", "show", f"{ref}:nucres/{filename}.py"], cwd=ROOT, text=True
            )
            source = source.replace("from .resonance import (", "from ._validation_old_resonance import (")
            path = Path(directory) / f"{filename}.py"
            path.write_text(source)
            name = f"nucres._validation_old_{filename}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.ResonanceSum


def direct_P(l, z1, z2, m1, m2, energy, radius):
    """Independent high-precision rho/(F^2+G^2), with no float conversion."""
    energy = mp.mpf(float(energy))
    mu = mp.mpf(m1) * m2 / (mp.mpf(m1) + m2)
    a = radius * (mp.root(m1, 3) + mp.root(m2, 3))
    rho = mp.mpf("0.218735") * a * mp.sqrt(mu * energy)
    eta = mp.mpf("0.157489") * z1 * z2 * mp.sqrt(mu / energy)
    return rho / (mp.coulombf(l, eta, rho) ** 2 + mp.coulombg(l, eta, rho) ** 2)


def entrance_P(rx, l, energy):
    return direct_P(l, rx.Z_proj, rx.Z_targ, rx.M_proj, rx.M_targ, energy, rx.R0_fm)


def exit_P(rx, l, energy):
    return direct_P(l, rx.Z_exit, rx.Z_proj + rx.Z_targ - rx.Z_exit,
                    rx.M_exit, rx.M_proj + rx.M_targ - rx.M_exit, energy, rx.R0_fm)


def relative_error(actual, reference):
    return np.abs(np.asarray(actual) / reference - 1)


def save_figure(fig, destination, name):
    fig.savefig(destination / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def constant_width_model(base):
    """Controlled constant-width limit of the production BW algebra.

    Only the width laws are frozen; sigma_one and bin_integrals are inherited
    unchanged. The energy-dependent model is checked separately by rate().
    """
    if hasattr(base, "_entrance_log"):
        class ConstantWidth(base):
            def _entrance_log(self, l):
                return lambda E: np.zeros_like(np.asarray(E, dtype=float))

            def _log_scaled(self, channel, G, L, Exf, E, E_r):
                return np.full_like(np.asarray(E, dtype=float), np.log(G) if G > 0 else -np.inf)
    else:
        class ConstantWidth(base):
            def _entrance(self, l):
                return lambda E: np.ones_like(np.asarray(E, dtype=float))

            def _scaled(self, channel, G, L, Exf, E, E_r):
                return np.full_like(np.asarray(E, dtype=float), G)
    return ConstantWidth


def underflow_check(old_sum):
    """Direct high-precision width ratios and physical on-shell normalization."""
    rx = load_reaction_params(ROOT / "input/22Mg(a,p)25Al.txt")
    gi = 0.01 * _wigner_limit_ev(rx.M_proj, rx.M_targ, rx.R0_fm)
    mr = rx.M_proj + rx.M_targ - rx.M_exit
    go = 0.0045 * _wigner_limit_ev(rx.M_exit, mr, rx.R0_fm)
    center = brentq(lambda e: float(mp.log(entrance_P(rx, 20, e))) + 749,
                    0.001, 0.01, xtol=1e-14)
    energies = np.linspace(center * 0.995, center * 1.005, 11)
    with mp.workdps(40):
        reference_log = np.array([float(mp.log(entrance_P(rx, 20, e))) for e in energies])
        reference_ratio = np.array([float(entrance_P(rx, 20, e) / entrance_P(rx, 20, center)) for e in energies])
        rows = [ResonanceRow(float(e), 20, float(2 * gi * entrance_P(rx, 20, e)), 20,
                            float(2 * gi * entrance_P(rx, 20, e)), 18, 0.0, 0, 0.0)
                for e in energies]
        # Distinct alpha/proton channels; allowed l for J=20. Matched exit
        # widths isolate the inelastic branching bound, not a spectrum prediction.
        for row in rows:
            assert 0 < row.G1 < 2 * go * exit_P(rx, 18, row.E_r + rx.S_proj_mev - rx.S_exit_mev)
    assert np.all(reference_log < -745)
    current = ResonanceSum(rx, rows, energies[0], energies[-1], npts=80)
    previous = old_sum(rx, rows, energies[0], energies[-1], npts=80)
    logP = current._entrance_log(20)
    ratio = np.exp(logP(energies) - float(logP(center)))
    oldP = previous._entrance(20)
    old_ratio = oldP(energies) / float(oldP(center))
    peak = np.array([float(current.sigma_one(row, row.E_r)) for row in rows])
    with np.errstate(all="ignore"):
        old_peak = np.array([float(previous.sigma_one(row, row.E_r)) for row in rows])
    bound = 41 * current.pi_lambda2_b_mev / energies
    ratio_error = float(relative_error(ratio, reference_ratio).max())
    peak_error = float(relative_error(peak, bound).max())
    assert ratio_error < 5e-3 and peak_error < 1e-9
    assert np.all(np.isfinite(peak)) and np.all(~np.isfinite(old_peak))
    return dict(energies=energies, reference_log=reference_log,
                reference_ratio=reference_ratio, ratio=ratio, old_ratio=old_ratio,
                peak=peak, old_peak=old_peak, bound=bound, rows=rows,
                ratio_error=ratio_error, peak_error=peak_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="588d3d549caf0955e262af472d626ca0334d6aae")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/validation/issue23")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    old_sum = baseline_sum(args.baseline_ref)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.2})

    # p + 24Mg(0+) -> 25Al + gamma: l=1 admits J=3/2-, and an E1
    # transition to the 25Al(5/2+) ground state is allowed.
    rx = load_reaction_params(ROOT / "input/24Mg(p,g)25Al.txt")
    row = ResonanceRow(1.5, 1.5, 1.0, 1, 0.2, 1, 0.0, 0, 0.0)
    omega = (2 * row.J + 1) / ((2 * rx.J_proj + 1) * (2 * rx.J_targ + 1))
    width_eV = row.G0 + row.G1
    width_MeV = width_eV * 1e-6
    current_constant = constant_width_model(ResonanceSum)(rx, [row], 0.5, 3.0)
    previous_constant = constant_width_model(old_sum)(rx, [row], 0.5, 3.0)
    offsets_eV = np.linspace(-10 * width_eV, 10 * width_eV, 1601)
    energies = row.E_r + offsets_eV * 1e-6
    # Independent textbook formula; no production cross-section helpers.
    mu = rx.M_proj * rx.M_targ / (rx.M_proj + rx.M_targ)
    prefactor = np.pi * 197.3269804**2 / (2 * mu * 931.49410242) / 100
    analytical = omega * prefactor / energies * (row.G0 * 1e-6) * (row.G1 * 1e-6) / (
        (energies - row.E_r)**2 + width_MeV**2 / 4)
    xs = current_constant.sigma(energies)
    old_xs = previous_constant.sigma(energies)
    bw_error = float(relative_error(xs, analytical).max())
    old_bw_error = float(relative_error(old_xs, analytical).max())
    assert bw_error < 1e-10 and old_bw_error < 1e-10
    half_max = xs.max() / 2
    mid = len(xs) // 2
    left = np.interp(half_max, xs[:mid+1], offsets_eV[:mid+1])
    right = np.interp(half_max, xs[mid:][::-1], offsets_eV[mid:][::-1])
    measured_width = right - left
    assert abs(measured_width / width_eV - 1) < 1e-3
    # Narrow-limit area, correcting the explicitly finite integration interval.
    half_range = 1e4 * width_MeV
    area = float(current_constant.bin_integrals(
        [row.E_r-half_range, row.E_r+half_range], n_base=2000, n_peak=800)[0])
    strength_MeV = omega * row.G0 * row.G1 / width_eV * 1e-6
    analytical_area = 2 * np.pi * prefactor / row.E_r * strength_MeV
    analytical_area *= 2 / np.pi * np.arctan(2 * half_range / width_MeV)
    area_error = abs(area / analytical_area - 1)
    assert area_error < 3e-3

    print("Computing energy-dependent single-resonance rates...", flush=True)
    current = ResonanceSum(rx, [row], 0.5, 3.0, npts=600)
    previous = old_sum(rx, [row], 0.5, 3.0, npts=600)
    temperatures = np.geomspace(0.8, 5.0, 35)
    rates = current.rate(temperatures, E_lo=0.5, E_hi=3.0, n_base=4000, n_peak=400)
    old_rates = previous.rate(temperatures, E_lo=0.5, E_hi=3.0, n_base=4000, n_peak=400)
    narrow_rates = 1.5394e11 * (mu * temperatures)**-1.5 * strength_MeV * np.exp(-11.605 * row.E_r / temperatures)
    # Refine the integration once to distinguish quadrature error from the
    # narrow-resonance approximation, finite bounds, and rounded constants.
    refined_rates = current.rate(temperatures, E_lo=0.5, E_hi=3.0, n_base=8000, n_peak=800)
    rate_error = float(relative_error(rates, narrow_rates).max())
    rate_change = float(relative_error(rates, old_rates).max())
    convergence = float(relative_error(rates, refined_rates).max())
    assert np.all(np.isfinite(rates)) and np.all(rates > 0)
    assert rate_error < 5e-3 and rate_change < 1e-9 and convergence < 1e-3

    print("Computing direct 40-digit underflow reference...", flush=True)
    stress = underflow_check(old_sum)
    fig, (ax, residual) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    ax.plot(offsets_eV, analytical, color="black", label="Analytical Breit–Wigner")
    ax.plot(offsets_eV, old_xs, "--", color="#d17b0f", label="Before fix")
    ax.plot(offsets_eV[::40], xs[::40], "o", ms=4, mfc="none", color="#176b87", label="After fix")
    ax.set(ylabel="Cross section (barn)")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.94, "²⁴Mg(p,γ)²⁵Al\nEᵣ = 1.5 MeV, Γp = 1.0 eV, Γγ = 0.2 eV", transform=ax.transAxes, va="top", fontsize=9)
    residual.plot(offsets_eV, old_xs / analytical, "--", color="#d17b0f")
    residual.plot(offsets_eV, xs / analytical, color="#176b87")
    residual.axhline(1, color="gray", ls=":")
    residual.set(xlabel="E − Eᵣ (eV)", ylabel="Numerical / analytical", ylim=(0.999, 1.001))
    residual.text(0.03, 0.12, f"Max relative error: {bw_error:.1e}", transform=residual.transAxes, fontsize=9)
    # Supplemental check tied directly to #23, separated from the ordinary
    # Lorentzian benchmark and explicitly labeled as a constructed stress case.
    inset = ax.inset_axes([0.62, 0.31, 0.34, 0.32])
    kev = stress['energies'] * 1000
    inset.plot(kev, stress['reference_ratio'], color="black", label="40-digit reference")
    inset.plot(kev, stress['old_ratio'], "--", color="#d17b0f", label="Before")
    inset.plot(kev, stress['ratio'], "o", ms=3, mfc="none", color="#176b87", label="After")
    inset.set(xlabel="E (keV)", ylabel="Γα(E) / Γα(Eᵣ)", title="Underflow check: ℓ = 20")
    inset.tick_params(labelsize=7)
    inset.xaxis.label.set_size(8)
    inset.yaxis.label.set_size(8)
    inset.title.set_size(8)
    inset.legend(fontsize=6, loc="upper left")
    save_figure(fig, destination, "breit_wigner_benchmark")

    fig, (ax, residual) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    ax.loglog(temperatures, narrow_rates, color="black", label="Analytical narrow-resonance rate")
    ax.loglog(temperatures, old_rates, "--", color="#d17b0f", label="Before fix: numerical integration")
    ax.loglog(temperatures[::2], rates[::2], "o", ms=4, mfc="none", color="#176b87", label="After fix: numerical integration")
    ax.set(ylabel=r"$N_A\langle\sigma v\rangle$ (cm³ mol⁻¹ s⁻¹)")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.06, "²⁴Mg(p,γ)²⁵Al\nEnergy-dependent widths; numerical integration: 0.5–3.0 MeV", transform=ax.transAxes, fontsize=9, ha="right")
    residual.semilogx(temperatures, old_rates / narrow_rates, "--", color="#d17b0f")
    residual.semilogx(temperatures, rates / narrow_rates, color="#176b87")
    residual.axhline(1, color="gray", ls=":")
    residual.set(xlabel="Temperature (GK)", ylabel="Numerical / analytical", ylim=(0.995, 1.005))
    residual.xaxis.set_major_locator(FixedLocator([0.8, 1, 2, 3, 5]))
    residual.xaxis.set_major_formatter(ScalarFormatter())
    residual.xaxis.set_minor_formatter(NullFormatter())
    save_figure(fig, destination, "narrow_resonance_rate_benchmark")

    np.savetxt(destination / "breit_wigner_benchmark.csv", np.c_[offsets_eV, analytical, old_xs, xs], delimiter=",", header="E_minus_Er_eV,analytical_barn,before_barn,after_barn")
    np.savetxt(destination / "narrow_resonance_rate_benchmark.csv", np.c_[temperatures, narrow_rates, old_rates, rates, refined_rates], delimiter=",", header="T_GK,analytical_rate,before_rate,after_rate,refined_rate; rates_in_cm3_mol_s")
    np.savetxt(destination / "underflow_benchmark.csv", np.c_[stress['energies'], stress['reference_log'], stress['reference_ratio'], stress['old_ratio'], stress['ratio'], stress['old_peak'], stress['peak'], stress['bound']], delimiter=",", header="E_MeV,reference_logP,reference_ratio,before_ratio,after_ratio,before_peak_barn,after_peak_barn,peak_bound_barn")
    metrics = {
        "baseline_commit": subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        "production_source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("nucres/resonance.py", "nucres/resonance_sum.py")},
        "reaction": asdict(rx), "resonance": asdict(row), "strength_MeV": strength_MeV,
        "max_bw_relative_error": bw_error, "max_before_bw_relative_error": old_bw_error,
        "measured_fwhm_eV": measured_width, "expected_fwhm_eV": width_eV,
        "finite_window_area_b_MeV": area, "finite_window_analytical_area_b_MeV": analytical_area,
        "relative_area_error": area_error,
        "max_narrow_rate_relative_error": rate_error, "max_rate_relative_change": rate_change,
        "max_rate_quadrature_relative_change": convergence,
        "max_underflow_ratio_relative_error": stress['ratio_error'],
        "max_underflow_peak_relative_error": stress['peak_error'],
        "before_nonfinite_stress_peaks": int(np.count_nonzero(~np.isfinite(stress['old_peak']))),
        "stress_resonances": [asdict(r) for r in stress['rows']],
        "software": {"python": sys.version, "numpy": np.__version__, "mpmath": mp.__version__, "matplotlib": matplotlib.__version__},
        "rate_reference": "https://reaclib.jinaweb.org/help.php?intCurrentNum=0&topic=rate_fitting",
    }
    (destination / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = f"""Issue #23: standard physics benchmarks

Reproduce from the repository root:
  uv run python scripts/validate_transmission_underflow.py

breit_wigner_benchmark.png
  Independent single-level Breit-Wigner formula versus before/after code.
  Constant widths are imposed only in the width-law methods of subclasses;
  production sigma_one and bin_integrals are used unchanged. This is an
  exact controlled limit, not the full energy-dependent physical model.
  Additional checks: FWHM and integrated resonance area. The small inset
  compares a constructed alpha-channel underflow case with direct 40-digit
  mpmath Coulomb functions; full stress-case data are in the CSV file.

narrow_resonance_rate_benchmark.png
  Actual energy-dependent production widths, Maxwell-Boltzmann integration,
  and the standard isolated narrow-resonance formula:
  NA<sigma v> = 1.5394e11 (mu*T9)^(-3/2) omega_gamma exp(-11.605*Er/T9).
  mu in atomic mass units; Er and omega_gamma in MeV; rate in cm^3 mol^-1 s^-1.
  Reference: {metrics['rate_reference']}
  Numerical bounds: 0.5--3.0 MeV; T9=0.8--5.0. Width=1.2 eV is small
  compared with Er and kT in this interval. The ratio need not equal one
  exactly: the analytical rate is an approximation, the integration bounds
  are finite, and the conventional numerical constants are rounded.
  A second, refined grid checks quadrature convergence separately.

Results:
  Maximum BW relative error: {bw_error:.6e}
  FWHM: {measured_width:.9g} eV (expected {width_eV:.9g} eV)
  Relative finite-window resonance-area error: {area_error:.6e}
  Maximum narrow-rate relative error: {rate_error:.6e}
  Maximum rate change from previous code: {rate_change:.6e}
  Maximum rate change on grid refinement: {convergence:.6e}
  Maximum underflow width-ratio relative error: {stress['ratio_error']:.6e}
  Maximum underflow on-shell peak relative error: {stress['peak_error']:.6e}
  Previous nonfinite stress-case peaks: {metrics['before_nonfinite_stress_peaks']} / 11

Scope:
  The 24Mg(p,g)25Al resonance is illustrative, not an identified measured
  level. l=1 permits J=3/2- and the E1 gamma transition to the 5/2+ ground
  state. This validates known algebra and the narrow-resonance limit,
  not a nuclear-model prediction or agreement with experiment.
  The supplemental 22Mg(a,p) stress example uses allowed angular momenta
  and widths below Wigner limits, with exit widths deliberately matched
  to tiny entrance widths. Its microscopic strengths are not predictions.
  Its widths are too narrow for a float64 energy grid to resolve; only
  width ratios and on-shell peaks are claimed. JWKB numerical cases are
  covered by unit tests; the figures do not calibrate an optical potential.

CSV data, metrics and source hashes accompany both PNG figures.
The earlier ordinary_regime/underflow_regime figures are superseded.
"""
    (destination / "validation.txt").write_text(report)
    print(report)
    print(f"Figures and data: {destination}")


if __name__ == "__main__":
    main()
