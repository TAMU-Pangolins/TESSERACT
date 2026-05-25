#!/usr/bin/env python3
"""Plot resonance-sampling distribution checks for one generated RatesMC run."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_ratesmc_input import (  # noqa: E402
    _PARTICLE_EXIT_CHANNELS,
    _infer_compound_nucleus,
    _infer_u_offset_mev,
    _parse_reaction_channels,
    _parse_template_metadata,
    _reaction_exit_symbol,
)
from common.densities_retrieval import spin_grid  # noqa: E402
from extract_resonance_data import extract_data  # noqa: E402
from nucres.generator import _interp_rhoJ_at_U  # noqa: E402
from nucres.hfb_adapter import load_hfb_record, resolve_density_paths  # noqa: E402
from nucres.read_qvals import AMU_TO_KG, mass_from_token  # noqa: E402
from nucres.resonance import (  # noqa: E402
    Resonance,
    make_penetrability_interp,
    penetrability_P_l_mev,
    sigma_bw_energy_dep,
)
try:  # noqa: E402
    from plot_talys_xs_overlay import load_xy, parse_optimized_xs_from_out
except ModuleNotFoundError:  # noqa: E402
    from scripts.plot_talys_xs_overlay import load_xy, parse_optimized_xs_from_out


WEIGHTED_SIGMA_FLOOR_MB = 1e-20


def parse_q_value_mev(lines: list[str]) -> float:
    s_proj = None
    s_exit = None
    for line in lines:
        raw, _, comment = line.partition("!")
        values = raw.strip().split()
        if not values:
            continue
        lower = comment.lower()
        if "projectile separation energy" in lower:
            s_proj = float(values[0])
        elif "exit particle separation energy" in lower:
            s_exit = float(values[0])
        if s_proj is not None and s_exit is not None:
            return (s_proj - s_exit) * 1e-3
    return 0.0


def load_run_table(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = _parse_template_metadata(lines)
    table = extract_data(path)
    return lines, metadata, table


def collect_run_inputs(ratesmc_dir: Path, reaction: str, limit: int | None = None) -> list[Path]:
    paths = sorted(ratesmc_dir.glob(f"RUN_*/{reaction}.in"))
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def recovered_reduced_widths(path: Path, channel: str) -> tuple[np.ndarray, str]:
    lines, metadata, table = load_run_table(path)
    q_mev = parse_q_value_mev(lines)
    projectile, ejectile, residual = _parse_reaction_channels(metadata.reaction)
    ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3

    if channel == "entrance":
        z1, a1 = projectile
        z2, a2 = metadata.Z, int(round(float(metadata.targ_A_token)))
        widths = np.asarray(table["G1"], dtype=float)
        l_vals = np.asarray(table["L1"], dtype=int)
        energies = ecm_mev
        label = r"Entrance reduced width"
    else:
        z1, a1 = ejectile
        z2, a2 = residual
        widths = np.asarray(table["G2"], dtype=float)
        l_vals = np.asarray(table["L2"], dtype=int)
        exf_mev = np.asarray(table["Exf"], dtype=float) * 1e-3 if "Exf" in table else 0.0
        energies = np.maximum(ecm_mev + q_mev - exf_mev, 0.0)
        label = r"Exit reduced width"
        if _reaction_exit_symbol(metadata.reaction) not in _PARTICLE_EXIT_CHANNELS:
            return widths[np.isfinite(widths) & (widths > 0)], label

    if None in (z1, a1, z2, a2):
        return widths[np.isfinite(widths) & (widths > 0)], label

    reduced = np.full_like(widths, np.nan, dtype=float)
    for i, (width, energy, l_val) in enumerate(zip(widths, energies, l_vals)):
        if not np.isfinite(width) or width <= 0.0 or energy <= 0.0:
            continue
        p_l = penetrability_P_l_mev(int(l_val), int(z1), int(z2), int(a1), int(a2), float(energy))
        if p_l > 0.0:
            reduced[i] = width / (2.0 * p_l)
    return reduced[np.isfinite(reduced) & (reduced > 0.0)], label


def porter_thomas_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x) / np.sqrt(2.0 * np.pi * np.maximum(x, 1e-300))


def plot_width_panel(ax, widths: np.ndarray, label: str) -> None:
    if widths.size < 3:
        ax.text(0.5, 0.5, "not enough widths", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        return
    normalized = widths / np.nanmean(widths)
    finite = normalized[np.isfinite(normalized) & (normalized > 0.0)]
    bins = np.logspace(np.log10(max(np.nanmin(finite), 1e-4)), np.log10(np.nanpercentile(finite, 98)), 26)
    hist, edges = np.histogram(finite, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.plot(centers, hist, "o-", color="C0", markersize=3.4, linewidth=1.2, label="Sample")
    x = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), 300)
    ax.plot(x, porter_thomas_pdf(x), "--", color="black", linewidth=1.4, label="Porter-Thomas")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Gamma / \langle\Gamma\rangle$")
    ax.set_ylabel("Density")
    ax.set_title(label)


def porter_thomas_cdf(x: np.ndarray) -> np.ndarray:
    vals = np.sqrt(np.maximum(x, 0.0) / 2.0)
    return np.asarray([math.erf(float(v)) for v in vals], dtype=float)


def plot_width_cdf_panel(ax, widths: np.ndarray) -> None:
    if widths.size < 3:
        ax.text(0.5, 0.5, "not enough widths", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reduced-width CDF")
        return
    normalized = widths / np.nanmean(widths)
    finite = np.sort(normalized[np.isfinite(normalized) & (normalized > 0.0)])
    y = np.arange(1, finite.size + 1, dtype=float) / finite.size
    ax.plot(finite, y, "o-", color="C0", markersize=2.6, linewidth=1.0, label="Sample")
    x = np.logspace(np.log10(max(finite[0], 1e-4)), np.log10(max(finite[-1], 1e-3)), 400)
    ax.plot(x, porter_thomas_cdf(x), "--", color="black", linewidth=1.4, label="Porter-Thomas")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Gamma / \langle\Gamma\rangle$")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Reduced-width CDF")


def expected_spin_probabilities(
    path: Path, spins: np.ndarray, parity: int
) -> tuple[np.ndarray, np.ndarray] | None:
    lines, metadata, table = load_run_table(path)
    try:
        z_comp, a_comp = _infer_compound_nucleus(metadata)
        u_offset = _infer_u_offset_mev(metadata)
        tab_path, cor_path = resolve_density_paths(z_comp, data_root=None)
        record = load_hfb_record(
            tab_path=str(tab_path),
            cor_path=(str(cor_path) if cor_path is not None else None),
            warn_if_ignored=False,
        )
    except Exception:
        return None

    ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
    j_grid = spin_grid(a_comp)
    accum = np.zeros_like(j_grid, dtype=float)
    block = record.positive if parity == +1 else record.negative
    for energy in ecm_mev:
        weights = np.clip(_interp_rhoJ_at_U(block, energy + u_offset), a_min=0.0, a_max=None)
        total = float(np.sum(weights))
        if total > 0.0:
            accum += weights / total
    if np.sum(accum) <= 0.0:
        return None
    probs = accum / np.sum(accum)
    keep = np.isin(j_grid, spins) | (probs > 1e-4)
    return j_grid[keep], probs[keep]


def plot_spin_panel(ax, path: Path, parity: int) -> None:
    _lines, _metadata, table = load_run_table(path)
    spins = np.asarray(table["Jr"], dtype=float)
    observed_spins, counts = np.unique(spins[np.isfinite(spins)], return_counts=True)
    if observed_spins.size == 0:
        ax.text(0.5, 0.5, "no spins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Spin distribution")
        return
    obs_prob = counts / counts.sum()
    ax.plot(observed_spins, obs_prob, "o-", color="C0", markersize=4.0, linewidth=1.2, label="Sample")
    expected = expected_spin_probabilities(path, observed_spins, parity)
    if expected is not None:
        exp_spins, exp_probs = expected
        ax.plot(exp_spins, exp_probs, "--", color="black", linewidth=1.4, label="HFB expectation")
    ax.set_xlabel(r"$J_r$")
    ax.set_ylabel("Probability")
    ax.set_title("Resonance spin distribution")


def expected_energy_density(path: Path, parity: int) -> tuple[np.ndarray, np.ndarray] | None:
    lines, metadata, table = load_run_table(path)
    try:
        z_comp, _a_comp = _infer_compound_nucleus(metadata)
        u_offset = _infer_u_offset_mev(metadata)
        tab_path, cor_path = resolve_density_paths(z_comp, data_root=None)
        record = load_hfb_record(
            tab_path=str(tab_path),
            cor_path=(str(cor_path) if cor_path is not None else None),
            warn_if_ignored=False,
        )
        block = record.positive if parity == +1 else record.negative
        ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
        emin = float(np.nanmin(ecm_mev))
        emax = float(np.nanmax(ecm_mev))
        grid = np.linspace(emin, emax, 800)
        rho = np.asarray(
            [
                np.sum(np.clip(_interp_rhoJ_at_U(block, energy + u_offset), 0.0, None))
                for energy in grid
            ],
            dtype=float,
        )
    except Exception:
        return None
    area = np.trapezoid(rho, grid)
    if area <= 0:
        return None
    return grid, rho / area


def expected_j_density(path: Path, parity: int, j_value: float) -> tuple[np.ndarray, np.ndarray] | None:
    lines, metadata, table = load_run_table(path)
    try:
        z_comp, a_comp = _infer_compound_nucleus(metadata)
        u_offset = _infer_u_offset_mev(metadata)
        tab_path, cor_path = resolve_density_paths(z_comp, data_root=None)
        record = load_hfb_record(
            tab_path=str(tab_path),
            cor_path=(str(cor_path) if cor_path is not None else None),
            warn_if_ignored=False,
        )
        block = record.positive if parity == +1 else record.negative
        j_grid = spin_grid(a_comp)
        j_idx = int(np.argmin(np.abs(j_grid - j_value)))
        ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
        emin = float(np.nanmin(ecm_mev))
        emax = float(np.nanmax(ecm_mev))
        grid = np.linspace(emin, emax, 800)
        rho = np.asarray(
            [
                max(float(_interp_rhoJ_at_U(block, energy + u_offset)[j_idx]), 0.0)
                for energy in grid
            ],
            dtype=float,
        )
    except Exception:
        return None
    area = np.trapezoid(rho, grid)
    if area <= 0.0:
        return None
    return grid, rho / area


def plot_energy_panel(ax, path: Path, parity: int) -> None:
    _lines, _metadata, table = load_run_table(path)
    ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
    ecm_mev = ecm_mev[np.isfinite(ecm_mev)]
    ax.hist(ecm_mev, bins=24, density=True, histtype="step", color="C0", linewidth=1.5)
    counts, edges = np.histogram(ecm_mev, bins=24, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, "o", color="C0", markersize=3.4, label="Sample")
    expected = expected_energy_density(path, parity)
    if expected is not None:
        x, y = expected
        ax.plot(x, y, "--", color="black", linewidth=1.4, label="HFB density")
    ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax.set_ylabel("Density")
    ax.set_title("Resonance-energy distribution")


def plot_level_density_recovery_panel(ax, paths: list[Path], parity: int) -> None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        ax.text(0.5, 0.5, "no run inputs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Level-density recovery")
        return
    energies = []
    template_path = paths[0]
    for path in paths:
        try:
            energies.append(observed_energies_mev(path))
        except Exception:
            continue
    energies = np.concatenate([e for e in energies if e.size]) if energies else np.array([])
    if energies.size == 0:
        ax.text(0.5, 0.5, "no energies", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Level-density recovery")
        return
    expected = expected_energy_density(template_path, parity)
    bins = min(40, max(10, int(np.sqrt(energies.size))))
    ax.hist(energies, bins=bins, density=True, histtype="step", color="C0", linewidth=1.5, label="Sample")
    counts, edges = np.histogram(energies, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, "o", color="C0", markersize=2.8)
    if expected is not None:
        grid, density = expected
        ax.plot(grid, density, "--", color="black", linewidth=1.4, label="HFB density")
    ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax.set_ylabel("Density")
    ax.set_title(f"Level-density recovery ({len(paths)} runs)")


def plot_spin_occupancy_recovery_panel(ax, paths: list[Path], parity: int) -> None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        ax.text(0.5, 0.5, "no run inputs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"$J^\pi$ occupancy recovery")
        return
    spins = []
    template_path = paths[0]
    for path in paths:
        try:
            _lines, _metadata, table = load_run_table(path)
            vals = np.asarray(table["Jr"], dtype=float)
            spins.append(vals[np.isfinite(vals)])
        except Exception:
            continue
    spins = np.concatenate([s for s in spins if s.size]) if spins else np.array([])
    if spins.size == 0:
        ax.text(0.5, 0.5, "no spins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"$J^\pi$ occupancy recovery")
        return
    observed_spins, counts = np.unique(spins, return_counts=True)
    ax.plot(observed_spins, counts / counts.sum(), "o-", color="C0", markersize=3.4, linewidth=1.2, label="Sample")
    expected = expected_spin_probabilities(template_path, observed_spins, parity)
    if expected is not None:
        exp_spins, exp_probs = expected
        ax.plot(exp_spins, exp_probs, "--", color="black", linewidth=1.4, label="HFB expectation")
    ax.set_xlabel(r"$J_r$")
    ax.set_ylabel("Probability")
    ax.set_title(r"$J^\pi$ occupancy recovery")


def observed_energies_mev(path: Path) -> np.ndarray:
    _lines, _metadata, table = load_run_table(path)
    ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
    return np.sort(ecm_mev[np.isfinite(ecm_mev)])


def poisson_spacing_pdf(s: np.ndarray) -> np.ndarray:
    return np.exp(-np.maximum(s, 0.0))


def wigner_spacing_pdf(s: np.ndarray) -> np.ndarray:
    s = np.maximum(s, 0.0)
    return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s * s)


def expected_bin_counts(path: Path, parity: int, edges: np.ndarray) -> np.ndarray | None:
    expected = expected_energy_density(path, parity)
    if expected is None:
        return None
    grid, density = expected
    n_obs = observed_energies_mev(path).size
    counts = np.zeros(edges.size - 1, dtype=float)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (grid >= a) & (grid <= b)
        if np.count_nonzero(mask) >= 2:
            counts[i] = np.trapezoid(density[mask], grid[mask]) * n_obs
        else:
            da = float(np.interp(a, grid, density))
            db = float(np.interp(b, grid, density))
            counts[i] = 0.5 * (da + db) * (b - a) * n_obs
    return counts


def plot_level_count_panel(ax, path: Path, parity: int) -> None:
    ecm_mev = observed_energies_mev(path)
    if ecm_mev.size == 0:
        ax.text(0.5, 0.5, "no energies", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Level counts")
        return
    bins = min(24, max(6, int(np.sqrt(ecm_mev.size))))
    counts, edges = np.histogram(ecm_mev, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, "o-", color="C0", markersize=3.4, linewidth=1.2, label="Sample")
    expected = expected_bin_counts(path, parity, edges)
    if expected is not None:
        ax.plot(centers, expected, "--", color="black", linewidth=1.4, label="HFB Poisson mean")
    ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax.set_ylabel("Levels / bin")
    ax.set_title("Level counts")


def unfolded_spacings(path: Path, parity: int) -> np.ndarray | None:
    energies = observed_energies_mev(path)
    if energies.size < 3:
        return None
    expected = expected_energy_density(path, parity)
    if expected is None:
        spacings = np.diff(energies)
        mean = np.mean(spacings)
        return spacings / mean if mean > 0.0 else None
    grid, density = expected
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(grid))]
    )
    unfolded = np.interp(energies, grid, cumulative)
    spacings = np.diff(unfolded)
    mean = np.mean(spacings)
    return spacings / mean if mean > 0.0 else None


def j_resolved_unfolded_spacings(path: Path, parity: int, min_levels: int = 4) -> np.ndarray | None:
    _lines, _metadata, table = load_run_table(path)
    energies = np.asarray(table["Ecm"], dtype=float) * 1e-3
    spins = np.asarray(table["Jr"], dtype=float)
    pieces = []
    for spin in np.unique(spins[np.isfinite(spins)]):
        mask = np.isfinite(energies) & np.isfinite(spins) & (spins == spin)
        seq = np.sort(energies[mask])
        if seq.size < min_levels:
            continue
        expected = expected_j_density(path, parity, float(spin))
        if expected is None:
            spacings = np.diff(seq)
        else:
            grid, density = expected
            cumulative = np.concatenate(
                [[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(grid))]
            )
            unfolded = np.interp(seq, grid, cumulative)
            spacings = np.diff(unfolded)
        mean = np.mean(spacings)
        if mean > 0.0:
            pieces.append(spacings / mean)
    return np.concatenate(pieces) if pieces else None


def plot_spacing_panel(ax, path: Path, parity: int) -> None:
    spacings = unfolded_spacings(path, parity)
    if spacings is None or spacings.size < 3:
        ax.text(0.5, 0.5, "not enough spacings", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Unfolded level spacing")
        return
    max_s = max(4.0, float(np.nanpercentile(spacings, 98)))
    bins = np.linspace(0.0, max_s, 24)
    hist, edges = np.histogram(spacings, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist, "o-", color="C0", markersize=3.2, linewidth=1.2, label="Sample")
    x = np.linspace(0.0, edges[-1], 400)
    ax.plot(x, poisson_spacing_pdf(x), "--", color="black", linewidth=1.4, label="Poisson")
    ax.plot(x, wigner_spacing_pdf(x), "--", color="C3", linewidth=1.2, label="Wigner")
    ax.set_xlabel(r"Unfolded spacing $s/\langle s\rangle$")
    ax.set_ylabel("Density")
    ax.set_title("Unfolded level spacing")


def plot_j_resolved_spacing_panel(ax, path: Path, parity: int) -> None:
    spacings = j_resolved_unfolded_spacings(path, parity)
    if spacings is None or spacings.size < 3:
        ax.text(0.5, 0.5, r"not enough same-$J^\pi$ spacings", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"Same-$J^\pi$ level spacing")
        return
    max_s = max(4.0, float(np.nanpercentile(spacings, 98)))
    bins = np.linspace(0.0, max_s, 20)
    hist, edges = np.histogram(spacings, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist, "o-", color="C0", markersize=3.2, linewidth=1.2, label="Sample")
    x = np.linspace(0.0, edges[-1], 400)
    ax.plot(x, wigner_spacing_pdf(x), "--", color="black", linewidth=1.4, label="Wigner")
    ax.plot(x, poisson_spacing_pdf(x), "--", color="0.55", linewidth=1.1, label="Poisson")
    ax.set_xlabel(r"Unfolded spacing $s/\langle s\rangle$")
    ax.set_ylabel("Density")
    ax.set_title(r"Same-$J^\pi$ level spacing")


def plot_spacing_wigner_panel(ax, path: Path, parity: int) -> None:
    spacings = unfolded_spacings(path, parity)
    if spacings is None or spacings.size < 3:
        ax.text(0.5, 0.5, "not enough spacings", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Unfolded level spacing")
        return
    max_s = max(4.0, float(np.nanpercentile(spacings, 98)))
    bins = np.linspace(0.0, max_s, 24)
    hist, edges = np.histogram(spacings, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist, "o-", color="C0", markersize=3.4, linewidth=1.2, label="Sample")
    x = np.linspace(0.0, edges[-1], 400)
    ax.plot(x, wigner_spacing_pdf(x), "--", color="black", linewidth=1.5, label="Wigner")
    ax.plot(x, poisson_spacing_pdf(x), "--", color="0.55", linewidth=1.1, label="Poisson")
    ax.set_xlabel(r"Unfolded spacing $s/\langle s\rangle$")
    ax.set_ylabel("Density")
    ax.set_title("Unfolded level spacing")


def plot_l_panel(ax, path: Path) -> None:
    _lines, _metadata, table = load_run_table(path)
    plotted = False
    for column, color, label in (("L1", "C0", r"$L_1$"), ("L2", "C1", r"$L_2$")):
        if column not in table:
            continue
        vals = np.asarray(table[column], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        levels, counts = np.unique(vals.astype(int), return_counts=True)
        probs = counts / counts.sum()
        ax.plot(levels, probs, "o-", color=color, markersize=3.4, linewidth=1.2, label=label)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no L values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel("Probability")
    ax.set_title("Angular-momentum counts")


def plot_width_energy_panel(ax, path: Path) -> None:
    _lines, _metadata, table = load_run_table(path)
    ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
    for column, color, label in (("G1", "C0", r"$G_1$"), ("G2", "C1", r"$G_2$")):
        if column not in table:
            continue
        widths = np.asarray(table[column], dtype=float)
        mask = np.isfinite(ecm_mev) & np.isfinite(widths) & (widths > 0.0)
        if np.any(mask):
            ax.plot(ecm_mev[mask], widths[mask], "o", color=color, markersize=2.7, alpha=0.82, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax.set_ylabel(r"Partial width [eV]")
    ax.set_title("Widths vs energy")


def _mass_or_amu(token: str | None, z_hint: int | None, fallback_a: int | None) -> float:
    mass = mass_from_token(token, z_hint)
    if mass is not None:
        return float(mass)
    if fallback_a is not None:
        return float(fallback_a) * AMU_TO_KG
    return AMU_TO_KG


def reconstructed_sigma(path: Path, npts: int = 2400) -> tuple[np.ndarray, np.ndarray] | None:
    _lines, metadata, table = load_run_table(path)
    ecm_mev = np.asarray(table["Ecm"], dtype=float) * 1e-3
    finite = ecm_mev[np.isfinite(ecm_mev)]
    if finite.size == 0:
        return None
    projectile, _ejectile, _residual = _parse_reaction_channels(metadata.reaction)
    z1 = metadata.proj_Z or projectile[0]
    z2 = metadata.Z
    a1 = projectile[1]
    a2 = metadata.A
    if z1 is None or z2 is None or a1 is None or a2 is None:
        return None
    m1 = _mass_or_amu(metadata.proj_A_token, metadata.proj_Z, projectile[1])
    m2 = _mass_or_amu(metadata.targ_A_token, metadata.Z, metadata.A)
    s1 = float(metadata.s1 if metadata.s1 is not None else 0.0)
    s2 = float(metadata.s2 if metadata.s2 is not None else 0.0)
    emin = max(1e-6, float(np.nanmin(finite)) * 0.8)
    emax = float(np.nanmax(finite)) * 1.05
    e_grid_mev = np.linspace(emin, emax, npts)
    e_grid_ev = e_grid_mev * 1e6
    sigma = np.zeros_like(e_grid_mev)
    spins = np.asarray(table["Jr"], dtype=float)
    g1_vals = np.asarray(table["G1"], dtype=float)
    g2_vals = np.asarray(table["G2"], dtype=float)
    l1_vals = np.asarray(table["L1"], dtype=float)
    p_interp_by_l = {}
    for er_mev, spin, g1, g2, l1 in zip(ecm_mev, spins, g1_vals, g2_vals, l1_vals):
        if not np.isfinite(er_mev + spin + g1 + g2 + l1) or g1 <= 0.0 or g2 <= 0.0:
            continue
        l1_int = int(l1)
        if l1_int not in p_interp_by_l:
            p_interp_by_l[l1_int] = make_penetrability_interp(
                l1_int,
                z1,
                z2,
                a1,
                a2,
                r0=1.25,
                Emin_mev=emin,
                Emax_mev=emax,
            )
        resonance = Resonance(
            E_r=float(er_mev) * 1e6,
            J=float(spin),
            s1=s1,
            s2=s2,
            m1=m1,
            m2=m2,
            Gamma_i=float(g1),
            Gamma_o=float(g2),
        )
        sigma += sigma_bw_energy_dep(
            e_grid_ev,
            resonance,
            z1,
            z2,
            a1,
            a2,
            l1_int,
            Gamma_i_Er_eV=float(g1),
            r0=1.25,
            P_interp=p_interp_by_l[l1_int],
        )
    return e_grid_mev, sigma


def plot_reconstructed_sigma_panel(ax, path: Path) -> None:
    result = reconstructed_sigma(path)
    if result is None:
        ax.text(0.5, 0.5, "could not reconstruct", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reconstructed cross section")
        return
    e_grid_mev, sigma = result
    mask = np.isfinite(sigma) & (sigma > 0.0)
    if np.any(mask):
        ax.plot(e_grid_mev[mask], sigma[mask], "-", color="C0", linewidth=1.0)
        ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax.set_ylabel(r"$\sigma$ [b]")
    ax.set_title("Reconstructed Breit-Wigner sum")


def run_idx_from_input_path(path: Path) -> int | None:
    name = path.parent.name
    if not name.startswith("RUN_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def reaction_file_stem(reaction: str) -> str:
    return reaction.replace("(a,p)", "_ap_").replace("(", "_").replace(")", "").replace(",", "_")


def default_integrated_xs_path(path: Path, reaction: str) -> Path | None:
    run_idx = run_idx_from_input_path(path)
    if run_idx is None:
        return None
    return (
        Path("outputs/analysis/talys_xs_overlay")
        / f"raw_run{run_idx}"
        / f"{reaction_file_stem(reaction)}_integrated_xs_dE_0.2_run{run_idx}.csv"
    )


def plot_weighted_sigma_curve(
    ax,
    energy_mev: np.ndarray,
    sigma_mb: np.ndarray,
    *,
    t9: float,
    label: str,
    color: str,
    linewidth: float = 1.4,
    linestyle: str = "-",
    marker: str | None = None,
) -> None:
    weighted = np.asarray(sigma_mb, dtype=float) * np.exp(-11.60451812 * np.asarray(energy_mev, dtype=float) / t9)
    mask = np.isfinite(energy_mev) & np.isfinite(weighted) & (weighted >= WEIGHTED_SIGMA_FLOOR_MB)
    if np.any(mask):
        ax.semilogy(
            np.asarray(energy_mev, dtype=float)[mask],
            weighted[mask],
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=3.0 if marker else None,
            label=label,
        )


def plot_gamow_window_integrand_panel(
    ax,
    path: Path,
    t9: float,
    integrated_xs: Path | None = None,
    talys_out: Path | None = None,
) -> None:
    result = reconstructed_sigma(path)
    if result is None:
        ax.text(0.5, 0.5, "could not reconstruct", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gamow window integrand")
        return
    e_grid_mev, sigma = result
    plot_weighted_sigma_curve(
        ax,
        e_grid_mev,
        1.0e3 * np.clip(sigma, 0.0, None),
        t9=t9,
        label="Generated resonances",
        color="C0",
        linewidth=1.3,
    )

    if integrated_xs is not None and integrated_xs.exists():
        int_energy, int_sigma = load_xy(integrated_xs)
        plot_weighted_sigma_curve(
            ax,
            int_energy,
            int_sigma,
            t9=t9,
            label=r"Generated integrated XS, $\Delta E=0.2$ MeV",
            color="C2",
            linewidth=1.5,
            linestyle="--",
            marker="o",
        )

    talys_out = talys_out or Path("outputs/raw/talys_opt/talys_optimization.out")
    if talys_out.exists():
        run_idx = run_idx_from_input_path(path)
        try:
            opt_energy, opt_sigma, metadata = parse_optimized_xs_from_out(talys_out, run_idx=run_idx)
        except ValueError:
            opt_energy, opt_sigma, metadata = parse_optimized_xs_from_out(talys_out)
        label = "TALYS optimized"
        if metadata.get("run_idx") is not None:
            label += f" (run {metadata['run_idx']})"
        plot_weighted_sigma_curve(
            ax,
            opt_energy,
            opt_sigma,
            t9=t9,
            label=label,
            color="C1",
            linewidth=2.0,
        )

    ax.axhline(WEIGHTED_SIGMA_FLOOR_MB, color="0.5", linewidth=0.8, linestyle=":")
    ax.set_ylim(bottom=WEIGHTED_SIGMA_FLOOR_MB)
    ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
    ax.set_ylabel(r"$\sigma(E)\exp(-11.6045 E/T_9)$ [mb]")
    ax.set_title(rf"Gamow window integrand ($T_9={t9:g}$)")


def style_axes(ax) -> None:
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_single_panel(output_path: Path, plot_func, *args) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    plot_func(ax, *args)
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_requested_individual_plots(
    input_path: Path,
    output_dir: Path,
    entrance_widths: np.ndarray,
    entrance_label: str,
    exit_widths: np.ndarray,
    exit_label: str,
    parity: int,
    integrand_t9: float,
    integrated_xs: Path | None,
    talys_out: Path | None,
) -> None:
    save_single_panel(output_dir / "level_widths_entrance_porter_thomas.png", plot_width_panel, entrance_widths, entrance_label)
    save_single_panel(output_dir / "level_widths_exit_porter_thomas.png", plot_width_panel, exit_widths, exit_label)
    save_single_panel(
        output_dir / "level_widths_combined_cdf_porter_thomas.png",
        plot_width_cdf_panel,
        np.concatenate([entrance_widths, exit_widths]),
    )
    save_single_panel(output_dir / "level_spacing_wigner.png", plot_spacing_wigner_panel, input_path, parity)
    save_single_panel(output_dir / "level_spacing_same_Jpi.png", plot_j_resolved_spacing_panel, input_path, parity)
    save_single_panel(
        output_dir / "gamow_window_integrand.png",
        plot_gamow_window_integrand_panel,
        input_path,
        integrand_t9,
        integrated_xs,
        talys_out,
    )


def default_input_for_run(run_idx: int, reaction: str, ratesmc_dir: Path) -> Path:
    return ratesmc_dir / f"RUN_{run_idx}" / f"{reaction}.in"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distribution checks for one generated resonance ensemble."
    )
    parser.add_argument("--input", type=Path, default=None, help="Generated RatesMC .in file.")
    parser.add_argument("--run-idx", type=int, default=956)
    parser.add_argument("--reaction", default="22Mg(a,p)25Al")
    parser.add_argument("--ratesmc-dir", type=Path, default=Path("outputs/reference_ratesmc"))
    parser.add_argument(
        "--parity",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Parity used for HFB spin/energy expectation curves.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/resonance_distribution_checks/resonance_distribution_checks.png"),
    )
    parser.add_argument(
        "--integrand-t9",
        type=float,
        default=2.0,
        help="T9 value used for the Gamow-window integrand diagnostic.",
    )
    parser.add_argument(
        "--integrated-xs",
        type=Path,
        default=None,
        help="Optional generated integrated cross-section table to overlay.",
    )
    parser.add_argument(
        "--talys-out",
        type=Path,
        default=Path("outputs/raw/talys_opt/talys_optimization.out"),
        help="TALYS optimization output containing optimized cross sections.",
    )
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=None,
        help="Optional directory for standalone width and level-spacing diagnostic images.",
    )
    parser.add_argument(
        "--recovery-limit",
        type=int,
        default=200,
        help="Maximum number of RUN_*/reaction inputs used for multi-run recovery panels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input or default_input_for_run(args.run_idx, args.reaction, args.ratesmc_dir)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    entrance_widths, entrance_label = recovered_reduced_widths(input_path, "entrance")
    exit_widths, exit_label = recovered_reduced_widths(input_path, "exit")
    all_widths = np.concatenate([entrance_widths, exit_widths])
    recovery_inputs = collect_run_inputs(args.ratesmc_dir, args.reaction, args.recovery_limit)
    integrated_xs = args.integrated_xs or default_integrated_xs_path(input_path, args.reaction)

    if args.individual_dir is not None:
        save_requested_individual_plots(
            input_path,
            args.individual_dir,
            entrance_widths,
            entrance_label,
            exit_widths,
            exit_label,
            args.parity,
            args.integrand_t9,
            integrated_xs,
            args.talys_out,
        )

    fig, axes = plt.subplots(5, 3, figsize=(10.5, 12.8))
    plot_width_panel(axes[0, 0], entrance_widths, entrance_label)
    plot_width_panel(axes[0, 1], exit_widths, exit_label)
    plot_width_cdf_panel(axes[0, 2], all_widths)
    plot_spin_panel(axes[1, 0], input_path, args.parity)
    plot_energy_panel(axes[1, 1], input_path, args.parity)
    plot_level_count_panel(axes[1, 2], input_path, args.parity)
    plot_spacing_panel(axes[2, 0], input_path, args.parity)
    plot_l_panel(axes[2, 1], input_path)
    plot_width_energy_panel(axes[2, 2], input_path)
    plot_reconstructed_sigma_panel(axes[3, 0], input_path)
    plot_gamow_window_integrand_panel(axes[3, 1], input_path, args.integrand_t9, integrated_xs, args.talys_out)
    plot_level_density_recovery_panel(axes[3, 2], recovery_inputs, args.parity)
    plot_spin_occupancy_recovery_panel(axes[4, 0], recovery_inputs, args.parity)
    plot_j_resolved_spacing_panel(axes[4, 1], input_path, args.parity)
    axes[4, 2].axis("off")

    for ax in axes.ravel():
        if ax.axison:
            style_axes(ax)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(input_path.parent.name.replace("_", " "), y=1.04, fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote distribution checks: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
