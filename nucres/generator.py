from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.densities_retrieval import spin_grid

from .hfb_adapter import (
    build_density_grid,
    build_total_density_grid,
    load_hfb_record,
    resolve_density_paths,
)
from .physics import MASS_PROTON
from .rates import ReactionRateResult, na_sigma_v_from_sigma
from .resonance import Resonance, sigma_bw_constant


@dataclass(frozen=True)
class HFBSamplerConfig:
    """Configuration for synthesizing resonance spectra from HFB level densities."""

    Z: int
    data_root: Optional[str | Path] = None
    A: int = 24
    J: float = 1.0
    pi: int = 1
    s1: float = 0.5
    s2: float = 0.5
    m1: float = MASS_PROTON
    m2: float = MASS_PROTON
    Gamma_i_mean_eV: float = 1.0
    Gamma_o_mean_eV: float = 1.0
    delta_E_mev: float = 0.05
    E_min_mev: float = 0.1
    E_max_mev: float = 2.0
    n_density_points: int = 2001
    n_sigma_points: int = 4000
    U_offset_mev: float = 8.0
    seed: Optional[int] = None
    sample_J: bool = True
    auto_l1: bool = True


@dataclass
class GeneratedSpectrum:
    """Sampled spectrum, resonance list, and auxiliary metadata."""

    energy_MeV: np.ndarray
    sigma_barns: np.ndarray
    resonances: List[Resonance]
    rho_energy_MeV: np.ndarray
    rho_levels_per_MeV: np.ndarray
    metadata: Dict[str, Any]

    @property
    def energy_eV(self) -> np.ndarray:
        """Return the sampled energy grid in eV."""
        return self.energy_MeV * 1e6

    def compute_rate(
        self,
        temperatures,
        *,
        temperature_unit: str = "GK",
        result_unit: str = "cm^3/mol/s",
        energy_unit: str = "eV",
        sigma_unit: str = "barn",
        m1: Optional[float] = None,
        m2: Optional[float] = None,
    ) -> ReactionRateResult:
        """Integrate the stored cross section over the requested temperatures."""
        m1_eff = m1 if m1 is not None else self.metadata.get("m1", MASS_PROTON)
        m2_eff = m2 if m2 is not None else self.metadata.get("m2", MASS_PROTON)
        return na_sigma_v_from_sigma(
            self.energy_eV,
            self.sigma_barns,
            temperatures,
            m1=m1_eff,
            m2=m2_eff,
            energy_unit=energy_unit,
            sigma_unit=sigma_unit,
            temperature_unit=temperature_unit,
            result_unit=result_unit,
        )


def _interp_rhoJ_at_U(block, U_mev: float) -> np.ndarray:
    U_grid = block.U
    rhoJ = block.rho_J
    if U_mev <= U_grid[0]:
        return rhoJ[0, :]
    if U_mev >= U_grid[-1]:
        return rhoJ[-1, :]
    idx = int(np.searchsorted(U_grid, U_mev) - 1)
    x0, x1 = U_grid[idx], U_grid[idx + 1]
    y0, y1 = rhoJ[idx, :], rhoJ[idx + 1, :]
    t = (U_mev - x0) / (x1 - x0)
    return y0 * (1.0 - t) + y1 * t


def _sample_J_from_hfb(
    record, A: int, pi: int, U_mev: float, rng, fallback_J: Optional[float]
) -> float:
    block = record.positive if pi == +1 else record.negative
    weights = np.clip(_interp_rhoJ_at_U(block, U_mev), a_min=0.0, a_max=None)
    total = float(weights.sum())
    J_vals = spin_grid(A)
    if total <= 0.0:
        if fallback_J is not None:
            return float(fallback_J)
        return float(J_vals[0])
    probs = weights / total
    idx = int(rng.choice(len(J_vals), p=probs))
    return float(J_vals[idx])


def _allowed_L_values(J: float, s1: float, s2: float, pi_res: int) -> List[int]:
    min_I = abs(s1 - s2)
    max_I = s1 + s2
    start = int(round(2 * min_I))
    end = int(round(2 * max_I))
    L_vals: List[int] = []
    for two_I in range(start, end + 1, 2):
        I = two_I / 2.0
        L_min = int(np.ceil(abs(J - I)))
        L_max = int(np.floor(J + I))
        for L in range(L_min, L_max + 1):
            if (pi_res == +1 and (L % 2 == 0)) or (pi_res == -1 and (L % 2 == 1)):
                L_vals.append(L)
    return sorted(set(L_vals))


def _pick_L1(J: float, s1: float, s2: float, pi_res: int) -> Optional[int]:
    allowed = _allowed_L_values(J, s1, s2, pi_res)
    if not allowed:
        return None
    return int(allowed[0])


def synthesize_sigma_from_hfb(config: HFBSamplerConfig) -> GeneratedSpectrum:
    """Generate a synthetic cross section by sampling resonances from HFB densities."""
    rng = np.random.default_rng(config.seed)

    def _U_of_E_mev(E_eV: np.ndarray) -> np.ndarray:
        return np.asarray(E_eV, dtype=float) * 1e-6 + float(config.U_offset_mev)

    record = None
    if config.sample_J:
        tab_p, cor_p = resolve_density_paths(config.Z, data_root=config.data_root)
        record = load_hfb_record(
            tab_path=str(tab_p),
            cor_path=(str(cor_p) if cor_p is not None else None),
            warn_if_ignored=False,
        )
        E_mev, rho_per_mev = build_total_density_grid(
            Z=config.Z,
            data_root=config.data_root,
            pi=config.pi,
            E_min_mev=config.E_min_mev,
            E_max_mev=config.E_max_mev,
            n_points=config.n_density_points,
            U_of_E_mev=_U_of_E_mev,
        )
    else:
        E_mev, rho_per_mev = build_density_grid(
            Z=config.Z,
            data_root=config.data_root,
            A=config.A,
            J_phys=config.J,
            pi=config.pi,
            E_min_mev=config.E_min_mev,
            E_max_mev=config.E_max_mev,
            n_points=config.n_density_points,
            U_of_E_mev=_U_of_E_mev,
        )

    total_levels = float(np.trapezoid(rho_per_mev, E_mev))

    n_bins = max(
        1, int(np.ceil((config.E_max_mev - config.E_min_mev) / config.delta_E_mev))
    )
    edges_mev = np.linspace(config.E_min_mev, config.E_max_mev, n_bins + 1)
    lambdas = np.zeros(n_bins, dtype=float)

    for k in range(n_bins):
        a, b = edges_mev[k], edges_mev[k + 1]
        mask = (E_mev >= a) & (E_mev <= b)
        if np.count_nonzero(mask) >= 2:
            lambdas[k] = np.trapezoid(rho_per_mev[mask], E_mev[mask])
        else:
            ra = np.interp(a, E_mev, rho_per_mev)
            rb = np.interp(b, E_mev, rho_per_mev)
            lambdas[k] = 0.5 * (ra + rb) * (b - a)

    counts = rng.poisson(lambdas)
    n_drawn = int(counts.sum())
    nz_bins = int((lambdas > 0).sum())

    E_plot_MeV = np.linspace(config.E_min_mev, config.E_max_mev, config.n_sigma_points)
    E_plot_eV = E_plot_MeV * 1e6
    sigma_tot = np.zeros_like(E_plot_MeV)
    resonances: List[Resonance] = []

    if n_drawn:

        def pt_width(mean_eV: float) -> float:
            return float(mean_eV * rng.chisquare(df=1))

        def sample_E_in_bin(a_mev: float, b_mev: float, n: int) -> np.ndarray:
            if n <= 0:
                return np.empty(0, dtype=float)
            mask = (E_mev >= a_mev) & (E_mev <= b_mev)
            Ex = E_mev[mask]
            rhx = rho_per_mev[mask]
            if Ex.size == 0:
                Ex = np.array([a_mev, b_mev], dtype=float)
                rhx = np.array(
                    [
                        np.interp(a_mev, E_mev, rho_per_mev),
                        np.interp(b_mev, E_mev, rho_per_mev),
                    ],
                    dtype=float,
                )
            else:
                if Ex[0] > a_mev:
                    Ex = np.concatenate([[a_mev], Ex])
                    rhx = np.concatenate([[np.interp(a_mev, E_mev, rho_per_mev)], rhx])
                if Ex[-1] < b_mev:
                    Ex = np.concatenate([Ex, [b_mev]])
                    rhx = np.concatenate([rhx, [np.interp(b_mev, E_mev, rho_per_mev)]])
            dE = np.diff(Ex)
            accum = np.concatenate([[0.0], np.cumsum(0.5 * (rhx[:-1] + rhx[1:]) * dE)])
            total = accum[-1]
            if total <= 0:
                return rng.uniform(a_mev, b_mev, n)
            u = rng.random(n) * total
            return np.interp(u, accum, Ex)

        for k, Nk in enumerate(counts):
            if Nk == 0:
                continue
            a_mev, b_mev = edges_mev[k], edges_mev[k + 1]
            Er_mev = sample_E_in_bin(a_mev, b_mev, int(Nk))
            for Er in Er_mev:
                Er_eV = float(Er * 1e6)
                if config.sample_J and record is not None:
                    U_mev = float(_U_of_E_mev(Er_eV))
                    J_val = _sample_J_from_hfb(
                        record, config.A, config.pi, U_mev, rng, config.J
                    )
                else:
                    J_val = config.J
                L1_val = (
                    _pick_L1(J_val, config.s1, config.s2, config.pi)
                    if config.auto_l1
                    else None
                )
                resonance = Resonance(
                    E_r=Er_eV,
                    J=J_val,
                    s1=config.s1,
                    s2=config.s2,
                    m1=config.m1,
                    m2=config.m2,
                    Gamma_i=pt_width(config.Gamma_i_mean_eV),
                    Gamma_o=pt_width(config.Gamma_o_mean_eV),
                    L1=L1_val,
                )
                resonances.append(resonance)
                sigma_tot += sigma_bw_constant(E_plot_eV, resonance)

    metadata: Dict[str, Any] = {
        "expected_levels": total_levels,
        "n_bins": n_bins,
        "nonzero_lambda_bins": nz_bins,
        "n_drawn": n_drawn,
        "seed": config.seed,
        "m1": config.m1,
        "m2": config.m2,
        "Gamma_i_mean_eV": config.Gamma_i_mean_eV,
        "Gamma_o_mean_eV": config.Gamma_o_mean_eV,
    }

    return GeneratedSpectrum(
        energy_MeV=E_plot_MeV,
        sigma_barns=sigma_tot,
        resonances=resonances,
        rho_energy_MeV=E_mev,
        rho_levels_per_MeV=rho_per_mev,
        metadata=metadata,
    )
