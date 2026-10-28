"""
Sum of Breit-Wigner resonances with RatesMC's energy dependence, and exact
bin integrals of it.

The cross section for each resonance is the one RatesMC integrates
numerically (src/Resonance.cpp in rlongland/RatesMC):

    sigma(E) = pi lambda-bar^2 omega Gamma_0(E) Gamma_1(E)
               / [(E - E_r)^2 + (Gamma_0(E) + Gamma_1(E) + Gamma_2(E))^2 / 4]

with omega = (2J+1)/((2j_p+1)(2j_t+1)) and

* entrance:  Gamma_0(E) = Gamma_0(E_r) P_l0(E) / P_l0(E_r)
* particle exit / spectator:
             Gamma_i(E) = Gamma_i(E_r) P_li(E_x) / P_li(E_x,r),
             E_x = E + S_proj - S_exit - E_xf (the exit-channel energy)
* gamma channel:
             Gamma_i(E) = Gamma_i(E_r) [(S_proj + E - E_xf)/(S_proj + E_r - E_xf)]^(2L+1)

Masses (in u), spins and R0 are taken from the RatesMC input exactly as
RatesMC reads them, so the "true" cross section and the RatesMC "true" rate
describe the same resonances.

The resonances are typically far narrower than any practical uniform energy
grid, so bin averages are integrated per resonance on a grid that is dense
around E_r (resolving the peak exactly) and logarithmic elsewhere (the
wings), instead of by trapezoids on one uniform grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .resonance import (
    make_jwkb_log_transmission_interp,
    make_log_penetrability_interp,
)

HBARC_MEV_FM = 197.3269804
AMU_MEV = 931.49410242


@dataclass(frozen=True)
class ReactionParams:
    """Reaction header of a RatesMC input (energies in MeV, masses in u)."""

    Z_proj: int
    Z_targ: int
    Z_exit: int
    M_proj: float
    M_targ: float
    M_exit: float
    J_proj: float
    J_targ: float
    S_proj_mev: float
    S_exit_mev: float
    R0_fm: float
    gamma_channel: int  # RatesMC "Gamma-ray channel number": 2 (exit) or 3 (spectator)

    @property
    def mu_u(self) -> float:
        return self.M_proj * self.M_targ / (self.M_proj + self.M_targ)


@dataclass(frozen=True)
class ResonanceRow:
    """One Resonant Contribution row (energies in MeV, widths in eV)."""

    E_r: float
    J: float
    G0: float
    L0: int
    G1: float
    L1: int
    G2: float
    L2: int
    Exf: float


class ResonanceSum:
    """
    Cross section of a set of resonances with RatesMC's energy dependence.

    Parameters
    ----------
    reaction : ReactionParams
    rows : sequence of ResonanceRow
        Only rows with E_r > 0 and nonzero entrance and exit widths are used
        (strength-only rows, wg != 0 with zero widths, carry no energy
        dependence and are not represented).
    E_min_mev, E_max_mev : float
        Energy range over which penetrability tables are built.
    penetrability_model : {"coulomb", "jwkb_real_omp"}
        Entrance-channel energy dependence ("jwkb_real_omp" replaces the
        Coulomb P_l by the real-OMP JWKB transmission; exit channels always
        use Coulomb P_l, as RatesMC does).
    penetrability_cache : dict, optional
        Shared store for the penetrability tables. Only share it between
        objects with the same reaction header, energy range, model and
        table sizes.

    Notes
    -----
    Transmission ratios, widths, and the Breit-Wigner denominator are
    evaluated in log space. No transmission floor is imposed, and each
    width retains its specified value at the resonance energy even when
    the absolute transmission underflows in float64. A nonzero exit or
    spectator width referenced at a closed channel energy raises ValueError.
    """

    def __init__(
        self,
        reaction: ReactionParams,
        rows: Sequence[ResonanceRow],
        E_min_mev: float,
        E_max_mev: float,
        penetrability_model: str = "coulomb",
        omp_model: str = "mcfadden_satchler",
        npts: int = 800,
        jwkb_npts: int = 300,
        jwkb_radial_npts: int = 2400,
        penetrability_cache: Optional[Dict[tuple, Callable]] = None,
    ):
        self.reaction = reaction
        self.rows = [
            r for r in rows if r.E_r > 0.0 and r.G0 > 0.0 and r.G1 > 0.0
        ]
        self.E_min = float(E_min_mev)
        self.E_max = float(E_max_mev)
        self.model = penetrability_model
        self.omp_model = omp_model
        self.npts = int(npts)
        self.jwkb_npts = int(jwkb_npts)
        self.jwkb_radial_npts = int(jwkb_radial_npts)
        # Penetrability tables depend only on the reaction header, the energy
        # range, the model and the table sizes. Several ResonanceSum objects
        # built with the same values (e.g. many runs of one reaction) may pass
        # the same dict to avoid rebuilding the tables for every run.
        self._logP: Dict[tuple, Callable] = (
            penetrability_cache if penetrability_cache is not None else {}
        )

        rx = reaction
        self.omega_den = (2 * rx.J_proj + 1) * (2 * rx.J_targ + 1)
        # pi * lambda-bar^2 = pi (hbar c)^2 / (2 mu c^2 E), fm^2 -> b (1 b = 100 fm^2)
        self.pi_lambda2_b_mev = np.pi * HBARC_MEV_FM**2 / (2 * rx.mu_u * AMU_MEV) / 100.0
        self.M_res = rx.M_proj + rx.M_targ - rx.M_exit
        self.Z_res = rx.Z_proj + rx.Z_targ - rx.Z_exit

    # -- penetrabilities ----------------------------------------------------
    def _entrance_log(self, l: int) -> Callable:
        key = ("in", l)
        if key not in self._logP:
            rx = self.reaction
            lo, hi = max(self.E_min * 0.5, 1e-6), self.E_max * 1.5
            if self.model == "jwkb_real_omp":
                self._logP[key] = make_jwkb_log_transmission_interp(
                    l, rx.Z_proj, rx.Z_targ, rx.M_proj, rx.M_targ,
                    omp_model=self.omp_model, Emin_mev=lo, Emax_mev=hi,
                    npts=self.jwkb_npts, radial_npts=self.jwkb_radial_npts,
                )
            else:
                self._logP[key] = make_log_penetrability_interp(
                    l, rx.Z_proj, rx.Z_targ, rx.M_proj, rx.M_targ,
                    r0=rx.R0_fm, Emin_mev=lo, Emax_mev=hi, npts=self.npts,
                )
        return self._logP[key]

    def _exit_log(self, l: int, offset: float) -> Callable:
        """Log P_l as a function of the exit-channel energy."""
        key = ("out", l, round(offset, 9))
        if key not in self._logP:
            rx = self.reaction
            lo = max(self.E_min * 0.5 + offset, 1e-6)
            hi = self.E_max * 1.5 + offset
            self._logP[key] = make_log_penetrability_interp(
                l, rx.Z_exit, self.Z_res, rx.M_exit, self.M_res,
                r0=rx.R0_fm, Emin_mev=lo, Emax_mev=max(hi, lo * 1.01), npts=self.npts,
            )
        return self._logP[key]

    def _log_scaled(self, channel: int, G: float, L: int, Exf: float, E, E_r) -> np.ndarray:
        """Log width of RatesMC channel 1 (exit) or 2 (spectator)."""
        rx = self.reaction
        E = np.asarray(E, dtype=float)
        if G <= 0.0:
            return np.full_like(E, -np.inf)
        exf = Exf if channel == 1 else 0.0
        if channel == rx.gamma_channel - 1:
            num = rx.S_proj_mev + E - exf
            den = rx.S_proj_mev + E_r - exf
            if den <= 0.0:
                raise ValueError("Nonzero gamma width referenced at a closed channel energy.")
            with np.errstate(divide="ignore"):
                return np.log(G) + (2 * L + 1) * (np.log(np.maximum(num, 0.0)) - np.log(den))
        offset = rx.S_proj_mev - rx.S_exit_mev - exf
        Ex = E + offset
        if E_r + offset <= 0.0:
            raise ValueError("Nonzero particle width referenced at a closed channel energy.")
        logP = self._exit_log(L, offset)
        return np.log(G) + logP(Ex) - float(logP(E_r + offset))

    # -- cross section ------------------------------------------------------
    def sigma_one(self, row: ResonanceRow, E) -> np.ndarray:
        """Cross section of one resonance, in barns, at CM energies E (MeV)."""
        E = np.asarray(E, dtype=float)
        out = np.zeros_like(E)
        positive = E > 0.0
        if not np.any(positive):
            return out
        energies = E[positive]
        logP0 = self._entrance_log(row.L0)
        # Widths are in MeV throughout the log-domain Breit-Wigner formula.
        logG0 = np.log(row.G0) + logP0(energies) - float(logP0(row.E_r)) + np.log(1e-6)
        logG1 = self._log_scaled(1, row.G1, row.L1, row.Exf, energies, row.E_r) + np.log(1e-6)
        logG2 = self._log_scaled(2, row.G2, row.L2, row.Exf, energies, row.E_r) + np.log(1e-6)
        omega = (2 * row.J + 1) / self.omega_den
        with np.errstate(divide="ignore", under="ignore"):
            logGt = np.logaddexp(np.logaddexp(logG0, logG1), logG2)
            log_detuning = 2 * np.log(np.abs(energies - row.E_r))
            log_den = np.logaddexp(log_detuning, 2 * logGt - np.log(4.0))
            # Exponentiate only the final cross section: individual width
            # ratios may exceed float64 even when their BW ratio is finite.
            out[positive] = np.exp(
                np.log(omega * self.pi_lambda2_b_mev) - np.log(energies)
                + logG0 + logG1 - log_den
            )
        return out

    def sigma(self, E) -> np.ndarray:
        """Total cross section in barns at CM energies E (MeV)."""
        E = np.asarray(E, dtype=float)
        total = np.zeros_like(E)
        for row in self.rows:
            total += self.sigma_one(row, E)
        return total

    # -- integrals ----------------------------------------------------------
    def _grid(self, row: ResonanceRow, edges: np.ndarray, n_base: int, n_peak: int) -> np.ndarray:
        """Log grid over the range, dense geometric grid about E_r, and the bin edges."""
        lo, hi = edges[0], edges[-1]
        base = np.geomspace(lo, hi, n_base)
        gamma = max((row.G0 + row.G1 + row.G2) * 1e-6, 1e-15)
        steps = gamma * np.geomspace(1e-3, max(hi - lo, gamma) / gamma, n_peak)
        peak = row.E_r + np.concatenate([-steps[::-1], [0.0], steps])
        grid = np.concatenate([base, peak, edges])
        grid = grid[(grid >= lo) & (grid <= hi)]
        return np.unique(grid)

    def bin_integrals_per_resonance(
        self, edges: Sequence[float], n_base: int = 4000, n_peak: int = 400
    ) -> np.ndarray:
        """
        Integral of each resonance's sigma over each bin, in b MeV.

        Returns an array of shape (len(self.rows), len(edges) - 1): entry
        [i, k] is the integral of resonance self.rows[i] over bin
        [edges[k], edges[k+1]]. Each resonance is integrated on its own grid
        (see _grid) with the trapezoid rule; the cumulative integral is
        evaluated at the bin edges, so a resonance's tails are assigned to
        the bins they fall in.
        """
        edges = np.asarray(edges, dtype=float)
        out = np.zeros((len(self.rows), len(edges) - 1))
        for i, row in enumerate(self.rows):
            grid = self._grid(row, edges, n_base, n_peak)
            s = self.sigma_one(row, grid)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(grid))])
            out[i] = np.diff(np.interp(edges, grid, cum))
        return out

    def bin_integrals(
        self, edges: Sequence[float], n_base: int = 4000, n_peak: int = 400
    ) -> np.ndarray:
        """
        Integral of sigma over each bin [edges[i], edges[i+1]], in b MeV.

        The sum over resonances of bin_integrals_per_resonance().
        """
        edges = np.asarray(edges, dtype=float)
        if not self.rows:
            return np.zeros(len(edges) - 1)
        return self.bin_integrals_per_resonance(edges, n_base, n_peak).sum(axis=0)


    def rate(self, T9, E_lo: float = 1e-3, E_hi: float = 10.0,
             n_base: int = 4000, n_peak: int = 400) -> np.ndarray:
        """
        N_A <sigma v> in cm^3 mol^-1 s^-1 implied by this cross section,

            3.7318e10 mu^-1/2 T9^-3/2 Int sigma(E)[b] E[MeV] exp(-11.605 E/T9) dE,

        over [E_lo, E_hi] MeV (RatesMC's numerical-integration range defaults
        to 1 keV - 10 MeV).
        """
        T9 = np.atleast_1d(np.asarray(T9, dtype=float))
        edges = np.array([E_lo, E_hi])
        out = np.zeros_like(T9)
        for row in self.rows:
            grid = self._grid(row, edges, n_base, n_peak)
            s = self.sigma_one(row, grid) * grid
            for k, t in enumerate(T9):
                f = s * np.exp(-11.605 * grid / t)
                out[k] += np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))
        return 3.7318e10 * self.reaction.mu_u ** -0.5 * T9 ** -1.5 * out


def rows_from_dataframe(df) -> list:
    """ResonanceRow list from extract_resonance_data.extract_data output."""
    def col(name, default=0.0):
        return df[name].to_numpy(dtype=float) if name in df else np.full(len(df), default)

    Jcol = "Jr" if "Jr" in df else "J"
    E = col("Ecm") * 1e-3
    return [
        ResonanceRow(E_r=E[i], J=float(col(Jcol)[i]),
                     G0=float(col("G1")[i]), L0=int(col("L1")[i]),
                     G1=float(col("G2")[i]), L1=int(col("L2")[i]),
                     G2=float(col("G3")[i]), L2=int(col("L3")[i]),
                     Exf=float(col("Exf")[i]) * 1e-3)
        for i in range(len(df))
    ]
