"""
Exit widths summed over final states: discrete levels plus the HFB continuum.

For a resonance (E_r, J^pi) decaying by a particle or a gamma ray, the exit
width given to RatesMC is the sum over every reachable final state f of a
Porter-Thomas (chi^2, nu = 1) draw about that state's mean partial width:

* particle exits:  <Gamma_f> = 2 <gamma^2> P_l(E_f),
  E_f = E_r + S_proj - S_exit - E_x,f, l = lowest allowed by J^pi, the
  ejectile's and the state's J^pi; one <gamma^2> (reduced width) for all f.
* gamma exits (Hauser-Feshbach strength-function form):
  <Gamma_f> = f_XL(E_gamma) E_gamma^(2L+1) D_Jpi(E_x,i),
  E_gamma = E_r + S_proj - E_x,f, L = lowest multipole (E or M by the
  parity change, L <= 2), D = 1 / rho_Jpi of the resonance from HFB.
  f_E1, f_E2: standard Lorentzians with RIPL giant-resonance systematics;
  f_M1: constant, from the Kopecky-Uhl ratio to f_E1 at 7 MeV.

Final states: the discrete levels of the TALYS/RIPL level file up to the
level to which the scheme is complete (n_high in the RIPL .cor file, or the
last level from the ground state up with measured J and pi), then the HFB
level density of the final nucleus in bins of width dU above that energy.
A continuum bin with N states of one J^pi contributes the sum of N draws,
i.e. <Gamma> chi^2_N.

RatesMC takes one exit width per resonance; it is given the sum, with the L
and E_xf of the largest single contribution so that its energy dependence
follows the dominant channel.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from common.densities_retrieval import read_hfb_cor, spin_grid
from .read_qvals import ELEMENT_SYMBOLS

HBARC_MEV_FM = 197.3269804
# 1 / (pi hbar c)^2 in mb^-1 MeV^-2: (pi * 197.327 MeV fm)^2 = 3.8434e5 MeV^2 fm^2, 1 fm^2 = 10 mb
_KAPPA = 1.0 / (np.pi * HBARC_MEV_FM) ** 2 / 10.0

_Z_TO_SYMBOL = {z: s for s, z in ELEMENT_SYMBOLS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Discrete levels
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Level:
    E: float      # excitation energy, MeV
    J: float
    parity: int   # +1 / -1
    measured: bool  # J and pi from experiment (not assigned by TALYS)


def default_levels_dir() -> Optional[Path]:
    """TALYS's structure/levels/final directory: $TALYS_LEVELS, else next to `talys` on PATH."""
    env = os.environ.get("TALYS_LEVELS")
    if env and Path(env).is_dir():
        return Path(env)
    exe = shutil.which("talys")
    if exe:
        cand = Path(exe).resolve().parent.parent / "structure" / "levels" / "final"
        if cand.is_dir():
            return cand
    return None


@lru_cache(maxsize=64)
def read_levels(levels_dir: str, Z: int, A: int) -> Tuple[Level, ...]:
    """All levels of (Z, A) from a TALYS/RIPL <Symbol>.lev file (fixed-format)."""
    sym = _Z_TO_SYMBOL.get(int(Z))
    if sym is None:
        raise ValueError(f"No element symbol for Z={Z}")
    path = Path(levels_dir) / f"{sym}.lev"
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        try:
            z, a = int(line[0:4]), int(line[4:8])
            n_lines, n_lev = int(line[8:13]), int(line[13:18])
        except ValueError:
            continue
        if (z, a) != (int(Z), int(A)):
            continue
        levels: List[Level] = []
        k = i + 1
        while len(levels) <= n_lev and k < len(lines):
            row = lines[k]
            E, J = float(row[4:15]), float(row[15:21])
            par, nb = int(row[24:26]), int(row[26:29])
            flags = row[58:60] if len(row) >= 60 else ""
            levels.append(Level(E, J, par, flags.strip() == ""))
            k += 1 + nb
        return tuple(levels)
    raise KeyError(f"No levels for Z={Z}, A={A} in {path}")


def discrete_final_states(levels_dir: str, Z: int, A: int,
                          cor_path: Optional[str] = None) -> Tuple[Tuple[Level, ...], float]:
    """
    Levels of the complete part of the scheme, and the energy above which
    the HFB continuum takes over (the last such level's energy).
    """
    levels = read_levels(str(levels_dir), Z, A)
    n_high = None
    if cor_path:
        entry = _cor_n_high(str(cor_path)).get((int(Z), int(A)))
        if entry is not None:
            n_high = entry
    if n_high is None:
        n_high = 0
        while n_high + 1 < len(levels) and levels[n_high + 1].measured:
            n_high += 1
    keep = tuple(lv for lv in levels[: n_high + 1] if lv.J >= 0 and lv.parity in (1, -1))
    return keep, (keep[-1].E if keep else 0.0)


@lru_cache(maxsize=32)
def _cor_n_high(cor_path: str) -> dict:
    out = {}
    try:
        for line in Path(cor_path).read_text(errors="replace").splitlines():
            p = line.split()
            if len(p) >= 6:
                try:
                    out[(int(p[0]), int(p[1]))] = int(p[3])
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Continuum (HFB)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Continuum:
    """HFB states above E_cut: bin centres U (MeV), and N[bin, J, parity] states."""
    U: np.ndarray            # (nU,)
    J: np.ndarray            # (nJ,)
    N_pos: np.ndarray        # (nU, nJ) expected number of + parity states per bin
    N_neg: np.ndarray        # (nU, nJ)


def hfb_continuum(record, A: int, E_cut: float, U_max: float, dU: float) -> Continuum:
    """Expected numbers of states of each J^pi in bins of width dU over (E_cut, U_max]."""
    J = spin_grid(A)
    if U_max <= E_cut:
        empty = np.zeros((0, len(J)))
        return Continuum(np.zeros(0), J, empty, empty)
    edges = np.arange(E_cut, U_max + dU, dU)
    edges = edges[edges <= U_max + 1e-12]
    if len(edges) < 2:
        edges = np.array([E_cut, U_max])
    U = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)

    def counts(block):
        return np.column_stack([
            np.clip(np.interp(U, block.U, block.rho_J[:, k], left=0.0), 0.0, None) * width
            for k in range(block.rho_J.shape[1])
        ])
    return Continuum(U, J, counts(record.positive), counts(record.negative))


def level_density_Jpi(record, A: int, U: float, J: float, parity: int) -> float:
    """rho_Jpi(U) in MeV^-1 from an HFB record (0 below the table)."""
    block = record.positive if parity > 0 else record.negative
    k = int(np.argmin(np.abs(spin_grid(A) - J)))
    return float(max(np.interp(U, block.U, block.rho_J[:, k], left=0.0), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Gamma strength functions (MeV^-(2L+1))
# ─────────────────────────────────────────────────────────────────────────────
def _gdr(A: int, Z: int):
    """RIPL E1 giant-dipole systematics: (E0 MeV, Gamma0 MeV, sigma0 mb)."""
    N = A - Z
    E0 = 31.2 * A ** (-1 / 3) + 20.6 * A ** (-1 / 6)
    G0 = 0.026 * E0 ** 1.91
    s0 = 1.2 * 120.0 * N * Z / (A * np.pi * G0)
    return E0, G0, s0


def _gqr(A: int, Z: int):
    """RIPL E2 giant-quadrupole systematics: (E0 MeV, Gamma0 MeV, sigma0 mb)."""
    E0 = 63.0 * A ** (-1 / 3)
    G0 = 6.11 - 0.012 * A
    s0 = 0.00015 * Z ** 2 * E0 ** 2 / (A ** (1 / 3) * G0)
    return E0, G0, s0


def _lorentzian(Eg, L, E0, G0, s0):
    """Standard Lorentzian strength function of multipolarity L."""
    Eg = np.asarray(Eg, dtype=float)
    return (_KAPPA / (2 * L + 1)) * s0 * G0 ** 2 * Eg ** (3 - 2 * L) / (
        (Eg ** 2 - E0 ** 2) ** 2 + Eg ** 2 * G0 ** 2)


def strength(XL: str, Eg, A: int, Z: int) -> np.ndarray:
    """f_XL(E_gamma) for XL in {"E1", "M1", "E2"}; 0 for higher multipoles."""
    if XL == "E1":
        return _lorentzian(Eg, 1, *_gdr(A, Z))
    if XL == "M1":
        f_e1_7 = float(_lorentzian(7.0, 1, *_gdr(A, Z)))
        return np.full_like(np.asarray(Eg, dtype=float), f_e1_7 / (0.0588 * A ** 0.878))
    if XL == "E2":
        return _lorentzian(Eg, 2, *_gqr(A, Z))
    return np.zeros_like(np.asarray(Eg, dtype=float))


def gamma_multipole(J_i: float, pi_i: int, J_f: float, pi_f: int) -> Optional[str]:
    """Lowest multipole (E1, M1 or E2) connecting J_i^pi_i -> J_f^pi_f, or None."""
    L = max(1, int(round(abs(J_i - J_f))))
    if L > int(round(J_i + J_f)):
        return None  # 0 -> 0
    change = pi_i * pi_f
    for L_try in (L, L + 1):
        if L_try > 2 or L_try > int(round(J_i + J_f)):
            break
        kind = "E" if change == (-1) ** L_try else "M"
        XL = f"{kind}{L_try}"
        if XL in ("E1", "M1", "E2"):
            return XL
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Summed exit width
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExitWidth:
    Gamma_eV: float     # summed exit width (eV)
    L: Optional[int]    # l or L of the largest contribution (None if Gamma = 0)
    Exf_MeV: float      # excitation energy of the largest contribution's final state
    n_open: int         # number of discrete states + continuum bins contributing


class FinalStateModel:
    """
    Summed exit widths for one reaction.

    kind: "particle" or "gamma".
    final_Z, final_A: nucleus the exit leaves behind (residual, or compound
      for gamma).
    levels: discrete final states; continuum: HFB states above them.
    S_proj, S_exit: separation energies (MeV) from the template.
    particle exits: ejectile spin/parity, penetrability factory pen(l) ->
      callable P(E), and the mean reduced width (eV) per final state.
    gamma exits: compound HFB record (for D) and its A.
    """

    def __init__(self, kind, final_Z, final_A, levels: Sequence[Level], continuum: Continuum,
                 S_proj, S_exit=0.0, ejectile_spin=0.5, ejectile_parity=1,
                 pen=None, gamma2_mean_eV=None, compound_record=None, compound_A=None,
                 l_max=8):
        self.kind = kind
        self.Z, self.A = int(final_Z), int(final_A)
        self.levels = list(levels)
        self.cont = continuum
        self.S_proj, self.S_exit = float(S_proj), float(S_exit)
        self.s_e, self.pi_e = float(ejectile_spin), int(ejectile_parity)
        self.pen = pen
        self.g2 = gamma2_mean_eV
        self.cn = compound_record
        self.cn_A = compound_A
        self.l_max = int(l_max)

    # -- selection rules (cached per J^pi pair) --------------------------------
    @staticmethod
    @lru_cache(maxsize=100000)
    def _l_particle(J, pi, s_e, pi_e, J_f, pi_f, l_max):
        from .generator import _allowed_L_values
        ls = [l for l in _allowed_L_values(J, s_e, J_f, pi, pi_e * pi_f) if l <= l_max]
        return ls[0] if ls else None

    @staticmethod
    @lru_cache(maxsize=100000)
    def _xl_gamma(J, pi, J_f, pi_f):
        return gamma_multipole(J, pi, J_f, pi_f)

    def _means(self, J, pi, J_f, pi_f, E_x, E_r, D_mev):
        """Mean partial widths (eV) to states J_f^pi_f at excitations E_x (array); and L."""
        E_x = np.asarray(E_x, dtype=float)
        if self.kind == "gamma":
            XL = self._xl_gamma(J, pi, J_f, pi_f)
            if XL is None or not D_mev:
                return None, None
            L = int(XL[1])
            Eg = E_r + self.S_proj - E_x
            m = np.where(Eg > 0.0,
                         strength(XL, np.clip(Eg, 1e-6, None), self.A, self.Z)
                         * np.clip(Eg, 0.0, None) ** (2 * L + 1) * D_mev * 1e6, 0.0)
            return m, L
        l = self._l_particle(J, pi, self.s_e, self.pi_e, J_f, pi_f, self.l_max)
        if l is None:
            return None, None
        E = E_r + self.S_proj - self.S_exit - E_x
        m = np.where(E > 0.0, 2.0 * self.g2 * self.pen(l)(np.clip(E, 1e-9, None)), 0.0)
        return m, l

    def exit_width(self, E_r: float, J: float, pi: int, rng) -> ExitWidth:
        """Summed exit width of a resonance at E_r (MeV, CM) with J^pi."""
        D = None
        if self.kind == "gamma":
            rho = level_density_Jpi(self.cn, self.cn_A, E_r + self.S_proj, J, pi)
            D = 1.0 / rho if rho > 0.0 else 0.0

        total, best, best_L, best_E, n_open = 0.0, 0.0, None, 0.0, 0
        for lv in self.levels:
            m, L = self._means(J, pi, lv.J, lv.parity, lv.E, E_r, D)
            if m is None or float(m) <= 0.0:
                continue
            g = float(m) * rng.chisquare(1)
            total += g
            n_open += 1
            if g > best:
                best, best_L, best_E = g, L, lv.E
        c = self.cont
        if c.U.size:
            for parity, N in ((1, c.N_pos), (-1, c.N_neg)):
                for k, J_f in enumerate(c.J):
                    n = N[:, k]
                    if not np.any(n > 1e-6):
                        continue
                    m, L = self._means(J, pi, float(J_f), parity, c.U, E_r, D)
                    if m is None:
                        continue
                    ok = (m > 0.0) & (n > 1e-6)
                    if not np.any(ok):
                        continue
                    g = m[ok] * rng.chisquare(n[ok])  # sum of n Porter-Thomas draws per bin
                    total += float(g.sum())
                    n_open += int(ok.sum())
                    i = int(np.argmax(g))
                    if g[i] > best:
                        best, best_L, best_E = float(g[i]), L, float(c.U[ok][i])
        return ExitWidth(total, best_L, best_E, n_open)


def residual_cor_path(data_root, Z: int) -> Optional[str]:
    p = Path(data_root) / f"z{int(Z):03d}.cor"
    return str(p) if p.exists() else None
