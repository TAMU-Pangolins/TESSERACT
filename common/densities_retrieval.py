from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------------------------- data classes ----------------------------

@dataclass
class HFBHeader:
    Z: int
    A: int
    B: float
    hw: float
    beta2: float
    beta3: float
    beta4: float

@dataclass
class HFBBlock:
    parity: int                 # +1 (positive), -1 (negative)
    U: np.ndarray               # (nU,) excitation energies (MeV)
    T: np.ndarray               # (nU,) nuclear temperatures (MeV)
    Ncumul: np.ndarray          # (nU,) cumulative # of levels (this parity)
    Rho_level: np.ndarray       # (nU,) total level density (Rhoobs), 1/MeV
    Rho_state: np.ndarray       # (nU,) total state density (Rhotot), 1/MeV
    rho_J: np.ndarray           # (nU, 50) spin-resolved level densities J=0..49, 1/MeV

@dataclass
class HFBRecord:
    header: HFBHeader
    positive: HFBBlock
    negative: HFBBlock

_BANNER_ZA = re.compile(r"\bZ\s*=\s*(\d+)\b.*\bA\s*=\s*(\d+)\b")


# Example header format:
# (25x,i3,3x,i3,67x,f6.2,4x,f6.2,3(4x,f6.3))

_HEADER_RE = re.compile(
    r".*?\b(?P<Z>\d{1,3})\b\s+(?P<A>\d{1,3})\b.*?(?P<B>-?\d+\.\d)\s+(?P<hw>-?\d+\.\d)\s+"
    r"(?P<beta2>-?\d+\.\d{3})\s+(?P<beta3>-?\d+\.\d{3})\s+(?P<beta4>-?\d+\.\d{3})"
)

# ---------------------------- core parsing ----------------------------

def _parse_header(line: str) -> HFBHeader:
    m = _HEADER_RE.match(line)
    if not m:
        raise ValueError("Header line does not match expected format:\n" + line)
    return HFBHeader(
        Z=int(m.group("Z")),
        A=int(m.group("A")),
        B=float(m.group("B")),
        hw=float(m.group("hw")),
        beta2=float(m.group("beta2")),
        beta3=float(m.group("beta3")),
        beta4=float(m.group("beta4")),
    )

def _parse_block(lines: List[str], parity: int) -> HFBBlock:
    """
    Each data line (Fortran '(f7.2, f7.3, 1x, 1p, 53e9.2)') has:
      U, T, Ncumul, Rhoobs (level), Rhotot (state), then 50 rho_J columns (J=0..49).
    """
    U_list, T_list, Nc_list, RhoL_list, RhoS_list = [], [], [], [], []
    rhoJ_rows: List[List[float]] = []
    for ln in lines:
        parts = ln.strip().split()
        # 55 numbers expected: 2 (U,T) + 53 exponentials
        if len(parts) < 55:
            raise ValueError(f"Data line has too few columns ({len(parts)}): {ln!r}")
        U = float(parts[0]); T = float(parts[1])
        Nc, RhoL, RhoS = map(float, parts[2:5])
        rhoJ = [float(x) for x in parts[5:55]]  # 50 spins
        U_list.append(U); T_list.append(T); Nc_list.append(Nc)
        RhoL_list.append(RhoL); RhoS_list.append(RhoS); rhoJ_rows.append(rhoJ)
    U_arr = np.array(U_list, dtype=float)
    return HFBBlock(
        parity=parity,
        U=U_arr,
        T=np.array(T_list, dtype=float),
        Ncumul=np.array(Nc_list, dtype=float),
        Rho_level=np.array(RhoL_list, dtype=float),
        Rho_state=np.array(RhoS_list, dtype=float),
        rho_J=np.array(rhoJ_rows, dtype=float),
    )

def _line_is_data_row(ln: str) -> bool:
    p = ln.strip().split()
    if len(p) < 55: return False
    try:
        # U, T, Ncumul, Rhoobs, Rhotot must be floats
        float(p[0]); float(p[1]); float(p[2]); float(p[3]); float(p[4])
        # and at least a few of the spin columns should parse
        float(p[5]); float(p[54])
        return True
    except Exception:
        return False

def _parse_header_flexible(lines: list[str]):
    """
    Try the full HFB header first; if it fails, try to extract Z,A from banners.
    Returns (header_dict_or_None, first_data_idx)
    """
    # First try: find a line matching the full header (with B, hw, beta2/3/4)
    for i, ln in enumerate(lines):
        m = _HEADER_RE.match(ln)
        if m:
            hdr = dict(Z=int(m["Z"]), A=int(m["A"]),
                       B=float(m["B"]), hw=float(m["hw"]),
                       beta2=float(m["beta2"]), beta3=float(m["beta3"]), beta4=float(m["beta4"]))
            # first data line should follow somewhere after; we'll scan anyway in read_hfb_tab
            return hdr, i+1
    # Fallback: look for a banner like "Z=  9 A= 17 ..."
    Z = A = None
    first_idx = 0
    for i, ln in enumerate(lines):
        m = _BANNER_ZA.search(ln)
        if m:
            Z = int(m.group(1)); A = int(m.group(2))
            first_idx = i + 1
            break
    if Z is not None and A is not None:
        hdr = dict(Z=Z, A=A, B=float("nan"), hw=float("nan"),
                   beta2=float("nan"), beta3=float("nan"), beta4=float("nan"))
        return hdr, first_idx
    # Nothing found
    return None, 0

def read_hfb_tab(path: str, strict: bool = True):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.rstrip("\n") for ln in f]

    hdr_dict, start_idx = _parse_header_flexible(raw)
    # Collect 60 numeric lines for +parity
    pos_lines = []
    i = start_idx
    while i < len(raw) and len(pos_lines) < 60:
        if _line_is_data_row(raw[i]):
            pos_lines.append(raw[i])
        i += 1
    # Collect 60 numeric lines for -parity
    neg_lines = []
    while i < len(raw) and len(neg_lines) < 60:
        if _line_is_data_row(raw[i]):
            neg_lines.append(raw[i])
        i += 1

    if len(pos_lines) != 60 or len(neg_lines) != 60:
        if strict:
            raise ValueError(f"Could not find two 60-row data blocks in {path} (got {len(pos_lines)} and {len(neg_lines)})")

    positive = _parse_block(pos_lines, parity=+1)
    negative = _parse_block(neg_lines, parity=-1)

    # Build header dataclass, tolerating missing extras
    if hdr_dict is None:
        header = HFBHeader(Z=-1, A=-1, B=float("nan"), hw=float("nan"),
                           beta2=float("nan"), beta3=float("nan"), beta4=float("nan"))
    else:
        # fill missing numeric fields with NaN if header came from banner
        for k in ("B","hw","beta2","beta3","beta4"):
            hdr_dict[k] = hdr_dict.get(k, float("nan"))
        header = HFBHeader(**hdr_dict)

    return HFBRecord(header=header, positive=positive, negative=negative)

# ---- optional: spin grid & index helpers ----
def spin_grid(A: int) -> np.ndarray:
    """
    First 50 spins used by the table (physical J values).
    Convention: even-A -> integers 0,1,2,...; odd-A -> half-integers 1/2,3/2,...
    """
    if A % 2 == 0:
        return np.arange(50, dtype=float)
    else:
        return 0.5 + np.arange(50, dtype=float)

def J_index(J_phys: float, A: int) -> int:
    """
    Map a physical spin (e.g. 3.5 for 7/2) to the column index 0..49 using spin_grid(A).
    """
    grid = spin_grid(A)
    return int(np.argmin(np.abs(grid - J_phys)))

# ---------------------------- corrections ----------------------------

def read_hfb_cor(path: str) -> Dict[Tuple[int, float], Dict[str, np.ndarray]]:
    """
    Read a zXXX.cor corrections file.
    Returns a dict keyed by (parity, U) with arrays to overwrite: T, Ncumul, Rho_level, Rho_state, rho_J.
    Rows are split into positive/negative blocks by detecting a U-sequence reset.
    """
    cor: Dict[Tuple[int, float], Dict[str, np.ndarray]] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except FileNotFoundError:
        return cor

    def try_parse_row(ln: str):
        parts = ln.split()
        if len(parts) >= 55:
            try:
                U = float(parts[0]); T = float(parts[1])
                Nc, RhoL, RhoS = map(float, parts[2:5])
                rhoJ = np.array([float(x) for x in parts[5:55]], dtype=float)
                return U, T, Nc, RhoL, RhoS, rhoJ
            except Exception:
                return None
        return None

    rows = []
    for ln in lines:
        r = try_parse_row(ln)
        if r is not None:
            rows.append(r)

    if not rows:
        return cor

    # Detect where the sequence restarts (switch from + to - parity)
    Uvals = np.array([r[0] for r in rows], dtype=float)
    reset_idx = None
    for i in range(1, len(Uvals)):
        if Uvals[i] < Uvals[i-1] - 1e-6:
            reset_idx = i
            break

    if reset_idx is None:
        # Couldn’t detect split; treat all as positive
        for U, T, Nc, RhoL, RhoS, rhoJ in rows:
            cor[(+1, U)] = {"T": T, "Ncumul": Nc, "Rho_level": RhoL, "Rho_state": RhoS, "rho_J": rhoJ}
        return cor

    pos_rows = rows[:reset_idx]
    neg_rows = rows[reset_idx:]
    for U, T, Nc, RhoL, RhoS, rhoJ in pos_rows:
        cor[(+1, U)] = {"T": T, "Ncumul": Nc, "Rho_level": RhoL, "Rho_state": RhoS, "rho_J": rhoJ}
    for U, T, Nc, RhoL, RhoS, rhoJ in neg_rows:
        cor[(-1, U)] = {"T": T, "Ncumul": Nc, "Rho_level": RhoL, "Rho_state": RhoS, "rho_J": rhoJ}
    return cor

def apply_hfb_corrections(record: HFBRecord, cor: Dict[Tuple[int, float], Dict[str, np.ndarray]], tol: float = 1e-6) -> HFBRecord:
    """
    Overwrite rows in 'record' with values from 'cor' where U matches within 'tol'.
    """
    for block in (record.positive, record.negative):
        pi = block.parity
        for (pi_row, Ucorr), data in cor.items():
            if pi_row != pi:
                continue
            idx = int(np.argmin(np.abs(block.U - Ucorr)))
            if abs(block.U[idx] - Ucorr) <= tol:
                block.T[idx] = float(data["T"])
                block.Ncumul[idx] = float(data["Ncumul"])
                block.Rho_level[idx] = float(data["Rho_level"])
                block.Rho_state[idx] = float(data["Rho_state"])
                block.rho_J[idx, :] = np.asarray(data["rho_J"], dtype=float)
    return record

# ---------------------------- utilities ----------------------------

def build_rho_interpolator(record: HFBRecord):
    """
    Returns rho(U, J, pi) in levels/MeV.
      - Linear interpolation in U, nearest (integer) for J, and explicit parity choice (+1/-1).
      - U can be scalar or array-like.
    """
    def rho(Uval, J: int, pi: int):
        block = record.positive if pi == +1 else record.negative
        Uarr = block.U
        rhoJ = block.rho_J
        if not (0 <= J < rhoJ.shape[1]):
            raise ValueError(f"J={J} out of range (expected 0..{rhoJ.shape[1]-1})")
        Uvals = np.atleast_1d(Uval).astype(float)
        out = np.empty_like(Uvals, dtype=float)
        # vectorized-ish linear interpolation
        for k, u in enumerate(Uvals):
            if u <= Uarr[0]:
                out[k] = rhoJ[0, J]
            elif u >= Uarr[-1]:
                out[k] = rhoJ[-1, J]
            else:
                i = np.searchsorted(Uarr, u) - 1
                x0, x1 = Uarr[i], Uarr[i+1]
                y0, y1 = rhoJ[i, J], rhoJ[i+1, J]
                t = (u - x0) / (x1 - x0)
                out[k] = y0 * (1 - t) + y1 * t
        return out if out.shape != () else float(out)
    return rho

def mean_spacing_D(record_or_rho, U: float, J: int, pi: int) -> float:
    """
    Return D(U,J,pi) = 1 / rho(U,J,pi) in MeV.
    Accepts either an HFBRecord (we build a rho interpolator) or a rho function from build_rho_interpolator.
    """
    if callable(record_or_rho):
        rho_fn = record_or_rho
    else:
        rho_fn = build_rho_interpolator(record_or_rho)
    val = rho_fn(U, J, pi)
    if val <= 0:
        raise ValueError(f"Non-positive rho(U={U},J={J},pi={pi}) = {val}")
    return 1.0 / val