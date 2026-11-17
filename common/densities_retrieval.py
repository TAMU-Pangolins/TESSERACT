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

def _isotope_block_starts(raw: List[str]) -> Dict[Tuple[int, int], int]:
    """
    Map (A, parity) -> index of the banner line for every isotope block.

    RIPL-3 zXXX.tab files hold every tabulated isotope of one element, each as
    a positive-parity block followed by a negative-parity block, introduced by
    a banner such as "Z= 14 A= 28: Positive-Parity Spin-dependent ...".
    """
    starts: Dict[Tuple[int, int], int] = {}
    for i, ln in enumerate(raw):
        m = _BANNER_ZA.search(ln)
        if not m:
            continue
        low = ln.lower()
        if "positive" in low:
            parity = +1
        elif "negative" in low:
            parity = -1
        else:
            continue
        starts.setdefault((int(m.group(2)), parity), i)
    return starts


def _collect_rows(raw: List[str], start: int, stop: int, n: int = 60) -> Tuple[List[str], int]:
    """Return up to `n` data rows from raw[start:stop] and the index after the last one."""
    rows = []
    i = start
    while i < stop and len(rows) < n:
        if _line_is_data_row(raw[i]):
            rows.append(raw[i])
        i += 1
    return rows, i


def read_hfb_tab(path: str, A: Optional[int] = None, strict: bool = True):
    """
    Read the HFB level-density tables for one isotope from a zXXX.tab file.

    `A` selects the isotope. It may be omitted only when the file contains a
    single isotope; otherwise a ValueError is raised rather than silently
    returning the first (lightest) isotope in the file.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.rstrip("\n") for ln in f]

    starts = _isotope_block_starts(raw)
    if starts:
        available = sorted({a for (a, _) in starts})
        if A is None:
            if len(available) != 1:
                raise ValueError(
                    f"{path} contains {len(available)} isotopes "
                    f"(A={available[0]}..{available[-1]}); pass A= to select one."
                )
            A = available[0]
        A = int(A)
        if (A, +1) not in starts or (A, -1) not in starts:
            raise ValueError(
                f"A={A} not found in {path}; available A values: {available}"
            )
        banners = sorted(starts.values())

        def _block_end(begin: int) -> int:
            later = [b for b in banners if b > begin]
            return later[0] if later else len(raw)

        pos_start = starts[(A, +1)]
        neg_start = starts[(A, -1)]
        pos_lines, _ = _collect_rows(raw, pos_start, _block_end(pos_start))
        neg_lines, _ = _collect_rows(raw, neg_start, _block_end(neg_start))
        Z_banner = int(_BANNER_ZA.search(raw[pos_start]).group(1))
        hdr_dict = dict(Z=Z_banner, A=A)
    else:
        # Non-RIPL layout without banners: fall back to the first two blocks.
        if A is not None:
            raise ValueError(f"No isotope banners found in {path}; cannot select A={A}.")
        hdr_dict, start_idx = _parse_header_flexible(raw)
        pos_lines, after_pos = _collect_rows(raw, start_idx, len(raw))
        neg_lines, _ = _collect_rows(raw, after_pos, len(raw))

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

def read_hfb_cor(path: str) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Read a RIPL-3 HFB zXXX.cor file of level-density normalisations.

    Each line is "Z A n_low n_high ctable ptable [nuclide]": the HFB density
    is renormalised to the discrete levels (and s-wave spacing where known)
    as

        rho(U, J, pi) = exp(ctable * sqrt(U - ptable)) * rho_HFB(U - ptable, J, pi)

    (TALYS ldmodel 5/6). Returns {(Z, A): (ctable, ptable)}; a missing file
    gives an empty dict.
    """
    cor: Dict[Tuple[int, int], Tuple[float, float]] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return cor
    for ln in lines:
        parts = ln.split()
        if len(parts) < 6:
            continue
        try:
            Z, A = int(parts[0]), int(parts[1])
            c, delta = float(parts[4]), float(parts[5])
        except ValueError:
            continue
        cor[(Z, A)] = (c, delta)
    return cor


def _correct_block(block: HFBBlock, c: float, delta: float) -> HFBBlock:
    U = block.U
    shifted = U - delta
    factor = np.where(shifted > 0.0, np.exp(c * np.sqrt(np.clip(shifted, 0.0, None))), 0.0)

    def rescale(y: np.ndarray) -> np.ndarray:
        # rho_HFB at U - delta (zero below the tabulated range), times exp(c sqrt(U - delta))
        return factor * np.interp(shifted, U, y, left=0.0, right=y[-1])

    return HFBBlock(
        parity=block.parity,
        U=U.copy(),
        T=block.T.copy(),
        Ncumul=block.Ncumul.copy(),  # not renormalised; unused by the samplers
        Rho_level=rescale(block.Rho_level),
        Rho_state=rescale(block.Rho_state),
        rho_J=np.column_stack([rescale(block.rho_J[:, k]) for k in range(block.rho_J.shape[1])]),
    )


def apply_hfb_corrections(record: HFBRecord, cor: Dict[Tuple[int, int], Tuple[float, float]]) -> HFBRecord:
    """
    Apply the (ctable, ptable) normalisation for the record's isotope (see
    read_hfb_cor). Returns a new record; raises KeyError if the isotope has
    no entry.
    """
    key = (record.header.Z, record.header.A)
    if key not in cor:
        raise KeyError(f"No level-density correction for Z={key[0]}, A={key[1]}.")
    c, delta = cor[key]
    return HFBRecord(
        header=record.header,
        positive=_correct_block(record.positive, c, delta),
        negative=_correct_block(record.negative, c, delta),
    )

# ------------------------- J^pi equilibration -------------------------

LEVEL_DENSITY_JPI_MODELS = ("hfb", "equilibrated")

# Fermi-gas level-density parameter systematics of TALYS (ldmodel 1, global
# default), a = alpha A + beta A^(2/3): A. J. Koning, S. Hilaire and
# S. Goriely, Nucl. Phys. A 810, 13 (2008), Table 3 (shell effects omitted).
_TALYS_ALPHA = 0.0722396
_TALYS_BETA = 0.195267


def fermi_gas_a(A: int) -> float:
    """Asymptotic level-density parameter a (MeV^-1), TALYS global systematics (see above)."""
    return _TALYS_ALPHA * A + _TALYS_BETA * A ** (2.0 / 3.0)


def pairing_shift(Z: int, A: int) -> float:
    """Pairing energy Delta = chi 12/sqrt(A) MeV, chi = 2 (even-even), 1 (odd A), 0 (odd-odd)."""
    N = A - Z
    chi = 2 if (Z % 2 == 0 and N % 2 == 0) else (0 if (Z % 2 and N % 2) else 1)
    return chi * 12.0 / math.sqrt(A)


def spin_cutoff_sigma2(U, Z: int, A: int, a: Optional[float] = None) -> np.ndarray:
    """
    Spin-cutoff parameter sigma^2(U) of the back-shifted Fermi gas,

        sigma^2 = 0.0146 A^(5/3) (1 + sqrt(1 + 4 a U*)) / (2 a),

    RIPL-3, R. Capote et al., Nucl. Data Sheets 110, 3107 (2009), Sec. 5
    (eq. for the BSFG spin cutoff; T. von Egidy and D. Bucurescu, PRC 72,
    044311 (2005)), with the effective excitation energy U* = max(U - Delta, 0),
    Delta = chi 12/sqrt(A) the pairing shift TALYS uses, and a from
    `fermi_gas_a` unless given.
    """
    a = fermi_gas_a(A) if a is None else float(a)
    if not a > 0.0:
        raise ValueError("Level-density parameter a must be positive.")
    Ueff = np.clip(np.asarray(U, dtype=float) - pairing_shift(Z, A), 0.0, None)
    return 0.0146 * A ** (5.0 / 3.0) * (1.0 + np.sqrt(1.0 + 4.0 * a * Ueff)) / (2.0 * a)


def spin_distribution(J: np.ndarray, sigma2) -> np.ndarray:
    """
    P(J; sigma^2) = (2J+1)/(2 sigma^2) exp(-(J+1/2)^2 / (2 sigma^2)), normalised
    to 1 over the spin grid J (axis -1). sigma2 may be a scalar or an array of
    shape (nU,), giving shape (nU, nJ).
    """
    J = np.asarray(J, dtype=float)
    s2 = np.asarray(sigma2, dtype=float)[..., None]
    if np.any(~(s2 > 0.0)):
        raise ValueError("sigma^2 must be positive.")
    P = (2.0 * J + 1.0) / (2.0 * s2) * np.exp(-((J + 0.5) ** 2) / (2.0 * s2))
    return P / P.sum(axis=-1, keepdims=True)


def hfb_sigma2(record: HFBRecord) -> np.ndarray:
    """
    The table's own spin cutoff on its U grid, sigma^2 = <(J+1/2)^2>/2 =
    (<J(J+1)> + 1/4)/2, averaged over both parities with rho_J weights (NaN
    where the table is empty).
    """
    J = spin_grid(record.header.A)
    w = np.clip(record.positive.rho_J, 0.0, None) + np.clip(record.negative.rho_J, 0.0, None)
    tot = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0.0, (w * (J + 0.5) ** 2).sum(axis=1) / tot / 2.0, np.nan)


def equilibrate_jpi(record: HFBRecord, sigma2: Optional[float] = None,
                    a: Optional[float] = None) -> HFBRecord:
    """
    Replace the table's J^pi split by a parity-equilibrated spin-cutoff one,

        rho(U, J, pi) = 1/2 rho_tot(U) P(J; sigma^2(U)),

    where rho_tot is the record's (possibly corrected) total, summed over its
    J columns and both parities, and P is `spin_distribution` on the
    isotope's spin grid. sigma^2(U) is `spin_cutoff_sigma2` (RIPL-3 BSFG
    systematic) unless a constant `sigma2` is given. The total level density
    is unchanged; Rho_level/Rho_state of each block become the new J sums.
    Apply after any (ctable, ptable) correction.
    """
    pos, neg = record.positive, record.negative
    if not np.array_equal(pos.U, neg.U):
        raise ValueError("Parity blocks must share one U grid to be equilibrated.")
    A = int(record.header.A)
    J = spin_grid(A)
    if pos.rho_J.shape[1] != J.size:
        raise ValueError("HFB spin columns do not match the spin grid.")
    if sigma2 is not None:
        s2 = np.full(pos.U.shape, float(sigma2))
    else:
        s2 = spin_cutoff_sigma2(pos.U, int(record.header.Z), A, a=a)
    rho_tot = np.clip(pos.rho_J, 0.0, None).sum(axis=1) + np.clip(neg.rho_J, 0.0, None).sum(axis=1)
    rhoJ = 0.5 * rho_tot[:, None] * spin_distribution(J, s2)

    def block(b: HFBBlock) -> HFBBlock:
        return HFBBlock(
            parity=b.parity,
            U=b.U.copy(),
            T=b.T.copy(),
            Ncumul=b.Ncumul.copy(),  # not recomputed; unused by the samplers
            Rho_level=rhoJ.sum(axis=1),
            Rho_state=(rhoJ * (2.0 * J + 1.0)).sum(axis=1),
            rho_J=rhoJ.copy(),
        )

    return HFBRecord(header=record.header, positive=block(pos), negative=block(neg))


def apply_level_density_jpi(record: HFBRecord, model: str = "hfb",
                            sigma2: Optional[float] = None) -> HFBRecord:
    """Apply the J^pi model `model` ("hfb": the table as is; "equilibrated": `equilibrate_jpi`)."""
    model = str(model).strip().lower()
    if model not in LEVEL_DENSITY_JPI_MODELS:
        raise ValueError(f"level_density_jpi must be one of {LEVEL_DENSITY_JPI_MODELS}; got {model!r}.")
    if model == "hfb":
        return record
    return equilibrate_jpi(record, sigma2=sigma2)

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