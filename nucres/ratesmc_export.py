from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .resonance import Resonance
from .physics import spin_stat_factor


@dataclass
class RatesMCExportOptions:
    """
    Controls how Resonance objects are converted into RatesMC resonant rows.
    """

    # Toggle between emitting an analytical strength (omega-gamma) or explicit widths
    use_strength: bool = False
    # If set, apply this fractional uncertainty to all energies/widths/strengths; else 0
    default_frac_unc: Optional[float] = None
    # Orbital angular momenta / multipolarities for channels 1–3
    l1: int = 0
    l2: int = 1
    l3: int = 0
    # Excitation energy of populated level (keV)
    exf_keV: float = 0.0
    # Int flag (0 = analytical, 1 = numerical)
    int_flag: int = 1
    # Whether to keep G3 columns; when False, G3 and DG3 are zeroed
    include_g3: bool = False
    # Optional override of projectile/target spins used in the statistical factor
    j_proj: Optional[float] = None
    j_targ: Optional[float] = None
    # Formatting
    precision: int = 3   # decimals for non-Ecm columns
    ecm_decimals: int = 3


@dataclass
class RatesMCRow:
    Ecm: float
    DEcm: float
    wg: float
    Dwg: float
    Jr: float
    G1: float
    DG1: float
    L1: int
    G2: float
    DG2: float
    L2: int
    G3: float
    DG3: float
    L3: int
    Exf: float
    Int: int

COLUMN_SPECS = [
    ("Ecm", 9, "float"),
    ("DEcm", 6, "float"),
    ("wg", 11, "sci"),
    ("Dwg", 5, "sci"),
    ("Jr", 3, "int"),
    ("G1", 11, "sci"),
    ("DG1", 5, "sci"),
    ("L1", 3, "int"),
    ("G2", 11, "sci"),
    ("DG2", 5, "sci"),
    ("L2", 3, "int"),
    ("G3", 11, "sci"),
    ("DG3", 5, "sci"),
    ("L3", 3, "int"),
    ("Exf", 6, "float"),
    ("Int", 3, "int"),
]


def omega_gamma(res: Resonance, j_proj: Optional[float] = None, j_targ: Optional[float] = None) -> float:
    """
    Compute strength omega-gamma (eV) from partial widths.
    """
    g = spin_stat_factor(res.J, j_proj if j_proj is not None else res.s1, j_targ if j_targ is not None else res.s2)
    gt = res.Gamma_i + res.Gamma_o
    if gt <= 0.0:
        return 0.0
    return g * (res.Gamma_i * res.Gamma_o) / gt


def _frac_unc(value: float, frac: Optional[float]) -> float:
    if frac is None:
        return 0.0
    return abs(value) * frac


def resonance_to_row(res: Resonance, opts: RatesMCExportOptions) -> RatesMCRow:
    """
    Map a Resonance object into a structured RatesMCRow (units converted to keV/eV).
    """
    Ecm_keV = res.E_r * 1e-3
    DEcm = _frac_unc(Ecm_keV, opts.default_frac_unc)

    if opts.use_strength:
        wg_val = omega_gamma(res, opts.j_proj, opts.j_targ)
        Dwg = _frac_unc(wg_val, opts.default_frac_unc)
    else:
        wg_val = 0.0
        Dwg = 0.0

    G1 = res.Gamma_i if not opts.use_strength else res.Gamma_i
    DG1 = _frac_unc(G1, opts.default_frac_unc)
    G2 = res.Gamma_o if not opts.use_strength else res.Gamma_o
    DG2 = _frac_unc(G2, opts.default_frac_unc)

    if opts.include_g3:
        G3 = 0.0
        DG3 = 0.0
    else:
        G3 = 0.0
        DG3 = 0.0

    return RatesMCRow(
        Ecm=Ecm_keV,
        DEcm=DEcm,
        wg=wg_val,
        Dwg=Dwg,
        Jr=res.J,
        G1=G1,
        DG1=DG1,
        L1=opts.l1,
        G2=G2,
        DG2=DG2,
        L2=opts.l2,
        G3=G3,
        DG3=DG3,
        L3=opts.l3,
        Exf=opts.exf_keV,
        Int=opts.int_flag,
    )


def _format_value(value: float, decimals: int, kind: str) -> str:
    if kind == "int":
        text = str(int(round(value)))
    else:
        scientific = kind == "sci"
        if abs(value) < 1e-300:
            text = "0"
        else:
            fmt = f"{{:.{decimals}{'e' if scientific else 'f'}}}"
            text = fmt.format(value)
    return text


def _row_as_strings(row: RatesMCRow, opts: RatesMCExportOptions) -> List[str]:
    values = [
        row.Ecm,
        row.DEcm,
        row.wg,
        row.Dwg,
        row.Jr,
        row.G1,
        row.DG1,
        row.L1,
        row.G2,
        row.DG2,
        row.L2,
        row.G3,
        row.DG3,
        row.L3,
        row.Exf,
        row.Int,
    ]
    formatted = []
    for (label, _, kind), value in zip(COLUMN_SPECS, values):
        decimals = opts.ecm_decimals if label == "Ecm" else opts.precision
        formatted.append(_format_value(value, decimals, kind))
    return formatted


def render_rows(resonances: Iterable[Resonance], opts: RatesMCExportOptions) -> Tuple[List[str], List[int]]:
    """
    Convert a list of Resonance objects to formatted RatesMC lines.
    """
    rows_raw: List[List[str]] = []
    for res in resonances:
        row = resonance_to_row(res, opts)
        rows_raw.append(_row_as_strings(row, opts))
    widths = [len(label) for label, _, _ in COLUMN_SPECS]
    for row in rows_raw:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))
    lines = [" ".join(val.rjust(widths[idx]) for idx, val in enumerate(row)) for row in rows_raw]
    return lines, widths


def resonant_header_line(widths: Optional[List[int]] = None) -> str:
    active_widths = widths or [len(label) for label, _, _ in COLUMN_SPECS]
    return " ".join(label.ljust(active_widths[idx]) for idx, (label, _, _) in enumerate(COLUMN_SPECS))


def write_resonant_block(dest: Path, rows: List[str], header: Optional[str] = None) -> None:
    """
    Write a resonant contribution block (header + rows) to a file.
    """
    parts: List[str] = []
    if header:
        parts.append(header.rstrip("\n"))
    parts.extend(rows)
    dest.write_text("\n".join(parts) + "\n", encoding="ascii")
