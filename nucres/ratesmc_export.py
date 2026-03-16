from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .physics import spin_stat_factor
from .resonance import Resonance


@dataclass
class RatesMCExportOptions:
    """
    Controls how `Resonance` objects are converted into RatesMC resonant rows.
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
    # Include Corr/Frac column when template expects it.
    include_corr_frac: bool = False
    # Value for Corr/Frac column (0 = no correlation by default).
    corr_frac_value: int = 0
    # Optional override of projectile/target spins used in the statistical factor
    j_proj: Optional[float] = None
    j_targ: Optional[float] = None
    # Formatting
    precision: int = 3  # decimals for non-Ecm columns
    ecm_decimals: int = 3


@dataclass
class RatesMCRow:
    """Structured representation of one formatted RatesMC resonance row."""

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
    corr_frac: int


BASE_COLUMN_SPECS = [
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


def _column_specs(opts: Optional[RatesMCExportOptions]) -> List[Tuple[str, int, str]]:
    specs = list(BASE_COLUMN_SPECS)
    if opts is not None and opts.include_corr_frac:
        specs.append(("Corr/Frac", 9, "int"))
    return specs


def omega_gamma(
    res: Resonance, j_proj: Optional[float] = None, j_targ: Optional[float] = None
) -> float:
    r"""
    Compute the resonance strength \(\omega \gamma\) in eV from partial widths.
    """
    g = spin_stat_factor(
        res.J,
        j_proj if j_proj is not None else res.s1,
        j_targ if j_targ is not None else res.s2,
    )
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
    Map a `Resonance` object into a `RatesMCRow`, converting units to keV/eV.
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
        L1=res.L1 if res.L1 is not None else opts.l1,
        G2=G2,
        DG2=DG2,
        L2=res.L2 if res.L2 is not None else opts.l2,
        G3=G3,
        DG3=DG3,
        L3=res.L3 if res.L3 is not None else opts.l3,
        Exf=opts.exf_keV,
        Int=opts.int_flag,
        corr_frac=opts.corr_frac_value,
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
    values = {
        "Ecm": row.Ecm,
        "DEcm": row.DEcm,
        "wg": row.wg,
        "Dwg": row.Dwg,
        "Jr": row.Jr,
        "G1": row.G1,
        "DG1": row.DG1,
        "L1": row.L1,
        "G2": row.G2,
        "DG2": row.DG2,
        "L2": row.L2,
        "G3": row.G3,
        "DG3": row.DG3,
        "L3": row.L3,
        "Exf": row.Exf,
        "Int": row.Int,
        "Corr/Frac": row.corr_frac,
    }
    formatted = []
    specs = _column_specs(opts)
    for label, _, kind in specs:
        decimals = opts.ecm_decimals if label == "Ecm" else opts.precision
        formatted.append(_format_value(values[label], decimals, kind))
    return formatted


def render_rows(
    resonances: Iterable[Resonance], opts: RatesMCExportOptions
) -> Tuple[List[str], List[int]]:
    """
    Convert resonances to aligned text rows ready for a RatesMC input block.
    """
    rows_raw: List[List[str]] = []
    for res in resonances:
        row = resonance_to_row(res, opts)
        rows_raw.append(_row_as_strings(row, opts))
    specs = _column_specs(opts)
    widths = [len(label) for label, _, _ in specs]
    for row in rows_raw:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))
    lines = [
        " ".join(val.rjust(widths[idx]) for idx, val in enumerate(row))
        for row in rows_raw
    ]
    return lines, widths


def resonant_header_line(
    widths: Optional[List[int]] = None, opts: Optional[RatesMCExportOptions] = None
) -> str:
    """Render the header line corresponding to the active RatesMC columns."""
    specs = _column_specs(opts)
    active_widths = widths or [len(label) for label, _, _ in specs]
    return " ".join(
        label.ljust(active_widths[idx]) for idx, (label, _, _) in enumerate(specs)
    )


def write_resonant_block(
    dest: Path, rows: List[str], header: Optional[str] = None
) -> None:
    """
    Write a resonant contribution block, including an optional header, to disk.
    """
    parts: List[str] = []
    if header:
        parts.append(header.rstrip("\n"))
    parts.extend(rows)
    dest.write_text("\n".join(parts) + "\n", encoding="ascii")
