"""
Read RatesMC rate tables (``RatesMC.out``) written by either RatesMC lineage.

The column layout differs between releases:

* RatesMC 2.11 (classic C code)::

     T9  RRate_low  Classical Rate  Median Rate  Mean Rate  RRate_high
         Log-Normal mu  Log-Normal sigma  A-D Statistic

* RatesMC 2.2+ (C++ rewrite, rlongland/RatesMC)::

     T9  RRate_low  Median Rate  RRate_high  f.u.

so columns are located by their header names, never by position.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Header labels (some contain spaces) -> canonical column names.
_HEADER_NAMES = [
    ("Classical Rate", "classical"),
    ("Median Rate", "median"),
    ("Mean Rate", "mean"),
    ("Log-Normal mu", "mu"),
    ("Log-Normal sigma", "sigma"),
    ("A-D Statistic", "ad"),
    ("RRate_low", "low"),
    ("RRate_high", "high"),
    ("f.u.", "fu"),
    ("T9", "T9"),
]


def _header_columns(line: str) -> List[str]:
    text = line.strip()
    for label, name in _HEADER_NAMES:
        text = text.replace(label, f" {name} ")
    return text.split()


def read_ratesmc_out(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Read a RatesMC ``.out`` rate table.

    Returns a dict of numpy arrays keyed by canonical column name. Always
    present: ``T9``, ``low``, ``median``, ``high``. Others (``fu``,
    ``classical``, ``mean``, ``mu``, ``sigma``, ``ad``) are present only if
    the file has them.

    Raises
    ------
    ValueError
        If no header line starting with ``T9`` or no rate rows are found, or a
        required column is missing.
    """
    path = Path(path)
    columns: Optional[List[str]] = None
    rows: List[List[float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if columns is None:
                if stripped.startswith("T9"):
                    columns = _header_columns(stripped)
                continue
            parts = stripped.split()
            if not parts:
                continue
            try:
                values = [float(p) for p in parts[: len(columns)]]
            except ValueError:
                if rows:
                    break  # end of the table (trailing notes)
                continue
            if len(values) < len(columns):
                continue
            rows.append(values)
    if columns is None:
        raise ValueError(f"{path}: no RatesMC header line starting with 'T9'.")
    if not rows:
        raise ValueError(f"{path}: no rate rows found.")
    for required in ("T9", "low", "median", "high"):
        if required not in columns:
            raise ValueError(
                f"{path}: column {required!r} not found in header {columns}."
            )
    table = np.asarray(rows, dtype=float)
    return {name: table[:, i] for i, name in enumerate(columns)}


# Run-directory layouts written by TESSERACT:
#   tesseract.py [ratesmc]      <root>/<reaction>/RUN_<j>/<reaction>.out
#   run_ratesmc_batches.sh      <root>/<reaction>/Run_<NN>/RatesMC.out
RUN_DIR_RE = re.compile(r"^(RUN|Run)_(\d+)$")


def find_ratesmc_outputs(reaction_dir: str | Path) -> List[Tuple[str, Path]]:
    """
    Find one RatesMC rate table per run directory under ``reaction_dir``.

    Accepts both ``RUN_<j>/<reaction>.out`` (tesseract.py) and
    ``Run_<NN>/RatesMC.out`` (run_ratesmc_batches.sh). In a ``RUN_<j>``
    directory ``<reaction>.out`` is preferred over ``RatesMC.out``. Returns
    ``(run_name, path)`` sorted by run number.
    """
    reaction_dir = Path(reaction_dir)
    found: List[Tuple[int, str, Path]] = []
    for run_dir in reaction_dir.iterdir() if reaction_dir.is_dir() else []:
        m = RUN_DIR_RE.match(run_dir.name)
        if not m or not run_dir.is_dir():
            continue
        for candidate in (run_dir / f"{reaction_dir.name}.out", run_dir / "RatesMC.out"):
            if candidate.exists():
                found.append((int(m.group(2)), run_dir.name, candidate))
                break
    return [(name, path) for _, name, path in sorted(found)]


def iter_reaction_dirs(root: str | Path, reaction: Optional[str] = None) -> Iterable[Path]:
    """Reaction directories under ``root`` (optionally just ``reaction``)."""
    root = Path(root)
    if reaction is not None:
        yield root / reaction
        return
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir():
            yield d
