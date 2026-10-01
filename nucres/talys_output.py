"""Helpers for the shared talys_optimization.out log written by talys_opt_with_unc.py."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

_INPUT_RE = re.compile(r"Input file\s*:\s*(.+)")
_RUN_RE = re.compile(r"Run index\s*:\s*(\S+)")
_BIN_RE = re.compile(r"Bin width\s*:\s*(.+)")
_REACTION_RE = re.compile(r"Reaction\s*:\s*(.+)")


def fit_key(block: str) -> tuple:
    """
    Identity of the fit a talys_optimization.out block belongs to.

    The input (integrated cross-section) file names one fit: its run index
    and bin width are in the name. Older blocks without it fall back to
    (reaction, run index, bin width).
    """
    m = _INPUT_RE.search(block)
    if m and m.group(1).strip():
        return ("input", m.group(1).strip())
    parts = []
    for rx in (_REACTION_RE, _RUN_RE, _BIN_RE):
        mm = rx.search(block)
        parts.append(mm.group(1).strip() if mm else "")
    return ("fields", *parts)


def keep_latest(records: Iterable[Dict], key: str = "fit_key") -> List[Dict]:
    """
    One record per fit: the last one in file order (talys_optimization.out
    is append-only, so the last block for a fit is its most recent result).
    Re-runs and resumed jobs append further blocks for the same fit; without
    this they were counted several times in percentiles and grid statistics.
    Order follows each fit's first appearance.
    """
    latest: Dict[tuple, Dict] = {}
    order: List[tuple] = []
    for rec in records:
        k = rec[key]
        if k not in latest:
            order.append(k)
        latest[k] = rec
    return [latest[k] for k in order]
