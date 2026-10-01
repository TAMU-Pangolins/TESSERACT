#!/usr/bin/env python3
"""
Convert old-format RatesMC templates to the format both RatesMC 2.11 and the
2.2+ rewrite read (the format of input/22Mg(a,p)25Al.txt).

Old -> current:
  * add the "(Res, E)" correlation line after the rate-output line (0 0);
  * add a Corr/Frac column (0) to the Resonant Contribution rows;
  * add a DPT column (0) after each PT in the Upper Limits and Interference
    rows.
These values reproduce the old behaviour (no correlations, no PT uncertainty).

Rows whose numbers were split by stray spaces (e.g. "0. 0", "6276 .2",
"5.0 e-3", "1 779.0"; apparently from copying tables out of a PDF) have one
token too many, which shifts every later column when RatesMC reads the row.
Splits of the form "x. y", "x .y" and "x e-3" are merged automatically.
Anything else is merged only if listed in MANUAL_MERGES below, after checking
it by hand; otherwise the file is left unconverted and reported.

Usage:
  python scripts/convert_ratesmc_templates.py input/*.txt          # dry run
  python scripts/convert_ratesmc_templates.py --write input/*.txt  # convert in place
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CORRELATION_LINE = "0 0 ! (Res, E) =0 for no correlation. =1 for correlation"

# Expected column counts (old, new) and where DPT is inserted (after these
# old-format PT indices).
SECTIONS = {
    "resonant contribution": {"old": 16, "new": 17},
    "upper limits": {"old": 17, "new": 20, "pt": (6, 10, 14)},
    "interference": {"old": 16, "new": 19, "pt": (6, 10, 14)},
}

UL_HEADER = "Ecm DEcm Jr G1 DG1 L1 PT DPT G2 DG2 L2 PT DPT G3 DG3 L3 PT DPT Exf Int"
INTF_HEADER = "Ecm DEcm Jr G1 DG1 L1 PT DPT G2 DG2 L2 PT DPT G3 DG3 L3 PT DPT Exf"

# (file name, original row) -> repaired row, for splits that are not
# self-evident. Each was checked against the final-state energies (Exf) of
# the residual nucleus.
MANUAL_MERGES: Dict[Tuple[str, str], str] = {
    # Exf = 1369 keV: first 2+ state of 24Mg
    ("20Ne(a,g)24Mg.txt",
     "215.93 0.10 2.0 1.08e-20 0.0 2 0.01 0.060 0.027 2 0 0.0 0.0 0 0 13 69.0 1"):
     "215.93 0.10 2.0 1.08e-20 0.0 2 0.01 0.060 0.027 2 0 0.0 0.0 0 0 1369.0 1",
    ("20Ne(a,g)24Mg.txt",
     "2653.1 1.0 0.0 0.0 0 2.4e+3 500.0 2 0.13 0.02 2 0.046 0.010 0 136 9.0 1"):
     "2653.1 1.0 0.0 0.0 0 2.4e+3 500.0 2 0.13 0.02 2 0.046 0.010 0 1369.0 1",
    # Exf = 1616 keV: 3/2- state of 19Ne
    ("18F(p,g)19Ne.txt",
     "287.0 6.0 2.5 2.4e-2 0.0 2 0.0045 0.29 0.15 1 0 1.2e3 0.3e3 3 0 16 16.0 1"):
     "287.0 6.0 2.5 2.4e-2 0.0 2 0.0045 0.29 0.15 1 0 1.2e3 0.3e3 3 0 1616.0 1",
    # Exf = 1779 keV: first 2+ state of 28Si
    ("24Mg(a,g)28Si.txt",
     "3655.0 1.0 0.0 0.0 2 656.0 117.0 2 0.16 6.0e-2 1 5.04e3 504.0 0 1 779.0 1"):
     "3655.0 1.0 0.0 0.0 2 656.0 117.0 2 0.16 6.0e-2 1 5.04e3 504.0 0 1779.0 1",
    ("24Mg(a,g)28Si.txt",
     "3888.8 1.2 0.0 0.0 3 2.35e3 868.0 3 5.63 2.21 1 4.75e3 1.45e3 1 1 779.0 1"):
     "3888.8 1.2 0.0 0.0 3 2.35e3 868.0 3 5.63 2.21 1 4.75e3 1.45e3 1 1779.0 1",
}

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _is_num(tok: str) -> bool:
    return bool(_NUM.match(tok))


def repair_split(tokens: List[str]) -> Optional[List[str]]:
    """Merge one self-evident split ("x. y", "x .y", "x e-3"); None if none found."""
    candidates = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        merged = a + b
        if not _is_num(merged):
            continue
        evident = (
            (a.endswith(".") and b[:1].isdigit())
            or (b.startswith(".") and a.replace("-", "").isdigit())
            or (b[:1] in "eE" and _is_num(a))
        )
        if evident:
            candidates.append(i)
    if len(candidates) != 1:
        return None
    i = candidates[0]
    return tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2:]


def _section_of(line: str) -> Optional[str]:
    low = line.strip().lower()
    for key in SECTIONS:
        if low.startswith(key):
            return key
    return None


def convert(path: Path) -> Tuple[Optional[List[str]], List[str]]:
    """Return (new lines or None if already current / not convertible, messages)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    msgs: List[str] = []
    if any("correlation" in ln.lower() for ln in lines[:25]):
        return None, ["already in the current format"]

    out: List[str] = []
    section: Optional[str] = None
    header_seen = False
    failed = False
    inserted_corr = False
    for ln in lines:
        stripped = ln.strip()

        if not inserted_corr and "rate output at all temperatures" in ln:
            out.append(ln)
            out.append(CORRELATION_LINE)
            inserted_corr = True
            continue

        sec = _section_of(ln)
        if sec is not None:
            section, header_seen = sec, False
            out.append(ln)
            continue
        if section is None or stripped.startswith("*"):
            section = None if stripped.startswith("*") else section
            out.append(ln)
            continue

        spec = SECTIONS[section]
        if not header_seen:
            if stripped.lower().startswith("ecm"):
                header_seen = True
                if section == "resonant contribution":
                    out.append(stripped + " Corr/Frac")
                elif section == "upper limits":
                    out.append(UL_HEADER)
                else:
                    out.append(INTF_HEADER)
            else:
                out.append(ln)  # Note: lines
            continue

        if not stripped or stripped.lstrip("!").strip() in ("+-", "+", "-"):
            out.append(ln)  # blank lines and interference sign markers
            continue

        comment = stripped.startswith("!")
        body = stripped.lstrip("!").strip()
        tokens = body.split()

        if len(tokens) > spec["old"] and all(float(t) == 0.0 for t in tokens if _is_num(t)) \
                and all(_is_num(t) for t in tokens):
            # All-zero placeholder row with surplus zeros: nothing to shift.
            msgs.append(f"[{section}] trimmed all-zero placeholder row from {len(tokens)} tokens")
            tokens = tokens[: spec["old"]]
        elif len(tokens) == spec["old"] + 1:
            key = (path.name, body)
            if key in MANUAL_MERGES:
                fixed = MANUAL_MERGES[key].split()
                msgs.append(f"[{section}] manual repair:\n      {body}\n   -> {' '.join(fixed)}")
            else:
                fixed = repair_split(tokens)
                if fixed is None:
                    msgs.append(f"[{section}] UNREPAIRED ({len(tokens)} tokens): {body}")
                    failed = True
                    out.append(ln)
                    continue
                msgs.append(f"[{section}] repaired: {body}\n   -> {' '.join(fixed)}")
            tokens = fixed

        if len(tokens) != spec["old"]:
            msgs.append(f"[{section}] UNEXPECTED {len(tokens)} tokens (want {spec['old']}): {body}")
            failed = True
            out.append(ln)
            continue

        if section == "resonant contribution":
            tokens = tokens + ["0"]
        else:
            for idx in sorted(spec["pt"], reverse=True):
                tokens.insert(idx + 1, "0")
        assert len(tokens) == spec["new"]
        out.append(("!" if comment else "") + " ".join(tokens))

    if not inserted_corr:
        msgs.append("UNEXPECTED: no 'rate output at all temperatures' line")
        failed = True
    return (None if failed else out), msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("templates", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true", help="Rewrite files in place.")
    args = ap.parse_args()

    n_conv = n_fail = 0
    for path in args.templates:
        new, msgs = convert(path)
        if new is None and msgs == ["already in the current format"]:
            continue
        status = "FAILED" if new is None else ("converted" if args.write else "would convert")
        print(f"{path.name}: {status}")
        for m in msgs:
            print("   " + m)
        if new is None:
            n_fail += 1
            continue
        n_conv += 1
        if args.write:
            path.write_text("\n".join(new) + "\n", encoding="utf-8")
    print(f"\n{n_conv} converted{'' if args.write else ' (dry run)'}, {n_fail} failed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
