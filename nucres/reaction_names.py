"""Parse reaction strings such as ``22Mg(a,p)25Al`` and derive file names from them."""

from __future__ import annotations

import re
from typing import NamedTuple

_REACTION_RE = re.compile(r"^\s*([^(\s]+)\(([^,]+),([^)]+)\)\s*(\S+)\s*$")


class Reaction(NamedTuple):
    target: str
    projectile: str
    ejectile: str
    residual: str


def parse_reaction(reaction: str) -> Reaction:
    """
    Split ``target(projectile,ejectile)residual``.

    >>> parse_reaction("22Mg(a,p)25Al")
    Reaction(target='22Mg', projectile='a', ejectile='p', residual='25Al')
    """
    m = _REACTION_RE.match(reaction)
    if not m:
        raise ValueError(f"Cannot parse reaction {reaction!r}; expected e.g. 22Mg(a,p)25Al.")
    return Reaction(*(g.strip() for g in m.groups()))


def file_stem(reaction: str) -> str:
    """
    File-name stem for a reaction: ``22Mg(a,p)25Al`` -> ``22Mg_ap_25Al``,
    ``24Mg(p,g)25Al`` -> ``24Mg_pg_25Al``.
    """
    r = parse_reaction(reaction)
    return f"{r.target}_{r.projectile}{r.ejectile}_{r.residual}"
