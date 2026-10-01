import re

import pandas as pd
from io import StringIO
import os


def load_nuclear_params(infile):
    """
    Read Z_proj, A_proj (projectile) and Z_tar, A_tar (target) from the
    header of a RatesMC input file.

    Looks for lines of the form:
        2  ! Zproj
       12  ! Ztarget
        4.003 ! Aproj
       22  ! Atarget

    Returns a dict with integer keys: Z_proj, A_proj, Z_tar, A_tar.
    A values are rounded to the nearest integer (mass number).
    """
    params = {}
    key_map = {
        'Zproj':   ('Z_proj', int),
        'Ztarget': ('Z_tar',  int),
        'Aproj':   ('A_proj', lambda x: int(round(float(x)))),
        'Atarget': ('A_tar',  lambda x: int(round(float(x)))),
    }

    with open(infile, 'r') as f:
        for line in f:
            comment_idx = line.find('!')
            if comment_idx == -1:
                continue
            comment = line[comment_idx + 1:].strip()
            # Match keyword at start of comment (e.g. "Ztarget", "Aproj")
            for keyword, (dest, cast) in key_map.items():
                if re.match(rf'\b{keyword}\b', comment, re.IGNORECASE):
                    val_str = line[:comment_idx].strip().split()[0]
                    try:
                        params[dest] = cast(val_str)
                    except (ValueError, IndexError):
                        pass
                    break
            if len(params) == 4:
                break

    missing = [k for k in ('Z_proj', 'A_proj', 'Z_tar', 'A_tar') if k not in params]
    if missing:
        raise ValueError(
            f"Could not parse nuclear parameters from {infile}: missing {missing}"
        )
    return params


_REACTION_KEYS = [
    # (comment prefix, field, cast)
    ("Zproj", "Z_proj", lambda v: int(round(float(v)))),
    ("Ztarget", "Z_targ", lambda v: int(round(float(v)))),
    ("Zexitparticle", "Z_exit", lambda v: int(round(float(v)))),
    ("Aproj", "M_proj", float),
    ("Atarget", "M_targ", float),
    ("Aexitparticle", "M_exit", float),
    ("Jproj", "J_proj", float),
    ("Jtarget", "J_targ", float),
    ("projectile separation energy", "S_proj_mev", lambda v: float(v) * 1e-3),
    ("exit particle separation energy", "S_exit_mev", lambda v: float(v) * 1e-3),
    ("Radius parameter", "R0_fm", float),
    ("Gamma-ray channel number", "gamma_channel", lambda v: int(round(float(v)))),
]


def load_reaction_params(infile):
    """
    Reaction header of a RatesMC input as nucres.resonance_sum.ReactionParams.

    Values are taken as RatesMC reads them: masses in u (not rounded),
    separation energies converted from keV to MeV.
    """
    from nucres.resonance_sum import ReactionParams

    values = {}
    with open(infile, 'r') as f:
        for line in f:
            if '!' not in line:
                continue
            value, comment = line.split('!', 1)
            comment = comment.strip()
            tokens = value.split()
            if not tokens:
                continue
            for prefix, field, cast in _REACTION_KEYS:
                if field not in values and comment.lower().startswith(prefix.lower()):
                    try:
                        values[field] = cast(tokens[0])
                    except ValueError as exc:
                        raise ValueError(
                            f"{infile}: cannot read {prefix!r} from {tokens[0]!r} "
                            "(nuclide names are not supported; give numbers)."
                        ) from exc
                    break
            if len(values) == len(_REACTION_KEYS):
                break
    missing = [field for _, field, _ in _REACTION_KEYS if field not in values]
    if missing:
        raise ValueError(f"{infile}: missing reaction header entries {missing}")
    return ReactionParams(**values)


def extract_data(file):
    """
    Read the Resonant Contribution table of a RatesMC input as a DataFrame.

    The section starts at a line beginning with "Resonant Contribution"
    (so "Non-Resonant Contribution" is not mistaken for it), the table at the
    following line beginning with "Ecm", and it ends at the next "*" divider
    or the Upper Limits section. Commented rows ("!") are skipped.
    """
    with open(file, 'r') as f:
        lines = f.read().splitlines()

    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().lower().startswith('resonant contribution')), None)
    if start is None:
        raise ValueError(f"{file}: no 'Resonant Contribution' section.")
    header = next((j for j in range(start + 1, len(lines))
                   if lines[j].strip().startswith('Ecm')), None)
    if header is None:
        raise ValueError(f"{file}: no 'Ecm' header in the Resonant Contribution section.")

    rows = []
    for ln in lines[header + 1:]:
        s = ln.strip()
        if s.startswith('*') or s.lower().startswith('upper limits'):
            break
        if not s or s.startswith('!'):
            continue
        rows.append(s)

    block = "\n".join([lines[header].strip()] + rows)
    df = pd.read_csv(StringIO(block), sep=r'\s+', header=0)
    return df


def load_resonance_data(infile):

    df = extract_data(infile)
    df = df.sort_values(by='Ecm', ascending=True)

    E_cm = df['Ecm'].values * 1e-3
    g1 = df['G1'].values
    g2 = df['G2'].values
    g3 = df['G3'].values
    Jr = df['Jr'].values
    l1 = df['L1'].values
    l2 = df['L2'].values

    L = l1 + l2
    G = g1 + g2 + g3

    return E_cm, g1, g2, g3, Jr, l1, l2, L, G