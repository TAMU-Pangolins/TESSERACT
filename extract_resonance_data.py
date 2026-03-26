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


def extract_data(file):
    start_marker = 'Resonant Contribution'
    end_marker = 'Upper Limits of Resonances'

    extracted_lines = []
    capture = False

    with open(file, 'r') as f:
        for line in f:
            if start_marker in line:
                capture = True
                skip_count = 0
                continue

            if end_marker in line and capture:
                break

            if capture:
                if skip_count < 2:
                    skip_count += 1
                    continue
                extracted_lines.append(line)

    if extracted_lines:
        extracted_lines.pop()

    block = "".join(extracted_lines)
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