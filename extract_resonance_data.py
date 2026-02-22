import pandas as pd
from io import StringIO
import os


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