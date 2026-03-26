#!/usr/bin/env python
"""
tesseract.py — TESSERACT pipeline driver.

Reads tesseract.in and runs, in sequence:
  Step 1: build_ratesmc_input.py   (generates RatesMC.in files)
  Step 2: generate_cross_sections_parallel.py  (computes σ(E) from that .in file)

At each step, if the expected output already exists the user is warned and
asked whether to overwrite before proceeding.

Usage:
    python tesseract.py [--input tesseract.in]
"""

import argparse
import re
import sys
import subprocess
from pathlib import Path

from nucres.physics import HBAR, MASS_PROTON, reduced_mass
from nucres.sampling import porter_thomas_factors


# ─────────────────────────────────────────────────────────────────────────────
def parse_input_file(path: str) -> dict:
    """Parse a simple key = value file, stripping inline # comments."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            key, _, val = line.partition('=')
            params[key.strip()] = val.strip()
    return params


def ask_overwrite(path: str) -> bool:
    """Warn that *path* exists and ask whether to overwrite. Returns True to proceed."""
    print(f"\n  Warning: output already exists — {path}")
    while True:
        ans = input("  Overwrite and re-run? [y/N]: ").strip().lower()
        if ans in ('y', 'yes'):
            return True
        if ans in ('', 'n', 'no'):
            return False


def compute_gamma(A_tar: int, mean_i: float, mean_o: float):
    """
    Compute entrance/exit channel mean widths and particle masses from the
    target mass number, using the same Wigner-limit fractions as buildratesMC.sh.

    Returns: (gamma_i_mean_eV, gamma_o_mean_eV, m_target_kg, m_alpha_kg)
    """
    A_alpha  = 4
    A_proton = 1

    m_alpha  = A_alpha  * MASS_PROTON
    m_target = A_tar    * MASS_PROTON

    # Entrance channel: alpha + target
    mu_i    = reduced_mass(m_target, m_alpha)
    R_sq_i  = (1.25e-15)**2 * (A_tar**(1/3) + A_alpha**(1/3))**2
    wig_i   = 3 * HBAR**2 / (2 * mu_i * R_sq_i) * 6.242e18
    gamma_i_mean = wig_i * mean_i   # mean_i = 0.010 for alpha channel

    # Exit channel: proton + residual
    mu_o    = reduced_mass(m_target, A_proton * MASS_PROTON)
    R_sq_o  = (1.25e-15)**2 * (A_tar**(1/3) + A_proton**(1/3))**2
    wig_o   = 3 * HBAR**2 / (2 * mu_o * R_sq_o) * 6.242e18
    gamma_o_mean = wig_o * mean_o   # mean_o = 0.0045 for proton channel

    return gamma_i_mean, gamma_o_mean, m_target, m_alpha


# ─────────────────────────────────────────────────────────────────────────────
def step1_build_ratesmc(p: dict) -> bool:
    """
    Run build_ratesmc_input.py for each RUN_N.
    Returns False if all runs were skipped, True if at least one ran.
    """
    reaction   = p['reaction']
    E_min      = p['E_min_mev']
    E_max      = p['E_max_mev']
    input_dir  = p['input_dir']
    output_dir = p['output_dir']
    n_samples  = p['n_samples']
    runs       = p['runs']
    seed       = p['seed']
    mean_i     = p['mean_i']   # Wigner-limit fraction for entrance channel (alpha)
    mean_o     = p['mean_o']   # Wigner-limit fraction for exit channel (proton)

    template = Path(input_dir) / f"{reaction}.txt"
    if not template.exists():
        sys.exit(f"[Step 1] Template not found: {template}")

    # Extract target mass number from reaction string (e.g. "22Mg(a,p)25Al" → 22)
    match = re.match(r'(\d+)', reaction)
    if not match:
        sys.exit(f"[Step 1] Cannot parse mass number from reaction: {reaction}")
    A_tar = int(match.group(1))

    gamma_i_mean, gamma_o_mean, m_target, m_alpha = compute_gamma(
        A_tar, float(mean_i), float(mean_o)
    )
    print(f"[Step 1] gamma_i_mean = {gamma_i_mean:.4e} eV  |  gamma_o_mean = {gamma_o_mean:.4e} eV")
    print(f"[Step 1] m_target = {m_target:.4e} kg  |  m_alpha = {m_alpha:.4e} kg")

    any_ran = False
    for j in range(runs):
        run_dir    = Path(output_dir) / reaction / f"RUN_{j}"
        output_in  = run_dir / f"{reaction}.in"

        if output_in.exists():
            if not ask_overwrite(str(output_in)):
                print(f"  Skipping RUN_{j}.")
                continue

        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "build_ratesmc_input.py",
            "--template",         str(template),
            "--output-dir",       str(output_dir),
            "--E-min-mev",        E_min,
            "--E-max-mev",        E_max,
            "--m1",               str(m_target),
            "--m2",               str(m_alpha),
            "--n-density-points", n_samples,
            "--Gamma-i-mean-eV",  str(gamma_i_mean),
            "--Gamma-o-mean-eV",  str(gamma_o_mean),
            "--seed",             seed,
        ]

        print(f"\n[Step 1] Building RatesMC input — {reaction} / RUN_{j} ...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"[Step 1] build_ratesmc_input.py failed for RUN_{j}.")

        # Mirror buildratesMC.sh: move the file into the RUN_N subdirectory
        generated = Path(output_dir) / reaction / f"{reaction}.in"
        if generated.exists() and generated != output_in:
            generated.rename(output_in)

        any_ran = True

    return any_ran


def step2_generate_xs(p: dict) -> None:
    """Run generate_cross_sections_parallel.py."""
    reaction = p['reaction']
    tag      = p['tag']
    dE_lst   = [x.strip() for x in p['dE'].split(",")]

    # ── Check unintegrated output ────────────────────────────────
    unint_file   = Path(f"{reaction}_xs_unintegrated_parallel{tag}.txt")
    skip_unint   = False
    if unint_file.exists():
        if not ask_overwrite(str(unint_file)):
            skip_unint = True
            print("  Keeping existing unintegrated cross sections.")

    # ── Check integrated outputs ─────────────────────────────────
    rxn_match  = re.match(r'([^(]+)\(a,p\)(.+)', reaction)
    target     = rxn_match.group(1) if rxn_match else reaction
    residual   = rxn_match.group(2) if rxn_match else ""
    tag_clean  = tag.lstrip('_')
    tag_suffix = f"_{tag_clean}" if tag_clean else ""
    int_files  = [Path(f"{target}_ap_{residual}_integrated_xs_dE_{dE}{tag_suffix}.csv")
                  for dE in dE_lst]
    existing_int = [f for f in int_files if f.exists()]
    skip_int     = False
    if existing_int:
        print(f"\n  Warning: {len(existing_int)} integrated output file(s) already exist:")
        for f in existing_int:
            print(f"    {f}")
        if not ask_overwrite("integrated cross section files"):
            skip_int = True
            print("  Keeping existing integrated cross sections.")

    # ── Nothing to do ────────────────────────────────────────────
    if skip_unint and skip_int:
        print("  All Step 2 outputs already exist and were kept. Nothing to run.")
        return

    cmd = [
        sys.executable, "generate_cross_sections_parallel.py",
        "--reaction",  reaction,
        "--E-min-mev", p['E_min_mev'],
        "--E-max-mev", p['E_max_mev'],
        "--dE",            p['dE'],
        "--n-grid-points", p['n_grid_points'],
        "--nproc",         p['nproc'],
        "--mean-i",    p['mean_i'],
        "--mean-o",    p['mean_o'],
        "--tag",       p['tag'],
    ]
    if skip_unint:
        cmd.append("--skip-unintegrated")
    if skip_int:
        cmd.append("--skip-integrated")

    print(f"\n[Step 2] Computing cross sections — {reaction} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("[Step 2] generate_cross_sections_parallel.py failed.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TESSERACT pipeline driver")
    parser.add_argument("--input", default="tesseract.in",
                        help="Path to the input file (default: tesseract.in)")
    args = parser.parse_args()

    raw = parse_input_file(args.input)

    # Normalise to the types each step needs (keep as strings for subprocess)
    required = ('reaction', 'dE')
    for key in required:
        if key not in raw:
            sys.exit(f"Missing required key '{key}' in {args.input}")

    p = {
        'reaction':   raw['reaction'],
        'E_min_mev':  raw.get('E_min_mev',  '0.1'),
        'E_max_mev':  raw.get('E_max_mev',  '10.0'),
        'mean_i':     raw['mean_i'],
        'mean_o':     raw['mean_o'],
        'input_dir':  raw.get('input_dir',  'input/'),
        'output_dir': raw.get('output_dir', 'outputs/'),
        'n_samples':  raw.get('n_samples',  '5000'),
        'runs':       int(raw.get('runs',   '1')),
        'seed':       raw.get('seed',       '42'),
        'dE':         raw['dE'],
        'n_grid_points': raw.get('n_grid_points', '10000'),
        'nproc':      raw.get('nproc',      '1'),
        'tag':        raw.get('tag',        ''),
    }

    print(f"TESSERACT pipeline  —  reaction: {p['reaction']}")
    print(f"  E_min = {p['E_min_mev']} MeV  |  E_max = {p['E_max_mev']} MeV")
    print(f"  dE bins = {p['dE']}  |  n_grid_points = {p['n_grid_points']}  |  nproc = {p['nproc']}")
    print("=" * 60)

    step1_build_ratesmc(p)
    step2_generate_xs(p)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
