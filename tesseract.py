#!/usr/bin/env python
"""
tesseract.py — TESSERACT pipeline driver.

Reads tesseract.in and runs any combination of:
  Step 1  [resonance]   — build_ratesmc_input.py
  Step 2  [integration] — generate_cross_sections_parallel.py
  Step 3  [talys]       — talys_opt_with_unc.py

Rules:
  • [basics]  is mandatory.
  • Every other section is optional.
  • If a section IS present  → its step is executed for all runs.
  • If a section is ABSENT   → its outputs are assumed to already exist.
      The pipeline checks for those output files before proceeding and
      exits with a clear error if any are missing.
  • All skip/resume decisions are file-existence checks — no interactive
      prompts, safe for nohup / background execution.

Output naming with multiple runs (runs = N):
  [resonance]   outputs/{reaction}/RUN_{j}/{reaction}.in          j = 0..N-1
  [integration] {target}_ap_{residual}_integrated_xs_dE_{dE}      one per (j, dE)
                  _{tag}_run{j}.csv
  [talys]       talys_results_{exp_file_stem}.npz                 one per file

Usage:
    python tesseract.py [--input tesseract.in]
    nohup python tesseract.py --input tesseract.in &
"""

import argparse
import re
import sys
import subprocess
from pathlib import Path

from nucres.physics import HBAR, MASS_PROTON, reduced_mass


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_input_file(path: str) -> dict:
    """
    Parse a sectioned INI-style input file into a nested {section → {key: value}} dict.
    Lines between \\opt markers in [talys] are stored under talys['_opt_lines'].
    Pairs before any section header go into '_global'.
    """
    sections: dict = {'_global': {}}
    current: str = '_global'
    in_opt: bool = False
    opt_lines: list = []

    with open(path) as fh:
        for raw_line in fh:
            raw      = raw_line.rstrip('\n')
            stripped = raw.strip()

            if stripped.startswith(r'\opt'):
                if not in_opt:
                    in_opt    = True
                    opt_lines = []
                else:
                    in_opt = False
                    sections.setdefault('talys', {})['_opt_lines'] = list(opt_lines)
                continue

            if in_opt:
                content = raw.split('#')[0].rstrip()
                if content.strip():
                    opt_lines.append(content.strip())
                continue

            line = raw.split('#')[0].strip()
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1].strip().lower()
                sections.setdefault(current, {})
                continue
            if not line or '=' not in line:
                continue
            key, _, val = line.partition('=')
            sections[current][key.strip()] = val.strip()

    return sections


def _bool(val, default: bool) -> bool:
    if val is None:
        return default
    return str(val).lower() in ('true', '1', 'yes')


def _banner(label: str, kind: str = "start") -> None:
    """Print a section-boundary banner to stdout (flush=True for nohup logs)."""
    width = 60
    if kind == "start":
        print(f"\n{'─' * width}", flush=True)
        print(f"  >> Starting [{label}]", flush=True)
        print(f"{'─' * width}", flush=True)
    else:
        print(f"{'─' * width}", flush=True)
        print(f"  << [{label}] finished", flush=True)
        print(f"{'─' * width}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output-path helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resonance_output_paths(reaction: str, output_dir: str, runs: int):
    return [
        Path(output_dir) / reaction / f"RUN_{j}" / f"{reaction}.in"
        for j in range(runs)
    ]


def _integrated_output_paths(reaction: str, dE_list: list, tag: str, runs: int):
    """
    Return all integrated XS file paths that [integration] will produce.
    One file per (run index, bin width).
    """
    rxn = re.match(r'([^(]+)\(a,p\)(.+)', reaction)
    target   = rxn.group(1) if rxn else reaction
    residual = rxn.group(2) if rxn else ""
    paths = []
    for j in range(runs):
        tag_j      = f"{tag}_run{j}"
        tag_suffix = f"_{tag_j.lstrip('_')}" if tag_j.lstrip('_') else ""
        for dE in dE_list:
            paths.append(
                Path(f"{target}_ap_{residual}_integrated_xs_dE_{dE}{tag_suffix}.csv")
            )
    return paths


def _talys_npz_path(exp_file: Path) -> Path:
    return Path(f"talys_results_{exp_file.stem}.npz")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks (called when a section is absent)
# ─────────────────────────────────────────────────────────────────────────────
def _check_resonance_outputs(reaction: str, output_dir: str, runs: int,
                              next_step: str,
                              run_idx: int = None) -> None:
    all_paths = (_resonance_output_paths(reaction, output_dir, runs)
                 if run_idx is None
                 else [Path(output_dir) / reaction / f"RUN_{run_idx}" / f"{reaction}.in"])
    missing = [p for p in all_paths if not p.exists()]
    if missing:
        sys.exit(
            f"\n[{next_step}] requires [resonance] outputs that are missing "
            f"(and [resonance] section is absent):\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nEither add a [resonance] section or ensure these files exist."
        )


def _check_integration_outputs(exp_file_str: str) -> None:
    """Check that the exp_file needed by [talys] exists."""
    if not exp_file_str:
        sys.exit(
            "\n[talys] requires an exp_file but none is set in [talys].\n"
            "Either add an [integration] section or set  exp_file = <path>  in [talys]."
        )
    if not Path(exp_file_str).exists():
        sys.exit(
            f"\n[talys] requires [integration] output that is missing "
            f"(and [integration] section is absent):\n  {exp_file_str}\n"
            "Either add an [integration] section or ensure this file exists."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Physics helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_gamma(A_tar: int, mean_i: float, mean_o: float):
    A_alpha  = 4
    A_proton = 1
    m_alpha  = A_alpha  * MASS_PROTON
    m_target = A_tar    * MASS_PROTON

    mu_i   = reduced_mass(m_target, m_alpha)
    R_sq_i = (1.25e-15)**2 * (A_tar**(1/3) + A_alpha**(1/3))**2
    wig_i  = 3 * HBAR**2 / (2 * mu_i * R_sq_i) * 6.242e18
    gamma_i = wig_i * mean_i

    mu_o   = reduced_mass(m_target, A_proton * MASS_PROTON)
    R_sq_o = (1.25e-15)**2 * (A_tar**(1/3) + A_proton**(1/3))**2
    wig_o  = 3 * HBAR**2 / (2 * mu_o * R_sq_o) * 6.242e18
    gamma_o = wig_o * mean_o

    return gamma_i, gamma_o, m_target, m_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — [resonance]
# ─────────────────────────────────────────────────────────────────────────────
def step1_build_ratesmc(basics: dict, resonance: dict,
                        run_idx: int = None) -> None:
    """
    Run build_ratesmc_input.py for each RUN_j.
    Skips a run silently if its .in file already exists.
    If run_idx is given, only process that single run.
    """
    reaction   = basics['reaction']
    E_min      = basics.get('E_min_mev',  '0.1')
    E_max      = basics.get('E_max_mev',  '10.0')
    input_dir  = resonance.get('input_dir',  'input/')
    output_dir = resonance.get('output_dir', 'outputs/')
    n_samples  = resonance.get('samples',    '5000')
    runs       = int(resonance.get('runs',   '1'))
    seed       = resonance.get('seed',       '42')
    mean_i     = resonance.get('mean_i',     '0.010')
    mean_o     = resonance.get('mean_o',     '0.0045')

    template = Path(input_dir) / f"{reaction}.txt"
    if not template.exists():
        sys.exit(f"[resonance] Template not found: {template}")

    match = re.match(r'(\d+)', reaction)
    if not match:
        sys.exit(f"[resonance] Cannot parse mass number from reaction: {reaction}")
    A_tar = int(match.group(1))

    gamma_i, gamma_o, m_target, m_alpha = compute_gamma(
        A_tar, float(mean_i), float(mean_o)
    )
    print(f"[resonance] gamma_i = {gamma_i:.4e} eV  |  gamma_o = {gamma_o:.4e} eV")
    run_range = [run_idx] if run_idx is not None else range(runs)
    print(f"[resonance] processing run(s): {list(run_range)}")

    n_skipped = 0
    for j in run_range:
        run_dir   = Path(output_dir) / reaction / f"RUN_{j}"
        output_in = run_dir / f"{reaction}.in"

        if output_in.exists():
            n_skipped += 1
            continue                        # already done — no prompt, just skip

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
            "--Gamma-i-mean-eV",  str(gamma_i),
            "--Gamma-o-mean-eV",  str(gamma_o),
            "--seed",             seed,
            "--l1",               resonance.get('l1',               '0'),
            "--l2",               resonance.get('l2',               '1'),
            "--l3",               resonance.get('l3',               '0'),
            "--pi",               resonance.get('pi',               '1'),
            "--delta-E-mev",      resonance.get('delta_E_mev',      '0.05'),
            "--n-sigma-points",   resonance.get('n_sigma_points',   '4000'),
            "--U-offset-mev",     resonance.get('U_offset_mev',     '8.0'),
            "--default-frac-unc", resonance.get('default_frac_unc', '0.001'),
            "--int-flag",         resonance.get('int_flag',         '1'),
            "--precision",        resonance.get('precision',        '3'),
            "--n-random-samples", resonance.get('n_random_samples', '1'),
        ]
        if _bool(resonance.get('use_strength'), False):
            cmd.append("--use-strength")
        if not _bool(resonance.get('sample_J'), True):
            cmd.append("--no-sample-J")
        if not _bool(resonance.get('auto_l1'), True):
            cmd.append("--no-auto-l1")

        print(f"[resonance] RUN_{j} ...", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"[resonance] build_ratesmc_input.py failed for RUN_{j}.")

        generated = Path(output_dir) / reaction / f"{reaction}.in"
        if generated.exists() and generated != output_in:
            generated.rename(output_in)

    if n_skipped:
        print(f"[resonance] {n_skipped}/{runs} run(s) already existed — skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — [integration]
# ─────────────────────────────────────────────────────────────────────────────
def step2_generate_xs(basics: dict, integration: dict, runs: int,
                      run_idx: int = None) -> None:
    """
    Run generate_cross_sections_parallel.py once per run index j = 0..runs-1.
    If run_idx is given, only process that single run.

    Output filenames encode the run index via --tag:
      unintegrated : {reaction}_xs_unintegrated_parallel_{base_tag}_run{j}.txt
      integrated   : {target}_ap_{residual}_integrated_xs_dE_{dE}_{base_tag}_run{j}.csv

    Skips a (run, file) silently if the outputs already exist.
    """
    reaction = basics['reaction']
    E_min    = basics.get('E_min_mev', '0.1')
    E_max    = basics.get('E_max_mev', '10.0')
    dE       = integration['dE']
    base_tag = integration.get('tag', '')          # e.g. "_test"
    dE_lst   = [x.strip() for x in dE.split(',')]

    rxn = re.match(r'([^(]+)\(a,p\)(.+)', reaction)
    target   = rxn.group(1) if rxn else reaction
    residual = rxn.group(2) if rxn else ""

    run_range = [run_idx] if run_idx is not None else range(runs)
    n_skipped = 0
    for j in run_range:
        tag_j       = f"{base_tag}_run{j}"           # e.g. "_test_run0"
        tag_clean_j = tag_j.lstrip('_')              # e.g.  "test_run0"
        tag_suffix  = f"_{tag_clean_j}" if tag_clean_j else ""

        # All expected outputs for this run
        unint = Path(f"{reaction}_xs_unintegrated_parallel{tag_j}.txt")
        int_files = [
            Path(f"{target}_ap_{residual}_integrated_xs_dE_{dE}{tag_suffix}.csv")
            for dE in dE_lst
        ]
        all_exist = unint.exists() and all(f.exists() for f in int_files)
        if all_exist:
            n_skipped += 1
            continue

        cmd = [
            sys.executable, "generate_cross_sections_parallel.py",
            "--reaction",      reaction,
            "--E-min-mev",     E_min,
            "--E-max-mev",     E_max,
            "--dE",            dE,
            "--n-grid-points", integration.get('n_grid_points', '10000'),
            "--nproc",         integration.get('nproc',         '1'),
            "--tag",           tag_j,
            "--run-idx",       str(j),
        ]
        # Skip sub-steps whose files already exist
        if unint.exists():
            cmd.append("--skip-unintegrated")
        existing_int = [f for f in int_files if f.exists()]
        if len(existing_int) == len(int_files):
            cmd.append("--skip-integrated")

        print(f"[integration] run {j}/{runs-1} ...", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"[integration] generate_cross_sections_parallel.py failed for run {j}.")

    if n_skipped:
        print(f"[integration] {n_skipped}/{runs} run(s) already existed — skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — [talys]
# ─────────────────────────────────────────────────────────────────────────────
def step3_talys_opt(talys: dict, exp_files: list, input_path: str,
                    run_idx: int = None) -> None:
    """
    Invoke talys_opt_with_unc.py once for every file in *exp_files*.
    Skips a file silently if its results .npz already exists.
    If run_idx is given, only process files whose name contains _run{run_idx}.

    exp_files : list of Path — the integrated XS files to fit
    """
    if run_idx is not None:
        exp_files = [f for f in exp_files if f"_run{run_idx}" in f.stem]
        if not exp_files:
            sys.exit(f"[talys] No exp_files found matching run index {run_idx}.")
    default_script = Path(__file__).resolve().parent / "talys_opt_with_unc.py"
    script_path    = Path(talys.get('talys_script', str(default_script)))

    if not script_path.exists():
        sys.exit(
            f"\n[talys] Optimisation script not found: {script_path}\n"
            "Set  talys_script = <path>  in the [talys] section of tesseract.in."
        )

    n_skipped = 0
    for exp_file in exp_files:
        npz = _talys_npz_path(exp_file)
        if npz.exists():
            n_skipped += 1
            continue

        print(f"[talys] Fitting {exp_file.name} ...", flush=True)
        cmd = [
            sys.executable, str(script_path),
            "--input",    input_path,
            "--exp-file", str(exp_file),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            # Log and continue — one failed fit should not abort the whole loop
            print(f"[talys] WARNING: optimisation failed for {exp_file.name}.", flush=True)

    if n_skipped:
        print(f"[talys] {n_skipped}/{len(exp_files)} file(s) already had results — skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TESSERACT pipeline driver")
    parser.add_argument("--input", default="tesseract.in",
                        help="Path to the input file (default: tesseract.in)")
    parser.add_argument("--run-idx", dest="run_idx", type=int, default=None,
                        help="Process only this run index (0-based). "
                             "If omitted, all runs are processed (nohup mode). "
                             "Use $(Process) in condor submit files.")
    args = parser.parse_args()

    raw = parse_input_file(args.input)

    # ── [basics] is mandatory ─────────────────────────────────────────────────
    basics = raw.get('basics', {})
    if not basics:
        sys.exit(f"[basics] section is mandatory but not found in {args.input}")
    reaction = basics.get('reaction')
    if not reaction:
        sys.exit(f"Missing 'reaction' in [basics] section of {args.input}")

    # ── Detect which sections are present ─────────────────────────────────────
    has_resonance   = 'resonance'   in raw
    has_integration = 'integration' in raw
    has_talys       = 'talys'       in raw

    resonance   = raw.get('resonance',   {})
    integration = raw.get('integration', {})
    talys       = raw.get('talys',       {})

    # output_dir and runs are needed for path checks even when [resonance] absent
    output_dir = resonance.get('output_dir', 'outputs/')
    runs       = int(resonance.get('runs',   '1'))

    STEPS   = ('resonance', 'integration', 'talys')
    active  = [s for s in STEPS if s in raw]
    skipped = [s for s in STEPS if s not in raw]

    run_idx = args.run_idx

    print(f"TESSERACT  —  reaction: {reaction}")
    print(f"  E_min = {basics.get('E_min_mev','0.1')} MeV  |  "
          f"E_max = {basics.get('E_max_mev','10.0')} MeV  |  runs = {runs}")
    if run_idx is not None:
        print(f"  Run index : {run_idx}  (condor mode — single run)")
    else:
        print(f"  Run index : all 0..{runs-1}  (nohup mode)")
    print(f"  Active  sections : {active or '(none)'}")
    if skipped:
        print(f"  Skipped sections : {skipped}  (outputs assumed to exist)")
    print("=" * 60)

    if not active:
        print("No pipeline sections found — nothing to run.")
        return

    # ── Step 1: [resonance] ───────────────────────────────────────────────────
    if has_resonance:
        _banner("resonance", "start")
        step1_build_ratesmc(basics, resonance, run_idx=run_idx)
        _banner("resonance", "end")
    elif has_integration or has_talys:
        _check_resonance_outputs(reaction, output_dir, runs,
                                 next_step='integration', run_idx=run_idx)

    # ── Step 2: [integration] ─────────────────────────────────────────────────
    if has_integration:
        if 'dE' not in integration:
            sys.exit("Missing 'dE' in [integration] section.")
        _banner("integration", "start")
        step2_generate_xs(basics, integration, runs, run_idx=run_idx)
        _banner("integration", "end")
    elif has_talys:
        _check_integration_outputs(talys.get('exp_file', ''))

    # ── Build the list of integrated XS files for step 3 ─────────────────────
    # When [integration] was active, enumerate every (run, dE) output.
    # When [integration] was absent, use the single exp_file from [talys].
    if has_talys:
        if has_integration:
            dE_lst   = [x.strip() for x in integration['dE'].split(',')]
            base_tag = integration.get('tag', '')
            exp_files = _integrated_output_paths(reaction, dE_lst, base_tag, runs)
            # Sanity-check: files for this run must exist before feeding to TALYS
            check_files = (
                [f for f in exp_files if f"_run{run_idx}" in f.stem]
                if run_idx is not None else exp_files
            )
            missing = [p for p in check_files if not p.exists()]
            if missing:
                sys.exit(
                    "[talys] Some integrated XS files are unexpectedly missing:\n"
                    + "\n".join(f"  {p}" for p in missing)
                )
        else:
            exp_files = [Path(talys['exp_file'])]

        _banner("talys", "start")
        step3_talys_opt(talys, exp_files, args.input, run_idx=run_idx)
        _banner("talys", "end")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
