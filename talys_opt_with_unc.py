#!/usr/bin/env python3
"""
TALYS parameter optimiser — driven by tesseract.in.

Optimises any set of TALYS parameters listed in the \\opt block of
tesseract.in by minimising a log-space chi-square plus a Gaussian prior on
the reaction's bin-averaged cross sections. TALYS is evaluated at
Gauss-Legendre points inside each data bin and averaged over the same bins
as the data (talys_points_per_bin; 1 = bin centres).  Three optimisers are available via  method = ...  in [talys]:

    Powell         scipy.optimize.minimize, derivative-free (default)
    Nelder-Mead    scipy.optimize.minimize, derivative-free
    least_squares  scipy.optimize.least_squares (trust-region reflective).
                   Exploits the sum-of-squares form of the loss and usually
                   needs far fewer TALYS runs.  Its Jacobian can be evaluated
                   with several TALYS runs in parallel (lsq_workers).

Usage:
    python talys_opt_with_unc.py --input /path/to/tesseract.in
    python talys_opt_with_unc.py --input /path/to/tesseract.in \\
                                  --exp-file my_xs_data.csv
"""

import argparse
import fcntl
import glob
import os
import re
import shutil
import tempfile
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from nucres.ratesmc_output import read_ratesmc_out
from nucres.read_qvals import atomic_mass_u

# ─────────────────────────────────────────────────────────────────────────────
# Keys in [talys] that are consumed by this script but NOT written to talys.inp.
# Everything else in [talys] is treated as a TALYS keyword → written verbatim.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_KEYS = frozenset({
    'method', 'maxiter', 'xtol', 'ftol', 'xatol', 'fatol',
    'exp_rel_err', 'e_fit_min', 'prior_rel_std', 'prior_abs_floor',
    'debug_every', 'exp_file', 'output_file', 'rates_mc_file',
    'rate_xmin', 'rate_xmax', 'plot_log_y_xs', 'plot_log_y_rate',
    'talys_output_dir', 'exp_energy_frame',
    'talys_xs_file', 'talys_points_per_bin', 'exp_bin_width',
    # least_squares-only settings
    'gtol', 'lsq_diff_step', 'lsq_workers',
})

# Accepted spellings of the least-squares method in tesseract.in
_LSQ_METHODS = frozenset({'least_squares', 'least-squares', 'lsq', 'trf'})

FLOOR = 1e-12
BIG   = 1e99


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OptParam:
    """One parameter to optimise, parsed from the \\opt block."""
    name:   str    # human-readable key, e.g. "rvadjust_a" or "T_13_25"
    prefix: str    # TALYS line prefix (everything before the value), e.g. "rvadjust a"
    x0:     float  # initial value
    lo:     float  # lower bound
    hi:     float  # upper bound


# ─────────────────────────────────────────────────────────────────────────────
# tesseract.in parser  (standalone — no imports from the TESSERACT package)
# ─────────────────────────────────────────────────────────────────────────────
def parse_tesseract_in(path: str) -> dict:
    """
    Parse tesseract.in into a nested {section → {key: value}} dict.
    Lines in the \\opt...\\opt block inside [talys] are stored as a list
    under sections['talys']['_opt_lines'] (inline # comments stripped).
    """
    sections: dict = {'_global': {}}
    current:  str  = '_global'
    in_opt:   bool = False
    opt_lines: list = []

    with open(path) as fh:
        for raw_line in fh:
            raw      = raw_line.rstrip('\n')
            stripped = raw.strip()

            # ── \opt block markers ────────────────────────────────────────────
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

            # ── Regular key = value lines ─────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Nuclear identity helpers
# ─────────────────────────────────────────────────────────────────────────────
_Z_TO_SYMBOL = {
    1:'H',  2:'He', 3:'Li', 4:'Be', 5:'B',  6:'C',  7:'N',  8:'O',
    9:'F',  10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P', 16:'S',
    17:'Cl',18:'Ar',19:'K', 20:'Ca',21:'Sc',22:'Ti',23:'V', 24:'Cr',
    25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',
}
_LIGHT_PARTICLE = {(0,0):'g', (0,1):'n', (1,1):'p', (1,2):'d', (1,3):'t', (2,3):'h', (2,4):'a'}

# TALYS exclusive-channel file names count the emitted n, p, d, t, h, a
# (written with "channels y"): (a,p) -> xs010000.tot, (p,g) -> xs000000.tot.
_EXCLUSIVE_ORDER = ('n', 'p', 'd', 't', 'h', 'a')


def talys_xs_candidates(cfg: dict) -> List[str]:
    """TALYS output files that may hold the fitted channel's cross section, in order."""
    names = []
    if cfg['script'].get('talys_xs_file'):
        names.append(cfg['script']['talys_xs_file'])
    proj, ejec = cfg['proj'], cfg['ejec']
    names.append(f"{proj}{ejec}.tot")
    counts = ''.join('1' if ejec == p else '0' for p in _EXCLUSIVE_ORDER)
    names.append(f"xs{counts}.tot")
    return names

def _particle_symbol(Z: int, A: int) -> str:
    return _LIGHT_PARTICLE.get((Z, A), _Z_TO_SYMBOL.get(Z, '?'))


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────
def _parse_opt_line(line: str) -> OptParam:
    """
    Parse one non-comment \\opt line.
    Format:  keyword [qualifiers...]  x0  lo  hi
    The last three tokens are always the numeric x0, lo, hi.
    """
    tokens = line.split()
    if len(tokens) < 4:
        raise ValueError(
            f"\\opt line needs ≥4 tokens (keyword... x0 lo hi), got: {line!r}"
        )
    try:
        hi  = float(tokens[-1])
        lo  = float(tokens[-2])
        x0  = float(tokens[-3])
    except ValueError as exc:
        raise ValueError(
            f"Last 3 tokens of \\opt line must be numeric (x0 lo hi): {line!r}"
        ) from exc
    prefix_toks = tokens[:-3]
    return OptParam(
        name   = "_".join(prefix_toks),
        prefix = " ".join(prefix_toks),
        x0     = x0,
        lo     = lo,
        hi     = hi,
    )


def load_config(tesseract_path: str) -> dict:
    """
    Parse tesseract.in and return a config dict with keys:
        basics      — nuclear identity (dict from [basics])
        talys_fixed — TALYS keywords written to talys.inp (dict)
        opt_params  — list of OptParam parsed from \\opt block
        script      — optimizer / script settings (dict)
        proj, tar, A_tar, ejec, residual, A_res  — derived convenience fields
    """
    raw       = parse_tesseract_in(tesseract_path)
    basics    = raw.get('basics',  {})
    talys_sec = raw.get('talys',   {})

    talys_fixed: dict = {}
    script_cfg:  dict = {}
    for k, v in talys_sec.items():
        if k == '_opt_lines':
            continue
        if k in _SCRIPT_KEYS:
            script_cfg[k] = v
        else:
            talys_fixed[k] = v

    opt_params: List[OptParam] = [
        _parse_opt_line(line)
        for line in talys_sec.get('_opt_lines', [])
    ]

    proj     = _particle_symbol(int(basics['projectile_Z']), int(basics['projectile_A']))
    tar      = _Z_TO_SYMBOL.get(int(basics['target_Z']),   '?')
    ejec     = _particle_symbol(int(basics['ejectile_Z']), int(basics['ejectile_A']))
    residual = _Z_TO_SYMBOL.get(int(basics['residual_Z']), '?')

    # TALYS takes, and tabulates ap.tot against, the projectile LAB energy.
    # TESSERACT's integrated cross sections are in the centre-of-mass frame,
    # so energies are converted on the way into TALYS and back out of it.
    m_proj = atomic_mass_u(int(basics['projectile_Z']), int(basics['projectile_A']))
    m_targ = atomic_mass_u(int(basics['target_Z']), int(basics['target_A']))
    frame  = script_cfg.get('exp_energy_frame', 'cm').strip().lower()
    if frame not in ('cm', 'lab'):
        raise ValueError(f"exp_energy_frame must be 'cm' or 'lab', got {frame!r}")
    # Multiply an exp_file energy by this to get the TALYS (lab) energy.
    exp_to_lab = (m_targ + m_proj) / m_targ if frame == 'cm' else 1.0

    return {
        'basics':      basics,
        'talys_fixed': talys_fixed,
        'opt_params':  opt_params,
        'script':      script_cfg,
        'proj':        proj,
        'tar':         tar,
        'A_tar':       int(basics['target_A']),
        'ejec':        ejec,
        'residual':    residual,
        'A_res':       int(basics['residual_A']),
        '_exp_to_lab': exp_to_lab,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TALYS I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def write_talys_files(
    workdir:    str,
    opt_values: np.ndarray,
    cfg:        dict,
    astro:      str = "n",
    astrogs:    str = "n",
) -> str:
    """
    Write energies.txt and talys.inp to workdir.

    talys.inp contains in order:
      0. strucpath $TALYS_STRUCPATH, if that is set
      1. Nuclear identity (projectile / element / mass) from [basics]
      2. Energy file reference
      3. Fixed TALYS keywords from [talys] key=value pairs
      4. Optimisation parameters with their current values
      5. astro / astrogs runtime flags

    Returns the path to the written input file.
    """
    energies_path = os.path.join(workdir, "energies.txt")
    with open(energies_path, "w") as fh:
        energies = cfg.get('_talys_E_cm', cfg['_x_exp'])
        for e in np.asarray(energies, dtype=float) * cfg['_exp_to_lab']:
            fh.write(f"{e:.10g}\n")

    inp_path = os.path.join(workdir, "talys.inp")
    lines = []

    # Each TALYS run opens ~330 structure files; thousands of fits doing that
    # against an NFS copy of the database saturated the file server, so a
    # job wrapper can point TALYS at a node-local copy instead. TALYS copies
    # this value over its compiled-in path without clearing it, so the
    # wrapper must give a path at least as long as that one.
    strucpath = os.environ.get('TALYS_STRUCPATH')
    if strucpath:
        lines.append(f"strucpath {strucpath}\n")

    # Nuclear identity
    lines.append(f"projectile {cfg['proj']}\n")
    lines.append(f"element    {cfg['tar']}\n")
    lines.append(f"mass       {cfg['A_tar']}\n")
    lines.append(f"energy     {energies_path}\n")

    # Fixed TALYS keywords
    for key, val in cfg['talys_fixed'].items():
        lines.append(f"{key} {val}\n")

    # Optimisation parameters (prefix carries keyword + qualifiers)
    for op, val in zip(cfg['opt_params'], opt_values):
        lines.append(f"{op.prefix} {val}\n")

    # Astro flags (runtime, not from config)
    lines.append(f"astro   {astro}\n")
    lines.append(f"astrogs {astrogs}\n")

    with open(inp_path, "w") as fh:
        fh.writelines(lines)

    return inp_path


def run_talys(workdir: str, inp_path: str, verbose: bool = False) -> bool:
    """Run TALYS in workdir, reading from inp_path. Returns True on success."""
    try:
        with open(inp_path, "rb") as fin:
            p = subprocess.run(
                ["talys"],
                cwd=workdir,
                stdin=fin,
                capture_output=True,
                text=True,
            )
        if p.returncode != 0:
            print(f"[TALYS] FAILED (returncode={p.returncode})", flush=True)
            if p.stdout.strip():
                print(f"[TALYS] stdout:\n{p.stdout.strip()}", flush=True)
            if p.stderr.strip():
                print(f"[TALYS] stderr:\n{p.stderr.strip()}", flush=True)
            return False
        if verbose:
            print(f"[TALYS] returncode=0", flush=True)
            if p.stdout.strip():
                print(f"[TALYS] stdout:\n{p.stdout.strip()}", flush=True)
            files = os.listdir(workdir)
            print(f"[TALYS] files in workdir: {sorted(files)}", flush=True)
        return True
    except FileNotFoundError:
        print("[TALYS] ERROR: 'talys' executable not found on PATH.", flush=True)
        return False
    except Exception as exc:
        print(f"[TALYS] ERROR: {exc}", flush=True)
        return False


def read_ap_tot(workdir: str, cfg: dict = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read the channel cross section (see talys_xs_candidates); returns (E, sigma) or None."""
    names = talys_xs_candidates(cfg) if cfg is not None else ["ap.tot"]
    path = next((os.path.join(workdir, n) for n in names
                 if os.path.exists(os.path.join(workdir, n))), None)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
        x  = df.iloc[:, 0].to_numpy(dtype=float)
        y  = df.iloc[:, 1].to_numpy(dtype=float)
        return None if (np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))) else (x, y)
    except Exception:
        return None


def read_astrorate(workdir: str, cfg: dict = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read astrorate.<ejectile> (e.g. astrorate.<ejectile>, astrorate.g); returns (T9, rate) or None."""
    ejec = cfg['ejec'] if cfg is not None else 'p'
    path = os.path.join(workdir, f"astrorate.{ejec}")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
        x  = df.iloc[:, 0].to_numpy(dtype=float)
        y  = df.iloc[:, 1].to_numpy(dtype=float)
        return None if (np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))) else (x, y)
    except Exception:
        return None


def read_ratesmc_file(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a RatesMC .out file; returns (T9, median rate) or None."""
    if not os.path.exists(path):
        return None
    try:
        table = read_ratesmc_out(path)
    except (OSError, ValueError):
        return None
    return table["T9"], table["median"]


def _cleanup_workdir(path: str, retries: int = 5, delay: float = 0.2) -> None:
    """
    Remove a TALYS scratch workdir, tolerating condor/NFS scratch races where
    a file TALYS just closed briefly isn't visible to rmtree's directory
    listing and then reappears before the final rmdir (OSError: Directory
    not empty). Retries a few times before giving up and leaving it behind,
    rather than crashing a long-running optimisation.
    """
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == retries - 1:
                print(f"[TALYS] WARNING: could not remove scratch dir {path} "
                      f"after {retries} attempts; leaving it behind.", flush=True)
                return
            time.sleep(delay)


def _ap_tot_in_exp_frame(xy, cfg: dict):
    """Convert ap.tot energies from TALYS's lab frame back to the exp_file frame."""
    if xy is None:
        return None
    x, y = xy
    return x / cfg['_exp_to_lab'], y


def run_talys_get_xs(
    opt_values: np.ndarray, cfg: dict
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run TALYS (no astro) and return the channel (E, sigma) or None, E in the exp_file frame."""
    wd = tempfile.mkdtemp(prefix="talys_xs_")
    try:
        inp = write_talys_files(wd, opt_values, cfg, astro="n", astrogs="n")
        return _ap_tot_in_exp_frame(read_ap_tot(wd, cfg), cfg) if run_talys(wd, inp) else None
    finally:
        _cleanup_workdir(wd)


def run_talys_get_rate(
    opt_values: np.ndarray, cfg: dict
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Run TALYS (astro=y) and return (astrorate_xy, ap_tot_xy); either may be None."""
    wd = tempfile.mkdtemp(prefix="talys_rate_")
    try:
        inp = write_talys_files(wd, opt_values, cfg, astro="y", astrogs="y")
        if not run_talys(wd, inp):
            return None, None
        return read_astrorate(wd, cfg), _ap_tot_in_exp_frame(read_ap_tot(wd, cfg), cfg)
    finally:
        _cleanup_workdir(wd)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints: one checkpoint.json per run directory, one entry per fit
# ─────────────────────────────────────────────────────────────────────────────
def _checkpoint_rmw(path: str, update) -> dict:
    """
    Locked read-modify-write of a checkpoint file holding {fit key: entry}.

    `update(data)` mutates the dict; the file is rewritten in place under an
    exclusive flock (so fits sharing a run directory cannot clobber each
    other) and removed when no entries remain. A file in the old single-fit
    format ({"params": ...}) cannot be attributed to a fit and is dropped.
    """
    import json as _json
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            text = fh.read()
            try:
                data = _json.loads(text) if text.strip() else {}
            except ValueError:
                data = {}
            if not isinstance(data, dict) or "params" in data:
                if data:
                    print(f"[checkpoint] ignoring old-format checkpoint in {path}", flush=True)
                data = {}
            update(data)
            if data:
                fh.seek(0)
                fh.truncate()
                _json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            else:
                os.remove(path)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return data


def checkpoint_load(path: str, key: str, param_names: List[str]) -> Optional[dict]:
    """This fit's checkpoint entry, or None (none saved, or the \\opt block changed)."""
    if not os.path.exists(path):
        return None
    found = {}
    _checkpoint_rmw(path, lambda data: found.update(data.get(key) or {}))
    if not found:
        return None
    if found.get("param_names") != list(param_names):
        print(f"[checkpoint] {key}: saved parameters {found.get('param_names')} "
              f"!= current {list(param_names)}; starting from x0.", flush=True)
        return None
    return found


def checkpoint_save(path: str, key: str, entry: dict) -> None:
    _checkpoint_rmw(path, lambda data: data.__setitem__(key, entry))


def checkpoint_clear(path: str, key: str) -> None:
    if os.path.exists(path):
        _checkpoint_rmw(path, lambda data: data.pop(key, None))


# ─────────────────────────────────────────────────────────────────────────────
# Objective (closure — no global mutable state)
# ─────────────────────────────────────────────────────────────────────────────
def read_exp_table(path: str) -> np.ndarray:
    """
    Numeric rows (E, sigma[, d_sigma]) of a data file.

    '#' lines are comments. A leading non-numeric line (a column header
    without '#') is skipped. Step 2's files have only a '#' header, which
    pd.read_csv(..., comment='#') used to take the FIRST DATA ROW as the
    column names for, silently dropping the lowest-energy bin.
    """
    df = pd.read_csv(path, comment="#", header=None)
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().any():
        df = df.iloc[1:]
    return df.apply(pd.to_numeric).to_numpy(dtype=float)


def exp_bin_width(exp_file: str, x_exp: np.ndarray, script: dict) -> float:
    """
    Width (MeV) of the data bins: exp_bin_width in [talys], else the
    'dE=' in the integrated file's header, else '_dE_<w>' in its name, else
    the median spacing of the bin centres.
    """
    if script.get('exp_bin_width'):
        return float(script['exp_bin_width'])
    try:
        with open(exp_file) as fh:
            m = re.search(r'dE\s*=\s*([0-9.eE+-]+)', fh.readline())
        if m:
            return float(m.group(1))
    except OSError:
        pass
    m = re.search(r'_dE_([0-9.]+?)(?:_|\.csv|$)', os.path.basename(exp_file))
    if m:
        return float(m.group(1))
    return float(np.median(np.diff(np.sort(x_exp))))


def talys_bin_grid(x_centres: np.ndarray, width: float, n_per_bin: int):
    """
    Gauss-Legendre energies inside each bin [c - w/2, c + w/2] and weights
    that average over the bin (they sum to 1 per bin). Interior nodes mean
    adjacent bins never share an energy. n_per_bin = 1 is the bin centre.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(n_per_bin))
    E = (np.asarray(x_centres, dtype=float)[:, None] + 0.5 * width * nodes[None, :]).ravel()
    return E, weights / 2.0


def talys_bin_average(y_sub: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bin averages from TALYS values at talys_bin_grid energies."""
    return np.asarray(y_sub, dtype=float).reshape(-1, len(weights)) @ weights


def make_objective(cfg: dict, x_exp: np.ndarray, y_exp: np.ndarray,
                   y_err: np.ndarray, mask: np.ndarray,
                   checkpoint_path: str = None, checkpoint_key: str = None):
    """
    Build the objective for the optimiser.

    Returns a callable ``objective(params) -> float`` giving chi2 + prior,
    with chi2 = sum((ln y_exp - ln y_TALYS)^2 / sigma_ln^2) over the fitted
    bins (not divided by the degrees of freedom; chi2/dof is reported
    separately) and TALYS averaged over each bin.  Powell and Nelder-Mead use
    this.

    Attached to it:
        objective.residuals(params) -> np.ndarray
            The same loss written as a residual vector r, such that
            sum(r**2) == objective(params).  method = least_squares uses this.
        objective.state['count']  — number of evaluations so far
        objective.state['best']   — (best_total, best_params) seen so far

    Both callables share a single evaluation routine, so the loss, prior,
    checkpointing and debug printout are identical for every optimiser.
    The routine is thread-safe, so the least-squares Jacobian can run
    several TALYS evaluations at the same time (see lsq_workers).

    If checkpoint_path is given, this fit's entry (checkpoint_key) in that
    run directory's checkpoint file is updated every time a new best is
    found, so the run can resume after eviction.
    """
    import json as _json
    import threading

    opt_params  = cfg['opt_params']
    script      = cfg['script']
    X0          = np.array([op.x0 for op in opt_params], dtype=float)
    EXP_REL_ERR = float(script.get('exp_rel_err',     '0.10'))
    PRIOR_STD   = float(script.get('prior_rel_std',   '0.15'))
    PRIOR_FLOOR = float(script.get('prior_abs_floor', '0.01'))
    DEBUG_EVERY = int(script.get(  'debug_every',     '10'))

    n_fit = int(np.count_nonzero(mask))
    dof   = max(n_fit - len(opt_params), 1)

    # Per-point uncertainty of ln(y): sigma_y / y from the data file's third
    # column where it is positive, else exp_rel_err.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.asarray(y_err, dtype=float) / np.asarray(y_exp, dtype=float)
    sigma_ln = np.where(np.isfinite(rel) & (rel > 0.0), rel, EXP_REL_ERR)[mask]

    # TALYS is run at cfg['_talys_E_cm'] (n points per bin) and averaged over
    # each bin with cfg['_bin_weights']; without them, the bin centres.
    bin_weights = cfg.get('_bin_weights')

    # Prior width per parameter (None → no prior term)
    prior_sigma = (PRIOR_STD * np.maximum(np.abs(X0), PRIOR_FLOOR)
                   if PRIOR_STD > 0.0 else None)

    # Length of the residual vector: data points (+ one entry per parameter
    # for the prior).  FAIL_RESID fills it when TALYS fails, so the least-
    # squares solver sees a large-but-finite loss instead of BIG (1e99),
    # which would wreck its trust-region steps.
    n_resid    = n_fit + (len(opt_params) if prior_sigma is not None else 0)
    FAIL_RESID = 1e3

    state = {'count': 0, 'best': (np.inf, None), 'best_parts': None}
    lock  = threading.Lock()

    def _evaluate(params: np.ndarray):
        """
        One TALYS evaluation.  Returns (residual_vector, total_loss);
        residual_vector is None when params are out of bounds or TALYS failed
        (total_loss is then BIG).
        """
        params = np.asarray(params, dtype=float)
        with lock:
            state['count'] += 1
            count = state['count']

        # ── Bounds penalty ───────────────────────────────────────────────────
        for i, op in enumerate(opt_params):
            if not (op.lo <= params[i] <= op.hi):
                return None, BIG

        # ── TALYS evaluation ─────────────────────────────────────────────────
        out = run_talys_get_xs(params, cfg)
        if out is None:
            return None, BIG
        _, y_t = out
        n_sub = 1 if bin_weights is None else len(bin_weights)
        if len(y_t) != len(y_exp) * n_sub:
            return None, BIG
        if bin_weights is not None:
            y_t = talys_bin_average(y_t, bin_weights)

        # ── Log-space chi-square, as residuals ───────────────────────────────
        # sum(r_data**2) = chi2; it is NOT divided by dof, so chi2 + prior is
        # the log-posterior (up to a constant) and the prior keeps its weight.
        # FLOOR only guards log(0); added to every value it would bias tiny sigma.
        r_data   = (np.log(np.maximum(y_exp[mask], FLOOR))
                    - np.log(np.maximum(y_t[mask], FLOOR))) / sigma_ln
        chi2     = float(np.sum(r_data ** 2))
        chi2_red = chi2 / dof

        # ── Gaussian prior, as residuals ─────────────────────────────────────
        if prior_sigma is not None:
            r_prior = (params - X0) / prior_sigma
        else:
            r_prior = np.zeros(0)
        prior = float(np.sum(r_prior ** 2))

        total = chi2 + prior

        with lock:
            if total < state['best'][0]:
                state['best'] = (total, params.copy())
                state['best_parts'] = {'chi2': chi2, 'chi2_red': chi2_red,
                                       'prior': prior, 'dof': dof}
                if checkpoint_path:
                    checkpoint_save(checkpoint_path, checkpoint_key or "fit", {
                        'params': params.tolist(),
                        'param_names': [op.name for op in opt_params],
                        'loss':   total,
                        'nfev':   state['count'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
            best_val = state['best'][0]

        if DEBUG_EVERY and (count % DEBUG_EVERY == 0):
            pstr = "  ".join(
                f"{op.name}={params[i]:.5g}" for i, op in enumerate(opt_params)
            )
            print(
                f"[eval {count:4d}]  {pstr}  "
                f"chi2={chi2:.4e} (red {chi2_red:.4e})  prior={prior:.4e}  total={total:.4e}  "
                f"| best={best_val:.4e}",
                flush=True,
            )

        return np.concatenate([r_data, r_prior]), total

    def objective(params: np.ndarray) -> float:
        """Scalar loss chi2 + prior (Powell / Nelder-Mead)."""
        return _evaluate(params)[1]

    def residuals(params: np.ndarray) -> np.ndarray:
        """Residual vector r with sum(r**2) = chi2 + prior (least_squares)."""
        r, _ = _evaluate(params)
        return np.full(n_resid, FAIL_RESID) if r is None else r

    objective.residuals = residuals
    objective.state     = state
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Least-squares optimiser  (method = least_squares)
# ─────────────────────────────────────────────────────────────────────────────
def run_least_squares(obj, x0: np.ndarray, cfg: dict):
    """
    Minimise chi2 + prior with scipy.optimize.least_squares.

    Why this is faster than Powell: the loss is a sum of squared residuals,
    so the solver can build a local linear model of TALYS from one Jacobian
    (N_opt extra TALYS runs) and jump straight towards the minimum, instead
    of doing one line search per parameter direction.

    Settings read from [talys] in tesseract.in:
        maxiter        cap on solver iterations (passed as max_nfev)
        xtol, ftol     relative step / loss-change tolerances
        gtol           gradient tolerance                      (default 1e-8)
        lsq_diff_step  relative finite-difference step for the Jacobian:
                       h_i = lsq_diff_step * max(|x_i|, 1)     (default 1e-2)
        lsq_workers    TALYS runs evaluated in parallel for each Jacobian
                       (default 1 = serial; useful up to N_opt)

    The Jacobian is computed here (forward differences) rather than by scipy
    so that its N_opt TALYS runs can go to a thread pool.  Threads are enough
    because each TALYS run is a separate process in its own scratch dir.

    Returns a scipy OptimizeResult with the fields main() expects:
        x, fun (scalar loss), success, message, nfev (total TALYS runs).
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.optimize import least_squares, OptimizeResult

    opt_params = cfg['opt_params']
    script     = cfg['script']
    lo = np.array([op.lo for op in opt_params], dtype=float)
    hi = np.array([op.hi for op in opt_params], dtype=float)

    maxiter   = int(script.get('maxiter', '250'))
    xtol      = float(script.get('xtol', '1e-3'))
    ftol      = float(script.get('ftol', '1e-3'))
    gtol      = float(script.get('gtol', '1e-8'))
    diff_step = float(script.get('lsq_diff_step', '1e-2'))
    workers   = max(1, min(int(script.get('lsq_workers', '1')), len(opt_params)))

    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)   # solver needs a feasible start

    # Remember the last residual so jac() can reuse it (saves one TALYS run
    # per iteration: scipy always calls fun(x) before jac(x) at the same x).
    last = {'x': None, 'r': None}

    def fun(x):
        r = obj.residuals(x)
        last['x'], last['r'] = np.array(x, dtype=float), r
        return r

    def jac(x):
        x = np.asarray(x, dtype=float)
        r0 = last['r'] if (last['x'] is not None and np.array_equal(last['x'], x)) else fun(x)

        steps = diff_step * np.maximum(np.abs(x), 1.0)
        steps = np.where(x + steps > hi, -steps, steps)    # step backwards at the upper bound
        shifted = []
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += steps[i]
            shifted.append(xp)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                r_shift = list(pool.map(obj.residuals, shifted))
        else:
            r_shift = [obj.residuals(xp) for xp in shifted]

        J = np.empty((len(r0), len(x)))
        for i in range(len(x)):
            J[:, i] = (r_shift[i] - r0) / steps[i]

        best_loss, _ = obj.state['best']
        vals = "  ".join(f"{op.name}={x[i]:.4g}" for i, op in enumerate(opt_params))
        print(f"[lsq] loss={float(np.sum(r0 ** 2)):.4e}  best={best_loss:.4e}  {vals}",
              flush=True)
        return J

    print(f"[lsq] trust-region reflective | diff_step={diff_step} | "
          f"workers={workers} | max iterations={maxiter}", flush=True)

    ls = least_squares(
        fun, x0, jac=jac, bounds=(lo, hi), method='trf',
        x_scale='jac', xtol=xtol, ftol=ftol, gtol=gtol,
        max_nfev=maxiter, verbose=1,
    )

    return OptimizeResult(
        x       = ls.x,
        fun     = float(np.sum(ls.fun ** 2)),   # = chi2 + prior, same scale as Powell
        success = ls.success,
        status  = ls.status,
        message = ls.message,
        nfev    = obj.state['count'],           # total TALYS runs, incl. Jacobians
        nit     = ls.nfev,
        njev    = ls.njev,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_best_fit_xs(cfg, x_exp, y_exp, x_t, y_t, params_best, y_err=None, out_dir=".",
                     y_bin=None, e_fit_min=None):
    script      = cfg['script']
    EXP_REL_ERR = float(script.get('exp_rel_err', '0.10'))
    log_y       = script.get('plot_log_y_xs', 'true').lower() == 'true'
    label       = "  ".join(
        f"{op.name}={params_best[i]:.4g}"
        for i, op in enumerate(cfg['opt_params'])
    )

    plt.figure()
    plt.errorbar(
        x_exp, y_exp,
        yerr=y_err if y_err is not None else EXP_REL_ERR * y_exp,
        fmt="o", capsize=3, label="Experimental",
    )
    plt.plot(x_t, y_t, linestyle="-", label=f"TALYS best-fit\n{label}")
    if y_bin is not None:
        plt.plot(x_exp, y_bin, "s", mfc="none", label="TALYS bin average")
    if e_fit_min is not None:
        plt.axvline(e_fit_min, color="grey", linestyle=":", linewidth=1,
                    label=f"fit window E >= {e_fit_min:g} MeV")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("Cross section [mb]")
    if log_y:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    #plt.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"talys_opt_{cfg['tar']}_{cfg['proj']}{cfg['ejec']}_{cfg['residual']}_xs.png")
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_reaction_rate(cfg, x_rate, y_rate, ratesmc_xy=None, out_dir="."):
    script = cfg['script']
    xmin   = float(script.get('rate_xmin',       '0.0'))
    xmax   = float(script.get('rate_xmax',        '2.0'))
    log_y  = script.get('plot_log_y_rate', 'true').lower() == 'true'

    sel = (x_rate >= xmin) & (x_rate <= xmax)
    plt.figure()
    plt.plot(x_rate[sel], y_rate[sel], label="TALYS best-fit rate")

    if ratesmc_xy is not None:
        x2, y2 = ratesmc_xy
        sel2 = (x2 >= xmin) & (x2 <= xmax)
        plt.plot(x2[sel2], y2[sel2], linestyle="--", label="RatesMC.out")

    plt.xlabel("T (GK)")
    plt.ylabel("Rate [cm³/s/mol]")
    if log_y:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.ylim(bottom=1e-3)
    fname = os.path.join(out_dir, f"reaction_rates_{cfg['tar']}_{cfg['proj']}{cfg['ejec']}_{cfg['residual']}.png")
    plt.savefig(fname, dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save / output helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_results(cfg, x_exp, y_exp, y_err, x_xs, y_xs, x_rate, y_rate, params_best, out_dir=".",
                 y_bin=None, fit_info=None):
    exp_file = cfg['script'].get('exp_file', 'exp')
    stem     = os.path.splitext(os.path.basename(exp_file))[0]
    path     = os.path.join(out_dir, f"talys_results_{stem}.npz")
    np.savez(
        path,
        x_exp       = x_exp,
        y_exp       = y_exp,
        y_err       = y_err,
        x_xs        = x_xs,
        y_xs        = y_xs,
        x_rate      = x_rate,
        y_best_rate = y_rate,
        params_best = params_best,
        param_names = np.array([op.name for op in cfg['opt_params']]),
        # TALYS averaged over the data bins (NaN if not computed)
        y_xs_binned = (np.asarray(y_bin, dtype=float) if y_bin is not None
                       else np.full(len(x_exp), np.nan)),
        **{k: np.asarray(v) for k, v in (fit_info or {}).items()},
    )
    print(f"\nResults saved to {path}")


_CHANNEL_MAP = {
    "ap":"(α,p)", "ag":"(α,γ)", "an":"(α,n)",
    "pg":"(p,γ)", "ng":"(n,γ)", "np":"(n,p)",
    "pa":"(p,α)", "na":"(n,α)", "aa":"(α,α)",
}
_FNAME_RE = re.compile(
    r"(?P<target>\d+[A-Za-z]+)_(?P<channel>[a-z]+)_(?P<product>\d+[A-Za-z]+)"
    r"_integrated_xs_dE_(?P<dE>[\d.]+)",
    re.IGNORECASE,
)

def _parse_exp_filename(fname: str) -> Tuple[str, str]:
    m = _FNAME_RE.search(os.path.basename(fname))
    if not m:
        return "unknown", "unknown"
    ch = _CHANNEL_MAP.get(m.group("channel").lower(), f"({m.group('channel')})")
    return f"{m.group('target')}{ch}{m.group('product')}", m.group("dE")


def write_optimization_output(
    cfg: dict,
    res,
    params_best: np.ndarray,
    x_xs:   np.ndarray,
    y_xs:   np.ndarray,
    x_rate: Optional[np.ndarray] = None,
    y_rate: Optional[np.ndarray] = None,
    out_file: str = "talys_optimization.out",
    fit_info: Optional[dict] = None,
) -> None:
    """
    Append a best-fit summary block to the shared output file.

    The write is protected by an exclusive flock so parallel workers do not
    interleave their output.  Block format:

        === header (reaction, run, optimizer stats) ===
        --- optimized parameters ---
        --- cross sections  E[MeV]  sigma[mb] ---
        --- reaction rates  T9[GK]  rate[cm3/s/mol]  (if available) ---
        ========================================
    """
    exp_file  = cfg['script'].get('exp_file', '')
    reaction, dE = _parse_exp_filename(exp_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w = max((len(op.name) for op in cfg['opt_params']), default=12)

    # Extract run index from filename suffix _run{j}
    run_m   = re.search(r'_run(\d+)', os.path.splitext(os.path.basename(exp_file))[0])
    run_idx = run_m.group(1) if run_m else "?"

    lines = []
    lines.append("=" * 72 + "\n")
    lines.append(f"Date/Time   : {timestamp}\n")
    lines.append(f"Input file  : {exp_file}\n")
    lines.append(f"Reaction    : {reaction}\n")
    lines.append(f"Bin width   : dE = {dE} MeV\n")
    lines.append(f"Run index   : {run_idx}\n")
    lines.append("-" * 72 + "\n")
    lines.append(
        f"Optimizer   : {cfg['script'].get('method','Powell')}  "
        f"(converged={res.success}, nfev={res.nfev})\n"
    )
    info = fit_info or {}
    chi2_red = info.get('chi2_red', res.fun)
    lines.append(f"Min reduced chi-square: {chi2_red:.6g}\n")
    if info:
        lines.append(f"Objective   : chi2 + prior = {res.fun:.6g} "
                     f"(chi2 = {info['chi2']:.6g}, prior = {info['prior']:.6g}, dof = {info['dof']})\n")
        lines.append(f"Fit window  : E >= {info['e_fit_min']} MeV; TALYS averaged over "
                     f"{info['talys_points_per_bin']} point(s) per {info['bin_width']:g} MeV bin\n")
    lines.append("-" * 72 + "\n")
    lines.append("Optimized parameters:\n")
    for i, op in enumerate(cfg['opt_params']):
        lines.append(f"  {op.name:<{w}} = {params_best[i]:+.6g}\n")
    lines.append("-" * 72 + "\n")
    lines.append("Cross sections  E[MeV]  sigma[mb]:\n")
    for e, s in zip(x_xs, y_xs):
        lines.append(f"  {e:.6e}  {s:.6e}\n")
    if x_rate is not None and len(x_rate):
        lines.append("-" * 72 + "\n")
        lines.append("Reaction rates  T9[GK]  rate[cm3/s/mol]:\n")
        for t, r in zip(x_rate, y_rate):
            lines.append(f"  {t:.6e}  {r:.6e}\n")
    lines.append("=" * 72 + "\n\n")

    with open(out_file, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.writelines(lines)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

    print(f"Results appended to {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Replot from saved .npz  (no re-running TALYS)
# ─────────────────────────────────────────────────────────────────────────────
def replot(npz_path: str, tesseract_path: str = "tesseract.in",
           ratesmc_path: str = None, out_dir: str = "."):
    """
    Reload saved results from a .npz and regenerate both plots without
    re-running TALYS.

    Usage:
        from talys_opt_with_unc import replot
        replot("talys_results_22Mg_ap_25Al.npz", "/path/to/tesseract.in")
    """
    cfg = load_config(tesseract_path)
    d   = np.load(npz_path, allow_pickle=True)

    params_best = d["params_best"]
    x_xs        = d["x_xs"]
    y_xs        = d["y_xs"]
    x_exp_      = d["x_exp"]
    y_exp_      = d["y_exp"]
    y_err_      = (d["y_err"] if "y_err" in d
                   else float(cfg['script'].get('exp_rel_err', '0.10')) * d["y_exp"])
    x_rate      = d["x_rate"]
    y_rate      = d["y_best_rate"]

    print(f"Loaded results from {npz_path}")
    print(f"  params: {dict(zip(d['param_names'], params_best))}")

    plot_best_fit_xs(cfg, x_exp_, y_exp_, x_xs, y_xs, params_best, y_err=y_err_,
                     out_dir=out_dir)

    rmc = ratesmc_path or cfg['script'].get('rates_mc_file', 'RatesMC.out')
    plot_reaction_rate(cfg, x_rate, y_rate, ratesmc_xy=read_ratesmc_file(rmc),
                       out_dir=out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Recover missing reaction rates from already-optimised runs
#
# Some runs finish optimisation fine but TALYS fails to produce astrorate.<ejectile>
# on that attempt (see the try/except around run_talys_get_rate() in main()).
# save_results() still writes the .npz unconditionally, just with empty
# x_rate / y_best_rate arrays, so tesseract.py's existence-only skip check
# (`if npz.exists(): continue`) treats that run as "done" forever.  These
# helpers find such npz files, reuse their already-saved params_best to run
# TALYS once more (astro-only, no re-optimisation), and patch just the rate
# fields in place.
# ─────────────────────────────────────────────────────────────────────────────
_BLOCK_RE = re.compile(r"={72}\n.*?\n={72}\n", re.DOTALL)


def _rate_is_missing(data: dict) -> bool:
    x = data.get('x_rate')
    y = data.get('y_best_rate')
    return x is None or y is None or len(np.asarray(x)) == 0 or len(np.asarray(y)) == 0


def _append_rate_recovered_note(shared_out: str, npz_path: str, n_points: int) -> bool:
    """
    Insert a short note immediately after this run's existing block in the
    shared talys_optimization.out, flagging that its rate was recomputed
    later. Uses the same exclusive flock as write_optimization_output() so
    it can never interleave with a concurrent worker appending a new block.
    Only the one matching block is touched — every other block in the file
    (i.e. every other run's rate data) is left byte-for-byte unchanged.

    Returns True if a matching block was found and annotated.
    """
    if not os.path.exists(shared_out):
        return False

    stem = re.sub(r'\.npz$', '', re.sub(r'^talys_results_', '', os.path.basename(npz_path)))

    with open(shared_out, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            content = fh.read()
            target = None
            for m in _BLOCK_RE.finditer(content):
                block = m.group(0)
                if "Run index" not in block:
                    continue
                mf = re.search(r"Input file\s*:\s*(.+)", block)
                if not mf:
                    continue
                block_stem = os.path.splitext(os.path.basename(mf.group(1).strip()))[0]
                if block_stem == stem:
                    target = m   # keep the LAST match — the most recent attempt

            if target is None:
                return False

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            note = (
                f"[UPDATE {timestamp}] Reaction rate for this run was recomputed "
                f"post-hoc from the saved best-fit parameters ({n_points} T9 points), "
                f"after the original run failed to produce astrorate.<ejectile>. See "
                f"{os.path.basename(npz_path)} for the updated rate.\n\n"
            )
            insert_at   = target.end()
            new_content = content[:insert_at] + note + content[insert_at:]
            fh.seek(0)
            fh.write(new_content)
            fh.truncate()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return True


def fix_missing_rate(npz_path: str, cfg: dict, shared_out: str) -> str:
    """
    Recompute the reaction rate for one saved run, in place, if it's missing.

    Only x_rate / y_best_rate are touched; x_exp, y_exp, y_err, x_xs, y_xs,
    params_best and param_names are re-saved exactly as loaded. No other
    run's .npz or .out block is read or written.

    Returns "skipped_has_rate", "fixed", "still_failed", or "param_mismatch".
    """
    data = dict(np.load(npz_path, allow_pickle=True))

    if not _rate_is_missing(data):
        return "skipped_has_rate"

    saved_names   = [str(n) for n in data['param_names']]
    current_names = [op.name for op in cfg['opt_params']]
    if saved_names != current_names:
        print(
            f"[fix-rates] {npz_path}: saved param_names {saved_names} != "
            f"current \\opt block {current_names} — skipping (tesseract.in "
            "\\opt block may have changed since this run)."
        )
        return "param_mismatch"

    params_best   = np.asarray(data['params_best'], dtype=float)
    run_cfg       = dict(cfg)                       # don't mutate caller's cfg
    run_cfg['_x_exp'] = np.asarray(data['x_exp'], dtype=float)

    print(f"[fix-rates] {npz_path}: re-running TALYS at saved best-fit params "
          f"{dict(zip(saved_names, params_best))} ...", flush=True)
    rate_best, _ = run_talys_get_rate(params_best, run_cfg)
    if rate_best is None:
        print(f"[fix-rates] {npz_path}: TALYS still failed to produce astrorate.<ejectile>.")
        return "still_failed"

    x_rate, y_rate = rate_best
    data['x_rate']      = x_rate
    data['y_best_rate'] = y_rate
    np.savez(npz_path, **data)
    print(f"[fix-rates] {npz_path}: rate recovered ({len(x_rate)} T9 points) — npz updated.")

    _append_rate_recovered_note(shared_out, npz_path, len(x_rate))
    return "fixed"


def fix_missing_rates_batch(out_root: str, cfg: dict, only_npz: str = None) -> None:
    """Run fix_missing_rate() over one npz (only_npz) or every npz under out_root."""
    shared_out = os.path.join(out_root, "talys_optimization.out")
    if only_npz:
        npz_paths = [only_npz]
    else:
        npz_paths = sorted(glob.glob(os.path.join(out_root, "RUN_*", "talys_results_*.npz")))

    if not npz_paths:
        print(f"[fix-rates] No talys_results_*.npz files found under {out_root}.")
        return

    tally = {"fixed": 0, "still_failed": 0, "skipped_has_rate": 0, "param_mismatch": 0}
    for npz_path in npz_paths:
        if not os.path.exists(npz_path):
            print(f"[fix-rates] {npz_path}: not found — skipping.")
            continue
        outcome = fix_missing_rate(npz_path, cfg, shared_out)
        tally[outcome] = tally.get(outcome, 0) + 1

    print(
        f"\n[fix-rates] Done. {tally['fixed']} fixed, "
        f"{tally['still_failed']} still failing, "
        f"{tally['skipped_has_rate']} already had rate data, "
        f"{tally['param_mismatch']} skipped (param mismatch)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="TALYS parameter optimiser driven by tesseract.in"
    )
    ap.add_argument(
        "--input", default="tesseract.in",
        help="Path to tesseract.in (default: tesseract.in in cwd)",
    )
    ap.add_argument(
        "--exp-file", default=None,
        help="Override exp_file from [talys] section of tesseract.in",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Root directory for talys_opt/ output (overrides talys_output_dir in tesseract.in)",
    )
    ap.add_argument(
        "--debug-talys", action="store_true",
        help="Run one TALYS call at x0 with verbose output, print workdir files, then exit",
    )
    ap.add_argument(
        "--fix-missing-rates", action="store_true",
        help="For runs whose talys_results_*.npz already exists but has empty "
             "x_rate/y_best_rate (TALYS failed to produce astrorate.<ejectile> on the "
             "original run), reuse the saved best-fit params to rerun TALYS "
             "for the rate only (no re-optimisation), patch the npz in place, "
             "and note the update in talys_optimization.out. With --run-idx or "
             "--exp-file, fixes only that one run; otherwise scans every "
             "RUN_*/talys_results_*.npz under the output root. Then exits.",
    )
    ap.add_argument(
        "--run-idx", type=int, default=None,
        help="With --fix-missing-rates, fix only RUN_{run-idx}'s npz (found as "
             "RUN_{run-idx}/talys_results_*.npz under the output root) — lets a "
             "condor job file reuse $(Process) directly without reconstructing "
             "the exp_file name. Ignored outside --fix-missing-rates mode.",
    )
    args = ap.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.input)
    if args.exp_file:
        cfg['script']['exp_file'] = args.exp_file

    # ── Output directory setup ────────────────────────────────────────────────
    script       = cfg['script']
    out_root     = args.output_dir or script.get('talys_output_dir', 'talys_opt')
    exp_file_cfg = script.get('exp_file', '')
    stem_cfg     = os.path.splitext(os.path.basename(exp_file_cfg))[0]
    run_m        = re.search(r'_run(\d+)', stem_cfg)
    run_idx      = run_m.group(1) if run_m else "0"
    run_dir      = os.path.join(out_root, f"RUN_{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    shared_out   = os.path.join(out_root, "talys_optimization.out")
    print(f"[talys_opt] output root : {out_root}")
    print(f"[talys_opt] run dir     : {run_dir}")
    print(f"[talys_opt] shared log  : {shared_out}")

    opt_params = cfg['opt_params']
    if not opt_params:
        raise SystemExit(
            "No optimisation parameters found in \\opt block of tesseract.in.\n"
            "Add lines of the form  'keyword [qualifiers] x0 lo hi'  inside \\opt."
        )

    # ── Rate-recovery mode: patch missing rates in existing npz, then exit ────
    # Doesn't need the original exp_file CSV — x_exp is read back from each
    # npz itself, so a single run's fix cannot disturb any other run's data.
    if args.fix_missing_rates:
        only_npz = None
        if args.run_idx is not None:
            run_glob = os.path.join(out_root, f"RUN_{args.run_idx}", "talys_results_*.npz")
            matches  = sorted(glob.glob(run_glob))
            if not matches:
                raise SystemExit(f"[fix-rates] No npz found matching {run_glob}.")
            if len(matches) > 1:
                raise SystemExit(
                    f"[fix-rates] Expected exactly one npz for RUN_{args.run_idx}, "
                    f"found {len(matches)}: {matches}"
                )
            only_npz = matches[0]
        elif args.exp_file:
            only_npz = os.path.join(run_dir, f"talys_results_{stem_cfg}.npz")
        fix_missing_rates_batch(out_root, cfg, only_npz=only_npz)
        raise SystemExit(0)

    # ── Load experimental data ────────────────────────────────────────────────
    exp_file = script.get('exp_file')
    if not exp_file or not os.path.exists(exp_file):
        raise FileNotFoundError(
            f"Experimental data file not found: {exp_file!r}\n"
            "Set exp_file in the [talys] section of tesseract.in."
        )

    exp_table   = read_exp_table(exp_file)
    x_exp       = exp_table[:, 0]
    y_exp       = exp_table[:, 1]
    EXP_REL_ERR = float(script.get('exp_rel_err', '0.10'))
    y_err       = (exp_table[:, 2] if exp_table.shape[1] >= 3 else EXP_REL_ERR * y_exp)
    E_FIT_MIN   = float(script.get('e_fit_min', '2.0'))
    mask        = (y_exp > 1e-10) & (x_exp >= E_FIT_MIN)

    if np.any(~np.isfinite(y_exp)):
        raise ValueError("Experimental y contains non-finite values.")
    if np.any(y_exp < 0):
        raise ValueError("Experimental y contains negative values.")

    # Attach energy grid to cfg so write_talys_files can use it
    cfg['_x_exp'] = x_exp

    # TALYS is averaged over the same bins as the data: n Gauss-Legendre
    # energies per bin (talys_points_per_bin; 1 = bin centres only).
    N_PER_BIN = int(script.get('talys_points_per_bin', '3'))
    BIN_WIDTH = exp_bin_width(exp_file, x_exp, script)
    if N_PER_BIN > 1:
        E_sub, w_sub = talys_bin_grid(x_exp, BIN_WIDTH, N_PER_BIN)
        cfg['_talys_E_cm'] = E_sub
        cfg['_bin_weights'] = w_sub
    print(f"[talys_opt] data bins: width {BIN_WIDTH:g} MeV; TALYS at {N_PER_BIN} "
          f"point(s) per bin; fit window E >= {E_FIT_MIN} MeV")

    # ── Debug mode: one verbose TALYS call at x0, then exit ──────────────────
    if args.debug_talys:
        X0_dbg = np.array([op.x0 for op in cfg['opt_params']], dtype=float)
        with tempfile.TemporaryDirectory(prefix="talys_debug_") as wd:
            inp = write_talys_files(wd, X0_dbg, cfg, astro="n", astrogs="n")
            run_talys(wd, inp, verbose=True)
            names = talys_xs_candidates(cfg)
            found = [n for n in names if os.path.exists(os.path.join(wd, n))]
            if found:
                print(f"\n[debug] {found[0]} contents:\n")
                with open(os.path.join(wd, found[0])) as fh:
                    print(fh.read())
            else:
                print(f"\n[debug] none of {names} was created "
                      "(exclusive-channel files need 'channels y' in [talys]).")
        raise SystemExit(0)

    # ── Checkpoint: resume x0 from this fit's entry if it was evicted ────────
    # One checkpoint.json per run directory, one entry per fit (keyed by the
    # data file), so fits for different bin widths never resume from each other.
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")
    checkpoint_key  = stem_cfg
    X0 = np.array([op.x0 for op in opt_params], dtype=float)
    try:
        ckpt = checkpoint_load(checkpoint_path, checkpoint_key, [op.name for op in opt_params])
    except Exception as exc:
        ckpt = None
        print(f"[checkpoint] WARNING: could not load checkpoint ({exc}), starting from x0.", flush=True)
    if ckpt:
        X0 = np.array(ckpt['params'], dtype=float)
        print(
            f"[checkpoint] Resuming {checkpoint_key} from {checkpoint_path}\n"
            f"  loss={ckpt['loss']:.6g}  nfev={ckpt['nfev']}  saved={ckpt['timestamp']}",
            flush=True,
        )

    # ── Optimisation setup ────────────────────────────────────────────────────
    method  = script.get('method',  'Powell')
    maxiter = int(script.get('maxiter', '250'))
    xtol    = float(script.get('xtol',  '1e-3'))
    ftol    = float(script.get('ftol',  '1e-3'))

    options = {"disp": True, "maxiter": maxiter}
    if method.lower() == "powell":
        options.update({"xtol": xtol, "ftol": ftol})
    else:
        options.update({
            "xatol": float(script.get('xatol', str(xtol))),
            "fatol": float(script.get('fatol', str(ftol))),
        })

    obj = make_objective(cfg, x_exp, y_exp, y_err, mask,
                         checkpoint_path=checkpoint_path, checkpoint_key=checkpoint_key)

    print(f"\nOptimising {len(opt_params)} parameter(s):")
    for op in opt_params:
        print(f"  {op.name:20s}  x0={op.x0}  bounds=[{op.lo}, {op.hi}]")
    print(f"\nMethod: {method}  |  maxiter: {maxiter}  |  E_fit_min: {E_FIT_MIN} MeV")
    print("=" * 60)

    def callback(xk):
        best_loss, _ = obj.state['best']
        vals = "  ".join(
            f"{opt_params[i].name}={xk[i]:.4g}" for i in range(len(opt_params))
        )
        print(f"loss={best_loss:.4e}  {vals}", flush=True)

    if method.lower() in _LSQ_METHODS:
        res = run_least_squares(obj, X0, cfg)
    else:
        res = minimize(obj, X0, method=method, options=options, callback=callback)

    print("\n=== Optimisation complete ===")
    print(res)
    params_best = res.x
    print("\nBest-fit parameters:")
    for i, op in enumerate(opt_params):
        print(f"  {op.name} = {params_best[i]:.6g}")
    parts = obj.state.get('best_parts') or {}
    fit_info = {
        'objective': float(res.fun),
        'chi2': parts.get('chi2', float('nan')),
        'chi2_red': parts.get('chi2_red', float('nan')),
        'prior': parts.get('prior', float('nan')),
        'dof': parts.get('dof', 0),
        'e_fit_min': E_FIT_MIN,
        'talys_points_per_bin': N_PER_BIN,
        'bin_width': BIN_WIDTH,
    }
    print(f"Objective chi2 + prior = {res.fun:.6g}  "
          f"(chi2 = {fit_info['chi2']:.6g}, prior = {fit_info['prior']:.6g})")
    print(f"Min reduced chi-square = {fit_info['chi2_red']:.6g}  (dof = {fit_info['dof']})")

    # ── Best-fit XS curve ─────────────────────────────────────────────────────
    out_xs = run_talys_get_xs(params_best, cfg)
    if out_xs is None:
        print("\nWARNING: TALYS failed at best-fit parameters.")
        return
    x_t, y_t = out_xs
    y_bin = (talys_bin_average(y_t, cfg['_bin_weights'])
             if cfg.get('_bin_weights') is not None and len(y_t) == len(x_exp) * N_PER_BIN
             else None)
    plot_best_fit_xs(cfg, x_exp, y_exp, x_t, y_t, params_best, y_err=y_err,
                     out_dir=run_dir, y_bin=y_bin, e_fit_min=E_FIT_MIN)

    # ── Reaction rate at best fit ─────────────────────────────────────────────
    x_rate: np.ndarray = np.array([])
    y_rate: np.ndarray = np.array([])
    try:
        rate_best, _ = run_talys_get_rate(params_best, cfg)
        if rate_best is None:
            print("\nWARNING: TALYS failed to produce astrorate.<ejectile> at best fit.")
        else:
            x_rate, y_rate = rate_best
            rmc_path   = script.get('rates_mc_file', 'RatesMC.out')
            ratesmc_xy = read_ratesmc_file(rmc_path)
            if ratesmc_xy is None:
                print(f"\nNOTE: {rmc_path} not found — it will not be plotted.")
            plot_reaction_rate(cfg, x_rate, y_rate, ratesmc_xy=ratesmc_xy,
                               out_dir=run_dir)
    except Exception as exc:
        print(f"\nWARNING: Could not compute/plot reaction rate: {exc!r}")

    # ── Write shared output file (flock-protected) and save .npz ─────────────
    write_optimization_output(cfg, res, params_best, x_t, y_t,
                               x_rate if len(x_rate) else None,
                               y_rate if len(y_rate) else None,
                               out_file=shared_out, fit_info=fit_info)
    save_results(cfg, x_exp, y_exp, y_err, x_t, y_t, x_rate, y_rate, params_best,
                 out_dir=run_dir, y_bin=y_bin, fit_info=fit_info)

    # This fit is done: drop its checkpoint entry (the file goes when empty)
    checkpoint_clear(checkpoint_path, checkpoint_key)


if __name__ == "__main__":
    main()
