#!/usr/bin/env python3
"""
TALYS parameter optimiser — driven by tesseract.in.

Optimises any set of TALYS parameters listed in the \\opt block of
tesseract.in by minimising log-space reduced chi-square on (a,p) cross
sections using scipy.optimize.minimize (Powell or Nelder-Mead).

Usage:
    python talys_opt_with_unc.py --input /path/to/tesseract.in
    python talys_opt_with_unc.py --input /path/to/tesseract.in \\
                                  --exp-file my_xs_data.csv
"""

import argparse
import fcntl
import os
import re
import tempfile
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────────────────
# Keys in [talys] that are consumed by this script but NOT written to talys.inp.
# Everything else in [talys] is treated as a TALYS keyword → written verbatim.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_KEYS = frozenset({
    'method', 'maxiter', 'xtol', 'ftol', 'xatol', 'fatol',
    'exp_rel_err', 'e_fit_min', 'prior_rel_std', 'prior_abs_floor',
    'debug_every', 'exp_file', 'output_file', 'rates_mc_file',
    'rate_xmin', 'rate_xmax', 'plot_log_y_xs', 'plot_log_y_rate',
    'talys_output_dir',
})

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
# tesseract.in parser  (standalone — no imports from the THICC package)
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
_LIGHT_PARTICLE = {(0,1):'n', (1,1):'p', (1,2):'d', (1,3):'t', (2,4):'a'}

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
      1. Nuclear identity (projectile / element / mass) from [basics]
      2. Energy file reference
      3. Fixed TALYS keywords from [talys] key=value pairs
      4. Optimisation parameters with their current values
      5. astro / astrogs runtime flags

    Returns the path to the written input file.
    """
    energies_path = os.path.join(workdir, "energies.txt")
    with open(energies_path, "w") as fh:
        for e in cfg['_x_exp']:
            fh.write(f"{e}\n")

    inp_path = os.path.join(workdir, "talys.inp")
    lines = []

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


def read_ap_tot(workdir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read ap.tot; returns (E, sigma) or None."""
    path = os.path.join(workdir, "ap.tot")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
        x  = df.iloc[:, 0].to_numpy(dtype=float)
        y  = df.iloc[:, 1].to_numpy(dtype=float)
        return None if (np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))) else (x, y)
    except Exception:
        return None


def read_astrorate(workdir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read astrorate.p; returns (T9, rate) or None."""
    path = os.path.join(workdir, "astrorate.p")
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
    """Read external MC rates file (RatesMC.out); returns (T9, rate) or None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
        x  = df.iloc[:, 0].to_numpy(dtype=float)
        y  = df.iloc[:, 1].to_numpy(dtype=float)
        return None if (np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))) else (x, y)
    except Exception:
        return None


def run_talys_get_xs(
    opt_values: np.ndarray, cfg: dict
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run TALYS (no astro) and return ap.tot (E, sigma) or None."""
    with tempfile.TemporaryDirectory(prefix="talys_xs_") as wd:
        inp = write_talys_files(wd, opt_values, cfg, astro="n", astrogs="n")
        return read_ap_tot(wd) if run_talys(wd, inp) else None


def run_talys_get_rate(
    opt_values: np.ndarray, cfg: dict
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Run TALYS (astro=y) and return (astrorate_xy, ap_tot_xy); either may be None."""
    with tempfile.TemporaryDirectory(prefix="talys_rate_") as wd:
        inp = write_talys_files(wd, opt_values, cfg, astro="y", astrogs="y")
        if not run_talys(wd, inp):
            return None, None
        return read_astrorate(wd), read_ap_tot(wd)


# ─────────────────────────────────────────────────────────────────────────────
# Objective (closure — no global mutable state)
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(cfg: dict, x_exp: np.ndarray, y_exp: np.ndarray,
                   y_err: np.ndarray, mask: np.ndarray):
    """
    Build and return an objective function (closure) over cfg and exp data.

    The returned callable accepts a 1-D params array (aligned with
    cfg['opt_params']) and returns the penalised reduced chi-square.

    Internal state is accessible via:
        obj.state['count']  — number of evaluations so far
        obj.state['best']   — (best_total, best_params) seen so far
    """
    opt_params  = cfg['opt_params']
    script      = cfg['script']
    X0          = np.array([op.x0 for op in opt_params], dtype=float)
    EXP_REL_ERR = float(script.get('exp_rel_err',     '0.10'))
    PRIOR_STD   = float(script.get('prior_rel_std',   '0.15'))
    PRIOR_FLOOR = float(script.get('prior_abs_floor', '0.01'))
    DEBUG_EVERY = int(script.get(  'debug_every',     '10'))

    state = {'count': 0, 'best': (np.inf, None)}

    def objective(params: np.ndarray) -> float:
        state['count'] += 1

        # ── Bounds penalty ───────────────────────────────────────────────────
        for i, op in enumerate(opt_params):
            if not (op.lo <= params[i] <= op.hi):
                return BIG

        # ── TALYS evaluation ─────────────────────────────────────────────────
        out = run_talys_get_xs(params, cfg)
        if out is None:
            return BIG
        _, y_t = out
        if len(y_t) != len(y_exp):
            return BIG

        # ── Log-space reduced chi-square ─────────────────────────────────────
        log_res  = (np.log(y_exp[mask] + FLOOR) - np.log(y_t[mask] + FLOOR)) / EXP_REL_ERR
        dof      = max(len(y_exp[mask]) - len(params), 1)
        chi2_red = float(np.sum(log_res ** 2) / dof)

        # ── Gaussian prior ───────────────────────────────────────────────────
        prior = 0.0
        if PRIOR_STD > 0.0:
            sigma = PRIOR_STD * np.maximum(np.abs(X0), PRIOR_FLOOR)
            prior = float(np.sum(((params - X0) / sigma) ** 2))

        total = chi2_red + prior

        if total < state['best'][0]:
            state['best'] = (total, params.copy())

        if DEBUG_EVERY and (state['count'] % DEBUG_EVERY == 0):
            best_val, _ = state['best']
            pstr = "  ".join(
                f"{op.name}={params[i]:.5g}" for i, op in enumerate(opt_params)
            )
            print(
                f"[eval {state['count']:4d}]  {pstr}  "
                f"chi2_red={chi2_red:.4e}  prior={prior:.4e}  total={total:.4e}  "
                f"| best={best_val:.4e}",
                flush=True,
            )

        return total

    objective.state = state
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_best_fit_xs(cfg, x_exp, y_exp, x_t, y_t, params_best, y_err=None, out_dir="."):
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
    plt.xlabel("Energy [MeV]")
    plt.ylabel("Cross section [mb]")
    if log_y:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=8)
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
def save_results(cfg, x_exp, y_exp, y_err, x_xs, y_xs, x_rate, y_rate, params_best, out_dir="."):
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
    lines.append(f"Min reduced chi-square: {res.fun:.6g}\n")
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

    # ── Load experimental data ────────────────────────────────────────────────
    exp_file = script.get('exp_file')
    if not exp_file or not os.path.exists(exp_file):
        raise FileNotFoundError(
            f"Experimental data file not found: {exp_file!r}\n"
            "Set exp_file in the [talys] section of tesseract.in."
        )

    df_exp      = pd.read_csv(exp_file, comment="#")
    x_exp       = df_exp.iloc[:, 0].to_numpy(dtype=float)
    y_exp       = df_exp.iloc[:, 1].to_numpy(dtype=float)
    EXP_REL_ERR = float(script.get('exp_rel_err', '0.10'))
    y_err       = (df_exp.iloc[:, 2].to_numpy(dtype=float)
                   if df_exp.shape[1] >= 3 else EXP_REL_ERR * y_exp)
    E_FIT_MIN   = float(script.get('e_fit_min', '2.0'))
    mask        = (y_exp > 1e-10) & (x_exp >= E_FIT_MIN)

    if np.any(~np.isfinite(y_exp)):
        raise ValueError("Experimental y contains non-finite values.")
    if np.any(y_exp < 0):
        raise ValueError("Experimental y contains negative values.")

    # Attach energy grid to cfg so write_talys_files can use it
    cfg['_x_exp'] = x_exp

    # ── Debug mode: one verbose TALYS call at x0, then exit ──────────────────
    if args.debug_talys:
        X0_dbg = np.array([op.x0 for op in cfg['opt_params']], dtype=float)
        with tempfile.TemporaryDirectory(prefix="talys_debug_") as wd:
            inp = write_talys_files(wd, X0_dbg, cfg, astro="n", astrogs="n")
            run_talys(wd, inp, verbose=True)
            ap_path = os.path.join(wd, "ap.tot")
            if os.path.exists(ap_path):
                print(f"\n[debug] ap.tot contents:\n")
                with open(ap_path) as fh:
                    print(fh.read())
            else:
                print("\n[debug] ap.tot was NOT created.")
        raise SystemExit(0)

    # ── Optimisation setup ────────────────────────────────────────────────────
    X0      = np.array([op.x0 for op in opt_params], dtype=float)
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

    obj = make_objective(cfg, x_exp, y_exp, y_err, mask)

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

    res = minimize(obj, X0, method=method, options=options, callback=callback)

    print("\n=== Optimisation complete ===")
    print(res)
    params_best = res.x
    print("\nBest-fit parameters:")
    for i, op in enumerate(opt_params):
        print(f"  {op.name} = {params_best[i]:.6g}")
    print(f"Min reduced chi-square = {res.fun:.6g}")

    # ── Best-fit XS curve ─────────────────────────────────────────────────────
    out_xs = run_talys_get_xs(params_best, cfg)
    if out_xs is None:
        print("\nWARNING: TALYS failed at best-fit parameters.")
        return
    x_t, y_t = out_xs
    plot_best_fit_xs(cfg, x_exp, y_exp, x_t, y_t, params_best, y_err=y_err,
                     out_dir=run_dir)

    # ── Reaction rate at best fit ─────────────────────────────────────────────
    x_rate: np.ndarray = np.array([])
    y_rate: np.ndarray = np.array([])
    try:
        rate_best, _ = run_talys_get_rate(params_best, cfg)
        if rate_best is None:
            print("\nWARNING: TALYS failed to produce astrorate.p at best fit.")
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
                               out_file=shared_out)
    save_results(cfg, x_exp, y_exp, y_err, x_t, y_t, x_rate, y_rate, params_best,
                 out_dir=run_dir)


if __name__ == "__main__":
    main()
