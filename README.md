# TESSERACT

**Thick-target Energy StudieS Estimating Reaction-rate Ambiguity from Cross–section Techniques**

[Documentation](https://tamu-pangolins.github.io/TESSERACT/)

## Installation guide

Start with Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The commands below install the Python dependencies and HFB data, then connect
your TALYS and RatesMC installations.

**1. Install TESSERACT and the HFB tables.**

```bash
git clone https://github.com/TAMU-Pangolins/TESSERACT.git
cd TESSERACT
uv sync --locked --python 3.11
uv run python download_data.py --dest data/densities/level-densities-hfb
```

The HFB archive is
[`hfb/hfb-level-densities-v1.zip`](https://raw.githubusercontent.com/TAMU-Pangolins/TESSERACT-data/main/hfb/hfb-level-densities-v1.zip)
in [TESSERACT-data](https://github.com/TAMU-Pangolins/TESSERACT-data).
`download_data.py` downloads it, verifies its checksum, extracts the tables into
`data/densities/level-densities-hfb`, and saves that location for subsequent runs.
No manual download or extraction is needed. Allow approximately 76 MB for the
archive and 488 MB for the extracted tables.

To store the tables elsewhere, replace the `--dest` path above. For an existing
dataset, set `NUCRES_DATA_ROOT` to the directory containing the `zXXX.tab` files.

**2. Connect TALYS and RatesMC.**

If needed, first follow the upstream installation instructions for
[TALYS](https://github.com/arjankoning1/talys#installation) and
[RatesMC](https://github.com/rlongland/RatesMC#installation-instructions), including
their data files and build dependencies. These programs are installed separately
from the Python package. Then set the paths to your installations:

```bash
export PATH="/absolute/path/to/talys/bin:$PATH"
export RATESMC_BIN="/absolute/path/to/RatesMC"

command -v talys
test -x "$RATESMC_BIN" && printf 'RatesMC executable is accessible\n'
```

The `PATH` entry is the directory containing the executable named `talys`.
`RATESMC_BIN` is the executable itself and is read by `run_ratesmc_batches.sh`.
Keep `mass_1.mas20` and `nubase_3.mas20` beside that executable when required by
your RatesMC installation; the batch script copies or links them into each run
directory.

Set these variables in the shell or batch-job environment that launches the
calculation. Add the exports to your shell configuration for persistent
interactive use.

## Example calculation

Calculate a single Breit–Wigner resonance using illustrative parameters.
This example requires neither TALYS nor RatesMC and does not download HFB data.

```bash
uv run python - <<'PY'
from nucres import Resonance, energy_grid, sigma_bw_constant
from nucres.physics import MASS_PROTON

resonance = Resonance(
    E_r=1000.0,  # resonance energy, eV
    J=1.0,
    s1=0.5,
    s2=0.5,
    m1=MASS_PROTON,  # kg
    m2=MASS_PROTON,
    Gamma_i=1.0,  # entrance partial width, eV
    Gamma_o=1.0,  # exit partial width, eV
)
energies_ev = energy_grid(resonance.E_r, half_width_eV=25.0, n=5)
cross_sections_barn = sigma_bw_constant(energies_ev, resonance)
print("Energy (eV):", energies_ev)
print("Cross section (barn):", cross_sections_barn)
PY
```

## Running a full example (22Mg(a,p)25Al)

This walks through the complete pipeline — resonance sampling, RatesMC,
cross-section integration, and TALYS fitting — for a single reaction. It
uses only `runs = 10` and a reduced TALYS optimiser so a first attempt
finishes in a reasonable time instead of running for ages; scale `runs` up
once you've confirmed everything works. Requires the Installation guide
steps above (HFB tables, TALYS, RatesMC) to be done first — RatesMC is
optional for this walkthrough (see step 2).

**1. Save this as `tesseract_22Mg_example.in`** in the repository root
(`*.in` files are gitignored, so this is a local working file, not something
you commit):

```ini
[basics]
reaction     = 22Mg(a,p)25Al
E_min_mev    = 0.1
E_max_mev    = 10.0

# Needed by the [talys] step below.
projectile_Z = 2
projectile_A = 4
target_Z     = 12
target_A     = 22
ejectile_Z   = 1
ejectile_A   = 1
residual_Z   = 13
residual_A   = 25

[resonance]
runs      = 10     # small ensemble for a quick first run; scale up later
samples   = 1000   # HFB density sampling points (default is 5000)
seed      = 1      # each run gets seed+run_index, so all 10 are reproducible but distinct
mean_i    = 0.010
mean_o    = 0.0045

[ratesmc]
# No ratesmc_bin set: auto-detects RatesMC on $PATH / $RATESMC_BIN (see
# Installation guide). If RatesMC isn't installed, this step prints a
# warning and is skipped -- it does not block the rest of the pipeline.

[integration]
dE = 0.2           # one bin width keeps this example to one TALYS fit per run

[talys]
ldmodel      = 1
alphaomp     = 6
xseps        = 1.e-20
transeps     = 1.e-20
outbasic     = y
channels     = y
filechannels = y

method          = least_squares   # far fewer TALYS calls than Powell/Nelder-Mead
maxiter         = 50
xtol            = 1e-3
ftol            = 1e-3
lsq_workers     = 1               # raise only if you can actually request that many cores
exp_rel_err     = 0.10
e_fit_min       = 0.0
prior_rel_std   = 2.00
prior_abs_floor = 0.1
debug_every     = 10

\opt
# Level density parameters for residual 25Al (Z=13, A=25), ldmodel 1 (BSFG).
# Format: keyword Z A  x0  lo  hi
T      13   25    2.167    0.1    10.0
E0     13   25   -1.328   -5.0    5.0
a      13   25    3.844    1.0   10.0
\opt
```

**2. Run it:**

```bash
uv run python tesseract.py --input tesseract_22Mg_example.in
```

If you just want to sanity-check the setup before committing to all 10 runs,
test a single one first:

```bash
uv run python tesseract.py --input tesseract_22Mg_example.in --run-idx 0
```

**3. Where the output lands:**

| Step | Output |
| --- | --- |
| `[resonance]` | `outputs/22Mg(a,p)25Al/RUN_0.../22Mg(a,p)25Al.in` |
| `[ratesmc]` (if installed) | `outputs/22Mg(a,p)25Al/RUN_0.../22Mg(a,p)25Al.out` |
| `[integration]` | `22Mg_ap_25Al_integrated_xs_dE_0.2_run0....csv` in the current directory (set `output_dir`/`unint_output_dir` under `[integration]` to put these somewhere tidier) |
| `[talys]` | `talys_opt/RUN_0.../talys_results_*.npz` |

**What to expect timing-wise:** don't assume `[resonance]`/`[integration]`
are instant just because `[talys]` isn't involved yet. The Coulomb
penetrability calculation each sampled resonance needs is done with `mpmath`
for precision, and that cost is paid once per resonance per energy point — so
runtime scales with how many resonances the HFB level density actually
produces for this reaction's energy window, which varies a lot by nuclide and
isn't always obvious up front. `[talys]` adds further time on top of that
(one external TALYS call per optimised parameter per iteration;
`least_squares` with the 3-parameter `\opt` block above needs only ~4 calls
per iteration versus hundreds for Powell, but it's still extra). If a first
attempt seems to be taking a while, that alone isn't a sign of a bug. Things
that genuinely shrink the work:
- Lower `samples` under `[resonance]` (density-grid points — fewer points,
  fewer sampled resonances).
- Narrow `E_min_mev`/`E_max_mev` under `[basics]` to a smaller window.
- Drop `[talys]` from your first attempt entirely while you confirm the
  earlier steps work — the pipeline runs `[resonance]` → `[ratesmc]` →
  `[integration]` and stops cleanly with no error when a later section is
  simply absent.

## Core features

- Breit–Wigner cross sections with constant or energy-dependent entrance widths.
- Coulomb penetrability and optional JWKB transmission through a real alpha optical potential.
- HFB-driven resonance sampling with Poisson or Wigner-surmise spacing within each spin/parity sequence.
- Cross-section averaging over energy bins and TALYS fitting workflows.
- Reaction-rate integration and RatesMC-compatible resonance inputs.

## Data and program requirements

| Requirement | Purpose |
| --- | --- |
| Python dependencies | NumPy, SciPy, mpmath, Matplotlib, and pandas; installed by `uv sync`. |
| TALYS and its nuclear data libraries | Cross-section fitting and reaction-rate calculations. |
| RatesMC and its support files | Monte Carlo reference reaction rates. |
| RIPL-3 HFB level-density tables | Resonance generation; downloaded automatically on first use or with `download_data.py`. |
| AME2020 mass data | Q-value utilities; included in `data/ame20.csv`. |
| Bash and ripgrep (`rg`) | Execution and log checks in `run_ratesmc_batches.sh`. |

## Project structure

| Path | Contents |
| --- | --- |
| `nucres/` | Public Python API, resonance models, sampling, cross sections, rate integration, and data access. |
| `common/` | Shared HFB table parsers. |
| `data/` | Bundled mass data and downloaded level-density tables. |
| `input/` | Reaction input templates. |
| `scripts/` | Analysis and plotting commands. |
| `tesseract.py` | Driver for resonance generation, cross-section averaging, and TALYS fitting. |
| `run_ratesmc_batches.sh` | RatesMC batch execution. |
| `docs/` | User documentation and API reference. |
| `tests/` | Numerical and interface checks. |

## License

TESSERACT is licensed under the [GNU General Public License, version 3](LICENSE)
(`GPL-3.0-only`). Third-party dependencies and data retain their own licenses.
