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
