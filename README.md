# TESSERACT
Physics helpers and Monte Carlo tools for nuclear resonance cross sections and rate modeling.

**Install**

```bash
git clone https://github.com/TAMU-Pangolins/TESSERACT.git
cd TESSERACT
uv sync --locked --python 3.11
uv run python download_data.py
```

`download_data.py` installs the complete HFB table and correction dataset from
[`TAMU-Pangolins/TESSERACT-data`](https://github.com/TAMU-Pangolins/TESSERACT-data). It
downloads a checksum-verified ZIP archive of roughly 76 MB and extracts roughly
488 MB of data. Set `NUCRES_DATA_ROOT` to choose a different cache directory.

**Example calculation**
```bash
uv run -- python - <<'PY'
from nucres import Resonance, energy_grid
from nucres.resonance import sigma_bw_constant
from nucres.physics import MASS_PROTON

r = Resonance(E_r=1000.0, J=0.5, s1=0.5, s2=0.5, m1=MASS_PROTON, m2=MASS_PROTON, Gamma_i=1.0, Gamma_o=1.0)
E = energy_grid(r.E_r, half_width_eV=25.0, n=5)
sigma = sigma_bw_constant(E, r)
print(E)
print(sigma)
PY
```

**Documentation (local)**
Run the docs site locally:
```bash
uv sync --locked --extra docs --python 3.11
uv run --extra docs mkdocs serve
```
Then open `http://127.0.0.1:8000`.

Check the site with `uv run --extra docs mkdocs build --strict`.
See [documentation deployment](docs/deployment.md) for the one-time GitHub Pages setup.

**Core Features**
- Breit–Wigner cross sections with constant or energy‑dependent widths
- HFB‑driven resonance sampling with selectable Poisson or same-\(J^\pi\)
  Wigner-surmise spacing
- Reaction rate computation from synthesized spectra
- Helper utilities for kinematics, units, and sampling distributions

**Verify Data Setup**
```bash
uv run -- python -c "import nucres; print('nucres ready, data root =', nucres.config.resolve_data_root())"
```

**Data Requirements**
- RIPL‑3 level density tables are fetched automatically for HFB-driven workflows.
- AME2020 mass data (`data/ame20.csv`) is used for Q‑value related utilities.

**Tests**
```bash
uv run -- python -m unittest discover -s tests -p "test_*.py"
```

**Project Structure**
- **nucres/**  
  - `api.py`, `__init__.py` - public API.  
  
  - `physics.py` - public API, physical constants, unit conversions, and basic kinematics helpers.  

  - `resonance.py` - resonance dataclass plus single level Breit-Wigner sigma(E) (constant and energy-dependent widths), Coulomb penetrability via Coulomb wave functions f and g. 

  - `sampling.py` - Porter–Thomas and inverse-CDF samplers, Poisson and unfolded Wigner-surmise placement of levels, mean widths from strength/penetrability.

  - `generator.py` - Monte Carlo synthesis of spectra using HFB level densities. Integrates sampled resonances with Breit-Wigner shapes. 

  - `model.py`, `rates.py` - high-level modeling API, cross section -> reaction rate conversion

  - `hfb_adapter.py` - parsing HFB combinatorial level density tables.

  - `plotting.py` - optional visualization helpers for resonances/spectra.

- **common/**  

  HFB density table parsers (`densities_retrieval.py`).

- **data/**

  All data, downloaded or otherwise set in the repository. 

- **tests/**  
  Unit tests for the generator and model APIs. (temporary) 


# Developer Standards


- Prefer stdlib first (`csv`, `pathlib.Path`, etc.) and keep dependencies to the existing set (`numpy`, `mpmath`, `matplotlib`)

- Keep units consistent (MeV/eV barns as used in `nucres`), document inputs/outputs in docstrings, and return informative errors for missing datasets or bad inputs.

## License

TESSERACT is licensed under the [GNU General Public License, version 3](LICENSE)
(`GPL-3.0-only`). Third-party dependencies and data retain their own licenses.
