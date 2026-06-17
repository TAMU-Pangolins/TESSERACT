# Data & Inputs

This section documents the inputs expected by the project and their units.

## RatesMC Templates (`input/*.txt`)

These text files define reactions in a RatesMC-compatible format.
Key points from the headers:

- `Ecm`, `Exf` are in **keV**
- `wg`, `G1`, `G2`, `G3` are in **eV**
- `S(keVb)` is in **keV·b** (nonresonant contribution)

The main resonant block is labeled:

```
Resonant Contribution
```

Followed by a header row starting with `Ecm`.

## Generated Inputs (`outputs/.../*.in`)

Generated `.in` files reuse the same format as templates.
The diagnostic script can read either templates or generated files.

## HFB Level Density Tables

HFB-driven synthesis uses the RIPL-3 `zXXX.tab` and `zXXX.cor` level-density
tables. On first use, the complete hosted dataset is downloaded automatically
from [`aldusv/TESSERACT-data`](https://github.com/aldusv/TESSERACT-data).
The checksum-verified ZIP archive is roughly 76 MB and extracts to roughly 488 MB.

Source checkouts with `data/densities/level-densities-hfb` use that existing
directory. Installed packages default to
`~/.cache/nucres/densities/level-densities-hfb`.

To prefetch the hosted dataset:

```bash
uv run python download_data.py
```

Set the data root when files should live somewhere else:

```bash
export NUCRES_DATA_ROOT="/path/to/level-densities-hfb"
```

`NUCRES_DATA_BASE_URL` can override the hosted source for mirrors or offline
test environments.

## AME Mass Data

Q-value utilities use `data/ame20.csv`.

## Units Summary

- Energies in the resonance API are **eV**
- `penetrability_P_l_mev` uses **MeV**
- Cross sections are **barns**
- Rate outputs are **cm^3/mol/s** by default
