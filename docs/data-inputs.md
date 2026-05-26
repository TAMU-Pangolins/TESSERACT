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

The HFB-driven synthesis requires RIPL-3 level density tables in:

```
data/densities/level-densities-hfb
```

By default, `nucres` looks for `zXXX.tab` (and optional `zXXX.cor`) files
through `NUCRES_DATA_ROOT` or the checkout-local default.

The helper script can download these tables:

```bash
python download_ripl.py
```

If the RIPL-3 server returns `HTTP Error 403: Forbidden`, the server or its
front-end protection may be blocking scripted downloads. The manual fallback is:

1. Open `https://www-nds.iaea.org/RIPL-3/densities/level-densities-hfb/` in a browser.
2. Download the needed `zXXX.tab` files and matching `zXXX.cor` correction files
   when present.
3. Place them in `data/densities/level-densities-hfb`.
4. Set the data root if the files are outside the checkout-local default:

```bash
export NUCRES_DATA_ROOT="/path/to/level-densities-hfb"
```

## AME Mass Data

Q-value utilities use `data/ame20.csv`.

## Units Summary

- Energies in the resonance API are **eV**
- `penetrability_P_l_mev` uses **MeV**
- Cross sections are **barns**
- Rate outputs are **cm^3/mol/s** by default
