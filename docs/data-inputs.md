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

## AME Mass Data

Q-value utilities use `data/ame20.csv`.

## Units Summary

- Energies in the resonance API are **eV**
- `penetrability_P_l_mev` uses **MeV**
- Cross sections are **barns**
- Rate outputs are **cm^3/mol/s** by default
