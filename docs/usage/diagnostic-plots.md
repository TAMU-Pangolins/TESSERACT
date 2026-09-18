# Diagnostic Plots

The diagnostic plot script is intended to sanity-check key physics quantities for a
single reaction.

**What it plots**

- Astrophysical S-factor `S(E)` in `keV·b`
- Dimensionless reduced widths `theta^2` (computed)
- Penetrability `P_l(E)` (dimensionless)

**Default reaction**

The default input file is `input/22Mg(a,p)25Al.txt`.

**Run locally**

```bash
python scripts/diagnostic_plot.py
```

**Common overrides**

```bash
python scripts/diagnostic_plot.py --input 'outputs/22Mg(a,p)25Al/RUN_0/22Mg(a,p)25Al.in'
python scripts/diagnostic_plot.py --emin 0.1 --emax 3.0 --npts 3000
python scripts/diagnostic_plot.py --output outputs/diagnostics/custom.png --no-show
```

**Notes**

- `S(E)` uses `S(E) = E_keV * sigma(E) * exp(2*pi*eta)` with `sigma(E)` in barns.
- `theta^2` is computed from the entrance width and the Wigner limit.
