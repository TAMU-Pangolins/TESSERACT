# THICC Documentation

Welcome. This documentation covers how to use the THICC tools locally.

Use this page for a quick sanity check and orientation:

- [Concepts](concepts.md) explains the core physics quantities
- [Workflows](workflows.md) shows end-to-end project flows
- [Examples](examples.md) collects short runnable snippets
- [API Reference](api/package.md) documents the `nucres` Python package

**Quick start**

1. Install project dependencies.

```bash
uv sync --python 3.11
```

2. Run a small sanity check.

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

This example is self-contained and does not require HFB density tables.

## HFB Data Prerequisite

Examples that use `HFBRateRequest`, `HFBSamplerConfig`, or `synthesize_sigma_from_hfb`
download the complete HFB level-density dataset from `aldusv/TESSERACT-data`
on first use and cache it locally. `NUCRES_DATA_ROOT` overrides the cache
directory.
