# THICC Documentation

Welcome. This documentation covers how to use the THICC tools locally.

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
