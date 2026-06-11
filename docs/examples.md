# Examples

This page is a grab bag of short runnable snippets. For broader step-by-step
flows, see [Workflows](workflows.md).

## Minimal Breit–Wigner

```bash
uv run -- python - <<'PY'
from nucres import Resonance, energy_grid
from nucres.resonance import sigma_bw_constant
from nucres.physics import MASS_PROTON

r = Resonance(
    E_r=1000.0,
    J=0.5,
    s1=0.5,
    s2=0.5,
    m1=MASS_PROTON,
    m2=MASS_PROTON,
    Gamma_i=1.0,
    Gamma_o=1.0,
)
E = energy_grid(r.E_r, half_width_eV=25.0, n=5)
sigma = sigma_bw_constant(E, r)
print(E)
print(sigma)
PY
```

This example is self-contained and does not require HFB density tables.

## Diagnostic Plot (Default Reaction)

```bash
uv run -- python scripts/diagnostic_plot.py
```

## Rate Table from HFB

The complete HFB level-density dataset is downloaded and cached on first use.

```bash
uv run -- python - <<'PY'
import numpy as np
from nucres.model import HFBRateRequest, compute_rate_table

req = HFBRateRequest(Z=12, A=24, J=1.0, pi=+1)
temps = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
result = compute_rate_table(req, temps, temperature_unit="GK")
print(result.temperature)
print(result.na_sigma_v)
PY
```
