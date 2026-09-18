# Workflows

This page outlines common end-to-end flows in the project. Unlike
[Examples](examples.md), the emphasis here is on prerequisites and where each
step fits into the larger TESSERACT process.

## A. Generate a Spectrum from HFB Densities

The complete HFB level-density dataset is downloaded and cached automatically
on first use. Create a `HFBSamplerConfig` and synthesize a spectrum.

```bash
uv run -- python - <<'PY'
from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb

cfg = HFBSamplerConfig(Z=12, A=24, J=1.0, pi=+1)
spec = synthesize_sigma_from_hfb(cfg)
print("grid points:", spec.energy_MeV.size)
print("resonances:", len(spec.resonances))
PY
```

## B. Compute a Reaction Rate Table

`HFBRateRequest` uses the same automatic HFB data cache as the generator layer.

```bash
uv run -- python - <<'PY'
import numpy as np
from nucres.model import HFBRateRequest, compute_rate_table

req = HFBRateRequest(Z=12, A=24, J=1.0, pi=+1)
temps = np.linspace(0.1, 2.0, 5)  # GK
result = compute_rate_table(req, temps, temperature_unit="GK")
print(result.temperature)
print(result.na_sigma_v)
PY
```

## C. Diagnostic Plot for a Reaction Input

This workflow does not depend on HFB density tables.

```bash
uv run -- python scripts/diagnostic_plot.py --input 'input/22Mg(a,p)25Al.txt' --no-show
```
