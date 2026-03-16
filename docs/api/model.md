# Model

High-level request/response layer that wraps the generator and rate integrator
into a simpler user-facing API.

## Example

This example shows the higher-level API for the same HFB-driven workflow. It is
useful when you want to specify the physical request once and let `nucres`
handle the lower-level sampler configuration internally.

```python
from pathlib import Path

from nucres.model import HFBRateRequest, generate_spectrum, compute_rate_table

request = HFBRateRequest(
    Z=9,
    A=18,
    J=1.0,
    pi=1,
    Gamma_i_mean_eV=0.5,
    Gamma_o_mean_eV=0.5,
    E_min_mev=0.2,
    E_max_mev=0.6,
    n_sigma_points=128,
    n_density_points=201,
    data_root=Path("data/densities/level-densities-hfb"),
    seed=123,
)

spectrum = generate_spectrum(request)
rate = compute_rate_table(request, [0.1, 0.2, 0.3], temperature_unit="GK")
```

Use this layer when you want a stable, user-facing API without manually
constructing `HFBSamplerConfig`.

This example requires local HFB level-density tables.

## See Also

- [Generator](generator.md) for the lower-level sampling API
- [Rates](rates.md) for the rate integration backend
- [Package](package.md) for the top-level import surface

::: nucres.model
    options:
      filters: ["!^_"]
