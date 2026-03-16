# HFB Adapter

HFB table loading, correction handling, and level-density grid construction.
Use this page when you need to understand how THICC maps density files into
callable \(\rho(E)\) functions.

## Example

```python
from pathlib import Path

from nucres.hfb_adapter import build_total_density_grid

E_mev, rho_per_mev = build_total_density_grid(
    Z=9,
    data_root=Path("data/densities/level-densities-hfb"),
    pi=1,
    E_min_mev=0.2,
    E_max_mev=0.6,
    n_points=201,
)
```

This is the adapter layer used underneath the generator when it builds the
level-density grid that drives resonance placement.

::: nucres.hfb_adapter
    options:
      filters: ["!^_"]

## See Also

- [Generator](generator.md) for the main consumer of these density grids
- [Config](config.md) for data-root resolution
