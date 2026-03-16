# Generator

Synthetic-spectrum generation from HFB level densities. This is the main low-level
entry point for building resonance populations and summed cross sections.

## Example

This example builds a synthetic resonance population from HFB level densities
for a narrow energy window, then converts the resulting cross section into a
reaction-rate sample. Use this layer when you want explicit control over the
sampling configuration.

```python
from pathlib import Path

from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb

data_root = Path("data/densities/level-densities-hfb")

cfg = HFBSamplerConfig(
    Z=9,
    data_root=data_root,
    A=18,
    J=1.0,
    pi=1,
    Gamma_i_mean_eV=0.5,
    Gamma_o_mean_eV=0.5,
    delta_E_mev=0.05,
    E_min_mev=0.2,
    E_max_mev=0.6,
    n_density_points=201,
    n_sigma_points=256,
    U_offset_mev=8.0,
    seed=42,
)

spectrum = synthesize_sigma_from_hfb(cfg)
rate = spectrum.compute_rate([0.1], temperature_unit="GK")
```

The resulting `spectrum` contains the cross-section grid, sampled resonances,
level-density grid, and metadata about the draw.

::: nucres.generator
    options:
      filters: ["!^_"]

## See Also

- [HFB Adapter](hfb_adapter.md) for loading and transforming HFB density tables
- [Resonance](resonance.md) for the per-resonance cross-section model
- [Model](model.md) for the higher-level request/response wrapper
