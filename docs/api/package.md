# Package

`nucres` is the top-level Python package for THICC's resonance sampling,
Breit-Wigner cross sections, reaction-rate integration, and HFB-driven spectrum
generation.

This page is the package overview, not the deepest reference page. The idea is:

- start here to understand the public entry points
- drop into the module pages for the detailed API

## What `nucres` Re-Exports

The top-level package is mostly a convenience import layer. In practice, these
are the main groups of objects you will reach for:

- Resonance modeling:
  `Resonance`, `sigma_bw_constant`, `sigma_bw_energy_dep`, `energy_grid`
- Reaction-rate integration:
  `na_sigma_v_from_sigma`, `ReactionRateResult`
- HFB-driven spectrum generation:
  `HFBSamplerConfig`, `GeneratedSpectrum`, `synthesize_sigma_from_hfb`
- Higher-level request/response API:
  `HFBRateRequest`, `generate_spectrum`, `compute_rate_table`, `rate_table_as_dict`
- Sampling utilities:
  `sample_er_sqrt_uniform`, `sample_er_increasing_pdf`,
  `porter_thomas_factors`, `nonhomogeneous_poisson_placements`,
  `mean_particle_width_from_strength`,
  `mean_particle_width_from_penetrability`, `fluctuate_widths`

## Suggested Reading Order

If you are new to the package, the shortest useful route is:

1. [Resonance](resonance.md) for the core cross-section model
2. [Rates](rates.md) for reaction-rate integration
3. [Generator](generator.md) for HFB-based spectrum synthesis
4. [Model](model.md) for the higher-level user-facing API

## Typical Usage Paths

Use the low-level path if you want direct control over resonance parameters:

```python
from nucres import Resonance, energy_grid, sigma_bw_constant
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
E = energy_grid(r.E_r, half_width_eV=25.0, n=200)
sigma = sigma_bw_constant(E, r)
```

Use the high-level path if you want THICC to build a synthetic spectrum and
rate table from HFB densities:

```python
from nucres import HFBRateRequest, compute_rate_table

request = HFBRateRequest(Z=12, A=24, J=1.0, pi=1)
result = compute_rate_table(request, temperatures=[0.1, 0.2, 0.3])
```

This high-level path requires HFB level-density tables to be available locally.

## Top-Level Namespace

The rendered API block below shows the actual objects exported from `nucres`.

::: nucres
    options:
      filters: ["!^_"]
