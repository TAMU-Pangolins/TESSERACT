# Sampling

Sampling primitives for resonance energies, widths, and Poisson placement models.

## Example

```python
import numpy as np

from nucres.sampling import porter_thomas_factors, sample_er_sqrt_uniform

rng = np.random.default_rng(0)
energies = sample_er_sqrt_uniform(8, e_min=900.0, e_max=1200.0, rng=rng)
width_factors = porter_thomas_factors(8, df=1, rng=rng)
```

These functions are useful when you want direct stochastic building blocks
without running the full HFB-driven generator.

::: nucres.sampling
    options:
      filters: ["!^_"]

## See Also

- [Generator](generator.md) for the higher-level synthesis workflow
- [Resonance](resonance.md) for turning sampled parameters into cross sections
