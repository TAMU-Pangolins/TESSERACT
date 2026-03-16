# Rates

Reaction-rate integration utilities for converting sampled cross sections into
\(N_A \langle \sigma v \rangle\) tables.

## Example

```python
from nucres.rates import na_sigma_v_from_sigma

E = [9.90e2, 1.00e3, 1.01e3]
sigma = [0.01, 0.2, 0.01]

result = na_sigma_v_from_sigma(
    E,
    sigma,
    [0.1, 0.2, 0.3],
    energy_unit="eV",
    sigma_unit="barn",
    temperature_unit="GK",
    result_unit="cm^3/mol/s",
)
```

This is the final integration step once you already have a cross section sampled
on an energy grid.

## See Also

- [Resonance](resonance.md) for producing \(\sigma(E)\)
- [Generator](generator.md) for building synthetic spectra
- [Model](model.md) for the higher-level end-to-end API

::: nucres.rates
    options:
      filters: ["!^_"]
