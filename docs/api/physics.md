# Physics

Physical constants, unit conversions, and small kinematic helpers used across
the package.

## Example

```python
from nucres.physics import MASS_PROTON, energy_grid, reduced_mass, wavenumber_from_E_eV

E = energy_grid(1000.0, half_width_eV=25.0, n=5)
mu = reduced_mass(MASS_PROTON, MASS_PROTON)
k = wavenumber_from_E_eV(E, mu)
```

::: nucres.physics
    options:
      filters: ["!^_"]

## See Also

- [Resonance](resonance.md) for the cross-section formulas that consume these helpers
- [Rates](rates.md) for the thermal integration step
