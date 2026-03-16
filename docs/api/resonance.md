# Resonance

Breit-Wigner cross sections, penetrability helpers, and the core `Resonance`
dataclass. Use this page for the resonance-level physics model.

## Example

This example evaluates a single narrow resonance near \(E_r = 1000\) eV using
constant entrance and exit widths. It is the simplest way to inspect the
Breit-Wigner line shape directly.

```python
from nucres import Resonance, energy_grid, sigma_bw_constant
from nucres.physics import MASS_PROTON

r = Resonance(
    E_r=1000.0,
    J=1.0,
    s1=0.5,
    s2=0.5,
    m1=MASS_PROTON,
    m2=MASS_PROTON,
    Gamma_i=1.0,
    Gamma_o=1.0,
)

E_vals = energy_grid(r.E_r, half_width_eV=25.0, n=200)
sigma_vals = sigma_bw_constant(E_vals, r)
```

For an energy-dependent entrance width, switch to `sigma_bw_energy_dep(...)`
and provide the channel parameters and reduced-width input.

## See Also

- [Physics](physics.md) for shared constants and kinematic helpers
- [Plotting](plotting.md) for quick visualization wrappers
- [Rates](rates.md) for integrating \(\sigma(E)\) into rate tables

::: nucres.resonance
    options:
      filters: ["!^_"]
