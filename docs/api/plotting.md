# Plotting

Quick plotting wrappers around the resonance cross-section helpers.

## Example

This example plots the line shape for a single resonance centered at
\(E_r = 1000\) eV. Use these wrappers when you want a quick diagnostic figure
without manually constructing the plotting grid and evaluating the cross section
yourself.

```python
from nucres.physics import MASS_PROTON
from nucres.plotting import plot_bw_constant
from nucres.resonance import Resonance

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

ax = plot_bw_constant(r, half_width_eV=25.0, n=200)
```

## See Also

- [Resonance](resonance.md) for the underlying evaluators
- [Physics](physics.md) for shared constants and grid helpers

::: nucres.plotting
    options:
      filters: ["!^_"]
