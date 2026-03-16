# Lab To CM

Utility for converting laboratory-frame projectile energies to center-of-mass
energies using AME masses.

## Example

```python
from nucres.lab_to_cm import lab_to_cm

E_cm = lab_to_cm(5.0, 22, "Ne", 4, "He")
```

## See Also

- [Read Qvals](read_qvals.md) for related AME-backed mass helpers

::: nucres.lab_to_cm
    options:
      filters: ["!^_"]
