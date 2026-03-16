# Read Qvals

AME mass-table helpers for Q-value calculations and nuclide-token parsing.

## Example

```python
from nucres.read_qvals import calc_qval, load_ame

ame = load_ame()
q_mev = calc_qval(1, 22, 12, 22, 1, 1, ame=ame)
```

This module is useful when you need mass-derived quantities without manually
reading the AME table.

## See Also

- [Lab To CM](lab_to_cm.md) for a related AME-backed conversion helper

::: nucres.read_qvals
    options:
      filters: ["!^_"]
