# RatesMC Export

Serialization helpers for converting sampled `Resonance` objects into RatesMC-style
rows and output blocks.

## Example

```python
from pathlib import Path

from nucres.physics import MASS_PROTON
from nucres.ratesmc_export import (
    RatesMCExportOptions,
    render_rows,
    resonant_header_line,
    write_resonant_block,
)
from nucres.resonance import Resonance

res = Resonance(
    E_r=1000.0,
    J=1.0,
    s1=0.5,
    s2=0.5,
    m1=MASS_PROTON,
    m2=MASS_PROTON,
    Gamma_i=1.0,
    Gamma_o=2.0,
    L1=0,
)
opts = RatesMCExportOptions()
rows, widths = render_rows([res], opts)
header = resonant_header_line(widths, opts)
write_resonant_block(Path("RatesMC_resonances.txt"), rows, header=header)
```

This page is most useful once you already have sampled or hand-built
`Resonance` objects.

::: nucres.ratesmc_export
    options:
      filters: ["!^_"]

## See Also

- [Resonance](resonance.md) for the source resonance objects
- [Generator](generator.md) for producing resonance populations
- [Model](model.md) if you want to start from a higher-level request
