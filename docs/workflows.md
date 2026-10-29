# Workflows

This page outlines common end-to-end flows in the project. Unlike
[Examples](examples.md), the emphasis here is on prerequisites and where each
step fits into the larger TESSERACT process.

## Full pipeline: 22Mg(a,p)25Al

The runnable configuration lives in
[`examples/22Mg_ap_25Al.in`](https://github.com/TAMU-Pangolins/TESSERACT/blob/main/examples/22Mg_ap_25Al.in).
It runs resonance sampling, optional RatesMC execution, cross-section integration,
and TALYS fitting for ten reproducible realisations. Edit that file to change
parameters; the configuration is maintained there rather than copied into this page.

First complete the
[installation instructions](https://github.com/TAMU-Pangolins/TESSERACT#installation-guide):
install the Python dependencies and HFB tables, make TALYS available on `PATH`,
and configure RatesMC with the documented source patch if you want its reference
rates. If RatesMC cannot be found, its step is skipped; the other stages still run.

From the repository root, start with one realisation:

```bash
uv run python tesseract.py --input examples/22Mg_ap_25Al.in --run-idx 0
```

Then run all ten:

```bash
uv run python tesseract.py --input examples/22Mg_ap_25Al.in
```

Existing output files are reused. When changing parameters, use a fresh output
location or remove the affected generated outputs before rerunning, so old results
are not reused with the new configuration.

All example outputs are under `outputs/example_22Mg_ap_25Al/`:

| Stage | Output relative to that directory |
| --- | --- |
| Resonance sampling | `resonances/22Mg(a,p)25Al/RUN_j/22Mg(a,p)25Al.in` |
| RatesMC, when available | `resonances/22Mg(a,p)25Al/RUN_j/22Mg(a,p)25Al.out` |
| Cross-section integration | `cross_sections/22Mg_ap_25Al_integrated_xs_dE_0.2_runj.csv` |
| Unintegrated cross sections | `cross_sections/22Mg(a,p)25Al_xs_unintegrated_parallel_runj.txt` |
| TALYS fitting | `talys/RUN_j/talys_results_*.npz` |

Here `j` is the run index, from 0 through 9. The configuration uses one bin width
and a limited least-squares optimiser. Even a single realisation can take time:
Coulomb penetrability calculations and repeated external TALYS calls dominate the
work. To shorten an initial check, narrow the energy window or temporarily remove
`[talys]` and its contents to run only the earlier stages. The `samples` setting
controls density-grid resolution, not the number of sampled resonances.

### TALYS reaction-output selection

The optimizer reads cross sections for the configured projectile and ejectile.
It checks, in order, `[talys] talys_xs_file` when specified,
`<projectile><ejectile>.tot`, and the exclusive-channel file
`xsNNNNNN.tot`. The six digits count emitted neutrons, protons, deuterons,
tritons, helium-3 nuclei, and alpha particles. For example, `(a,p)` uses
`xs010000.tot`, while `(p,g)` uses `xs000000.tot`. Enable `channels = y`
and `filechannels = y` in `[talys]` to request exclusive output files.

The first existing file is used. An absent explicit override produces a warning
before checking the standard channel files; an invalid selected file is reported
as a failure. Cross sections retain TALYS's millibarn units and the configured
experimental energy frame.

Rates are read only from `astrorate.<ejectile>`, such as `astrorate.g` for
radiative capture or `astrorate.p` for a proton exit. Temperatures remain in GK
and rates in cm³ mol⁻¹ s⁻¹. A missing channel output is reported; the optimizer
does not substitute another ejectile's rate. This selection does not cause TALYS
to generate rate files that the installed version does not support.

## A. Generate a Spectrum from HFB Densities

The complete HFB level-density dataset is downloaded and cached automatically
on first use. Create a `HFBSamplerConfig` and synthesize a spectrum.

```bash
uv run -- python - <<'PY'
from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb

cfg = HFBSamplerConfig(Z=12, A=24, J=1.0, pi=+1)
spec = synthesize_sigma_from_hfb(cfg)
print("grid points:", spec.energy_MeV.size)
print("resonances:", len(spec.resonances))
PY
```

## B. Compute a Reaction Rate Table

`HFBRateRequest` uses the same automatic HFB data cache as the generator layer.

```bash
uv run -- python - <<'PY'
import numpy as np
from nucres.model import HFBRateRequest, compute_rate_table

req = HFBRateRequest(Z=12, A=24, J=1.0, pi=+1)
temps = np.linspace(0.1, 2.0, 5)  # GK
result = compute_rate_table(req, temps, temperature_unit="GK")
print(result.temperature)
print(result.na_sigma_v)
PY
```

## C. Diagnostic Plot for a Reaction Input

This workflow does not depend on HFB density tables.

```bash
uv run -- python scripts/diagnostic_plot.py --input 'input/22Mg(a,p)25Al.txt' --no-show
```
