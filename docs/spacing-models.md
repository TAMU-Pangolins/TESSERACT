# Resonance Spacing Models

TESSERACT supports two resonance-energy placement modes through
`spacing_model`.

## Poisson mode

```ini
[resonance]
spacing_model = poisson
```

This is the default and preserves the historical generator behavior. Level
counts are Poisson distributed and energies are conditionally sampled from the
HFB density. Levels do not repel one another.

## Wigner mode

```ini
[resonance]
spacing_model = wigner
```

This mode treats each fixed-spin, fixed-parity sequence as an independent
ladder. It integrates the corresponding HFB density to create an unfolded
coordinate, generates unit-mean nearest-neighbour spacings from the GOE Wigner
surmise, and maps the levels back to physical energy.

Only levels in the same \(J^\pi\) sequence repel. Independent sequences may
still lie close together in the combined spectrum, as required physically.

The current implementation is a stationary nearest-neighbour renewal model.
It reproduces the Wigner-surmise spacing distribution without pinning a level
to an energy-window boundary. It is not a full GOE matrix calculation and does
not reproduce long-range GOE spectral correlations.

## Validation and limits

Unit tests cover the sampled mean spacing, suppression of small spacings,
reproducibility, energy bounds, preservation of expected level counts,
zero-density regions, and HFB spectrum generation. Run them with:

```bash
python -m unittest tests.test_sampling tests.test_generator tests.test_model
```

Local ensemble comparisons also checked unfolded spacing histograms, cumulative
distributions, separate spin/parity sequences, number variance, and average
level populations for the historical alpha-proton reactions. The comparisons
used 300 realizations per parity and model over 1.5–3.5 MeV. Spacing panels
with insufficient expected populations were omitted. These comparisons used
explicitly selected isotope records; they do not resolve the existing HFB
isotope-selection limitation or validate exported spins and reaction rates.

Spacing diagnostics must compare neighbours within the same spin and parity.
Pooling spacings after separate unfolding is appropriate; taking neighbours
from a spectrum with different spin/parity sequences mixed together can hide
level repulsion.

Poisson remains the default. Validation of the complete averaging and TALYS
inference workflow remains separate from this spacing-model integration.
