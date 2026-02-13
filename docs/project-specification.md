# Project Specification and Physical Motivation

This page captures the physics motivation, project scope, and implementation mapping for THICC.

## Physical Motivation

Stellar nuclear reaction rates are computed by folding the cross section with the Maxwell-Boltzmann distribution:

\[
\langle \sigma v \rangle = \int_0^\infty E\,e^{-E/kT}\,\sigma(E)\,dE
\]

Direct measurements are the gold standard because they measure the target quantity, \(\sigma(E)\), from controlled beam-target yields. For many astrophysical reactions this is difficult because:

- Cross sections are very small in the relevant energy region.
- Beam intensities are limited, especially for radioactive ion beams.
- Beam purity is often poor for inverse-kinematics measurements.
- Some key reactions involve \(\alpha\)-induced channels requiring gaseous or implanted targets.

Active-target techniques address these constraints by using gas as both target and detector medium, enabling:

1. High efficiency because reactions occur inside the detector volume.
2. Beam tracking and particle-ID support for contamination rejection.
3. High luminosity with thick targets while preserving event-by-event energy tracking.
4. Broad excitation-function coverage in one beam setting, reducing retuning overhead.

Two major active-target classes are relevant:

- Multi-sampling ionization chambers (MuSIC-style systems with segmented anodes).
- Active-target TPC systems (AT-TPC) with full track reconstruction from segmented readouts.

Key limitation: thick-target measurements typically have center-of-mass energy resolution broader than narrow resonance scales. This smooths resonance structure and can bias inferred rates when extrapolated with statistical models (for example TALYS). THICC focuses on quantifying this information-loss effect.

## Scope and Terms

THICC evaluates uncertainty introduced by finite energy resolution in thick-target active-target measurements.

Definitions used throughout the framework:

- `TCS` (true cross section): synthetic cross section generated from sampled resonance parameters.
- `TRR` (true reaction rate): reaction rate computed from `TCS`.
- `ATCS` (active-target cross section): energy-averaged cross section after thick-target/binning effects.
- `ATRR` (active-target reaction rate): reaction rate inferred from `ATCS` workflow.

Important clarification: `TCS` is not claimed as a prediction of the real physical cross section. It is a controlled assumed truth used to measure bias and uncertainty introduced by the measurement+inference chain.

## End-to-End Pipeline

High-level flow:

1. Sample resonances from level-density-informed Monte Carlo.
2. Build `TCS` from summed Breit-Wigner resonance contributions.
3. Export resonance set to RatesMC-style inputs for reproducibility and TRR computation.
4. Apply thick-target averaging/binning to obtain `ATCS`.
5. Compare active-target-informed rate results against reference workflows (RatesMC and TALYS-based analyses).
6. Quantify uncertainty in inferred reaction rates due to energy-resolution information loss.

## Resonance Monte Carlo Generation

Implemented in `nucres/generator.py` using `HFBSamplerConfig` and `synthesize_sigma_from_hfb`.

Core method:

- Interpolate HFB level density \(\rho(E)\) on the configured energy range.
- Partition energy interval into bins of width `delta_E_mev`.
- Sample resonance counts per bin from Poisson statistics:

\[
N_k \sim \mathrm{Poisson}\!\left(\int_{E_k}^{E_{k+1}}\rho(E)\,dE\right)
\]

- Sample resonance energies within occupied bins by inverse-CDF logic.
- Sample resonance attributes (for example `J`, widths, optional `L1`) using configured distributions and Porter-Thomas fluctuations.

Cross section model contribution per resonance is generated with Breit-Wigner calculations in `nucres/resonance.py` (`sigma_bw_constant` and energy-dependent variants).

## RatesMC Pipeline and Outputs

Project components:

- `build_ratesmc_input.py`: injects sampled resonances into a `RatesMC.in` template.
- `nucres/ratesmc_export.py`: formats resonance rows and headers (`resonance_to_row`, `render_rows`).
- `run_ratesmc_batches.sh`: executes batch Runs with structured output directories.
- `scripts/analyze_ratesmc_outputs.py`: aggregates run outputs and computes uncertainty bands.
- `scripts/overlay_ratesmc_talys.py`: overlays RatesMC summaries with TALYS reference curves.

Default output pattern:

- `outputs/<reaction>/Run_NNN/RatesMC.out`

## Thick-Target Cross-Section Computation

Implemented by `plot_cross_sections.py`.

Current behavior:

1. Parse resonance data from the RatesMC resonant-contribution section.
2. Compute total cross section from resonance sum.
3. Apply bin-wise integration over user-defined energy range and bin width.
4. Write integrated cross sections for downstream TALYS parameter studies.

Entrance-channel energy dependence follows penetrability scaling:

\[
\Gamma_i(E) = \Gamma_i(E_r)\,\frac{P_\ell(E)}{P_\ell(E_r)}
\]

with \(\Gamma_i(E_r)=2\gamma^2 P_\ell(E_r)\).

## Analysis Positioning

The framework supports three statistically analyzable rate outputs:

- THICC internal model outputs.
- RatesMC outputs generated from the same sampled resonance population.
- TALYS-based external reference outputs.

Because THICC and RatesMC share resonance-level structures, direct distribution-level comparison is possible when sufficient Monte Carlo samples are produced. TALYS is used as an external modeling benchmark.

## Documentation Value (Software Perspective)

This content strengthens standard software documentation in six ways:

1. It defines domain intent and non-goals, preventing misuse of outputs (`TCS` as assumed truth, not prediction).
2. It creates a stable glossary (`TCS`, `TRR`, `ATCS`, `ATRR`) used consistently across code and analysis.
3. It maps physics concepts to concrete modules/scripts, improving maintainability and onboarding.
4. It documents pipeline boundaries and artifacts, which supports reproducibility and debugging.
5. It states known limitations (resolution-induced information loss), guiding validation priorities.
6. It clarifies cross-tool comparisons (THICC vs RatesMC vs TALYS), supporting test and review strategy.
