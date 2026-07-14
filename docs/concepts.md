# Concepts

This section explains the core physics quantities and how they map to the codebase.

## Resonances and Cross Sections

The main building block is a single-level Breit–Wigner resonance.
In code this is represented by `nucres.resonance.Resonance`, with:

- `E_r` resonance energy (eV)
- `Gamma_i` entrance width (eV)
- `Gamma_o` exit width (eV)
- `J` resonance spin
- `s1`, `s2` projectile/target spins
- `m1`, `m2` projectile/target masses (kg)

The cross section can be computed with constant widths:

- `sigma_bw_constant(E_eV, r)` in `nucres.resonance`

Or with an energy-dependent entrance width using penetrability:

- `sigma_bw_energy_dep(E_eV, r, Z1, Z2, A1, A2, l, gamma2, ...)`

## Penetrability and Reduced Width

Energy-dependent widths use the Coulomb penetrability `P_l(E)`:

- `penetrability_P_l_mev(l, Z1, Z2, A1, A2, E_mev)`

When `Gamma_i` is not known, `sigma_bw_energy_dep` uses:

- `Gamma_i(E_r) = 2 * gamma2 * P_l(E_r)`

Here `gamma2` is the reduced width (in eV), and `P_l` is dimensionless.

The default `penetrability_model="coulomb"` preserves the original behavior.
For alpha-induced reactions, `penetrability_model="jwkb_real_omp"` replaces the
Coulomb penetrability energy dependence with a JWKB transmission ratio:

- `Gamma_i(E) = Gamma_i(E_r) * T_l(E) / T_l(E_r)`

The sampled `Gamma_i(E_r)` is not recomputed. The first implemented alpha OMP
is McFadden-Satchler. Its real Woods-Saxon term is included with finite-size
Coulomb and centrifugal terms in the JWKB barrier; the imaginary OMP part is
recorded as ignored and is not included in the tunneling integral. If no
forbidden region or valid turning-point pair is found, the JWKB transmission is
set to `T=1`.

The vectorized cross-section driver exposes this through:

- `--penetrability-model jwkb_real_omp`
- `--omp-model mcfadden_satchler`

Runs write a metadata JSON sidecar describing the penetrability model, OMP
radii, Coulomb geometry, diffuseness, transmission floor, and fallback behavior.

## Sampling and HFB Level Densities

The HFB-driven workflow samples resonance energies from level densities.
The top-level entry point is:

- `synthesize_sigma_from_hfb(HFBSamplerConfig)`

This returns a `GeneratedSpectrum` with an energy grid, total cross section,
and the list of sampled `Resonance` objects.

## Reaction Rates

Reaction rates are computed by integrating a cross section over a
Maxwell–Boltzmann distribution:

- `na_sigma_v_from_sigma(E, sigma, T, ...)` in `nucres.rates`

Convenience wrappers live in `nucres.model`:

- `generate_spectrum(request)`
- `compute_rate_table(request, temperatures, ...)`
