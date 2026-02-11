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
