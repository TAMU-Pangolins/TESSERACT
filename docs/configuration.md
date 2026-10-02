# Configuration (`tesseract.in`)

`tesseract.py --input tesseract.in` reads a sectioned `key = value` file
(`#` starts a comment). `[basics]` is required; each other section runs its
step when present and is skipped otherwise, in which case that step's
outputs must already exist:

| Section | Step | Script |
| --- | --- | --- |
| `[resonance]` | 1: sample resonances, write `RUN_j/<reaction>.in` | `build_ratesmc_input.py` |
| `[ratesmc]` | 1b: run RatesMC on each `RUN_j` (optional) | RatesMC executable |
| `[integration]` | 2: true and bin-averaged cross sections | `generate_cross_sections_vectorized.py` |
| `[talys]` | 3: fit TALYS to each bin-averaged file | `talys_opt_with_unc.py` |

A step whose outputs already exist is skipped, run by run.
`--run-idx j` processes a single run (e.g. `$(Process)` in a condor job).

## [basics]

| Key | Default | Meaning |
| --- | --- | --- |
| `reaction` | required | e.g. `22Mg(a,p)25Al`; the template is `<input_dir>/<reaction>.txt`. |
| `E_min_mev`, `E_max_mev` | `0.1`, `10.0` | Centre-of-mass energy range (MeV) for resonance sampling and step 2. |
| `python_exec` | current interpreter | Python used for the sub-steps (useful under condor). |
| `projectile_Z`, `projectile_A`, `target_Z`, `target_A`, `ejectile_Z`, `ejectile_A`, `residual_Z`, `residual_A` | required by `[talys]` | Nuclear identity written to `talys.inp`, and the masses for the CM/lab conversion. |

## [resonance]

Step 1 samples resonances from HFB level densities and writes them into the
template's Resonant Contribution table, once per run.

| Key | Default | Meaning |
| --- | --- | --- |
| `input_dir` | `input/` | Template directory. |
| `output_dir` | `outputs/` | Writes `<output_dir>/<reaction>/RUN_j/<reaction>.in`. |
| `runs` | `1` | Number of independent resonance ladders (`RUN_0` ... `RUN_{runs-1}`). |
| `seed` | unset | Base seed (an integer); run `j` uses `seed + j`. Unset: each run takes fresh OS entropy, so it cannot be reproduced. |
| `samples` | `5000` | Grid points for the level-density interpolation. |
| `delta_E_mev` | `0.05` | Bin width (MeV) for placing levels. |
| `spacing_model` | `poisson` | `poisson` or `wigner` (Wigner-surmise repulsion within each J^pi ladder). |
| `U_offset_mev` | projectile separation energy (AME2020) | Excitation offset added to E for the density lookup. |
| `hfb_corrections` | `false` | Renormalise the HFB densities with the RIPL-3 `.cor` (ctable, ptable) entry, rho(U) = exp(ctable sqrt(U - ptable)) rho_HFB(U - ptable), as TALYS does. Isotopes without an entry keep the raw table. |
| `pi` | `0` | Compound parity to sample: `0` both (default), `1` or `-1` one parity. |
| `target_parity`, `projectile_parity` | NUBASE2020 ground state (`data/nubase_3.mas20`) | Intrinsic parities (`1`/`-1`). The run stops if NUBASE gives no definite parity and none is set. |
| `drop_forbidden` | `true` | Drop J^pi sequences the entrance channel cannot form (no l with \|J - S\| <= l <= J + S and pi = pi_p pi_t (-1)^l). `false` keeps them. |
| `sample_J` | `true` | Sample J from the HFB rho_J; `false` uses a fixed J. |
| `auto_l1` | `true` | L1 = smallest allowed entrance l per resonance; `false` uses `l1`. |
| `l1` | `0` | Entrance l when `auto_l1 = false`. |
| `exit_width_model` | `summed` | `summed`: the exit width is the sum of one Porter-Thomas draw per open final state (discrete levels, then the HFB continuum); RatesMC gets the sum with the l/L and E_xf of the largest contribution. `ground-state`: one width to the ground state (or to `final_spin`/`final_parity`). Resonances with no open final state are dropped. |
| `levels_dir` | `$TALYS_LEVELS`, else `structure/levels/final` next to `talys` on `PATH` | TALYS/RIPL discrete-level files, needed by the summed model. Levels are used up to the level where the scheme is complete (`n_high` in the RIPL `.cor` file, else the last level from the ground state with measured J^pi). |
| `continuum_dU` | `0.1` | Bin width (MeV) of the HFB continuum of final states (summed model). |
| `auto_l2` | `true` | Ground-state model: lowest exit l (particle) or multipolarity (gamma) allowed by the resonance, ejectile and final-state J^pi; resonances that cannot decay are dropped. `false` uses the fixed `l2` (ground-state model only; the summed model needs it `true`). |
| `l2`, `l3` | `1`, `0` | Fixed exit l (with `auto_l2 = false`) and spectator l. |
| `final_spin`, `final_parity` | NUBASE ground state of the residual (compound for gamma exits) | Final state for the ground-state model. |
| `mean_i` | `0.010` | Mean entrance reduced width as a fraction of the Wigner limit of projectile + target (template masses in u and R0). |
| `mean_o` | `0.0045` | Particle exits: mean exit reduced width as a fraction of the Wigner limit of ejectile + residual (one value for every final state in the summed model). Gamma exits: the mean gamma width in eV in the ground-state model; unused in the summed model, where gamma widths come from strength functions (E1/M1/E2) and the HFB density. |
| `Gamma_i_dof` | `1` | chi^2 degrees of freedom of the entrance-width fluctuation (1 = Porter-Thomas). |
| `Gamma_o_dof` | `10` for gamma exits, `1` for particle exits | chi^2 degrees of freedom of the exit-width fluctuation; `inf` for none. Ground-state model only: the summed model draws one Porter-Thomas width per final state. |
| `n_random_samples` | `1` | Overrides the template's number of RatesMC samples. 1 on purpose: the sampled resonances are treated as perfectly known, so there is no experimental uncertainty to sample. |
| `default_frac_unc` | `0.001` | Fractional uncertainty written for Ecm and widths. |
| `int_flag` | `1` | RatesMC integration flag (0 analytic, 1 numerical). |
| `precision` | `3` | Mantissa digits of the non-Ecm columns. |
| `n_sigma_points` | `4000` | Grid of the generator's own diagnostic sigma (not used downstream). |
| `use_strength` | `false` | Also write omega-gamma; the widths are written either way. |

The template's upper-limit resonances (every *Upper Limits of Resonances*
section) are removed from the generated inputs, since TESSERACT treats its
sampled resonances as the complete set; `build_ratesmc_input.py
--keep-upper-limits` keeps them (not a `tesseract.in` key).
`build_ratesmc_input.py --help` lists further options for standalone use
(e.g. `--exf-kev`, which then requires `--final-spin`/`--final-parity`).

## [ratesmc]

Optional step 1b. For each run it copies `RUN_j/<reaction>.in` to
`RUN_j/RatesMC.in`, links `mass_1.mas20` and `nubase_3.mas20` into `RUN_j`,
runs RatesMC there with no arguments (no release accepts an input path), and
copies `RatesMC.out` to `<reaction>.out`. A run succeeds when `RatesMC.out`
contains rate rows; the exit code is ignored (the 2.2+ rewrite returns 1 on
success). Console output goes to `RUN_j/RatesMC.stdout`, since RatesMC writes
its own `RatesMC.log`. Nothing downstream reads RatesMC's output, so if no
executable is found the step is skipped with a warning, and by default
RatesMC runs in the background while `[integration]` and `[talys]` proceed
(give a condor job two CPUs). The driver waits for it before finishing.

| Key | Default | Meaning |
| --- | --- | --- |
| `ratesmc_bin` | `$RATESMC_BIN`, else `RatesMC`/`ratesmc` on `PATH` | Executable (path or command name). |
| `mass_dir` | beside the executable, else one level up (upstream `build/`) | Directory holding `mass_1.mas20` and `nubase_3.mas20`. |
| `concurrent` | `true` | Run RatesMC alongside the later steps; `false` runs it before them. |
| `keep_integrand` | `false` | Keep `RatesMC.integ`, RatesMC's dump of every integrand evaluation (12–18 GB per run for the (a,p) cases); otherwise it is linked to `/dev/null`. |

Use RatesMC 2.3.0 (upstream [rlongland/RatesMC](https://github.com/rlongland/RatesMC)),
which runs all 64 templates. RatesMC 2.11 hangs while reading 8 of them.

## [integration]

Step 2 builds the cross section from the generated RatesMC input with
RatesMC's own integrand (energy-dependent entrance, exit and spectator
widths, template masses, spins and R0) and writes one bin-averaged file per
run and bin width,
`<output_dir>/<target>_<proj><ejec>_<residual>_integrated_xs_dE_<dE>_<tag>_run<j>.csv`
(e.g. `22Mg_ap_25Al_...`). Bin averages are exact per-resonance integrals
over bins of exactly `dE`.

| Key | Default | Meaning |
| --- | --- | --- |
| `dE` | required | Bin width(s) in MeV, comma-separated (e.g. `0.1,0.2,0.5`). |
| `n_grid_points` | `10000` | Points of the unintegrated file. That file is a plotting grid only (resonances narrower than its spacing are not resolved); the bins do not depend on it. |
| `tag` | empty | Label inserted in the output names. |
| `output_dir` | `.` | Directory of the integrated files. |
| `unint_output_dir` | parent of `output_dir` | Directory of `<reaction>_xs_unintegrated_parallel_<tag>_run<j>.txt`. |

The entrance-width penetrability model (`--penetrability-model coulomb |
jwkb_real_omp`, with `--omp-model`, `--jwkb-npts`, `--jwkb-radial-npts`) is an
option of `generate_cross_sections_vectorized.py` only; `tesseract.py` does
not pass it, so the pipeline uses the default `coulomb`.

## [talys]

Step 3 fits TALYS to each integrated file. Keys listed below are read by
`talys_opt_with_unc.py`; **every other key is written to `talys.inp`
verbatim** as `key value` (e.g. `channels = y`, `ldmodel = 5`). The
parameters to fit go between two `\opt` lines, one per line as
`keyword [qualifiers] x0 lo hi`.

The objective is chi^2 + prior in ln(sigma) (not divided by dof, so it is the
log-posterior up to a constant); chi^2/dof is reported and written as
`Min reduced chi-square`. The prior is Gaussian about x0 with width
`prior_rel_std * max(|x0|, prior_abs_floor)` per parameter.

| Key | Default | Meaning |
| --- | --- | --- |
| `exp_file` | from `[integration]` | File to fit when `[integration]` is absent; `{run}` is replaced by the run index. |
| `talys_output_dir` | `talys_opt` | Output root; results go to `RUN_j/`, the shared log to `talys_optimization.out`. |
| `exp_energy_frame` | `cm` | Frame of the data energies. `cm`: converted to projectile lab energies for TALYS (AME2020 masses) and back. `lab`: no conversion. |
| `talys_points_per_bin` | `3` | Gauss-Legendre energies per bin at which TALYS runs; TALYS is averaged over the same bins as the data. `1` = bin centres. |
| `exp_bin_width` | from the file | Bin width (MeV); else the `dE=` in the file header, the `_dE_<w>` in its name, or the median spacing. |
| `e_fit_min` | `2.0` | Only bins with E >= e_fit_min (MeV) enter chi^2. |
| `exp_rel_err` | `0.10` | Relative uncertainty of points without one (the data file's third column, an absolute uncertainty, is used where present). |
| `prior_rel_std` | `0.15` | Relative prior width; `0` disables the prior. |
| `prior_abs_floor` | `0.01` | Floor on \|x0\| in the prior width. |
| `talys_xs_file` | unset | TALYS output file to read. Otherwise `<proj><ejec>.tot` (e.g. `ap.tot`), then the exclusive-channel `xsNNNNNN.tot` (e.g. `xs010000.tot` for (a,p), which needs `channels = y`). The rate is read from `astrorate.<ejectile>`. |
| `method` | `Powell` | `Powell`, `Nelder-Mead` (scipy `minimize`), or `least_squares` (also `lsq`, `trf`). |
| `maxiter` | `250` | Iteration (minimize) or evaluation (least_squares) limit. |
| `xtol`, `ftol` | `1e-3`, `1e-3` | Tolerances; Nelder-Mead uses `xatol`/`fatol` (default: `xtol`/`ftol`). |
| `gtol`, `lsq_diff_step`, `lsq_workers` | `1e-8`, `1e-2`, `1` | least_squares only: gradient tolerance, relative finite-difference step, parallel TALYS runs for the Jacobian. |
| `debug_every` | `10` | Print every n-th evaluation. |
| `rates_mc_file` | `RatesMC.out` | RatesMC table drawn on the rate plot. |
| `rate_xmin`, `rate_xmax`, `plot_log_y_xs`, `plot_log_y_rate` | `0.0`, `2.0`, `true`, `true` | Plot settings. |
| `output_file` | | Reserved (not passed to TALYS). |

`RUN_j/checkpoint.json` holds one entry per fit (keyed by the data file), so
an evicted fit resumes from its own best point only; a finished fit removes
its entry. `talys_opt_with_unc.py --debug-talys` runs TALYS once at x0 and
prints the output it finds.

## Templates

The 64 templates in `input/` are in the current RatesMC format (that of
`input/22Mg(a,p)25Al.txt`), converted by `scripts/convert_ratesmc_templates.py`.
The seven (a,p) templates `14O`, `18Ne`, `22Mg`, `26Si`, `30S`, `34Ar`,
`38Ca(a,p)` have empty resonance tables (commented placeholder rows); step 1
fills the Resonant Contribution section.
