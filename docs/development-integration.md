# Development integration record

This integration reconciles the remaining work from
[PR #22](https://github.com/TAMU-Pangolins/TESSERACT/pull/22) with `main`
at `617960fb6ad002d6a011168fceb08b481eabe776`. Its development source is
`89c8bec0e8a2a7c788468dbcf5c1e4774e060e99`, before the later session merge.
The component commits record their source commits and any deliberate changes.

The purpose is to retain development's remaining tools without reintroducing
problems already corrected on main. This is an implementation integration,
not a claim that the optional physical models have completed calibration.

## Remaining-work accounting

Identifiers refer to the earlier 17-item comparison of development and main.
Every item has an explicit disposition below. Source hashes identify the
original development commits; integration subjects identify the corresponding
commits in this branch.

| Item | Problem or capability | Development source | Integration disposition |
| --- | --- | --- | --- |
| D01 | Native RatesMC execution and failure reporting | `5401a60`, `f6b9e0e` | **Run RatesMC in isolated run directories and propagate failures**. Uses native `RatesMC.in`, separates console/program logs, validates rate rows, recomputes invalid cached output, and reports batch/background failures. |
| D02 | Old-format RatesMC templates | `a24582d` | **Convert the 56 old-format RatesMC templates to the current format**. Converter and template-integrity tests retained; all 64 templates checked. |
| D03 | Copied neutron-channel tables under seven alpha-proton reaction headers | `69b91ce` | **Empty the copied 22Ne(a,n) data from the seven (a,p) templates**. Genuine upper-limit fixtures now test retention; generated-input upper-limit handling already existed on main. |
| D04 | Configurable width-fluctuation degrees of freedom | `f057f36` | **Add opt-in final-state width models while preserving current defaults**. Mean-preserving chi-square fluctuations; positive infinity suppresses fluctuations; invalid parameters fail. Both defaults remain one degree of freedom. |
| D05 | Gamma widths summed over discrete and continuum final states | `5335b0f` | Same final-state commit. Available through explicit `exit_width_model = summed`; E1/M1/E2 strength functions and continuum-bin checks retained. Calibration remains under #20. |
| D06 | Summed particle exits, per-resonance final excitation, shared exit tables | `5335b0f` | Same final-state commit plus **Share logarithmic exit penetrability tables across final states**. Exports the dominant contribution's angular momentum and excitation energy. Retains main's log-domain widths and threshold checks. The dominant-channel approximation still requires physical validation. |
| D07 | Hard-coded alpha-proton cross-section and rate readers | `b3b84d3` | **Select TALYS outputs for the configured reaction** and its CI mapping commit. Supports configured output, projectile/ejectile selection, exclusive-channel fallback, and the appropriate `astrorate` channel. |
| D08 | Additive logarithm-floor bias | `b3b84d3` | **Already on main through PR #27**, commit `617960f`; not reapplied. Existing numerical regression tests remain. |
| D09 | Missing binned predictions and fit diagnostics in archives | `b3b84d3` | **Save binned TALYS predictions and consistent fit diagnostics**. Saves the predictions and statistics for the actual saved parameters, including after an optimizer evaluation limit. Older archives remain readable. |
| D10 | Gamow-window interpolation and missing/ambiguous coverage | `bb0daa1` | **Validate Gamow-window coverage and refine TALYS percentiles**. Ported into main's current ensemble interface; no extrapolation across unavailable energies; integrates partial boundary intervals and handles empty/ambiguous inputs explicitly. |
| D11 | Incorrect filtering/ranking of requested parameter pairs | `fd6c98c` | **Correct parameter-pair selection and rate-ratio output names**. Retains requested nearly constant pairs, rejects unknown names, and ranks only selected fits. |
| D12 | Ratio plots named for requested rather than available bin widths | `fd6c98c` | Same analysis commit. Names normal and axes-free outputs after the widths actually plotted. |
| D13 | Unused RatesMC diagnostic output | `ec16e67`, `802522a` | **Make RatesMC diagnostic output retention configurable**. Suppresses new integrand/S-factor writes by default and supports retention. Existing regular output is archived rather than deleted. |
| D14 | Optional numerical/analytical integration selection | `a7924e1` | **Add optional width-based RatesMC integration selection**. Unset by default; validates a finite nonnegative cutoff. This is a heuristic, not an accuracy guarantee. |
| D15 | Runtime TALYS structure-database selection | `505914e` | **Allow runtime selection of the TALYS structure database**. Environment override is emitted once; otherwise preserves configured `strucpath`. This is partial work for #25, not a staging system. |
| D16 | TALYS cutoffs for small fitted cross sections | `89c8bec` | Configuration reference explains `popeps`, `xseps`, and `transeps`; the example adds `popeps` under `[talys]`. Reaction-specific convergence work remains under #6. |
| D17 | Complete section-by-section configuration reference | `4607131` | **Document integrated configuration and native program operation**. Updates original reference to the reconciled defaults, output handling, model limitations, and current native invocation. |

## Work already resolved on main

Main's implementations remain authoritative for isotope selection (#11), parity
and entrance-state selection (#12), half-integer spins (#13), energy frames
(#14), per-run random streams (#15), generated upper-limit removal and reaction
masses (#16), Sommerfeld interpolation (#17), RatesMC output parsing (#18),
objective scaling and bin averaging (#19), the previously resolved #21
corrections, and transmission underflow/log-space cross sections (#23).
The runnable example already addresses #10. The Python CI from PR #28 and
the floor correction from PR #27 remain in the base.

In particular, development's older ordinary-space Breit–Wigner implementation
does not replace main's logarithmic calculation. Shared multi-final-state exit
tables use logarithmic penetrabilities, retain closed-channel checks, and divide
wide energy ranges at powers of ten to avoid undersampling the MeV region with
a single Sommerfeld grid extending down to 1 eV. Each band uses the configured
table size. A single final-state offset retains its previous grid exactly.
The cache includes the energy domain so different realizations cannot silently
reuse a table with narrower coverage.
For multiple offsets, the lower-energy margin is applied after the exit-energy
shift. This avoids building a 1-eV grid for a channel that stays wholly open
over the requested range. Threshold-crossing domains retain coverage down to
the existing 1-eV table limit.

## How mixed source commits were separated

Development's `b3b84d3` contains several responsibilities. Its channel readers
became `b4bfde8`; their targeted CI mapping became `c93e71c`; its archive and
diagnostic work became `1f391ba`. Its logarithm-floor correction was already
merged through PR #27. Objective scaling and bin averaging were already on
main and were not imported again.

Similarly, `f057f36` includes HFB correction work that main already contains;
only the remaining fluctuation controls were incorporated with the final-state
models in `e61daa3`. Those closely related model changes retain one component
commit, while the integration-selection policy remains separate in `31ca8b9`.
The template conversion and removal of copied data remain distinct commits.

These are ports with explicit provenance, rather than a wholesale cherry-pick
of mixed commits or the later session merge. Existing component commits do not
need to be split again before publication. A normal PR merge commit preserves
their individual explanations and review history.

## Corrections found during reconciliation

The component commits also identify corrections discovered during verification:

- Recompute final fit diagnostics for the saved parameter vector. Previously,
  an evaluation limit could leave statistics from a different, better probe.
- Refer to the TALYS energy file relative to its working directory. A long
  absolute scratch path can exceed TALYS's native input-line limit even when
  the process returns success.
- Retain existing diagnostic files when suppressing future output, validate
  native rate rows, and propagate requested native calculation failures.
- Validate fluctuation parameters and partial continuum bins; preserve default
  ground-state behavior instead of silently adopting development's changed
  model and gamma-fluctuation defaults.
- Extend targeted CI selection to every integrated component. The CI job still
  runs the complete regression suite as well as relevant focused tests.

## Verification scope

Local Python 3.13 verification ran **204 tests: 203 passed and one optional
fixture test was skipped**. It also checked tracked Python/shell syntax,
correctness lint, workflow syntax, and a strict documentation build. The GitHub
workflow uses Python 3.11; its results must still pass after publication.
Native checks use TALYS
2.2 and RatesMC 2.3.0 (`3649f24baa`), with no functional source changes to either
program. Local executable copies require only ELF interpreter/library-path
adjustments for NixOS; RatesMC additionally needs its GSL CBLAS library loaded.

Native end-to-end cases use one seeded `22Mg(a,p)25Al` realization, 151
resonances, 1.8–2.8 MeV centre-of-mass energy, four 0.25-MeV bins, and 32
RatesMC samples. They exercise the ground-state and explicit summed models,
native rate output, binned cross sections, fitting, saved predictions, and
diagnostics. The fit is deliberately capped at two least-squares evaluations;
it verifies execution and record consistency, not a converged physical fit.
Ground-state generated resonance rows and both cross-section output files
match the baseline from before the optional models were integrated exactly.
The ground-state and summed pipelines completed in approximately 159 and 181
seconds respectively, each producing 60 native RatesMC temperature rows,
12 TALYS cross-section points, 30 TALYS rate points, and four saved binned
predictions with consistent fit diagnostics. The pre-sharing summed attempt
exceeded a 600-second limit. These are local verification timings, not general
performance estimates. A separate native TALYS check also passed where the
absolute energy-file path contains 191 characters.

A separate two-resonance RatesMC comparison covers numerical, analytical, and
width-selected integration. Over 26 temperature points from 0.1 to 3 GK,
the hybrid result differs from all-numerical integration by at most **1.41%**;
all-analytical integration differs by at most **49.55%** on that fixture.
These figures do not establish an error bound or speedup for other reactions.
The former blanket 0.1% documentation claim is not retained.

Template tests cover all 64 files and converter idempotence; native calculations
do not validate every evaluation represented in those files. Automated tests
cover gamma final-state strengths, continuum behavior, sampling moments,
multi-final-state Coulomb interpolation, and main's underflow safeguards.
This does not establish calibrated inclusive gamma or particle rates.

## Issue and PR disposition

Use one integration PR referencing #22 and this record. A new issue for every
already implemented item is unnecessary. Preserve the component commits with
**Create a merge commit** rather than squash merging if their individual
history is to remain visible on main.

Do not automatically close #20 (physical width calibration), #25 (general data
staging/resource control), #6 (cutoff convergence), or #24 (container/native CI).
They extend beyond the capabilities integrated here. Inclusive particle-model
validation can become a distinct follow-up issue if it needs separate tracking;
it should assess the dominant-channel approximation and rate changes, rather
than restating an implementation task that is already complete.

After this PR is reviewed and merged, compare this accounting with #22 and
close that original PR as superseded when its remaining scope is accounted for.
No GitHub issue, comment, pull request, merge, branch deletion, or commit push
is performed by preparing this local integration.
