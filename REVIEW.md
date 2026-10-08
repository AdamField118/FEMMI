# FEMMI science, benchmark protocol, and documentation patch

Base: `fce327f3b179abc7b15247d1427bac9c48bc979d` (origin/main, fetched and verified during this task).

Apply from a clean checkout of that revision:

```bash
git apply --check --index /path/to/femmi-fce327f.patch
git apply --index /path/to/femmi-fce327f.patch
```

The patch has been applied with these commands in a fresh worktree. Every resulting file was compared byte-for-byte with the edited source. Nothing was pushed. The patch uses full object IDs and includes binary-file deletions; it does not duplicate deleted binary payloads. Git history retains the removed artifacts.

## What changed

1. Scientific diagnostics: exposed mesh-ring padding/resolution; added FE-L2 adjoint, cross-response, fixed-geometry halo-superposition, and conditional B-response checks. `FEMMapper.diagnose_b()` separates raw B into the response to the fitted E prediction and the response to the residual shear. Survey users can request the two additional FITS maps with `b_diagnostics=True`. Their interpretation is explicit in the API, mathematics, and FITS headers.
2. One maintained benchmark driver: synthetic catalogues round-trip through production FITS ingestion before all methods see them. Recipes freeze separate calibration/evaluation seeds; each FEM arm jointly tunes lambda/length; KS arms tune grid/smoothing; unresolved calibration edges and KS+ budget instability stop publication evaluation. Source hashes, versions, candidates, failures, paired statistics, and raw maps are retained in run directories. Removed the duplicated SMPy reconstruction during spatial scoring.
3. Matched aperture comparison: use upstream SMPy's Schneider Q and its paired compensated convergence U at a fixed physical radius and common complete support. Keep aperture statistics separate from convergence errors. A dense noiseless control quantifies finite-grid discrepancies; a calibration gate requires refinement before fitting when they exceed the chosen budget. Lognormal scoring uses 256 pixels after calibration-grid refinement checks.
4. Performance preparation: a serial fresh-process NumPy/Numba matrix runner and portable Slurm submission script separate compilation, cache load, assembly, factorization, repeated solves, total cost, and memory. Numerical parity is required before ratios are emitted. No new Turing speed claim is made.
5. Repository cleanup: removed tracked generated benchmark products, sample run outputs, compiled paper PDF, output figures, and development audit reports. Kept source recipes under `configs/benchmarks/`, runners, and small regression fixtures. Generated outputs are ignored. A report generator no longer writes tables into MATH.md.
6. Documentation: rebuilt README, mathematical reference, installation, quickstart, survey/API, calibration, benchmark, profiling, prior/sampling, contribution, and troubleshooting guidance. Removed stale result tables, hardware reports, process narratives, and unsupported scientific claims. Applied the Human-First Academic Writing Loop principles to make descriptions concrete and faithful to current behavior. Added a strict documentation CI build. This is preparation for software review, not a claim of JOSS acceptance.

## Scientific findings and limits

The Gaussian and analytic GalSim NFW sweeps each cover P3, Argyris, and HCT; 64 and 180 sources; ring padding 1.12 and 1.5; single and two off-centre halos; and tighter-solver comparisons. These are uncalibrated diagnostics at lambda=0.03 and length=0.3 arcmin, not method rankings.

Across the 48 rows, the largest FE-L2 superposition discrepancy was below 7.6e-8, the largest change from tighter solves below 9e-8, and B-response closure below 1.7e-8. The driver requires adjoint/skew errors below 1e-8 and superposition/closure below 2e-5. The multi-halo reconstruction is consistent with the linear estimator at fixed geometry/prior. This does not settle reconstruction accuracy after changing geometry or tuning.

A large rotated-shear response persists for inputs manufactured by the same discrete E operator. Its right-hand side contains `F1.T W F2 - F2.T W F1`, which need not vanish. Tightening CG cannot make this an orthogonal E/B decomposition. The new leakage/residual split is conditional on the E fit and prior; neither component is a pure-B estimator or a calibrated significance measure. A validated pure-mode or joint E/B estimator remains separate scientific work.

The final smoke run completed 15 convergence fits and 18 aperture comparisons without numerical failures. It deliberately allows diagnostic gate failures. Argyris has an unresolved search edge at this short budget. The coarse 24-pixel aperture grid has about 15% filter discrepancy. KS+ changes by 32.2% and 19.8% on the DC-removed physical field between 100 and 200 iterations for calibration seeds 100 and 101. This exceeds the 5% protocol budget. A supplementary longer-budget probe also showed substantial sensitivity; its recorded metric is the full grid and is labeled separately.

Upstream SMPy's threshold decay time defaults to one quarter of its total iteration budget. Increasing that budget changes the path, not just the stopping time. Do not describe the check as a solver residual or assume that more iterations will repair it. Publication evaluation remains blocked until KS+ budget/schedule sensitivity and unresolved tuning are addressed on calibration data. The upstream algorithm was not modified.

At fixed lognormal truth, increasing the scoring grid from 64 to 256 reduced the dense filter discrepancy to 0.25–0.34% across calibration seeds 100–104. This is a numerical-control result, not evidence for a method advantage.

## Validation

- Full non-slow suite with Astropy, GalSim, Numba, JAX, and pinned SMPy: **333 passed, 1 skipped, 3 deselected**. The neural module was skipped because Flax/Optax were absent. Existing convergence-fit and upstream FFT-division warnings remain.
- Added the aperture-gate regression afterward; final protocol/calibration/mapper subset: **27 passed**.
- Survey, scientific, protocol, and SMPy tests on a freshly patched checkout: **36 passed** before the final iteration-metric refinement; the final protocol subset above covers that refinement.
- Critical-error Ruff check, whitespace check, strict MkDocs build, and all local Markdown links passed.
- Executed the documented synthetic quickstart. Checked the rendered home, quickstart, API, survey, and benchmark pages; no horizontal overflow was detected.
- Fresh-process NumPy/Numba checks: all three element families, 24 sources, two processes per backend, two reconstructions per mapper; objectives, coefficients, and predictions passed the comparison utility. These are workflow checks, not representative Turing measurements.
- Earlier unmasked/masked integration runs completed 30 convergence and 36 aperture comparisons with no numerical failures. The included `protocol-validated` run uses the final gate definitions.

## Next work

1. Resolve KS+ budget/schedule sensitivity and complete calibration without unresolved edges. Preserve the frozen held-out evaluation split.
2. Run the supplied CPU matrix on Turing at representative catalogue sizes, with matched reconstruction quality and enough memory. Use those measurements to choose any further optimization.
3. Run the publication scenarios and archive outputs separately from Git. Inspect failures, paired effect sizes, aperture controls, and boundary/mask behavior before drawing rankings.
4. Use the estimator on real data with survey-specific shear calibration and source-plane assumptions. A pure E/B analysis, if required scientifically, needs its own validated estimator.

The evidence directory contains raw diagnostics, integration outputs, CPU smoke outputs, environment information, and test/build logs. They accompany this delivery and are not added to FEMMI's tracked documentation.
