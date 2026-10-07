# Mapper / SMPy change validation

Base main: `9f0394aec74e51ee3792fc07eb462dbfffcf919a`.
No commits, branches or pushes were made. Apply the accompanying binary-capable
patch with `git apply` on this base.

## Changes and scope

- One reusable, explicit P3/Argyris/HCT quadratic catalogue mapper now serves
  production and calibration. Masks/weights, coordinate/source assumptions,
  residual checks, saved outputs and physical FE evaluation are explicit.
- Held-out shear cross-validation supports unknown real catalogues. It reports
  boundary winners and requires a wider caller search before acceptance.
- Removed the obsolete catalogue wrapper, historical density parameter anchors,
  duplicate density runners and replaced comparison scripts. Research prior,
  posterior, low-level JAX and adjoint paths remain available.
- Replaced ranking assertions about old default configurations with estimator,
  geometry, paired-data and upstream-convention checks.
- Fixed FITS weights discarded by the experimental pipeline.
- Pinned and checked actual SMPy KS/KS+; weighted binning, availability masks and
  linear-shear assumptions match the common catalogue contract. The upstream
  aperture adapter is tested but excluded from convergence-map rankings.
- Profiling now uses the production quadratic solve, separates startup and
  factorization, and checks paired outputs before reporting timings.

## Executed validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

303 passed, one skipped module (Flax/Optax neural-prior dependencies absent),
126.93 s. Two existing two-resolution convergence-fit warnings; two upstream
SMPy division-by-zero warnings at its discarded Fourier DC coefficient. Its
returned maps are finite and analytic sign/B-mode checks pass. No upstream
source is patched or silently substituted.

After final metadata/CLI edits, focused tests were rerun with the NumPy BEM
backend; their final count is recorded below. The full-suite count above predates
the two added CLI/weight-persistence regression tests.

The new mapper reproduces saved density-5, seed-0 baseline reconstructions for
P3, Argyris and HCT at their recorded calibration settings. Maximum relative
map difference is 1.02e-8; fresh residuals are below 7.4e-9. See
`calibration-parity.json`.

18 fresh-process CPU runs (three elements, two backends, three processes) include
54 accepted reconstructions. All nine paired parity comparisons pass. Raw data
and the measured absence of an end-to-end Numba advantage at this size are in
`RESULTS.md` and `runs/`.

Two held-out SMPy integration scenarios (NFW and a central mask) use two
calibration seeds and three disjoint evaluation seeds. All ten method/scenario
searches resolved their boundaries; all 30 evaluation maps succeeded. Raw
catalogues, candidates, maps and paired summaries are in
`benchmarks/smpy/integration`. These small integration experiments are not
publication-level evidence of superiority.

The larger six-scenario config is supplied but not claimed as executed. Target
hardware scaling, broader survey sweeps, KS+ iteration stability, a matched
upstream aperture-statistic experiment, MassiveNuS and real-data evaluation
remain publication work. No CUDA or automatic iterative-solver switch is added.

Final focused run: `FEMMI_BEM_BACKEND=numpy python -m pytest tests/test_mapping.py tests/test_smpy.py tests/test_calibration.py -q` with single-thread BLAS/OpenMP: **26 passed**, 7.26 s (the same two upstream DC warnings).
