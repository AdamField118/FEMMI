# Validation record

Base main: `8cb095ccd6333b057a7c931935b52bf42d035450`.
Rechecked against the remote during the task; no commits, branches or pushes.

## Executed checks

- `FEMMI_REQUIRE_OPTIONAL=1 FEMMI_BEM_BACKEND=numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`: **310 passed**, 364.23 seconds. All required optional dependencies were installed. Two existing warnings intentionally flag convergence slopes fitted from only two mesh resolutions.
- After the final failure-audit and SciPy compatibility edits, `python -m pytest tests/test_calibration.py -q` under the same settings: **13 passed**, 13.64 seconds. This includes the additional failed-search audit test; the current suite contains 311 tests.
- `FEMMI_BEM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_calibration.py tests/test_dual_operators.py tests/test_coherence.py -q`: **15 passed**, 13.48 seconds, before the one additional failure-audit test.
- Dense quadratic optimum versus matrix-free MAP; weighted and gauge-projected adjoint checks for all three FEM arms; converged C¹ dense-catalogue regression; forward/transpose and multi-RHS equilibrated LU checks.
- Seed overlap, wrong catalogue hashes, duplicated pairs, search expansion, unresolved-edge reporting, masked clustering, MassiveNuS content overlap and actual patch-mean preservation.
- Dual operator singular-quadrature convergence, symmetry, constant null mode and full preconditioned spectral growth. Matched-residual iterative/direct measurements and dense/ACA matrix parity.

## Completed synthetic study

16 scenarios × four methods × six held-out evaluation catalogues = **384 evaluations**.
**2,786 parameter candidates**, each scored on three calibration catalogues.
**Zero failed candidates, zero failed evaluations, zero unresolved search edges** in the accepted run.

The maximum freshly recomputed relative normal-equation residual was
**2.425e-8 across FEM calibration fits** and **9.970e-9 across FEM evaluation fits**.
The explicitly recorded solver acceptance limit is 1e-6; actual accepted
solutions were substantially more accurate. All method pairings were checked
against saved catalogue fingerprints. Calibration seeds 100–102 and evaluation
seeds 0–5 are disjoint. Intermediate runs that exposed numerical defects are
not mixed into these accepted results.

P3/KS and C¹ arms were rerun independently under the same configurations and
merged only after verifying catalogue fingerprints and complete method/seed
coverage. This is equivalent to independent method dispatch in the provided
single-command runner. Reproduction commands are in `docs/calibration.md`;
`requirements-calibration.txt` and `environment.json` record the environment.

The plot was visually inspected. MATH.md uses the generated baseline table;
`report_calibration.py --update-math` updates its marked block along with the
standalone results. The old-to-new section mapping has been checked, including
all 18 headings and references in code, examples and tests.

## Delivery checks

`git diff --check` and `git apply --check` against an untouched checkout of the
base passed. The patch contains code, tests, documentation, configurations,
raw calibration/evaluation JSON, catalogues, sampled reconstruction maps, the
figure and numerical measurements. Patch checksums are supplied with delivery.

MassiveNuS and real-cluster/Chandra validation remain explicitly blocked by
missing external data. Timing under concurrent experiment load is not a
publication-quality speed measurement.
