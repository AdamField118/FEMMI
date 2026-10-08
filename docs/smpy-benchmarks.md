# Benchmark protocol

Run comparisons through `examples/paper/benchmark_smpy.py`. It uses the production
FEM estimator, pinned upstream SMPy Kaiser–Squires (KS), and an explicitly
labeled correction to the pinned SMPy KS+ forward transform. The upstream aperture mapper is evaluated against an aperture
statistic, not against an unfiltered convergence map.

## Install and run

```bash
python -m pip install -e '.[io,galsim,speed]'
python -m pip install -r requirements-benchmark.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FEMMI_BEM_BACKEND=numba \
  python examples/paper/benchmark_smpy.py \
  --config configs/benchmarks/smoke.json --output results/smpy-smoke
```

The smoke recipe is explicitly exploratory. It allows unresolved search edges
and unstable iteration checks so that a short integration run can finish. Its
outputs cannot establish a publication ranking.

Use `configs/benchmarks/publication.json` for the full protocol. `--names nfw
two-halos` selects named scenarios; use a new output directory for every run.
The driver refuses to overwrite a nonempty directory. Run serially when comparing
wall times. For dedicated backend timings, use the [profiling guide](production-performance.md).

## Freeze the experiment

The run writes `protocol.json` before fitting. It records the selected recipes,
configuration hash, code provenance, pinned SMPy revision, selection metric,
and catalogue transport. Archive the exact checkout and environment with the
outputs, particularly when running a modified working tree.

Each realization is written as a FITS table and read through the same production
survey ingestion used by `map_mass`. Both method families then receive those
same positions, component signs, shear values, and weights. FEM fits individual
rows; SMPy receives weighted mean bins over a fixed square extent. That binning
is part of the SMPy estimator, not a change in input catalogue.

Catalogue generation uses mean-one weights and independent component noise
`noise_std/sqrt(weight)`. Nominal count density uses the gross circular field
area. Weighted effective density is `(sum w)^2/sum(w^2)/area`; masks do not
change that area definition. A shared near-duplicate selection happens before
any estimator is constructed. The FITS tables and catalogue hashes preserve the
actual observations used for pairing.

## Calibration and iteration checks

Each FEM arm jointly selects lambda and physical correlation length. HCT,
Argyris, and P3 are calibrated independently. Each KS arm selects grid size and
Gaussian smoothing in pixels. All minimize mean source-position DC-removed
relative L2 on calibration catalogues with truth convergence below one.

Winning search edges expand before acceptance; local refinement adds candidates
near an interior winner. Zero length or smoothing is a valid physical endpoint.
A positive unresolved boundary stops publication evaluation. A finite search
still does not certify a global optimum.

The publication recipe requires at least five calibration and twenty evaluation
seeds, with no overlap. These are minimum protocol budgets, not a power analysis.
Choose the sample size needed for the intended precision before inspecting the
evaluation results. Parameters are frozen before generating evaluation catalogues.

KS+ uses one outer reduced-shear pass because the input is linear shear.
Always set `ks_plus_threshold_tau` explicitly. Upstream defaults to one quarter
of the total iteration budget, which changes the threshold trajectory whenever
the budget changes. A fixed positive tau gives the same trajectory at every
checkpoint. Tau is part of the estimator configuration, not a residual tolerance.

`ks_plus_forward="corrected"` is the adapter default. The pinned upstream
`_kappa_to_gamma` discards the B contribution when taking real parts. The
correction implements `gamma1=D1 E-D2 B`, `gamma2=D2 E+D1 B`. Thresholding,
wavelet constraints, inverse transforms, padding, and masks remain upstream.
The real-valued even-grid Nyquist convention also remains upstream; the
round-trip contract applies to resolved modes away from DC/Nyquist. Use
`ks_plus_forward="upstream"` only to reproduce or diagnose that pinned behavior.
Every fit records the variant. Do not describe corrected results as unmodified
upstream SMPy. This correction changes the estimator, so recalibrate it.

Choose the stopping policy **before** evaluation:

- `ks_plus_iteration_policy="stable"` (default) keeps the stability gate.
  Set `ks_plus_iterations` and include it plus at least two larger budgets in
  `ks_plus_iteration_check`. After grid/smoothing calibration, the check compares
  unsmoothed E and B maps on the physical field, with a common E+B norm after
  removing each component's field mean. The selected and all later checkpoints
  must agree with the longest run within `ks_plus_stability_tolerance`.
  A B excursion can fail the gate even when E appears steady. Means and full-grid
  changes are retained separately. Failure stops publication evaluation.
- `ks_plus_iteration_policy="calibrated_budget"` treats iteration count as a
  hyperparameter of a finite-budget estimator. Supply at least two counts in
  `ks_plus_iteration_candidates`. Grid and smoothing are recalibrated for every
  count, then the lowest calibration error selects the complete configuration.
  `ks_plus_iteration_check` must include all candidates and at least two budgets
  above the largest one. The same plateau diagnostic is saved, even if it fails.
  Evaluation is allowed, with an explicit **finite-budget, no convergence claim**
  label. A winner at the largest allowed count is marked `iteration_budget_limited`.
  Report the declared cost cap; this is not an unconstrained optimal KS+ claim.
  A winner at the smallest count stops evaluation if that count exceeds one;
  include one or extend the lower search range before acceptance.

The finite-budget policy does not make an unstable iteration converge. It asks
a different, well-defined benchmark question: how well does an independently
tuned, computationally bounded estimator predict held-out catalogues? Keep
failed candidates and unresolved spatial tuning boundaries under either policy.
Never change the budget set or tau after examining evaluation outcomes.

`configs/benchmarks/ks-plus-budget-smoke.json` demonstrates that policy with
small exploratory splits. Publication splits still require at least five
calibration and twenty evaluation seeds. The existing publication recipe remains
in strict `stable` mode; it has not been relabeled to bypass a failed gate.

To inspect schedules independently of a full calibration:

```bash
python examples/diagnostics/ks_plus_iterations.py --iterations 100 400 1600 \
  --tau 25 --output results/ks-plus-iterations.json
```

It compares native and corrected forward transforms under native and fixed-tau
schedules. This is an empirical budget check, not a fixed-point residual or an
accuracy guarantee. Hard thresholding and gap-power rescaling do not provide a
monotone quadratic objective, so more iterations need not improve the map.

## Spatial and aperture scoring

All convergence methods are evaluated at the same physical pixel centres.
One field-wide mean error is removed before calculating interior, boundary,
and mask RMSE; regions are not recentered independently. Source-position error,
mean error, field shape error, and central core-minus-annulus contrast error
are retained separately.

With `aperture_comparison=true`, every convergence map is filtered with the
compensated U paired with SMPy's Schneider Q (`l=3`). The physical radius is
`aperture_radius_arcmin`, defaulting to one quarter of the field radius; it is
not a tuning parameter. SMPy's aperture mapper receives the shared catalogue
binned on the evaluation grid. Both are scored against U applied to the truth
map, only where the full aperture fits within valid scoring support. Catalogue
holes remain in the challenge.

The sampled U kernel has its support mean removed to preserve exact discrete
compensation. SMPy's Q implementation is unchanged. `aperture_control.json`
records dense noiseless Q-versus-U discrepancies, which must be small relative
to any claimed method difference. Refine `evaluation_grid` when that control
is material. Before tuning, calibration controls must pass
`aperture_quadrature_tolerance` (default relative L2 0.05) unless the run is
explicitly exploratory. This is a numerical budget, not evidence that a smaller
method difference is meaningful. Evaluation controls remain in the saved outputs.
Aperture mass and convergence have separate summaries.

## Files and interpretation

| Product | Contents |
|---|---|
| `protocol.json` | Frozen recipes, hashes, source and method provenance |
| `scenario/catalogue-*.fits` | Calibration and evaluation observations with truth and source IDs |
| `scenario/calibration.json` | Every candidate, failed candidate, expansion, chosen setting, and KS+ check |
| `scenario/evaluation.json` | Per-method, per-seed errors, timings, solver diagnostics, and failures |
| `scenario/map-*.npz` | Source values and common-grid predictions |
| `scenario/aperture.json` | Matched aperture metrics and failures |
| `scenario/aperture_control.json` | Dense noiseless filter discretization control |
| `summary.json`, `RESULTS.md` | Paired summaries regenerated from saved evaluations |

```bash
python examples/paper/report_calibration.py --root results/smpy-smoke
```

Paired differences use shared catalogue hashes, preserve failures, and report
sample counts, standard errors, Student-t intervals, wins, and exact sign tests.
Intervals describe repeated catalogues for the chosen scenario. Many exploratory
comparisons do not support an unadjusted universal superiority claim. Examine
failures and effect sizes as well as mean ranks.

Store these generated products outside Git. For publication, archive them with
an immutable source revision, environment, and reproduction commands in a data
repository or other durable research archive.

## External simulation maps

Edit `configs/benchmarks/massivenus-template.json` to point at separate calibration
and evaluation directories. Content hashes reject reused files, including
renamed copies. Split independent simulations or lines of sight before cropping;
separate random seeds do not make overlapping patches independent. The loader
retains the supplied convergence reference level.
