# Calibrated catalogue comparisons

The maintained comparison is `examples/paper/calibrated_comparison.py`.
It independently calibrates **P3, Argyris, HCT, and binned Kaiser–Squires**.
Historical `density_sweep` defaults remain available for compatibility; they
are not the settings used by this experiment.

## Observation and estimator contract

Angles are arcminutes. The synthetic catalogue contains true shear, not reduced
shear. NFW redshift assumptions are the defaults in `femmi.truth.nfw_truth`;
changing them requires new calibration. One catalogue is generated before any
mesh: positions, both noisy shear components, and mean-one inverse-variance
weights. A single near-duplicate filter is shared by all methods. Gaussian
noise has component standard deviation `noise_std/sqrt(weight)`. SHA-256
fingerprints verify pairing. The FEM guard ring has radius `1.12 * field_radius`
and supplies no measurements. All arms score the same retained galaxies.

P3 recovers nodal Hessians; Argyris carries Hessian vertex DOFs; HCT uses its
area-weighted recovered Hessian. Each uses the same prior family
`lambda * kappa.T @ (M + length**2 * K) @ kappa`, jointly tuned in both knobs.
Length is physical in arcminutes. At zero length this family is mass-only;
the legacy MAP APIs' zero-length stiffness-only convention is distinct.
The coupled forward solve continues to use SuperLU. C¹ systems are diagonally
equilibrated before factorization, with matching scaling for transpose solves;
this fixes residual drift from mixed value/derivative coefficient units. The outer **quadratic MAP**
uses prior-preconditioned CG, checks the freshly evaluated normal-equation
residual, and performs residual refinement when needed. Internal target is
1e-8; the explicitly recorded acceptance tolerance is 1e-6. A returned CG
success flag alone is insufficient. This path does not replace nonquadratic
MAP, custom JAX derivatives, or posterior sampling.

KS uses weighted bins, fixed physical square extent, and jointly tuned grid
size and Gaussian smoothing in pixel units. Empty bins follow the existing
zero-filled FFT convention. The comparison is to FEMMI's KS implementation,
not yet a comprehensive SMPy algorithm benchmark.

## Reproduction

```bash
pip install -e '.[dev,neural,galsim,io,speed]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FEMMI_BEM_BACKEND=numba \
  python examples/paper/calibrated_comparison.py \
  --config benchmarks/calibration/configs.json \
  --output benchmarks/calibration/runs
python examples/paper/report_calibration.py --update-math
FEMMI_REQUIRE_OPTIONAL=1 pytest -q
```

Use `--names density-5 density-10` to select scenarios. Each method starts from
the same search budget appropriate to its two knobs. Winning edges expand;
then neighboring values are refined geometrically. A zero smoothing endpoint
is valid; unresolved positive edges are flagged. Calibration seeds 100–102
are disjoint from evaluation seeds 0–5. Final parameters are saved before
blind evaluation. Searching different finite grids can still change an optimum:
these experiments do not establish global optimality or an optimal prior.

The objective for calibration is mean DC-removed relative L2 across calibration
catalogues, evaluated at galaxy positions with truth κ<1. It measures shape
recovery, not recovery of the mass sheet. Raw relative error and mean error are
saved too. Six evaluation seeds permit paired comparisons, with SE, t intervals,
win counts, and exact sign tests; they do not warrant broad superiority claims.
No ranking should silently exclude solver failures or unmatched catalogues.

Results, all candidate scores and search histories, configurations, calibration
and evaluation catalogues, and sampled reconstructed maps are stored in
`benchmarks/calibration/`. `RESULTS.md`, `means.json`, `paired.json`, and the
figure are generated from the raw files without refitting. Evaluation timing includes setup and solution. Calibration candidate rows
record solves on reused models and omit the one-time model construction cost.
Concurrent load makes this an accuracy study,
not a controlled runtime benchmark. Dedicated speed measurements remain in
`docs/cpu-performance.md`.

## Broader scope and external data

The supplied scenarios vary noise (0.05–0.30), halo position/multiplicity and
mass/concentration, radius (1.5/3/6 arcmin at fixed count density), weights,
masks, clustering, and lognormal truth. Each scenario is independently tuned.
The field/count convention uses gross area; nonuniform-weight effective density
is `(sum w)**2 / sum(w*w) / area`. Source counts at noise 0.05 should not be
presented as a forecast for a named survey at a different shape-noise level.

`massivenus-template.json` requires actual simulation maps in separate
calibration/evaluation directories. Content hashes reject overlapping files,
including renamed copies. Split by independent simulations/lines of sight,
not merely overlapping crops with different random seeds. The truth loader
now retains the actual patch mean; subtracting its minimum is not restoration
of the simulation's DC mode. A real MassiveNuS run has not been supplied here.

`femmi.coherence` computes annular Fourier cross-coherence on already registered
κ and tracer maps and propagates supplied κ samples through threshold crossings.
It reports unresolved/censored coherence lengths rather than forcing finite
values or a Fisher error estimate. WCS alignment, common pixel/PSF treatment,
mask/window bias, source redshifts, actual Chandra data, and a validated posterior
are prerequisites for the proposed real-cluster experiment. No real-cluster
validation is claimed by the synthetic diagnostic tests.
