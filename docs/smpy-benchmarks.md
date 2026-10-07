# SMPy comparisons

Install the immutable upstream implementation:

```bash
python -m pip install -r requirements-benchmark.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/paper/benchmark_smpy.py \
  --config benchmarks/smpy/configs.json --output results/smpy
```

The driver checks the installed VCS revision before any experiment. A missing or
wrong SMPy installation fails explicitly; it never substitutes FEMMI's KS.
The pinned upstream commit is `26d231f5b4b22b41e76cb3bd3143d799c2e7ebfe` (0.5.0).

## Methods and matching

| Upstream method | Adapter | Comparison |
|---|---|---|
| Kaiser–Squires | `smpy_ks` | convergence E/B maps |
| KS+ | `smpy_ks_plus` | convergence E/B maps with inpainting |
| Aperture mass | `smpy_aperture` | tested adapter; separate statistic, excluded from convergence rankings |

KS and KS+ independently tune grid size and smoothing. P3, Argyris and HCT
independently tune lambda and physical correlation length. Every method uses the
same calibration catalogues and disjoint evaluation seeds. Winning grid edges
expand and refine; unresolved edges stop evaluation unless `allow_unresolved`
is explicitly set for exploratory runs. This is a common selection protocol,
not an equal wall-time tuning budget. All candidate costs and scores are saved.

The source catalogue and weights are shared. Upstream Fourier methods receive
weighted mean shear in square pixels. Empty pixels are zero for KS; KS+ receives
the weight-sum availability grid, so zero observed shear remains valid data.
There is no shear sign flip or image transpose. Pixel arrays have order [y,x].
The convention tests recover an analytic Fourier E-mode and zero B-mode.

These are **linear shear** comparisons. KS+ uses one outer reconstruction pass
(`reduced_shear_iterations=1`), disabling nonlinear reduced-shear updates. Its
other settings follow the pinned upstream defaults, with 100 inpainting steps.
`ks_plus_iterations` is explicit and should be varied for publication-level
iteration/stability checks; it is not a FEM residual tolerance. No iteration
success flag is invented for SMPy's direct/fixed-iteration algorithms.

## Truth, evaluation and artifacts

The supplied suite covers GalSim NFW, separated halos, masks, variable weights,
independent lognormal fields and high noise. GalSim supplies analytic shear;
lognormal truth uses the repository's independent truth path. It does not use
FEMMI's forward operator to manufacture shear.

Selection uses source-position DC-removed relative L2, retained from the verified
calibration protocol. Evaluation additionally uses one fixed grid for every
method, including unobserved mask interiors. Regional errors remove **one common
field offset**, not separate offsets in each region. Reported quantities are
field shape error, interior/boundary/mask RMSE and compensated aperture contrast
error. Strong pixels with truth kappa>=1 are excluded equally from scoring;
that rule is recorded rather than treating those pixels as linear-shear evidence.

The runner saves full catalogues, candidates, selected settings, source maps,
spatial maps/truth/masks, upstream E/B grids, failure records and paired statistics.
`summary.json` includes paired errors and wall times; `RESULTS.md` is generated
from those records. A failed evaluation is retained, not silently dropped.
The timing column covers setup and reconstruction at sources; spatial scoring,
truth generation, imports and file I/O are outside it. Use the separate process
profiler for controlled phase and memory comparisons.

## Remaining publication runs

The supplied six-scenario configuration is a starting suite, not a comprehensive
survey claim. Expand density, field size, mask geometry, clustering, seeds and
truth families on target hardware. Validate iteration stability for KS+, add a
matched aperture-statistic comparison (the upstream aperture output is not kappa),
and supply independent MassiveNuS/real catalogues. The existing MassiveNuS split
checks are reused. The short integration run validates the complete workflow; it
cannot establish general superiority.
