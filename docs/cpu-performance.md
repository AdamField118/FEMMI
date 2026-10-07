# CPU acceleration: correctness and performance

Base: `e55e1412035ba2b215d2dbb9633f5536da7beea8` (main, 7 October 2026).
Measurements started at `fa079a9468a19cb1713f814304270723f2025e44`.
The intervening commits only removed an accidentally tracked patch and added
`*.patch` to `.gitignore`; all numerical source files are identical.

## Changes

- P3 and degree-general straight-edge single/double layers now share one
  float64 assembly implementation, with optional serial Numba kernels.
- Gauss-Legendre and logarithmic Gauss-Laguerre nodes, weights, and basis
  evaluations are cached by degree/rule. Duffy self interactions retain the same
  quadrature, factoring the geometry into `L²/(2π) * (log(L) A + B)`.
  Geometry-dependent quadrature points are computed once per assembly.
- The NumPy fallback also benefits from precomputation and staged matrix
  contractions. Numba uses `fastmath=False`, no parallel scatter, and disk caching.
- C¹ spaces lazily retain each element's coefficient solve. HCT evaluates
  monomials and subtriangle membership in batches, preserving first-match traces
  and the original tolerance. Reference triangle quadrature is cached.
- SciPy's sparse factorization, dense solves, optimizers, and JAX differentiation
  paths are unchanged. Curved BEM, hypersingular assembly and ACA remain separate.

`C1Space` now owns read-only copies of vertices/connectivity; construct a new
space when geometry changes. Cached element arrays are read-only too.
`space.clear_element_cache()` releases element storage without changing the mesh.

## Measurement design

AMD EPYC 9V74 host, Linux x86-64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
JAX 0.11.2, Numba 0.68.0. BLAS/OpenMP/Numba thread counts were fixed at one.
These are shared-host timings: use the raw ranges, not extra decimal places.

Three fresh processes per workload/backend, each building one mesh and fitting
the same deterministic analytic Gaussian shear catalogue three times from zero.
Seed 2718, per-component noise 0.01, lambda 1, correlation length 0.5.
The driver applies identical L-BFGS-B stopping options to both revisions:
`gtol=1e-6`, `ftol=1e-14`, `maxcor=20`, `maxiter=20000`.
This is a fixed performance workload, not a choice of calibrated science settings.
All 81 final fits reported convergence.

P3: square nx=ny=20, 3,721 DOFs, 240 boundary DOFs; boundary observations
excluded. Argyris/HCT: circular mesh with 24 boundary vertices, 61 total
vertices and 96 triangles; 522/339 volume DOFs and 120/72 BEM DOFs respectively.
Boundary observations are valid for the explicit C¹ observation operators used.

The table reports median seconds. Total is setup + catalogue construction +
three reconstructions; imports, parity probes and file writes are excluded.
Numba is initialized before setup and reported separately below. Setup includes
mesh construction, volume/shear/BEM assembly, boundary coupling and factorization.
The sparse-factorization column is only SuperLU; the dense boundary solve is
included in setup. Independent medians need not add exactly.

| Workload / implementation | Setup | Single layer | Double layer | Sparse factorization | Three fits | Total |
|---|---:|---:|---:|---:|---:|---:|
| p3 / baseline | 5.8052 | 2.9131 | 0.0490 | 0.0731 | 0.5119 | 6.3286 |
| p3 / numpy | 2.2742 | 0.1746 | 0.0320 | 0.0634 | 0.5324 | 2.8073 |
| p3 / numba | 2.3305 | 0.0655 | 0.0037 | 0.0788 | 0.5637 | 2.9394 |
| argyris / baseline | 4.5639 | 4.1853 | 0.0123 | 0.0093 | 19.3067 | 24.0615 |
| argyris / numpy | 0.3406 | 0.0303 | 0.0060 | 0.0101 | 18.5959 | 18.8640 |
| argyris / numba | 0.3382 | 0.0194 | 0.0009 | 0.0138 | 18.7570 | 19.0833 |
| hct / baseline | 4.9872 | 1.9524 | 0.0092 | 0.0027 | 0.2392 | 5.2351 |
| hct / numpy | 0.4475 | 0.0286 | 0.0046 | 0.0028 | 0.1795 | 0.6274 |
| hct / numba | 0.4755 | 0.0175 | 0.0009 | 0.0049 | 0.1772 | 0.6482 |

| Workload | Setup speedup (baseline / Numba) | Total speedup | Total including disk-cache initialization |
|---|---:|---:|---:|
| p3 | 2.49× | 2.15× | 3.384 s |
| argyris | 13.49× | 1.26× | 19.498 s |
| hct | 10.49× | 8.08× | 1.071 s |

Most of the gain is precomputation and HCT batching. Numba further reduces
single-layer time by about 2.7× / 1.6× / 1.6× against the **new NumPy path**
(P3 / Argyris / HCT), but these workloads show no additional end-to-end win over
that fallback within host timing variation. The reported overall speedups belong
to the combined changes, not to Numba alone. Repeated reconstruction does not
reassemble BEM, so it should not inherit the assembly speedup.

HCT volume assembly fell from 2.689 s to 0.374 s; Argyris volume assembly stayed
near 0.28 s. Cached trace assembly fell from 0.056 to 0.016 s for Argyris and
0.070 to 0.011 s for HCT. A separate profile recorded 96 element constructions
instead of 120 for Argyris and 216 for HCT. HCT basis evaluation fell from
4.16 profiled seconds to 0.060; profiles include instrumentation overhead and
are not used for the timing ratios above.

## Initialization and memory

A fresh `NUMBA_CACHE_DIR` measured 0.253 s importing/initializing Numba and
1.080 s compiling the kernel. A second process using that cache measured
0.267 s import plus 0.140 s cache load. The first tiny execution was below
0.001 s. The whole cold HCT workload (including compilation) was 1.894 s;
the corresponding disk-cache run was 1.060 s. These separate startup observations
are single runs, not timing distributions.

Median process peak RSS rose from approximately 349/244/239 MiB to
454/347/341 MiB for P3/Argyris/HCT with Numba. The optimized NumPy path used
353/246/240 MiB. These are whole-process high-water marks, including JAX and
Numba runtime overhead; they are not isolated element-cache measurements.

## Numerical checks and limitations

- Full optional-dependency suite with Numba required: **295 passed**; two existing
  warnings flag convergence slopes fitted from only two resolutions.
- Final NumPy-forced integration/parity checks on the latest main: **32 passed**.
- Frozen matrices from fa079a9 cover P3/degree-5, quadrature orders 7/25, irregular
  boundaries, two scales, singular self blocks and adjacent blocks. Both backends
  satisfy `rtol=2e-12, atol=3e-15` against the old implementation.
- All 18 paired final comparisons passed: identical input arrays, forward/gradient
  relative errors below `1e-9`, objective agreement within `1e-5` relative,
  prediction and vertex-kappa differences below `1e-3` relative.
  Actual largest operator probe error was about `9.5e-12`.
  Argyris's Numba vertex-map difference was `2.83e-4` relative, predictions
  `7.92e-5`; mixed value/derivative coefficient norms are also saved.
- Ordinary floating-point reassociation changes some last bits and optimizer
  trajectories. The initial lambda=0.03, 300-iteration experiment did not
  converge; Argyris even hit 10,000 iterations under a stricter follow-up.
  Those exploratory records are retained in the evidence bundle but excluded
  from the matched-quality speedup table. Acceleration does not fix that existing
  optimizer-conditioning issue.

The remaining P3 setup profile is dominated by JAX reference-Hessian construction
(about 1.66 instrumented seconds); repeated fits spend much of their time in
SuperLU solves. C¹ setup now spends most of its time constructing/evaluating
volume bases. Argyris's final workload needs roughly 8,200 optimizer iterations,
so its total speedup is modest despite a large assembly improvement. These are
the next measured CPU targets. No CUDA speedup, production catalogue scaling,
or SMPy accuracy claim is established here.

## Reproduce

Install `pip install -e ".[dev,speed,neural,galsim,io]"`. Use the updated driver
against both an unmodified baseline checkout and the updated checkout. Example:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
FEMMI_BEM_BACKEND=numpy python examples/diagnostics/profile_cpu.py --kind p3 --size 20 --lam 1 --length .5 --gtol 1e-6 --maxiter 20000 --repeats 3 --source-root /path/to/baseline --output /tmp/baseline-p3.json
FEMMI_BEM_BACKEND=numba python examples/diagnostics/profile_cpu.py --kind p3 --size 20 --lam 1 --length .5 --gtol 1e-6 --maxiter 20000 --repeats 3 --warm-numba --output /tmp/numba-p3.json
python examples/diagnostics/compare_cpu.py /tmp/baseline-p3.json /tmp/numba-p3.json
```

Use `--kind argyris --size 24` or `--kind hct --size 24` for C¹; repeat each
command three times with unique output filenames. Run the new NumPy backend too
(omit `--warm-numba`). All runs must be sequential to avoid competing workloads.
To profile, add `--profile /tmp/run.prof` in a separate run; inspect it with
`python -m pstats /tmp/run.prof`. To measure first compilation, supply a fresh
`NUMBA_CACHE_DIR` together with `--warm-numba`; reuse it for cache-load timing.
The driver uses Unix `resource` for peak RSS (Linux units).

```bash
FEMMI_REQUIRE_OPTIONAL=1 FEMMI_BEM_BACKEND=numba python -m pytest -q
FEMMI_BEM_BACKEND=numpy python -m pytest -q tests/test_cpu_acceleration.py tests/test_hct_observations.py tests/test_observation_consistency.py::test_transpose_numpy_jax_and_hessian tests/test_c1_inverse.py::test_adjoint_gradient_matches_finite_differences
```

The backend parity test requires Numba to be installed to exercise both paths;
otherwise the Numba parameterizations skip. Explicit `FEMMI_BEM_BACKEND=numba`
raises when Numba is absent. Invalid backends and JIT/runtime errors are not
silently replaced by another implementation. See the optional-dependency extra
and [Numba compilation documentation](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html).

Machine-readable timing and comparison records are in
`benchmarks/cpu/2026-10-07/`. The accompanying evidence bundle also includes
NPZ outputs, binary/readable profiles, exploratory failures and test logs.
The drivers regenerate those artifacts; no revised science benchmark tables
are inferred from these timing workloads.
