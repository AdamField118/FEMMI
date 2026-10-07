# Production mapper performance

Use `examples/diagnostics/profile_cpu.py` to measure the same `FEMMapper` used
by catalogue mapping and calibration. Geometry is fixed while each repeated fit
receives a deterministic new noise realization. Every fit starts from zero.
Settings and actual observations are stored with results. This replaces the
older L-BFGS-only timing driver; historical CPU results remain historical data.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/femmi-fresh-cache
FEMMI_BEM_BACKEND=numba python examples/diagnostics/profile_cpu.py \
  --kind hct --sources 200 --repeats 3 --warm-numba --output cold.json
FEMMI_BEM_BACKEND=numba python examples/diagnostics/profile_cpu.py \
  --kind hct --sources 200 --repeats 3 --warm-numba --output warm.json
FEMMI_BEM_BACKEND=numpy python examples/diagnostics/profile_cpu.py \
  --kind hct --sources 200 --repeats 3 --output numpy.json
python examples/diagnostics/compare_cpu.py numpy.json warm.json --output parity.json
```

Create a genuinely new cache directory for the cold run. Repeat for P3 and
Argyris, multiple catalogue sizes and at least three fresh processes per case.
Run serially on an otherwise idle node. `--profile` writes a cProfile file;
profiled runs are rejected by the speedup comparison utility.

Reported phases: Numba import, compilation/cache loading, first kernel execution,
catalogue generation, full setup, single/double-layer assembly, C1 volume assembly,
coupled sparse factorization, prior factorization and repeated reconstruction.
The factorization/assembly subtimers are contained in setup or reconstruction;
never add them again to the total. Setup also contains meshing, dense boundary
solves and, for P3, JAX reference-basis work. RSS is the process high-water mark,
including imports, rather than an allocation count for FEM matrices alone.

`total_s` is catalogue generation + setup + repeated fits.
`total_including_jit_s` additionally includes explicitly separated Numba startup.
It excludes Python/module startup, output writing and cProfile overhead. These
labels should accompany any reported speed ratio.

Comparison requires identical workload/configuration, accepted fresh residuals,
matching objectives and FE coefficients/predicted shear. Tolerances are numerical
parity thresholds, not speed assertions. CUDA and automatic solver switching
remain outside this measured CPU task.
