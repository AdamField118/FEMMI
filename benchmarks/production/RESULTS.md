# Production mapper CPU measurements

Base: `9f0394aec74e51ee3792fc07eb462dbfffcf919a`, plus this patch.
Three fresh processes per backend/element; 200 sources and three new-noise fits per mesh. Serial runs with BLAS/OpenMP/Numba threads fixed to one. Independent median columns need not add exactly.

| Element | Backend | DOFs | Setup (s) | Single layer (s) | Three fits (s) | Total (s) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| p3 | numpy | 1972 | 1.2883 | 0.0276 | 0.1049 | 1.3950 | 391.9688 |
| p3 | numba | 1972 | 1.3134 | 0.0128 | 0.1003 | 1.4275 | 513.2969 |
| argyris | numpy | 2037 | 0.9273 | 0.0299 | 0.2626 | 1.2077 | 349.3203 |
| argyris | numba | 2037 | 1.0221 | 0.0141 | 0.2704 | 1.3098 | 467.1758 |
| hct | numpy | 1347 | 1.3110 | 0.0276 | 0.1156 | 1.4234 | 329.3711 |
| hct | numba | 1347 | 1.3452 | 0.0137 | 0.1226 | 1.4635 | 449.8672 |

All 9 comparisons passed. Largest relative FE-coefficient/prediction difference: 8.11e-09. All 54 timed fits passed the fresh residual acceptance criterion.

Numba reduces single-layer assembly time by roughly a factor of two at this size, but there is no measured end-to-end gain over the optimized NumPy path. The additional runtime memory and initialization cost remain visible. This is not a comparison with the original unoptimized implementation.

First-process compilation: 0.6921 s; next-process cache loading: 0.0857 s. Import and first execution are separate raw fields. These startup values are single observations, not distributions.

The diagnostic profiles locate the next CPU work: P3 reference Hessian construction through JAX dominates fresh setup; Argyris volume assembly is dominated by element construction and monomial evaluation. Optimizing these requires separate parity checks. The profiles are instrumented and are excluded from speed comparisons.

Reproduce:

```bash
python examples/diagnostics/run_profiles.py --sources 200 --output benchmarks/production/runs
```

The saved JSON/NPZ pairs are the measurement and parity evidence. See `docs/production-performance.md` for phase definitions, cold-cache instructions and individual profiling commands. Repeat on target publication hardware before making a speed claim.
