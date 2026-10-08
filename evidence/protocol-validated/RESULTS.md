# SMPy comparison results

Timings are per-run wall times, including FEM setup. These are local measurements, not target-hardware speed claims.

| Scenario | Method | Fits | Failures | Mean field shape L2 | Mean total seconds |
|---|---|---:|---:|---:|---:|
| nfw-integration | argyris | 3 | 0 | 0.8498 | 0.14702 |
| nfw-integration | hct | 3 | 0 | 0.81513 | 0.21016 |
| nfw-integration | p3 | 3 | 0 | 0.80842 | 0.015636 |
| nfw-integration | smpy_ks | 3 | 0 | 0.82709 | 0.005062 |
| nfw-integration | smpy_ks_plus | 3 | 0 | 0.91517 | 0.060283 |
