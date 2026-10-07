# SMPy comparison results

Timings are per-run wall times, including FEM setup. These are local measurements, not target-hardware speed claims.

| Scenario | Method | Fits | Failures | Mean field shape L2 | Mean total seconds |
|---|---|---:|---:|---:|---:|
| masked-integration | argyris | 3 | 0 | 0.91829 | 0.14416 |
| masked-integration | hct | 3 | 0 | 0.90271 | 0.20548 |
| masked-integration | p3 | 3 | 0 | 0.91435 | 0.014871 |
| masked-integration | smpy_ks | 3 | 0 | 0.8969 | 0.0048108 |
| masked-integration | smpy_ks_plus | 3 | 0 | 0.97077 | 0.10124 |
| nfw-integration | argyris | 3 | 0 | 0.84436 | 0.14293 |
| nfw-integration | hct | 3 | 0 | 0.80947 | 0.19928 |
| nfw-integration | p3 | 3 | 0 | 0.87082 | 0.014563 |
| nfw-integration | smpy_ks | 3 | 0 | 0.82709 | 0.0053074 |
| nfw-integration | smpy_ks_plus | 3 | 0 | 0.89256 | 0.053327 |
