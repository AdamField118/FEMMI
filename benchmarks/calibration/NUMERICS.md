# Numerical follow-up

These timings were collected with one BLAS thread and the Numba BEM backend while other experiments ran concurrently. They are illustrative; rerun serially on target hardware for publication. Complete trials and residuals are in `numerics/numerics.json`.

## Actual dual-grid operator spectrum

Positive kernel −log(r/σ)/(2π), σ twice polygon diameter; dual P0 V, continuous P1 W, stabilized constant mode; P = G⁻ᵀ Wₛ G⁻¹. The spectrum is computed through symmetric similarity, not nonsymmetric singular values.

| Nodes | Spacing | cond(V) | cond(PV) |
|---:|---|---:|---:|
| 16 | uniform | 26.16 | 1.3133 |
| 32 | uniform | 52.11 | 1.3296 |
| 64 | uniform | 104.10 | 1.3338 |
| 128 | uniform | 208.15 | 1.3348 |
| 16 | perturbed | 26.58 | 1.3131 |
| 32 | perturbed | 53.02 | 1.3296 |
| 64 | perturbed | 105.95 | 1.3338 |
| 128 | perturbed | 211.87 | 1.3348 |

## SuperLU / ILU-GMRES, checked at the same residual target

Both use the same operator and RHS. Columns combine measured setup cost with median solve time over five independent RHS, once or five times. Common assembly is excluded. The current builder already constructs an LU, so these cold alternatives do not represent an automatic end-to-end speedup.

| DOFs | LU + 1 solve (s) | ILU + 1 solve (s) | LU + 5 solves (s) | ILU + 5 solves (s) | GMRES iteration range | Maximum GMRES residual |
|---:|---:|---:|---:|---:|---|---:|
| 169 | 0.00162 | 0.00490 | 0.00178 | 0.01852 | 17–19 | 4.70e-11 |
| 625 | 0.00557 | 0.00768 | 0.00613 | 0.02092 | 19–20 | 6.39e-11 |
| 2401 | 0.02868 | 0.03183 | 0.03333 | 0.07441 | 20–21 | 5.97e-11 |
| 5329 | 0.09487 | 0.08699 | 0.10602 | 0.18993 | 21–22 | 8.17e-11 |
| 9409 | 0.23476 | 0.21332 | 0.25688 | 0.50836 | 22–22 | 2.51e-11 |

The first measured cold single-RHS advantage for ILU-GMRES is at 5329 DOFs (between the tested 2401 and 5329 sizes). It disappears by five RHS at every tested size. Every reuse-only solve is faster with the existing LU. This load-sensitive result does not support a universal DOF switch; expected reuse and target-machine measurements matter.

## Warm dense versus ACA single-layer assembly

Each route is warmed before three timed trials. A ratio below one favours dense assembly. ACA tolerance is 1e-9; the table reports the measured global matrix error, which need not equal that internal tolerance.

| Geometry | Degree | Boundary DOFs | Median dense/ACA time | Maximum relative matrix error |
|---|---:|---:|---:|---:|
| circle | 3 | 96 | 0.00250 | 4.75e-16 |
| circle | 3 | 192 | 0.00423 | 5.29e-09 |
| circle | 3 | 384 | 0.00679 | 8.60e-10 |
| circle | 5 | 95 | 0.00084 | 6.45e-16 |
| circle | 5 | 190 | 0.00185 | 2.02e-10 |
| circle | 5 | 380 | 0.00315 | 7.40e-10 |
| catalogue ring | 5 | 120 | 0.00117 | 4.79e-16 |
| catalogue ring | 5 | 240 | 0.00224 | 2.68e-08 |

ACA remains off by default. No crossover beyond the measured range is inferred.
