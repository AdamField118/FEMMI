# Catalogue mapper

`FEMMapper` is the production quadratic estimator for P3, Argyris and HCT.
`MapperConfig` requires the element, regularization strength, physical correlation
length and field radius. There are no density-dependent historical defaults.

```python
import numpy as np
from femmi import FlatCatalog, MapperConfig, FEMMapper

catalogue = FlatCatalog(x, y, gamma1, gamma2, weight)
config = MapperConfig(method='hct', lam=0.3, length=0.6, radius=3.)
mapper = FEMMapper(catalogue, config)
result = mapper.reconstruct()
result.save('mass-map.npz')
kappa_grid = result.evaluate(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
# Same positions/weights: reuse the mesh, coupled LU and prior factorization.
second = mapper.reconstruct(other_gamma1, other_gamma2)
```

The numbers above illustrate syntax, not a calibration for an unknown cluster.
`evaluate` uses the element's actual polynomial basis, including HCT subcells,
and returns NaN outside the computational mesh. The ring radius is 1.12 times
the configured field radius. Ring vertices have no observations. Values inside
holes are prior-dependent predictions, not measured pixels.

## Observation and noise contract

Coordinates and correlation length are in **arcminutes**, on an East/North
tangent plane. `config.center` is the centre of the computational field in that
plane; `FlatCatalog.center` records the celestial tangent point. Use
`read_fits_catalog(...).to_tangent_plane(...)` for sky coordinates and check the
survey's shear sign and response convention. Select masks explicitly with
`FlatCatalog.select`; the estimator never silently drops observations.

Inputs are calibrated **shear**, for one effective source plane. Reduced shear
and per-source redshifts are rejected rather than silently interpreted as shear.
If a catalogue contains redshifts, any effective-plane approximation must be
made explicitly before constructing `FlatCatalog` for this estimator.

The objective is

$$\sum_i w_i[(F_1\kappa-\gamma_1)_i^2+(F_2\kappa-\gamma_2)_i^2]
 +\lambda\kappa^T(M+\ell^2K)\kappa.$$

Weights are used exactly as supplied. For a likelihood, use inverse component
variance. With relative weights and a common variance, lambda must be in the
same relative-weight convention. Multiplying every weight and lambda by the
same factor leaves the estimate unchanged. Zero weights remove likelihood rows;
nonfinite arrays, negative weights and duplicate positions require explicit
cleaning. Covariance between galaxies or components is not modeled here.

## Regularization without truth

```python
selection = mapper.select_regularization(
    lambdas=[0.03, 0.3, 3.], lengths=[0., 0.3, 1., 3.], folds=3, seed=42)
if selection['boundary_unresolved']:
    raise RuntimeError('expand the search grid before accepting its winner')
best = selection['best']
result = mapper.reconstruct(lam=best['lam'], length=best['length'])
```

This uses held-out weighted shear prediction. Validation positions remain in the
fixed mesh, but validation shear does not enter the training likelihood.
It selects predictive regularization, not an optimal mass-map error or a calibrated
confidence interval. Record the complete selection report and examine stability
across splits. Independent synthetic calibration remains available separately.

## Numerical contract and output

The solver uses prior-preconditioned CG with an internal relative target of
1e-8 and independently recomputed normal-equation residual acceptance of 1e-6.
Both are configurable. A failed solve raises; there is no silent unconverged map.
This contract is inherited from FEMMI's verified solver, not copied from an
unrelated iteration count. These tolerances describe numerical error, not the
observational chi-square acceptance region.

`MassMap` contains FE coefficients, source-position convergence, predicted shear,
actual input shear, the effective configuration and convergence diagnostics.
Its NPZ contains arrays and JSON metadata, readable with `allow_pickle=False`.
It includes mesh vertices/connectivity; load values directly for plotting or
rebuild the mapper from saved catalogue/config to evaluate arbitrary positions.
Setup and solve times are reported separately.

```bash
femmi map --catalogue catalogue.npz --config mapper.yaml --output mass-map.npz
```

The input NPZ contains `x,y,g1,g2,weight`, optionally `units` (default `arcmin`).
A `z` array is rejected. Generate the YAML with `MapperConfig.save(path)`.
`femmi run` remains the separate experimental prior/posterior workflow, including
nonquadratic penalties and sampling. Those research capabilities and their JAX
adjoints are retained; the production mapper is a NumPy/SciPy interface and
is not itself a JAX-traceable function.

## API design references

- [SMPy, commit 26d231f](https://github.com/GeorgeVassilakis/SMPy/tree/26d231f5b4b22b41e76cb3bd3143d799c2e7ebfe):
  `api.map_mass`, `Config`, and mapper `create_maps` separate caller-facing input,
  explicit method settings and numerical mapping. FEMMI follows that separation,
  while keeping catalogues unbinned and requiring prior parameters.
- [jax-fem, commit 9a79b4b](https://github.com/deepmodeling/jax-fem/tree/9a79b4bb47460a90a6fafe26f1fbd58d2d3fed08):
  problem objects and solver options separate assembly from solve control.
  FEMMI keeps reusable geometry and explicit residual checks; Newton tolerances
  are not substituted for FEMMI's linear MAP residual definition.
- [Clawpack/PyClaw Controller](https://github.com/clawpack/pyclaw/blob/master/src/pyclaw/controller.py):
  separate solver, state and output responsibilities. FEMMI keeps persistence
  out of the solve and uses a distinct result object.

The obsolete `reconstruct_catalog`, density runners/interpolated KS anchors and
one-off comparison scripts were removed. Use `map_mass`/`FEMMapper` for data and
`calibrated_comparison.py`/`benchmark_smpy.py` for experiments. There are no aliases
preserving the obsolete defaults.
