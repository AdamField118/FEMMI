# Catalogue mapper

`FEMMapper` is the production quadratic estimator for P3, Argyris and HCT.
`MapperConfig` requires the element, regularization strength, physical correlation
length and field radius. All four choices are explicit.

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
and returns NaN outside the computational mesh. The ring radius is `boundary_padding` times
the configured field radius (default 1.12). `boundary_nodes` optionally fixes
the ring resolution. Ring vertices have no observations. Values inside
holes are prior-dependent predictions, not measured pixels.

## Observation and noise contract

Coordinates and correlation length are in **arcminutes**, on an East/North
tangent plane. `config.center` is the centre of the computational field in that
plane; `FlatCatalog.center` records the celestial tangent point. Use
`read_fits_catalog(...).to_tangent_plane(...)` for sky coordinates and check the
survey's shear sign and response convention. Select masks explicitly with
`FlatCatalog.select`; the estimator never silently drops observations.

Inputs are calibrated **shear**, for one effective source plane. Reduced shear
is rejected. Redshifts may accompany sources for selection and provenance but
never change the forward operator or become per-source lensing efficiencies.

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
These tolerances describe numerical error, not the
observational chi-square acceptance region.

`MassMap` contains FE coefficients, source-position convergence, predicted shear,
actual input shear, the effective configuration and convergence diagnostics.
Its NPZ contains arrays and JSON metadata, readable with `allow_pickle=False`.
It includes mesh vertices/connectivity; load values directly for plotting or
rebuild the mapper from saved catalogue/config to evaluate arbitrary positions.
Setup and solve times are reported separately.

For survey FITS input and astronomical image products, use the separate
[SMPy-style survey API and CLI](survey-io.md):

```bash
femmi map --catalogue shear.fits --config configs/survey.yaml --output-dir results
```

`map_catalogue(flat, config)` is the one-call low-level estimator. Its `MassMap.save`
method remains the portable NPZ export for numerical work. `femmi run` remains the separate experimental prior/posterior workflow, including
nonquadratic penalties and sampling. Those research capabilities and their JAX
adjoints are retained; the production mapper is a NumPy/SciPy interface and
is not itself a JAX-traceable function.

## Conditional B diagnostic

```python
response = mapper.diagnose_b(e_fit=result)
print(response['diagnostics']['closure_relative'])
```

The returned `e`, `raw`, `leakage`, and `residual` entries are `MassMap` objects.
`raw` fits rotated observations; `leakage` fits the rotated E prediction;
`residual` fits the rotated shear residual. Their interpretation and finite-field
limitations are described in the [observation model](observation-model.md).
The diagnostic requires two extra solves when matching E and B fits are supplied.

See the [API reference](api.md) for every configuration field and result attribute.
