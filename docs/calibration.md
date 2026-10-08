# Choosing regularization

The production estimator needs `lam` and `length`. Lambda controls the penalty
relative to the supplied weights; length is measured in arcminutes. Each element
family needs its own calibration.

For a real catalogue without convergence truth, use held-out shear prediction:

```python
selection = mapper.select_regularization(
    lambdas=[.03, .3, 3.], lengths=[0., .3, 1., 3.], folds=3, seed=42)
if selection['boundary_unresolved']:
    raise RuntimeError('expand the grid and repeat selection')
```

The selection dictionary's `best` entry contains `lam`, `length`, and `score`.
Pass only the parameters to reconstruction:

```python
best = selection['best']
result = mapper.reconstruct(lam=best['lam'], length=best['length'])
```

Validation positions remain part of the mesh, but their shear values do not
enter the training likelihood. The criterion is weighted predictive shear MSE,
not mass-map error or posterior coverage. Repeat splits to assess stability.

For synthetic method comparisons, follow the [benchmark protocol](smpy-benchmarks.md).
It uses separate calibration and evaluation catalogues and jointly tunes both
parameters, expanding winning search boundaries. Recipes for density, field
size, masks, clustering, noise, and weights live in `configs/benchmarks/calibration.json`.
Choose the scenarios and budgets before evaluation; do not retune from the
held-out scores.

The research pipeline also offers Morozov discrepancy selection. It requires a
noise scale in the same weight convention. Noise estimates from fitted
residuals can absorb signal or model error; they should not be treated as an
independent survey noise calibration. See [configuration](configuration.md)
for that separate interface.
