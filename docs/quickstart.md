# Quickstart

This example creates a noisy analytic shear catalogue and reconstructs it.
Run it from an installed checkout; it does not need a downloaded data set.

```python
from pathlib import Path
import numpy as np
from femmi import FlatCatalog, MapperConfig, FEMMapper
from femmi.catalog import analytic_gaussian_catalog

c = analytic_gaussian_catalog(n_gal=80, field_radius=3.,
    sigma=.7, amp=.1, shape_noise=.01, seed=5)
flat = FlatCatalog(c['x'], c['y'], c['g1'], c['g2'], np.ones(len(c['x'])))
mapper = FEMMapper(flat, MapperConfig('p3', lam=.3, length=.6, radius=3.))
result = mapper.reconstruct()
Path('results').mkdir(exist_ok=True)
result.save('results/mass-map.npz')
print(result.diagnostics['relative_residual'])

axis = np.linspace(-2., 2., 40)
x, y = np.meshgrid(axis, axis)
kappa = result.evaluate(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
```

`kappa` is dimensionless; coordinates and `length` are in arcminutes. The prior
settings illustrate the API and must be chosen for the intended data. The saved
NPZ can be opened with `numpy.load(..., allow_pickle=False)`.

To repeat a reconstruction on the same positions and weights, call
`mapper.reconstruct(other_g1, other_g2)`. This reuses assembly and factorization.
Changing positions or weights requires a new mapper.

For a FITS table, use [map_mass](survey-io.md). It handles column selection,
coordinate conversion, WCS evaluation, and astronomical output products.
