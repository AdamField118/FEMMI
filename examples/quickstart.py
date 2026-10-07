"""Run from the repository root: python examples/quickstart.py."""
import numpy as np
from femmi import FlatCatalog, MapperConfig, FEMMapper
from femmi.catalog import analytic_gaussian_catalog

cat = analytic_gaussian_catalog(n_gal=200, field_radius=3., shape_noise=.02, seed=3)
catalogue = FlatCatalog(cat['x'],cat['y'],cat['g1'],cat['g2'],np.ones(len(cat['x'])))
# Demonstration values, not calibrated defaults for real observations.
mapper = FEMMapper(catalogue, MapperConfig('p3',lam=.3,length=.6,radius=3.))
result = mapper.reconstruct()
result.save('mass-map.npz')
print(result.diagnostics)
