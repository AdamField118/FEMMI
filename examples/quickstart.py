"""
examples/quickstart.py
The shortest useful FEMMI program: reconstruct a mass map from a shear catalog.

If you read one file to understand the API, read this one. Everything else in
examples/ is a diagnostic or a benchmark built on top of these same few calls.

Run:
    python examples/quickstart.py
"""

import numpy as np
from femmi.catalog import reconstruct_catalog, analytic_gaussian_catalog

# 1. Get a shear catalog: (x, y) galaxy positions and their measured shear (g1, g2).
#    Here we make a synthetic one; on real data you would read a FITS catalog
#    (see femmi.io) and pass its columns instead.
cat = analytic_gaussian_catalog(n_gal=1500, sigma=0.5, shape_noise=0.06, seed=0)

# 2. Reconstruct the convergence kappa directly on the galaxy positions.
#    That's it -- the FEM-BEM mesh, far-field boundary condition, and automatic
#    regularisation are all handled inside.
rec = reconstruct_catalog(cat["x"], cat["y"], cat["g1"], cat["g2"])

# 3. rec.kappa_gal is the reconstructed kappa at each galaxy. A few galaxies that
#    fall on the mesh boundary (or are near-duplicates) carry no estimate and come
#    back as NaN, so compare on the finite ones.
kappa = rec.kappa_gal
truth = cat["kappa_true"]
ok = np.isfinite(kappa)
err = np.linalg.norm(kappa[ok] - truth[ok]) / np.linalg.norm(truth[ok])
print(f"reconstructed {ok.sum()} of {len(kappa)} galaxies   relative L2 error = {err:.3f}")

# Want a different prior? Pass prior='tv' | 'sparse' | 'maxent' | 'neural'.
# Want uncertainty (posterior mean + per-galaxy std)? See examples/uncertainty_demo.py.
