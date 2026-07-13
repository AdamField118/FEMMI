# Quickstart

The fastest path is the CLI with a config file (see
[Configuration & CLI](configuration.md)):

```bash
femmi run --config configs/default.yaml
python examples/plot_npz.py runs/run.npz     # truth / kappa / std panels
```

## The Python API

`examples/quickstart.py` is the minimal end-to-end reconstruction from a galaxy
shear catalog:

```python
import numpy as np
from femmi.operators import build_operators_catalog
from femmi.catalog   import analytic_gaussian_catalog, reconstruct_catalog

# a synthetic catalog (x, y, g1, g2) with a known truth
cat = analytic_gaussian_catalog(n_gal=2000, sigma=0.5, shape_noise=0.05, seed=0)

# build the FEM-BEM operators with nodes AT the galaxy positions
ops, cm = build_operators_catalog(cat["x"], cat["y"], radius=None, n_boundary=48)

# MAP reconstruction with automatic lambda (Morozov noise matching)
kappa = reconstruct_catalog(ops, cm, cat["g1"], cat["g2"], noise_std=0.05)
print("reconstructed kappa on", len(kappa), "nodes")
```

## What you get back

A `femmi run` writes `runs/<name>.npz` with node-aligned fields:

| key | meaning |
|---|---|
| `kappa` | the reconstruction (MAP, or posterior point estimate) |
| `std`   | per-node posterior uncertainty (sampling runs only) |
| `truth` | the input $\kappa$ when a synthetic/known truth exists |
| `nodes` | the `(N, 2)` mesh-node coordinates |

Plot them with `examples/plot_npz.py`, which draws each field as a `tripcolor`
over the mesh triangulation.

More teaching examples — priors, uncertainty, KS head-to-head — are indexed in
[`examples/README.md`](https://github.com/AdamField118/FEMMI/blob/main/examples/README.md).
