"""
tests/test_catalog_pipeline.py
End-to-end catalog-native reconstruction (femmi.catalog):

  - reconstruct_catalog runs a raw (x, y, g1, g2) catalog all the way to a
    kappa map, placing observed shear only on galaxy nodes,
  - the data weight is a correct binary galaxy selection (0 on guard/boundary),
  - FEMMI recovers an analytic Gaussian convergence from its exact tangential
    shear, and so does the SMPy-style binned Kaiser-Squires path,
  - the galaxy -> node kappa mapping is consistent.

Run:
    python -m pytest tests/test_catalog_pipeline.py -v
    python tests/test_catalog_pipeline.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.catalog import (
    analytic_gaussian_catalog, reconstruct_catalog,
    bin_shear_to_grid, kaiser_squires_binned,
    load_frontier_model, field_to_catalog,
)


def _corr_inner(pred, truth, x, y, r=1.5):
    m = (np.hypot(x, y) < r) & np.isfinite(pred)
    return np.corrcoef(pred[m], truth[m])[0, 1]


def test_reconstruct_catalog_runs_and_maps():
    cat = analytic_gaussian_catalog(n_gal=700, sigma=0.5, shape_noise=0.02, seed=3)
    cr  = reconstruct_catalog(cat['x'], cat['y'], cat['g1'], cat['g2'],
                              center=(0., 0.), n_boundary=72, wiener_length=0.5,
                              use_morozov=False, lam_reg=1e-2, maxiter=250,
                              verbose=False)

    # kappa defined at every node, and per-galaxy values are finite where kept
    assert cr.kappa_nodes.shape[0] == cr.ops.n_nodes
    assert len(cr.kappa_gal) == len(cat['x'])
    kept = np.isfinite(cr.kappa_gal)
    assert kept.sum() > 0.8 * len(cat['x'])

    # galaxy-node kappa matches the flat kappa_nodes at the mapped nodes
    cm = cr.catalog_mesh
    assert np.allclose(cr.kappa_gal[cm.source_index], cr.kappa_nodes[cm.galaxy_nodes],
                       equal_nan=True)


def test_data_weight_is_galaxy_selection():
    cat = analytic_gaussian_catalog(n_gal=500, seed=4)
    cr  = reconstruct_catalog(cat['x'], cat['y'], cat['g1'], cat['g2'],
                              center=(0., 0.), n_boundary=72, use_morozov=False,
                              lam_reg=1e-2, maxiter=1, verbose=False)
    cm = cr.catalog_mesh
    w  = cr.data_weight

    # weight is 1 exactly on galaxy nodes, 0 everywhere else
    assert np.all(w[cm.galaxy_nodes] == 1.0)
    off = np.ones(len(w), dtype=bool); off[cm.galaxy_nodes] = False
    assert np.all(w[off] == 0.0)
    # boundary nodes (no data) must be excluded
    assert np.all(w[np.array(cr.ops.mesh.boundary)] == 0.0)


def test_femmi_recovers_gaussian_from_catalog():
    cat = analytic_gaussian_catalog(n_gal=1200, sigma=0.5, shape_noise=0.05, seed=1)
    cr  = reconstruct_catalog(cat['x'], cat['y'], cat['g1'], cat['g2'],
                              center=(0., 0.), n_boundary=96, wiener_length=0.5,
                              use_morozov=False, lam_reg=1e-2, maxiter=300,
                              verbose=False)
    corr = _corr_inner(cr.kappa_gal, cat['kappa_true'], cat['x'], cat['y'])
    assert corr > 0.85, f"FEMMI catalog-native corr={corr:.3f}"


def test_ks_binned_recovers_gaussian():
    cat = analytic_gaussian_catalog(n_gal=1200, sigma=0.5, shape_noise=0.05, seed=1)
    eval_pts = np.column_stack([cat['x'], cat['y']])
    kks = kaiser_squires_binned(cat['x'], cat['y'], cat['g1'], cat['g2'],
                                grid_size=48, smoothing_px=1.5, eval_pts=eval_pts)
    corr = _corr_inner(kks, cat['kappa_true'], cat['x'], cat['y'])
    assert corr > 0.85, f"KS-binned corr={corr:.3f}"


def test_bin_shear_conserves_mean():
    cat = analytic_gaussian_catalog(n_gal=800, seed=2)
    g1g, g2g, counts, ext = bin_shear_to_grid(cat['x'], cat['y'],
                                               cat['g1'], cat['g2'], grid_size=32)
    assert g1g.shape == (32, 32)
    assert counts.sum() == len(cat['x'])
    # occupied-pixel mean shear is close to the catalog mean (weighted binning)
    occ = counts > 0
    assert abs(g1g[occ].mean() - cat['g1'].mean()) < 0.05


def _write_synthetic_frontier(tmpdir, n=161, s=22.0, A=1000.0):
    """Analytic Gaussian potential -> exact kappa/gamma; write kappa+psi+deflect."""
    from astropy.io import fits
    yy, xx = np.mgrid[0:n, 0:n]
    x = xx - n // 2; y = yy - n // 2; r2 = x**2 + y**2
    psi = A * np.exp(-r2 / (2 * s**2))
    kap = 0.5 * (-2 / s**2 + r2 / s**4) * psi
    g1a = 0.5 * (x**2 - y**2) / s**4 * psi
    g2a = (x * y) / s**4 * psi
    fits.writeto(os.path.join(tmpdir, "hlsp_x_kappa.fits"), kap, overwrite=True)
    fits.writeto(os.path.join(tmpdir, "hlsp_x_psi.fits"), psi, overwrite=True)
    fits.writeto(os.path.join(tmpdir, "hlsp_x_x-arcsec-deflect.fits"), -x / s**2 * psi, overwrite=True)
    fits.writeto(os.path.join(tmpdir, "hlsp_x_y-arcsec-deflect.fits"), -y / s**2 * psi, overwrite=True)
    return x, y, r2, g1a, g2a


def test_frontier_loader_derives_correct_shear():
    """psi-Hessian shear matches the analytic gamma; kappa truth is preserved."""
    import pytest
    pytest.importorskip("astropy")
    import tempfile
    tmp = tempfile.mkdtemp()
    x, y, r2, g1a, g2a = _write_synthetic_frontier(tmp)

    fld = load_frontier_model(tmp, source="psi", pixscale_arcsec=1.0,
                              downsample=1, verbose=False)
    interior = (np.abs(x) < 45) & (np.abs(y) < 45) & (r2 > 1)
    def med_rel(a, b):
        return np.median(np.abs(a[interior] - b[interior]) /
                         (np.abs(b[interior]) + 1e-6))
    assert med_rel(fld["g1"], g1a) < 0.05
    assert med_rel(fld["g2"], g2a) < 0.05
    assert fld["X"].shape == fld["kappa_true"].shape

    cat = field_to_catalog(fld, n_gal=400, shape_noise=0.0, rmax_arcmin=0.6, seed=0)
    assert len(cat["x"]) == 400
    assert np.isfinite(cat["g1"]).all() and np.isfinite(cat["kappa_true"]).all()


def test_frontier_deflection_crosscheck():
    """Deflection-derived shear matches the psi-Hessian shear (independent path)."""
    import pytest
    pytest.importorskip("astropy")
    import tempfile
    tmp = tempfile.mkdtemp()
    x, y, r2, g1a, g2a = _write_synthetic_frontier(tmp)

    fp = load_frontier_model(tmp, source="psi", pixscale_arcsec=1.0,
                             downsample=1, verbose=False)
    fd = load_frontier_model(tmp, source="deflect", pixscale_arcsec=1.0,
                             downsample=1, verbose=False)
    interior = (np.abs(x) < 45) & (np.abs(y) < 45) & (r2 > 1)
    # deflection shear reproduces the analytic gamma and agrees with psi
    err = np.median(np.abs(fd["g1"][interior] - g1a[interior]) /
                    (np.abs(g1a[interior]) + 1e-6))
    assert err < 0.05, f"deflection g1 error {err:.3f}"
    cc = np.corrcoef(fp["g1"][interior], fd["g1"][interior])[0, 1]
    assert cc > 0.99, f"psi vs deflection shear agreement corr={cc:.3f}"


if __name__ == "__main__":
    tests = [
        test_reconstruct_catalog_runs_and_maps,
        test_data_weight_is_galaxy_selection,
        test_femmi_recovers_gaussian_from_catalog,
        test_ks_binned_recovers_gaussian,
        test_bin_shear_conserves_mean,
        test_frontier_loader_derives_correct_shear,
        test_frontier_deflection_crosscheck,
    ]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn(); passed += 1; print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}"); failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}"); failed.append(fn.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if not failed else 1)
