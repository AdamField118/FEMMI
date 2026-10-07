"""Binning and independent truth loading; mapper integration is in test_mapping."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.catalog import (
    analytic_gaussian_catalog,
    bin_shear_to_grid, kaiser_squires_binned,
    load_frontier_model, field_to_catalog,
)


def _corr_inner(pred, truth, x, y, r=1.5):
    m = (np.hypot(x, y) < r) & np.isfinite(pred)
    return np.corrcoef(pred[m], truth[m])[0, 1]


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
