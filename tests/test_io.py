"""
tests/test_io.py
Tests for femmi/io.py: gnomonic projection, spin-2 shear transforms,
metacal response, catalog cleaning, and the FITS reader.

The FITS reader test is skipped if astropy is not installed; everything
else is pure numpy.

Run:
    python -m pytest tests/test_io.py -v
    python tests/test_io.py
"""

import sys, os, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.io import (
    gnomonic_project, gnomonic_deproject, projection_rotation,
    rotate_shear, tangential_cross_shear,
    apply_response, clean_catalog,
    ShearCatalog, FlatCatalog,
    _find_column, _detect_response, _DEFAULT_COLUMNS,
)

A2744_CENTER = (3.5862, -30.4003)  # degrees (RA, Dec)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

def test_projection_roundtrip():
    ra0, dec0 = np.deg2rad(A2744_CENTER[0]), np.deg2rad(A2744_CENTER[1])
    rng = np.random.default_rng(0)
    ra  = ra0 + np.deg2rad(rng.uniform(-0.06, 0.06, 3000))
    dec = dec0 + np.deg2rad(rng.uniform(-0.06, 0.06, 3000))
    xi, eta = gnomonic_project(ra, dec, ra0, dec0)
    ra2, dec2 = gnomonic_deproject(xi, eta, ra0, dec0)
    assert np.max(np.hypot(ra2 - ra, dec2 - dec)) < 1e-12


def test_projection_center_and_north():
    ra0, dec0 = np.deg2rad(A2744_CENTER[0]), np.deg2rad(A2744_CENTER[1])
    x0, y0 = gnomonic_project(ra0, dec0, ra0, dec0)
    assert abs(float(x0)) < 1e-14 and abs(float(y0)) < 1e-14
    # a point due North maps to +eta, zero xi
    xiN, etaN = gnomonic_project(ra0, dec0 + np.deg2rad(0.01), ra0, dec0)
    assert abs(float(xiN)) < 1e-12
    assert float(etaN) > 0.0


# --------------------------------------------------------------------------- #
# Spin-2 shear transforms
# --------------------------------------------------------------------------- #

def test_shear_rotation_algebra():
    g1, g2 = 0.3, -0.1
    # rotation by pi/2: g -> g*exp(-i*pi) = -g
    a1, a2 = rotate_shear(g1, g2, np.pi / 2)
    assert np.isclose(a1, -g1) and np.isclose(a2, -g2)
    # rotation by pi/4: g -> g*exp(-i*pi/2) = -i*g  => g1'=g2, g2'=-g1
    b1, b2 = rotate_shear(g1, g2, np.pi / 4)
    assert np.isclose(b1, g2) and np.isclose(b2, -g1)
    # magnitude preserved
    assert np.isclose(np.hypot(a1, a2), np.hypot(g1, g2))
    assert np.isclose(np.hypot(b1, b2), np.hypot(g1, g2))


def test_projection_rotation_small_and_invertible():
    ra0, dec0 = np.deg2rad(A2744_CENTER[0]), np.deg2rad(A2744_CENTER[1])
    assert abs(float(projection_rotation(ra0, dec0, ra0, dec0))) < 1e-10

    rng = np.random.default_rng(3)
    ra  = ra0 + np.deg2rad(rng.uniform(-0.06, 0.06, 1000))
    dec = dec0 + np.deg2rad(rng.uniform(-0.06, 0.06, 1000))
    omega = projection_rotation(ra, dec, ra0, dec0)
    # sub-degree over an HST-size field
    assert np.rad2deg(np.max(np.abs(omega))) < 0.1

    # rotating into the plane then back recovers the input shear
    cat = ShearCatalog.from_arrays(np.rad2deg(ra), np.rad2deg(dec),
                                   rng.normal(0, 0.2, ra.size),
                                   rng.normal(0, 0.2, ra.size))
    flat = cat.to_tangent_plane(center=A2744_CENTER)
    g1b, g2b = rotate_shear(flat.g1, flat.g2, -omega)
    assert np.max(np.abs(g1b - cat.g1)) < 1e-10
    assert np.max(np.abs(g2b - cat.g2)) < 1e-10
    assert flat.units == "arcmin"


def test_flip_g2():
    cat = ShearCatalog.from_arrays(
        np.array([A2744_CENTER[0]]), np.array([A2744_CENTER[1]]),
        np.array([0.2]), np.array([0.1]))
    a = cat.to_tangent_plane(center=A2744_CENTER, flip_g2=False)
    b = cat.to_tangent_plane(center=A2744_CENTER, flip_g2=True)
    assert np.isclose(a.g1[0], b.g1[0])
    assert np.isclose(a.g2[0], -b.g2[0])


def test_tangential_cross_pure_tangential():
    rng = np.random.default_rng(5)
    x = rng.uniform(-3, 3, 800)
    y = rng.uniform(-3, 3, 800)
    phi = np.arctan2(y, x)
    gt_true = 0.05
    g1 = -gt_true * np.cos(2 * phi)
    g2 = -gt_true * np.sin(2 * phi)
    gt, gx = tangential_cross_shear(g1, g2, x, y)
    assert np.allclose(gt, gt_true)
    assert np.sqrt(np.mean(gx**2)) < 1e-12


# --------------------------------------------------------------------------- #
# Metacal response
# --------------------------------------------------------------------------- #

def test_apply_response_forms():
    e1 = np.array([0.1, 0.2])
    e2 = np.array([-0.05, 0.0])

    g1, g2 = apply_response(e1, e2, 0.7)
    assert np.allclose(g1, e1 / 0.7) and np.allclose(g2, e2 / 0.7)

    g1, g2 = apply_response(e1, e2, [0.7, 0.8])
    assert np.allclose(g1, e1 / 0.7) and np.allclose(g2, e2 / 0.8)

    R = np.array([[0.7, 0.02], [0.01, 0.8]])
    g1, g2 = apply_response(e1, e2, R)
    Rinv = np.linalg.inv(R)
    assert np.allclose(g1, Rinv[0, 0] * e1 + Rinv[0, 1] * e2)
    assert np.allclose(g2, Rinv[1, 0] * e1 + Rinv[1, 1] * e2)

    Rper = np.stack([R, np.array([[0.6, 0.0], [0.0, 0.9]])])
    g1, g2 = apply_response(e1, e2, Rper)
    assert np.isclose(g1[1], e1[1] / 0.6) and np.isclose(g2[1], e2[1] / 0.9)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def test_clean_catalog():
    ra  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    dec = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    g1  = np.array([0.1, np.nan, 5.0, 0.2, 0.3])     # idx1 nan, idx2 too large
    g2  = np.array([0.0, 0.0, 0.0, 0.0, np.inf])     # idx4 inf
    cat = ShearCatalog.from_arrays(ra, dec, g1, g2)
    cleaned = clean_catalog(cat)
    assert cleaned.n == 2
    assert cleaned.meta["n_dropped"] == 3


def test_select_and_mask_core():
    rng = np.random.default_rng(7)
    n = 200
    cat = ShearCatalog.from_arrays(
        A2744_CENTER[0] + rng.uniform(-0.05, 0.05, n),
        A2744_CENTER[1] + rng.uniform(-0.05, 0.05, n),
        rng.normal(0, 0.2, n), rng.normal(0, 0.2, n))
    flat = cat.to_tangent_plane(center=A2744_CENTER)
    inner = flat.mask_core(0.5)
    assert inner.n < flat.n
    assert np.all(inner.radius() >= 0.5)


# --------------------------------------------------------------------------- #
# Column resolution / response detection (no astropy needed, mock table)
# --------------------------------------------------------------------------- #

class _MockColumns:
    def __init__(self, names):
        self.names = list(names)


class _MockTable:
    """Minimal stand-in for a FITS_rec: supports tbl[name], len(tbl), tbl.columns.names."""
    def __init__(self, data):
        self._data = {k: np.asarray(v) for k, v in data.items()}
        self.columns = _MockColumns(self._data.keys())

    def __len__(self):
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        return self._data[key]


def test_find_column():
    lower = {"alpha_j2000": "ALPHA_J2000", "delta_j2000": "DELTA_J2000"}
    assert _find_column(lower, _DEFAULT_COLUMNS["ra"]) == "ALPHA_J2000"
    assert _find_column(lower, _DEFAULT_COLUMNS["dec"]) == "DELTA_J2000"
    assert _find_column(lower, _DEFAULT_COLUMNS["weight"]) is None


def test_detect_response_mock():
    tbl = _MockTable({
        "R11": [0.7, 0.6],
        "R22": [0.8, 0.9],
        "R12": [0.02, 0.0],
        "R21": [0.01, 0.0],
    })
    lower = {nm.lower(): nm for nm in tbl.columns.names}
    R = _detect_response(tbl, lower)
    assert R.shape == (2, 2, 2)
    assert np.isclose(R[0, 0, 0], 0.7) and np.isclose(R[0, 1, 1], 0.8)
    assert np.isclose(R[0, 0, 1], 0.02) and np.isclose(R[0, 1, 0], 0.01)

    # no R columns -> None
    tbl2 = _MockTable({"g1": [0.1], "g2": [0.2]})
    lower2 = {nm.lower(): nm for nm in tbl2.columns.names}
    assert _detect_response(tbl2, lower2) is None


# --------------------------------------------------------------------------- #
# FITS reader (requires astropy)
# --------------------------------------------------------------------------- #

def _astropy_available():
    try:
        import astropy  # noqa: F401
        return True
    except ImportError:
        return False


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print(f"  SKIP {msg}")
        return True
    return True


def test_read_fits_catalog():
    if not _astropy_available():
        _skip("astropy not installed")
        return
    from astropy.io import fits
    from femmi.io import read_fits_catalog

    rng = np.random.default_rng(11)
    n = 50
    ra  = A2744_CENTER[0] + rng.uniform(-0.05, 0.05, n)
    dec = A2744_CENTER[1] + rng.uniform(-0.05, 0.05, n)
    e1  = rng.normal(0, 0.25, n)
    e2  = rng.normal(0, 0.25, n)
    R11 = np.full(n, 0.7)
    R22 = np.full(n, 0.8)
    zb  = rng.uniform(0.5, 3.0, n)
    # inject two bad rows that clean_catalog should drop
    e1[0] = np.nan
    e1[1] = 9.0

    cols = fits.ColDefs([
        fits.Column(name="ALPHA_J2000", format="D", array=ra),
        fits.Column(name="DELTA_J2000", format="D", array=dec),
        fits.Column(name="g1_noshear",  format="D", array=e1),
        fits.Column(name="g2_noshear",  format="D", array=e2),
        fits.Column(name="R11",         format="D", array=R11),
        fits.Column(name="R22",         format="D", array=R22),
        fits.Column(name="ZB",          format="D", array=zb),
    ])
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "mock_catalog.fits")
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)

    cat = read_fits_catalog(path)
    # auto-detected RA/Dec/shear, dropped the 2 bad rows
    assert cat.n == n - 2
    assert cat.meta["n_dropped"] == 2
    assert cat.z is not None
    # metacal response auto-applied: g1 = e1 / R11 on the surviving rows
    keep = np.arange(n) >= 2
    assert np.allclose(cat.g1, (e1 / R11)[keep])
    assert np.allclose(cat.g2, (e2 / R22)[keep])

    # column_map override + projection
    cat2 = read_fits_catalog(path, column_map={"z": "ZB"})
    flat = cat2.to_tangent_plane(center=A2744_CENTER)
    assert flat.n == cat2.n
    assert np.all(np.isfinite(flat.x)) and np.all(np.isfinite(flat.y))


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        test_projection_roundtrip,
        test_projection_center_and_north,
        test_shear_rotation_algebra,
        test_projection_rotation_small_and_invertible,
        test_flip_g2,
        test_tangential_cross_pure_tangential,
        test_apply_response_forms,
        test_clean_catalog,
        test_select_and_mask_core,
        test_find_column,
        test_detect_response_mock,
        test_read_fits_catalog,
    ]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("Failed:", failed)
    sys.exit(0 if not failed else 1)
