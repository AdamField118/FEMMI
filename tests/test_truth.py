"""
tests/test_truth.py
The INDEPENDENT ground-truth generators (femmi.truth).

These are the fields the paper's headline comparison is measured against, so the
things worth pinning down are (a) that they agree with an independently derived
analytic result, and (b) that they use the same shear sign convention as the rest
of FEMMI -- a sign slip here would silently invert the whole comparison.

Run:
    python -m pytest tests/test_truth.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.truth import aperiodic_shear_from_kappa, galsim_nfw_truth
from femmi.catalog import analytic_gaussian_shear

galsim = pytest.importorskip("galsim", reason="GalSim needed for the NFW truth")


def test_aperiodic_kernel_matches_the_analytic_gaussian():
    """The real-space convolution kernel must reproduce the closed-form shear of a
    Gaussian convergence -- this validates its normalisation AND its sign."""
    n, hw = 192, 5.0
    pix = 2 * hw / n
    c = (np.arange(n) + 0.5) * pix - hw
    X, Y = np.meshgrid(c, c)
    sigma, amp = 0.6, 1.0
    kappa = amp * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    g1, g2 = aperiodic_shear_from_kappa(kappa, pix=pix)
    _, a1, a2 = analytic_gaussian_shear(np.stack([X.ravel(), Y.ravel()], 1),
                                        sigma=sigma, amp=amp)
    a1 = a1.reshape(n, n); a2 = a2.reshape(n, n)

    r = np.hypot(X, Y)
    m = (r > 1.5 * sigma) & (r < 0.6 * hw)      # off the core, inside the padding
    num = np.sqrt(np.mean((g1 - a1)[m]**2 + (g2 - a2)[m]**2))
    den = np.sqrt(np.mean(a1[m]**2 + a2[m]**2))
    assert num / den < 5e-3


def test_nfw_truth_uses_femmi_shear_convention():
    """For a mass concentration the tangential shear must be POSITIVE, matching
    catalog.analytic_gaussian_shear. If GalSim's convention ever flips relative to
    FEMMI's, this catches it before it silently inverts the KS comparison."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [1.5, 1.5], [-2.0, 0.7]])
    kappa, g1, g2 = galsim_nfw_truth(pts)
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    gamma_t = -(g1 * np.cos(2 * phi) + g2 * np.sin(2 * phi))
    assert np.all(kappa > 0)
    assert np.all(gamma_t > 0)


def test_nfw_truth_is_finite_at_the_halo_centre():
    """A mesh node sitting exactly on the halo centre must not produce NaN/inf
    (GalSim reads uninitialised memory at zero separation)."""
    pts = np.array([[0.0, 0.0], [0.5, 0.0]])
    kappa, g1, g2 = galsim_nfw_truth(pts)
    assert np.all(np.isfinite(kappa)) and np.all(np.isfinite(g1)) and np.all(np.isfinite(g2))


def test_halos_superpose():
    """Two halos give the sum of the two single-halo fields (linear weak-lensing)."""
    pts = np.array([[1.0, 0.3], [-0.8, 1.2]])
    a = (2e14, 4.0, (-0.5, 0.0))
    b = (1e14, 3.5, (0.6, 0.4))
    ka, a1, a2 = galsim_nfw_truth(pts, halos=(a,))
    kb, b1, b2 = galsim_nfw_truth(pts, halos=(b,))
    kab, ab1, ab2 = galsim_nfw_truth(pts, halos=(a, b))
    assert np.allclose(kab, ka + kb)
    assert np.allclose(ab1, a1 + b1) and np.allclose(ab2, a2 + b2)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
