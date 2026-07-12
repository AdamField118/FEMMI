"""
tests/test_bc_ablation.py
Boundary-condition machinery for the ablation study:

  - dirichlet_from_operators actually enforces psi = 0 on the boundary,
  - it reuses the same K/M/S1/S2 and plugs into the same MAP inversion,
  - the Dirichlet forward reproduces the analytic shear well in the interior,
  - both reconstructions are finite, and with the Steinbach coupling the BEM
    far-field is at least as accurate as a Dirichlet truncation on a central
    aperture when the boundary sits near the mass (the project's thesis).

These pin the mechanics and the corrected far-field behaviour.

Run:
    python -m pytest tests/test_bc_ablation.py -v
    python tests/test_bc_ablation.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators, dirichlet_from_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor
from femmi.catalog   import analytic_gaussian_shear


def _recon(ops, g1, g2, lam=1e-4):
    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(fwd, maxiter=250, wiener_length=0.5, noise_std=None)
    k, _ = rec.reconstruct(g1, g2, verbose=False)
    return k


def test_dirichlet_enforces_zero_boundary():
    ops = build_operators(14, 14, -2.0, 2.0, -2.0, 2.0, verbose=False)
    dr  = dirichlet_from_operators(ops)
    nodes = np.array(dr.mesh.nodes)
    kt, _, _ = analytic_gaussian_shear(nodes, sigma=0.5)
    psi = dr.psi_from_kappa(kt)
    bnd = np.array(dr.mesh.boundary)
    assert np.max(np.abs(np.asarray(psi)[bnd])) < 1e-9, "psi not zero on boundary"
    # BEM shares K/M/S1/S2 with its Dirichlet counterpart
    assert dr.K is ops.K and dr.M is ops.M and dr.S1 is ops.S1


def test_dirichlet_forward_accurate_interior():
    ops = build_operators(20, 20, -2.5, 2.5, -2.5, 2.5, verbose=False)
    dr  = dirichlet_from_operators(ops)
    nodes = np.array(ops.mesh.nodes)
    kt, g1a, g2a = analytic_gaussian_shear(nodes, sigma=0.5)
    sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 0.8
    g1, g2 = (np.asarray(a) for a in dr.forward(kt))
    err = np.linalg.norm(np.hypot(g1[sel] - g1a[sel], g2[sel] - g2a[sel])) / \
        np.linalg.norm(np.hypot(g1a[sel], g2a[sel]))
    assert err < 0.15, f"Dirichlet interior forward error {err:.3f} too large"


def test_bem_at_least_as_accurate_as_dirichlet():
    # boundary at L=2.0 sits near a sigma=0.5 lens, so the far-field matters
    ops = build_operators(16, 16, -2.0, 2.0, -2.0, 2.0, verbose=False)
    dr  = dirichlet_from_operators(ops)
    nodes = np.array(ops.mesh.nodes)
    kt, g1, g2 = analytic_gaussian_shear(nodes, sigma=0.5)
    sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 0.8

    kb = _recon(ops, g1, g2)
    kd = _recon(dr, g1, g2)
    assert np.all(np.isfinite(kb)) and np.all(np.isfinite(kd))

    def l2(k):
        return np.linalg.norm(k[sel] - kt[sel]) / np.linalg.norm(kt[sel])
    # with the Steinbach coupling the BEM far-field is no worse than Dirichlet
    # on the central aperture (a small tolerance guards against optimiser jitter)
    assert l2(kb) <= l2(kd) + 0.02, \
        f"BEM L2={l2(kb):.3f} should be <= Dirichlet L2={l2(kd):.3f} near the mass"


if __name__ == "__main__":
    tests = [
        test_dirichlet_enforces_zero_boundary,
        test_dirichlet_forward_accurate_interior,
        test_bem_at_least_as_accurate_as_dirichlet,
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
