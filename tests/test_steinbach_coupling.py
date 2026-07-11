"""
tests/test_steinbach_coupling.py
Regression tests for the corrected FEM-BEM coupling (coupling='steinbach'):

    A_coupled = K - P^T M_b V_sigma^{-1} (0.5 M_b - K_h) P,
    V_sigma = V_h - (ln sigma / 2pi) w w^T,  w = M_b 1,  sigma = diam(Gamma).

Each test pins one property established during the investigation:
  1. sigma-scaling moves ONLY the n=0 (log-capacity) mode; every n>=1 eigenvalue
     of the exterior DtN is frozen to machine precision.
  2. the forward error is scale-invariant (was the original bug).
  3. it beats a Dirichlet truncation when the boundary is near the mass (the
     method's central claim; the legacy coupling failed this).
  4. it is translation-invariant (sigma depends on diam, not absolute position).
  5. the full MAP reconstruction pipeline runs with the new coupling.

Run:
    python -m pytest tests/test_steinbach_coupling.py -v
    python tests/test_steinbach_coupling.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators, dirichlet_from_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor
from femmi.catalog   import analytic_gaussian_shear
from femmi.bem       import circular_boundary_mesh, assemble_bem_matrices


def _fwd_err(ops, kt, g1a, g2a, sel):
    g1, g2 = (np.asarray(a) for a in ops.forward(kt))
    return float(np.linalg.norm(np.hypot(g1[sel] - g1a[sel], g2[sel] - g2a[sel])) /
                 np.linalg.norm(np.hypot(g1a[sel], g2a[sel])))


def test_sigma_scaling_moves_only_constant_mode():
    """The cleanest regression: only lambda_0 changes; n>=1 frozen to ~machine."""
    from scipy.spatial.distance import pdist
    R = 2.0
    bnd = circular_boundary_mesh(radius=R, n_boundary=600)
    V, K, M = assemble_bem_matrices(bnd, 25, 8)
    th = np.arctan2(bnd.nodes[:, 1], bnd.nodes[:, 0])
    w  = M @ np.ones(V.shape[0])
    sigma = float(pdist(bnd.nodes).max())
    Vs = V - (np.log(sigma) / (2 * np.pi)) * np.outer(w, w)

    def lam(Vm, n):
        phi = np.ones_like(th) if n == 0 else np.cos(n * th)
        t = np.linalg.solve(Vm, 0.5 * M - K) @ phi
        return float(phi @ M @ t) / float(phi @ M @ phi)

    for n in range(1, 8):
        assert abs(lam(V, n) - lam(Vs, n)) < 1e-6, f"n={n} eigenvalue moved"
        assert abs(lam(Vs, n) - (-n / R)) < 5e-3, f"n={n} not ~ -n/R"
    assert abs(lam(V, 0) - lam(Vs, 0)) > 0.1, "n=0 mode did not move"


def test_steinbach_scale_invariant():
    """Forward error is independent of the absolute coordinate scale."""
    def err(s):
        L = 1.5 * s
        ops = build_operators(20, 20, -L, L, -L, L, verbose=False, coupling='steinbach')
        nd = np.array(ops.mesh.nodes); sel = np.hypot(nd[:, 0], nd[:, 1]) < 0.5 * L
        kt, g1, g2 = analytic_gaussian_shear(nd, sigma=0.4 * s, amp=1.0)
        return _fwd_err(ops, kt, g1, g2, sel)
    e1, e2 = err(1.0), err(50.0)
    assert abs(e1 - e2) < 0.05 * e1, f"not scale-invariant: {e1:.4f} vs {e2:.4f}"


def test_steinbach_beats_dirichlet_when_boundary_matters():
    """Boundary close to the mass: steinbach BEM << Dirichlet truncation."""
    L = 1.0                                      # kappa(boundary) ~ 0.14, BC matters
    ops = build_operators(24, 24, -L, L, -L, L, verbose=False, coupling='steinbach')
    dr  = dirichlet_from_operators(build_operators(24, 24, -L, L, -L, L, verbose=False))
    nd = np.array(ops.mesh.nodes); sel = np.hypot(nd[:, 0], nd[:, 1]) < 0.6
    kt, g1, g2 = analytic_gaussian_shear(nd, sigma=0.5, amp=1.0)
    e_bem = _fwd_err(ops, kt, g1, g2, sel)
    e_dir = _fwd_err(dr, kt, g1, g2, sel)
    assert e_bem < 0.5 * e_dir, f"BEM {e_bem:.3f} not << Dirichlet {e_dir:.3f}"


def test_steinbach_translation_invariant():
    """sigma = diam(Gamma) is translation-invariant, so the shear must be too."""
    L, T = 1.5, 1000.0
    o0 = build_operators(20, 20, -L, L, -L, L, verbose=False, coupling='steinbach')
    oT = build_operators(20, 20, T - L, T + L, T - L, T + L, verbose=False, coupling='steinbach')
    n0 = np.array(o0.mesh.nodes); nT = np.array(oT.mesh.nodes)
    sel = np.hypot(n0[:, 0], n0[:, 1]) < 0.5 * L
    k0, _, _ = analytic_gaussian_shear(n0, sigma=0.5, center=(0., 0.))
    kT, _, _ = analytic_gaussian_shear(nT, sigma=0.5, center=(T, T))
    g0 = np.hypot(*[np.asarray(a) for a in o0.forward(k0)])
    gT = np.hypot(*[np.asarray(a) for a in oT.forward(kT)])
    assert np.allclose(g0[sel], gT[sel], atol=1e-8), "shear changed under translation"


def test_steinbach_reconstruction_pipeline():
    """The full MAP pipeline runs and recovers a lens with the new coupling."""
    ops = build_operators(16, 16, -2.5, 2.5, -2.5, 2.5, verbose=False, coupling='steinbach')
    nd = np.array(ops.mesh.nodes)
    kt = np.exp(-(nd[:, 0]**2 + nd[:, 1]**2) / (2 * 0.5**2))
    g1, g2 = (np.asarray(a) for a in ops.forward(kt))
    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    rec = MAPReconstructor(fwd, maxiter=300, wiener_length=0.5, noise_std=None)
    krec, _ = rec.reconstruct(g1, g2, verbose=False)
    inner = np.hypot(nd[:, 0], nd[:, 1]) < 1.5
    corr = np.corrcoef(krec[inner], kt[inner])[0, 1]
    assert np.all(np.isfinite(krec)) and corr > 0.95, f"reconstruction corr={corr:.3f}"


if __name__ == "__main__":
    tests = [
        test_sigma_scaling_moves_only_constant_mode,
        test_steinbach_scale_invariant,
        test_steinbach_beats_dirichlet_when_boundary_matters,
        test_steinbach_translation_invariant,
        test_steinbach_reconstruction_pipeline,
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
