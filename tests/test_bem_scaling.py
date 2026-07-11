"""
tests/test_bem_scaling.py
Characterises the BEM coupling scaling issue found in the forward-accuracy
investigation (see examples/bem_scaling_diagnostic.py):

  - the lensing forward map is scale-invariant, and Dirichlet respects it;
  - the DEFAULT BEM coupling does not -- its forward error grows with the
    absolute coordinate scale (this test documents the bug so a future fix can
    be validated against it);
  - ||C_dense|| falls as 1/L with domain size (the mechanism);
  - build_operators(..., couple_scale='auto') restores scale-invariance.

Run:
    python -m pytest tests/test_bem_scaling.py -v
    python tests/test_bem_scaling.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators, dirichlet_from_operators
from femmi.catalog   import analytic_gaussian_shear


def _fwd_err(ops, nx_scale):
    L, sig = 1.5 * nx_scale, 0.4 * nx_scale
    nodes = np.array(ops.mesh.nodes)
    sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 0.5 * L
    kt, g1a, g2a = analytic_gaussian_shear(nodes, sigma=sig, amp=1.0)
    g1, g2 = (np.asarray(a) for a in ops.forward(kt))
    return float(np.linalg.norm(np.hypot(g1[sel] - g1a[sel], g2[sel] - g2a[sel])) /
                 np.linalg.norm(np.hypot(g1a[sel], g2a[sel])))


def _ops(scale, **kw):
    L = 1.5 * scale
    return build_operators(16, 16, -L, L, -L, L, verbose=False, **kw)


def test_dirichlet_is_scale_invariant():
    """The reference: Dirichlet forward error is independent of coordinate scale."""
    e1 = _fwd_err(dirichlet_from_operators(_ops(1.0)), 1.0)
    e2 = _fwd_err(dirichlet_from_operators(_ops(25.0)), 25.0)
    assert abs(e1 - e2) < 0.02 * e1, f"Dirichlet not scale-invariant: {e1:.4f} vs {e2:.4f}"


def test_default_bem_is_scale_dependent():
    """Documents the bug: default BEM forward error grows sharply with scale."""
    e1 = _fwd_err(_ops(1.0), 1.0)
    e2 = _fwd_err(_ops(25.0), 25.0)
    assert e2 > 3.0 * e1, f"expected BEM to degrade with scale: {e1:.4f} -> {e2:.4f}"


def test_coupling_norm_scales_as_inverse_L():
    """||C_dense|| ~ 1/L: the coupling weakens with domain size (the mechanism)."""
    c1 = np.linalg.norm(_ops(1.0).C_dense)
    c2 = np.linalg.norm(_ops(25.0).C_dense)
    ratio = (c1 / c2)                       # expect ~ 25
    assert 15.0 < ratio < 40.0, f"||C|| ratio {ratio:.1f} not ~ 1/L (expected ~25)"


def test_circular_boundary_mesh_is_valid():
    """Boundary-only circular mesh assembles consistent BEM matrices (no interior)."""
    from femmi.bem import circular_boundary_mesh, assemble_bem_matrices
    R = 2.0
    bnd = circular_boundary_mesh(radius=R, n_boundary=120)
    assert bnd.n_boundary_dofs % 3 == 0
    V, K, M = assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8)
    one = np.ones(bnd.n_boundary_dofs)
    # boundary mass integrates to the perimeter; Calderon identity (0.5M+K)1 ~ 0
    assert abs(float((M @ one).sum()) - 2 * np.pi * R) < 0.05
    assert np.linalg.norm((0.5 * M + K) @ one) / np.linalg.norm(M @ one) < 1e-2


def test_couple_scale_auto_restores_invariance():
    """The experimental fix makes the BEM forward error scale-invariant."""
    e1 = _fwd_err(_ops(1.0, couple_scale='auto'), 1.0)
    e2 = _fwd_err(_ops(25.0, couple_scale='auto'), 25.0)
    assert abs(e1 - e2) < 0.1 * e1, f"fix not scale-invariant: {e1:.4f} vs {e2:.4f}"
    # and it is at least as accurate as the (scale-dependent) default at unit scale
    assert e1 <= _fwd_err(_ops(1.0), 1.0) + 1e-6


if __name__ == "__main__":
    tests = [
        test_dirichlet_is_scale_invariant,
        test_default_bem_is_scale_dependent,
        test_coupling_norm_scales_as_inverse_L,
        test_circular_boundary_mesh_is_valid,
        test_couple_scale_auto_restores_invariance,
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
