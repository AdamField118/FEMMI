"""
tests/test_aca.py
Hierarchical compression of the BEM operator (femmi.aca).

The properties that matter for a fast BEM are: the cluster tree and
admissibility are sane, ACA reproduces an admissible block to the requested
tolerance from only a few sampled rows/columns, and the assembled H-matrix
applies the true operator accurately while storing less than the dense one.

Run:
    python -m pytest tests/test_aca.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.aca import (build_cluster_tree, is_admissible, aca_partial,
                       build_hmatrix, single_layer_entry_fn)
from femmi.bem import circular_boundary_mesh


def test_cluster_tree_partitions_every_point_exactly_once():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 2))
    root = build_cluster_tree(pts, min_size=16)

    leaves = []
    stack = [root]
    while stack:
        c = stack.pop()
        if c.is_leaf:
            leaves.append(c.idx)
        else:
            stack += [c.left, c.right]
    allidx = np.sort(np.concatenate(leaves))
    assert np.array_equal(allidx, np.arange(len(pts)))
    assert all(len(l) <= 16 for l in leaves)


def test_admissibility_separates_far_from_near():
    """Two well-separated blobs are admissible; a cluster against itself is not."""
    a = np.random.default_rng(1).normal(size=(50, 2)) * 0.1
    b = a + np.array([10.0, 0.0])
    pts = np.vstack([a, b])
    ca = build_cluster_tree(pts[:50], min_size=99)
    cb = build_cluster_tree(pts[50:], min_size=99)
    assert is_admissible(ca, cb, eta=1.0)
    assert not is_admissible(ca, ca, eta=1.0)


def test_aca_recovers_a_smooth_low_rank_block():
    """On an asymptotically-smooth kernel between separated point sets, ACA must
    reach the requested accuracy at a rank far below the block size."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, size=(80, 2))
    y = rng.uniform(-1, 1, size=(90, 2)) + np.array([8.0, 0.0])
    A = np.log(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1))

    res = aca_partial(lambda i: A[i], lambda j: A[:, j], 80, 90, tol=1e-8)
    assert res is not None
    U, V = res
    assert U.shape[1] < 20                                  # genuinely low rank
    assert np.linalg.norm(U @ V.T - A) / np.linalg.norm(A) < 1e-6


def test_hmatrix_applies_the_operator_accurately():
    """H-matvec must match the dense operator built from the same kernel."""
    bnd = circular_boundary_mesh(radius=2.5, n_boundary=240)
    pts, entry = single_layer_entry_fn(bnd, n_quad=8)
    H = build_hmatrix(pts, entry, min_size=32, eta=1.0, tol=1e-6)

    n = H.shape[0]
    D = entry(np.arange(n), np.arange(n))
    x = np.random.default_rng(3).normal(size=n)
    assert np.linalg.norm(H.matvec(x) - D @ x) / np.linalg.norm(D @ x) < 1e-6
    assert H.compression <= 1.0


def test_hmatrix_reproduces_the_TRUE_single_layer_with_exact_near_field():
    """Regression for a real bug: build_hmatrix used the far-field evaluator for
    INADMISSIBLE blocks too, where the log singularity needs Duffy/log-Gauss. The
    result silently approximated a different operator (69% error in the coupled
    solve). Comparing H against the same naive kernel could never catch it -- this
    compares against the properly assembled V_h."""
    from femmi.bem import assemble_bem_matrices

    bnd = circular_boundary_mesh(radius=2.5, n_boundary=240)
    V_h, _, _ = assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8)
    pts, far = single_layer_entry_fn(bnd, n_quad=10)
    near = lambda r, c: V_h[np.ix_(np.atleast_1d(r), np.atleast_1d(c))]

    H = build_hmatrix(pts, far, min_size=24, eta=1.0, tol=1e-8, near_block=near)
    assert np.linalg.norm(H.to_dense() - V_h) / np.linalg.norm(V_h) < 1e-6

    # and the far-field-only version must NOT match, or the test proves nothing
    H_bad = build_hmatrix(pts, far, min_size=24, eta=1.0, tol=1e-8)
    assert np.linalg.norm(H_bad.to_dense() - V_h) / np.linalg.norm(V_h) > 1e-3


def test_compression_improves_with_problem_size():
    """The point of an H-matrix: the numerical rank of an admissible block stays
    bounded as the boundary is refined, so the stored fraction shrinks. At small
    N it correctly falls back to (near-)dense rather than compressing badly."""
    fracs = {}
    for nb in (240, 480):
        bnd = circular_boundary_mesh(radius=2.5, n_boundary=nb)
        pts, entry = single_layer_entry_fn(bnd, n_quad=8)
        H = build_hmatrix(pts, entry, min_size=32, eta=1.0, tol=1e-6)
        fracs[nb] = H.compression
        ranks = [b[3][0].shape[1] for b in H.blocks if b[2] == "lr"]
        if ranks:
            assert max(ranks) < 20                          # rank stays small
    assert fracs[480] < fracs[240]


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
