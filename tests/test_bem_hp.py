"""
tests/test_bem_hp.py
Hypersingular operator and near-field-only ACA assembly.

W is built through the Nedelec / Maue identity, which turns the hypersingular
form into the SINGLE-LAYER form applied to arc-length derivatives -- so the tests
target the properties that identity guarantees (symmetry, constants in the null
space) rather than re-deriving the kernel.

The near-field assembler is what makes ACA self-contained: before it, the only
way to get correct inadmissible blocks was to look them up in an exactly
assembled dense V_h, which is the very thing ACA exists to avoid.

Run:
    python -m pytest tests/test_bem_hp.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.bem import circular_boundary_mesh, assemble_single_layer
from femmi.bem_hp import (assemble_hypersingular_hp, assemble_single_layer_hp,
                          assemble_boundary_mass_hp, boundary_basis_deriv,
                          boundary_basis)
from femmi.aca import near_field_entry_fn, single_layer_entry_fn, build_hmatrix


def test_derivative_basis_is_the_derivative_of_the_basis():
    """Checked against a central difference, plus the partition-of-unity identity
    that the derivatives must sum to zero."""
    t = np.linspace(0.05, 0.95, 11)
    for deg in (3, 5):
        d = boundary_basis_deriv(deg, t)
        h = 1e-6
        fd = (boundary_basis(deg, t + h) - boundary_basis(deg, t - h)) / (2 * h)
        assert np.abs(d - fd).max() < 1e-6
        assert np.abs(d.sum(axis=1)).max() < 1e-12


def test_hypersingular_is_symmetric_and_annihilates_constants():
    """Both follow from the Nedelec form: it pairs du/ds with dv/ds, and the
    derivative of a constant is zero. A W that failed either would be wrong."""
    bnd = circular_boundary_mesh(radius=2.5, n_boundary=96)
    W = assemble_hypersingular_hp(bnd, 3, n_quad=25)
    assert np.abs(W - W.T).max() < 1e-12 * max(np.abs(W).max(), 1e-30)
    one = np.ones(bnd.n_boundary_dofs)
    assert np.linalg.norm(W @ one) / np.linalg.norm(W) < 1e-10


def test_calderon_improves_conditioning_by_a_constant_factor():
    """Honest scope: pairing V and W on the SAME mesh gives a constant-factor
    improvement (~2.3x), NOT the mesh-independence true Calderon preconditioning
    provides -- that needs dual (Buffa-Christiansen) bases. This pins the
    behaviour so the claim in MATH.md stays accurate."""
    from scipy.spatial.distance import pdist

    ratios = []
    for nb in (48, 96):
        bnd = circular_boundary_mesh(radius=2.5, n_boundary=nb)
        N = bnd.n_boundary_dofs
        V = assemble_single_layer_hp(bnd, 3, n_quad=25)
        W = assemble_hypersingular_hp(bnd, 3, n_quad=25)
        Mb = assemble_boundary_mass_hp(bnd, 3)
        w = Mb @ np.ones(N)
        Veff = V - (np.log(float(pdist(bnd.nodes).max())) / (2 * np.pi)) * np.outer(w, w)
        Wst = W + np.abs(W).max() * np.outer(w, w) / float(w @ w)
        P = Veff @ np.linalg.solve(Mb, Wst)
        ratios.append(np.linalg.cond(Veff) / np.linalg.cond(P))
    assert all(r > 1.5 for r in ratios)              # it does help
    assert max(ratios) < 10.0                        # but only by a constant


def test_chebyshev_clustering_is_a_valid_basis_and_clusters_at_the_ends():
    """Gauss-Lobatto nodes must still form a Lagrange basis (partition of unity,
    cardinal at its own nodes) and must be denser near the element ends than
    equispaced -- that is the point of using them at degree 5."""
    from femmi.bem_hp import node_positions

    uni = node_positions(5, "uniform")
    che = node_positions(5, "chebyshev")
    assert np.allclose(che[[0, -1]], [0.0, 1.0])
    assert che[1] < uni[1] and che[-2] > uni[-2]          # clustered at the ends

    t = np.linspace(0.0, 1.0, 21)
    for c in ("uniform", "chebyshev"):
        phi = boundary_basis(5, t, c)
        assert np.abs(phi.sum(axis=1) - 1.0).max() < 1e-12
        at_nodes = boundary_basis(5, node_positions(5, c), c)
        assert np.abs(at_nodes - np.eye(6)).max() < 1e-10


def test_curved_elements_put_nodes_exactly_on_the_circle():
    """A chord-based boundary is an inscribed polygon: its nodes sit inside the
    circle by O(h^2) and its perimeter is short. Curved elements remove both --
    on a method whose selling point is the exactness of the exterior condition."""
    from femmi.bem_hp import build_circular_boundary_mesh, build_boundary_mesh

    R = 2.5
    for ne in (8, 16, 32):
        corners = np.array([[R * np.cos(a), R * np.sin(a)]
                            for a in np.linspace(0, 2 * np.pi, ne, endpoint=False)])
        straight = build_boundary_mesh(corners, 5)
        curved = build_circular_boundary_mesh(ne, 5, radius=R)

        r_s = np.hypot(straight.nodes[:, 0], straight.nodes[:, 1])
        r_c = np.hypot(curved.nodes[:, 0], curved.nodes[:, 1])
        assert np.abs(r_c - R).max() < 1e-12
        assert np.abs(r_s - R).max() > np.abs(r_c - R).max()
        assert abs(curved.element_lengths.sum() - 2 * np.pi * R) < 1e-10
        assert straight.element_lengths.sum() < 2 * np.pi * R


def test_near_field_assembler_reproduces_the_assembled_single_layer():
    """The near-field evaluator must match the tuned assembly on near/self blocks,
    singular treatment included -- that is what lets ACA drop the dense V_h."""
    bnd = circular_boundary_mesh(radius=2.5, n_boundary=96)
    V_h = assemble_single_layer(bnd, n_quad=25)
    _, near = near_field_entry_fn(bnd, degree=3, n_quad=25)
    idx = np.arange(20)
    got = near(idx, idx)
    ref = V_h[np.ix_(idx, idx)]
    assert np.abs(got - ref).max() < 1e-12 * max(np.abs(ref).max(), 1e-30)


def test_hmatrix_needs_no_dense_matrix_and_still_matches():
    """End to end: build H from the far-field evaluator plus the near-field
    assembler only, and compare against the true V_h."""
    bnd = circular_boundary_mesh(radius=2.5, n_boundary=120)
    V_h = assemble_single_layer(bnd, n_quad=25)
    pts, far = single_layer_entry_fn(bnd, n_quad=25)
    _, near = near_field_entry_fn(bnd, degree=3, n_quad=25)
    H = build_hmatrix(pts, far, min_size=24, eta=1.0, tol=1e-8, near_block=near)
    assert np.linalg.norm(H.to_dense() - V_h) / np.linalg.norm(V_h) < 1e-7


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
