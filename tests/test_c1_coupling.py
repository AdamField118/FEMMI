"""
tests/test_c1_coupling.py
FEM-BEM coupling for the C^1 spaces (femmi.bem_hp, femmi.c1_coupling).

The chain that has to hold:

  * the degree-generalised BEM assembly must reproduce femmi.bem's tuned P3
    assembly exactly at degree 3 -- that keeps the trusted path as the reference
    instead of re-deriving it;
  * the boundary loop must be the actual boundary, counter-clockwise;
  * the trace operator must genuinely evaluate the Argyris field at the boundary
    nodes (a selection matrix would silently drop the derivative DOFs);
  * and the payoff: on a field whose potential does NOT vanish at the boundary,
    the coupled solve must beat a Dirichlet pin by a wide margin.

Run:
    python -m pytest tests/test_c1_coupling.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.bem import (circular_boundary_mesh, assemble_bem_matrices,
                       _p3_boundary_basis)
from femmi.bem_hp import (boundary_basis, build_boundary_mesh, assemble_bem_hp,
                          assemble_boundary_mass_hp)
from femmi.elements import C1Space, structured_triangulation
from femmi.c1_coupling import boundary_loop, trace_operator, C1CoupledOperators
from femmi.c1_assembly import solve_poisson_c1, _quad
from femmi.catalog import analytic_gaussian_shear


def test_generalised_basis_matches_p3_at_degree_three():
    t = np.linspace(0.0, 1.0, 17)
    assert np.allclose(boundary_basis(3, t), _p3_boundary_basis(t), atol=1e-14)


def test_generalised_assembly_reproduces_the_tuned_p3_operators():
    """Degree 3 must reproduce femmi.bem bit-for-bit, singular treatment included.
    If this drifts, the degree-5 path cannot be trusted either."""
    bnd = circular_boundary_mesh(radius=2.5, n_boundary=48)
    V0, K0, M0 = assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8)
    V1, K1, M1 = assemble_bem_hp(bnd, 3, n_quad_sl=25, n_quad_dl=8)
    for a, b in ((V0, V1), (K0, K1), (M0, M1)):
        assert np.abs(a - b).max() <= 1e-12 * max(np.abs(a).max(), 1e-30)


def test_boundary_mass_sums_to_the_perimeter_at_any_degree():
    corners = np.array([[2.5 * np.cos(t), 2.5 * np.sin(t)]
                        for t in np.linspace(0, 2 * np.pi, 9)[:-1]])
    for deg in (3, 5):
        bnd = build_boundary_mesh(corners, deg)
        M = assemble_boundary_mass_hp(bnd, deg)
        assert abs(M.sum() - bnd.element_lengths.sum()) < 1e-10
        assert bnd.n_boundary_dofs == bnd.n_elements * deg
        # consecutive elements must share an endpoint or the trace is discontinuous
        for e in range(bnd.n_elements):
            assert bnd.elements[e][-1] == bnd.elements[(e + 1) % bnd.n_elements][0]


def test_boundary_loop_is_the_boundary_and_counter_clockwise():
    verts, tris = structured_triangulation(4, 2.5)
    S = C1Space(verts, tris, kind="argyris")
    loop = boundary_loop(S)
    assert len(loop) == 4 * 4                       # perimeter vertices of a 4x4 grid
    p = verts[loop]
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    assert area2 > 0                                # CCW -> outward normals
    assert abs(area2 / 2 - 25.0) < 1e-9             # encloses the square


def test_trace_operator_evaluates_the_field_not_just_selects_nodes():
    """P must reproduce the field at the boundary nodes for a function whose
    Argyris DOFs include nonzero derivatives -- a selection matrix would fail."""
    verts, tris = structured_triangulation(4, 2.5)
    S = C1Space(verts, tris, kind="argyris")
    ops = C1CoupledOperators(S, degree=5)

    def g(p, dx=0, dy=0):
        x, y = p[0], p[1]
        if (dx, dy) == (0, 0): return x * x - 0.5 * y * y
        if (dx, dy) == (1, 0): return 2 * x
        if (dx, dy) == (0, 1): return -y
        if (dx, dy) == (2, 0): return 2.0
        if (dx, dy) == (0, 2): return -1.0
        return 0.0

    u = S.interpolate(g)
    got = ops.P @ u
    want = np.array([g(pt) for pt in ops.bnd.nodes])
    assert np.abs(got - want).max() < 1e-10


def test_coupled_solve_beats_dirichlet_on_a_non_compact_field():
    """The whole reason the coupling exists. A Gaussian convergence has
    psi ~ log r far away, so psi does not vanish on the boundary and a Dirichlet
    pin is simply the wrong condition. The BEM far-field must do markedly
    better -- measured 5-9x on the shear.
    """
    hw, sigma, amp = 2.5, 0.6, 1.0
    qp, qw = _quad(7)
    kappa = lambda P: amp * np.exp(-(P[:, 0]**2 + P[:, 1]**2) / (2 * sigma**2))

    verts, tris = structured_triangulation(6, hw)
    S = C1Space(verts, tris, kind="argyris")
    psi_d = solve_poisson_c1(S, kappa, hw)
    psi_c = C1CoupledOperators(S, degree=5).psi_from_kappa(kappa)

    def err(psi):
        num = den = 0.0
        for t in range(len(tris)):
            v = verts[tris[t]]
            area = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0]]))) / 2
            pts = v[0] + qp[:, 0:1] * (v[1] - v[0]) + qp[:, 1:2] * (v[2] - v[0])
            keep = np.hypot(pts[:, 0], pts[:, 1]) < 0.8 * hw
            if not keep.any():
                continue
            a1, a2 = S.eval_shear(psi, t, pts)
            _, e1, e2 = analytic_gaussian_shear(pts, sigma=sigma, amp=amp)
            num += area * np.sum((qw * keep) * ((a1 - e1)**2 + (a2 - e2)**2))
            den += area * np.sum((qw * keep) * (e1**2 + e2**2))
        return np.sqrt(num / max(den, 1e-300))

    assert err(psi_c) < 0.35 * err(psi_d)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
