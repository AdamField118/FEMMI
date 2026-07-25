"""
tests/test_c1_assembly.py
FEM assembly and Poisson solves on the C^1 spaces (femmi.c1_assembly).

tests/test_elements.py shows the elements can REPRESENT a second derivative well.
This checks the operators built on them behave: the quadrature is strong enough
for quintics, the stiffness matrix has the right null space, and the shear coming
out of an actual SOLVE converges at the element's theoretical rate.

Run:
    python -m pytest tests/test_c1_assembly.py -v
"""

import sys, os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.elements import C1Space, structured_triangulation
from femmi.c1_assembly import (_quad, assemble_c1, solve_poisson_c1,
                               c1_shear_at_vertices, boundary_dofs,
                               solved_shear_convergence)
from femmi.experiments import manufactured_potential_derivs as F


def test_quadrature_is_exact_through_degree_ten():
    """Argyris mass integrands are degree 10; femmi.assembly tops out at degree 7,
    which is why this module generates its own rule. If the rule were too weak the
    measured convergence order would be quietly capped."""
    pts, wts = _quad(7)
    assert abs(wts.sum() - 1.0) < 1e-12
    for a, b in [(0, 0), (2, 3), (5, 5), (6, 4), (4, 6)]:
        exact = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
        num = 0.5 * np.sum(wts * pts[:, 0]**a * pts[:, 1]**b)   # 0.5 = ref area
        assert abs(num - exact) <= 1e-12 * max(exact, 1e-12)


def test_stiffness_annihilates_constants_and_mass_is_positive():
    """K must kill the constant function (pure Neumann null space) and M must be
    SPD -- the basic sanity check that the C^1 assembly is not scrambled."""
    verts, tris = structured_triangulation(4, 2.5)
    S = C1Space(verts, tris, kind="argyris")
    K, M = assemble_c1(S, quad_order=7)

    # the constant 1 has value DOFs 1 and every derivative DOF 0
    one = np.zeros(S.n_dofs)
    one[0:S.n_vertices * 6:6] = 1.0
    assert np.linalg.norm(K @ one) < 1e-8 * max(abs(K).max(), 1.0)
    assert one @ (M @ one) > 0


def test_boundary_dofs_cover_the_square_ring():
    verts, tris = structured_triangulation(4, 2.5)
    S = C1Space(verts, tris, kind="argyris")
    bd = boundary_dofs(S, 2.5)
    on = (np.abs(np.abs(verts[:, 0]) - 2.5) < 1e-9) | \
         (np.abs(np.abs(verts[:, 1]) - 2.5) < 1e-9)
    assert len(bd) >= 6 * on.sum()               # all vertex DOFs of the ring
    assert len(bd) < S.n_dofs                    # and it is not everything


def test_solved_shear_converges_at_fourth_order_for_argyris():
    """The payoff, from a SOLVE rather than interpolation: psi is obtained by
    solving grad^2 psi = 2 kappa and the shear read back off it still converges
    at O(h^4)."""
    d = solved_shear_convergence(kind="argyris", nxs=(6, 8, 12, 16))
    assert d["local"][-1] > 3.3
    assert d["err"][-1] < d["err"][0]


def test_vertex_shear_is_a_pure_dof_selection_for_argyris():
    """Argyris stores {u_xx, u_xy, u_yy} at vertices, so shear extraction is a
    selection -- single-valued, and no averaging over adjacent elements. Checked
    against the exact manufactured shear on the interpolant."""
    verts, tris = structured_triangulation(12, 2.5)
    S = C1Space(verts, tris, kind="argyris")
    u = S.interpolate(lambda p, dx=0, dy=0: F(p, dx, dy))
    g1, g2 = c1_shear_at_vertices(S, u)

    e1 = np.array([0.5 * (F(p, 2, 0) - F(p, 0, 2)) for p in verts])
    e2 = np.array([F(p, 1, 1) for p in verts])
    # interpolation reproduces the vertex Hessian DOFs exactly
    assert np.allclose(g1, e1, atol=1e-10)
    assert np.allclose(g2, e2, atol=1e-10)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
