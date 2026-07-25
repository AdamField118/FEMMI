"""
tests/test_iterative.py
Matrix-free coupled FEM-BEM solves (femmi.iterative).

The whole point is to solve A_coupled x = b WITHOUT an assembled A_coupled, so
that femmi.aca can supply the BEM block. The things that must hold:

  * the operator reproduces the assembled matrix, and its TRANSPOSE reproduces
    the assembled transpose (the adjoint drives the MAP gradient, and the gauge
    row makes the transpose easy to get subtly wrong);
  * GMRES reaches the same solution as the direct LU factorisation;
  * the iteration count does not blow up with mesh size;
  * the BEM block can be supplied through a `v_solve` callable, which is what an
    H-matrix plugs into.

Run:
    python -m pytest tests/test_iterative.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.experiments import square_ops
from femmi.iterative import CoupledOperator, solve_coupled, fem_block_preconditioner


def _rhs(ops, seed=0):
    r = np.random.default_rng(seed).normal(size=ops.n_nodes)
    r[int(ops.bnd_mesh.node_indices[0])] = 0.0        # gauge node
    return r


def test_operator_matches_the_assembled_matrix():
    ops = square_ops(10, 2.5)
    A = CoupledOperator(ops)
    x = np.random.default_rng(0).normal(size=ops.n_nodes)
    ref = ops.A_coupled @ x
    assert np.linalg.norm(A @ x - ref) / np.linalg.norm(ref) < 1e-12


def test_transpose_matches_the_assembled_transpose():
    """The gauge fix zeroes ROW g but leaves COLUMN g populated, so (A^T x)_g must
    ADD the gauge term to the column contribution rather than overwrite it.
    Overwriting passes a casual smoke test and breaks the adjoint at the 1e-4
    level, which would quietly corrupt every MAP gradient."""
    ops = square_ops(10, 2.5)
    A = CoupledOperator(ops)
    x = np.random.default_rng(1).normal(size=ops.n_nodes)
    ref = ops.A_coupled.T @ x
    assert np.linalg.norm(A.rmatvec(x) - ref) / np.linalg.norm(ref) < 1e-12


def test_gmres_matches_the_direct_solve():
    ops = square_ops(10, 2.5)
    b = _rhs(ops)
    x, info = solve_coupled(ops, b, return_info=True)
    assert info["converged"]
    ref = ops._solve_psi(b)
    assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-8


def test_iteration_count_stays_bounded_with_refinement():
    """With the ILU(K) preconditioner the count should be flat-ish in the mesh
    size; a blow-up here means the preconditioner has stopped capturing the
    operator."""
    counts = []
    for nx in (8, 12, 16):
        ops = square_ops(nx, 2.5)
        _, info = solve_coupled(ops, _rhs(ops), return_info=True)
        assert info["converged"]
        counts.append(info["iterations"])
    assert max(counts) < 60
    assert counts[-1] <= 3 * counts[0]


def test_bem_block_can_be_supplied_through_v_solve():
    """The hook an H-matrix uses: rebuild C from V_eff/Xm/M_b with a pluggable
    solve instead of the precomputed dense block, and get the same operator."""
    import scipy.linalg as sla
    from scipy.spatial.distance import pdist
    from femmi.bem import assemble_bem_matrices

    ops = square_ops(10, 2.5)
    bnd = ops.bnd_mesh
    V_h, K_h, M_b = assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8)
    w = M_b @ np.ones(bnd.n_boundary_dofs)
    Xm = 0.5 * M_b - K_h
    cap = np.log(float(pdist(bnd.nodes).max())) / (2.0 * np.pi)
    V_eff = V_h - cap * np.outer(w, w)

    lu = sla.lu_factor(V_eff)
    A = CoupledOperator(ops, v_solve=lambda u: sla.lu_solve(lu, u),
                        V_eff=V_eff, Xm=Xm, M_b=M_b)
    x = np.random.default_rng(2).normal(size=ops.n_nodes)
    ref = ops.A_coupled @ x
    assert np.linalg.norm(A @ x - ref) / np.linalg.norm(ref) < 1e-10

    b = _rhs(ops)
    xs, info = solve_coupled(ops, b, operator=A, return_info=True)
    assert info["converged"]
    assert np.linalg.norm(xs - ops._solve_psi(b)) / np.linalg.norm(ops._solve_psi(b)) < 1e-8


def test_preconditioner_is_applicable():
    ops = square_ops(8, 2.5)
    Mp = fem_block_preconditioner(ops)
    v = np.random.default_rng(3).normal(size=ops.n_nodes)
    assert np.all(np.isfinite(Mp @ v))


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
