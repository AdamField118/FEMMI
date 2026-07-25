"""
femmi/iterative.py
Matrix-free solves of the coupled FEM-BEM system.

WHY
---
`build_operators` forms A_coupled = K + P^T C P as an explicit sparse matrix and
LU-factorises it. That is fine while C is a small dense block, but it is exactly
what stops `femmi.aca` from being useful: an H-matrix has no entries to scatter
into a sparse matrix, so as long as the solve needs an assembled A_coupled the
BEM stays dense and O(N_b^2).

This module applies A_coupled as an operator instead:

    A x  =  K x  +  scatter( C_apply( gather(x) ) ),      C = -M_b V_eff^{-1} X_m

with the gauge row enforced explicitly. `C_apply` reaches the BEM only through
matvecs and one V_eff solve, so an ACA/H-matrix version can be dropped in by
passing `v_solve` without touching anything else.

PRECONDITIONING -- and an honest note
-------------------------------------
The preconditioner here is an incomplete LU of the sparse FEM block K (with the
gauge row), which is the dominant and best-conditioned part of the operator.

It is NOT Calderon preconditioning. Calderon preconditioning of the single-layer
operator V uses the HYPERSINGULAR operator W and the identity that VW is a
compact perturbation of -I/4; femmi.bem assembles V, K and M_b but not W, so the
ingredient does not exist yet. (`bem.calderon_matrix` is the coupling operator
V^{-1}(1/2 M_b + K_h), not a preconditioner for V -- an easy name to misread.)
Adding W is the natural follow-up and is what would make the iteration count
genuinely mesh-independent.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CoupledOperator(spla.LinearOperator):
    """A_coupled applied matrix-free.

    ops      : FEMOperators (supplies K, the boundary index map and the BEM block)
    v_solve  : optional callable u -> V_eff^{-1} u. Defaults to a dense LU of
               V_eff. Pass an H-matrix-backed solve to make the BEM sub-linear.
    """

    def __init__(self, ops, v_solve=None, V_eff=None, Xm=None, M_b=None):
        n = ops.n_nodes
        super().__init__(dtype=np.float64, shape=(n, n))
        self.ops = ops
        self.K = ops.K.tocsr()
        self.bnd_idx = np.asarray(ops.bnd_mesh.node_indices, int)
        self.idx_gauge = int(self.bnd_idx[0])

        # Reconstruct the BEM pieces when not supplied. C_dense is what
        # build_operators already computed, so by default we reuse it and this
        # class is purely a restructuring of the same operator.
        self.C_dense = None if V_eff is not None else np.asarray(ops.C_dense)
        self.V_eff, self.Xm, self.M_b = V_eff, Xm, M_b
        if V_eff is not None:
            self._v_solve = v_solve or _dense_lu_solve(V_eff)

    def _apply_C(self, u):
        if self.C_dense is not None:
            return self.C_dense @ u
        return -(self.M_b @ self._v_solve(self.Xm @ u))

    def _matvec(self, x):
        x = np.asarray(x, float).reshape(-1)
        y = self.K @ x
        y[self.bnd_idx] += self._apply_C(x[self.bnd_idx])
        # gauge row: replaces the equation at one boundary node with x_g = rhs_g
        y[self.idx_gauge] = x[self.idx_gauge]
        return y

    def _rmatvec(self, x):
        """Transpose action, needed for the adjoint solve in the MAP gradient.

        The gauge fix zeroes ROW g and puts a 1 on the diagonal, but leaves COLUMN
        g populated. So (A^T x)_g is the whole of column g -- the contributions
        from every row i != g PLUS the diagonal 1 -- and the gauge term must be
        ADDED, not written over the column contribution.
        """
        x = np.asarray(x, float).reshape(-1)
        xg = x[self.idx_gauge]
        x = x.copy(); x[self.idx_gauge] = 0.0          # row g contributes only x_g
        y = self.K.T @ x
        if self.C_dense is not None:
            y[self.bnd_idx] += self.C_dense.T @ x[self.bnd_idx]
        else:
            y[self.bnd_idx] += -(self.Xm.T @ self._v_solve(self.M_b.T @ x[self.bnd_idx]))
        y[self.idx_gauge] += xg
        return y


def _dense_lu_solve(V_eff):
    import scipy.linalg as sla
    lu = sla.lu_factor(np.asarray(V_eff, float))
    return lambda u: sla.lu_solve(lu, u)


def fem_block_preconditioner(ops, drop_tol=1e-4, fill_factor=10):
    """ILU of the sparse FEM block K with the gauge row applied.

    K is the dominant, sparse, well-understood part of A_coupled; the BEM block
    is a dense low-rank-ish correction on the boundary ring only. Preconditioning
    with K alone therefore captures most of the spectrum at sparse cost.
    """
    K = ops.K.tolil()
    g = int(np.asarray(ops.bnd_mesh.node_indices, int)[0])
    K[g, :] = 0.0
    K[g, g] = 1.0
    ilu = spla.spilu(K.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    return spla.LinearOperator(K.shape, matvec=ilu.solve, dtype=np.float64)


def solve_coupled(ops, rhs, tol=1e-10, maxiter=500, operator=None,
                  precond=None, return_info=False):
    """Solve A_coupled x = rhs with GMRES, matrix-free.

    Returns x (and, with return_info=True, the iteration count and residual).
    Falls back to reporting non-convergence rather than silently returning a bad
    solution.
    """
    A = operator if operator is not None else CoupledOperator(ops)
    Mp = precond if precond is not None else fem_block_preconditioner(ops)
    rhs = np.asarray(rhs, float).reshape(-1)

    n_it = {"k": 0}

    def _count(_):
        n_it["k"] += 1

    try:
        x, info = spla.gmres(A, rhs, rtol=tol, restart=100, maxiter=maxiter,
                             M=Mp, callback=_count, callback_type="pr_norm")
    except TypeError:                                  # older scipy: tol= not rtol=
        x, info = spla.gmres(A, rhs, tol=tol, restart=100, maxiter=maxiter,
                             M=Mp, callback=_count, callback_type="pr_norm")

    res = float(np.linalg.norm(A @ x - rhs) / (np.linalg.norm(rhs) + 1e-300))
    if return_info:
        return x, dict(iterations=n_it["k"], converged=(info == 0), residual=res)
    return x
