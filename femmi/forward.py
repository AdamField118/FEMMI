"""
femmi/forward.py
Differentiable forward model: kappa -> (gamma1, gamma2).

"""

import numpy as np
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple

from .operators import FEMOperators, build_operators, build_laplacian


def _make_fem_solve(K_lu, boundary, n_nodes):
    """Return a JAX-traceable solve for K x = b with custom VJP."""
    shape_struct = jax.ShapeDtypeStruct((n_nodes,), jnp.float64)

    def _solve_np(b):
        return K_lu.solve(np.array(b, dtype=np.float64))

    def _solve_np_T(b):
        return K_lu.solve(np.array(b, dtype=np.float64), trans='T')

    @jax.custom_vjp
    def fem_solve(b):
        return jax.pure_callback(_solve_np, shape_struct, b)

    def fem_solve_fwd(b):
        x = fem_solve(b)
        return x, x

    def fem_solve_bwd(x, g):
        lam = jax.pure_callback(_solve_np_T, shape_struct, g)
        return (lam,)

    fem_solve.defvjp(fem_solve_fwd, fem_solve_bwd)
    return fem_solve


def _make_matvec(A_np, n_nodes):
    """Return a JAX-traceable y = A x with custom VJP."""
    shape_struct = jax.ShapeDtypeStruct((n_nodes,), jnp.float64)
    AT_np        = A_np.T.tocsr()

    def _fwd_np(x):
        return np.array(A_np @ np.array(x, dtype=np.float64))

    def _bwd_np(g):
        return np.array(AT_np @ np.array(g, dtype=np.float64))

    @jax.custom_vjp
    def matvec(x):
        return jax.pure_callback(_fwd_np, shape_struct, x)

    def matvec_fwd(x):
        return matvec(x), None

    def matvec_bwd(_, g):
        return (jax.pure_callback(_bwd_np, shape_struct, g),)

    matvec.defvjp(matvec_fwd, matvec_bwd)
    return matvec


class DifferentiableForward:
    """
    Differentiable kappa -> (gamma1, gamma2) forward model.

    """

    def __init__(self, ops: FEMOperators, lam_reg: float = 1e-3):
        self.ops      = ops
        self.lam_reg  = lam_reg
        self.n_nodes  = ops.n_nodes
        self.boundary = ops.boundary
        self.interior = ops.interior

        L = build_laplacian(ops)
        n = ops.n_nodes

        self._fem_solve = _make_fem_solve(ops.A_coupled_lu, ops.boundary, n)
        self._M_mv      = _make_matvec(ops.M,  n)
        self._S1_mv     = _make_matvec(ops.S1, n)
        self._S2_mv     = _make_matvec(ops.S2, n)
        self._L_mv      = _make_matvec(L,       n)

    def rhs_from_kappa(self, kappa):
        rhs = -2.0 * self._M_mv(kappa)
        idx = int(self.ops.bnd_mesh.node_indices[0])
        return rhs.at[idx].set(0.0)

    def psi_from_kappa(self, kappa):
        return self._fem_solve(self.rhs_from_kappa(kappa))

    def gamma_from_kappa(self, kappa):
        psi = self.psi_from_kappa(kappa)
        return self._S1_mv(psi), self._S2_mv(psi)

    def data_fidelity(self, kappa, g1_obs, g2_obs):
        g1, g2 = self.gamma_from_kappa(kappa)
        return jnp.sum((g1 - g1_obs)**2) + jnp.sum((g2 - g2_obs)**2)

    def regularizer(self, kappa):
        return jnp.dot(kappa, self._L_mv(kappa))

    def loss_fn(self, kappa, g1_obs, g2_obs):
        return (self.data_fidelity(kappa, g1_obs, g2_obs)
                + self.lam_reg * self.regularizer(kappa))

    def grad_fn(self, kappa, g1_obs, g2_obs):
        return jax.value_and_grad(self.loss_fn)(kappa, g1_obs, g2_obs)

    def validate_gradients(self, kappa, g1_obs, g2_obs,
                           n_checks=8, eps=1e-5, verbose=True):
        """Compare autodiff vs central-difference finite differences."""
        kj  = jnp.array(kappa)
        g1j = jnp.array(g1_obs)
        g2j = jnp.array(g2_obs)

        _, grad_ad = self.grad_fn(kj, g1j, g2j)
        grad_ad    = np.array(grad_ad)

        np.random.seed(0)
        interior_idx = np.where(self.interior)[0]
        probe = np.random.choice(interior_idx,
                                 size=min(n_checks, len(interior_idx)),
                                 replace=False)

        if verbose:
            print(f"{'node':>6}  {'AD':>14}  {'FD':>14}  {'rel err':>10}")

        rel_errors = []
        for j in probe:
            kp = kappa.copy(); kp[j] += eps
            km = kappa.copy(); km[j] -= eps
            Lp  = float(self.loss_fn(jnp.array(kp), g1j, g2j))
            Lm  = float(self.loss_fn(jnp.array(km), g1j, g2j))
            g_fd = (Lp - Lm) / (2 * eps)
            g_ad = grad_ad[j]
            rel  = abs(g_ad - g_fd) / (abs(g_fd) + 1e-14)
            rel_errors.append(rel)
            if verbose:
                ok = "ok" if rel < 1e-4 else "FAIL"
                print(f"{j:6d}  {g_ad:14.6e}  {g_fd:14.6e}  {rel:10.3e}  {ok}")

        max_rel = max(rel_errors)
        passed  = max_rel < 1e-4
        if verbose:
            print(f"max rel error: {max_rel:.3e}  ->  {'PASS' if passed else 'FAIL'}")
        return {'max_rel_error': max_rel, 'passed': passed, 'rel_errors': rel_errors}

    def hvp(self, kappa, v, g1_obs, g2_obs):
        """Hessian-vector product H*v without forming H."""
        f = lambda k: self.loss_fn(k, g1_obs, g2_obs)
        return jax.jvp(jax.grad(f), (kappa,), (v,))[1]
