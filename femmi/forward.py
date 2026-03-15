"""
forward.py
==========
Uses jax.pure_callback to call scipy/numpy from inside JAX tracing.
This is the only correct pattern for custom_vjp + external solvers.
"""

import numpy as np
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple, Callable, Optional

from .operators import FEMOperators, build_operators, build_laplacian


# ── custom_vjp sparse solve ────────────────────────────────────────────────────

def _make_fem_solve(K_lu, boundary: np.ndarray, n_nodes: int):
    """
    JAX-traceable K x = b solver using jax.pure_callback.
    VJP: K^{-T} g = K^{-1} g  (K symmetric).
    """
    shape_struct = jax.ShapeDtypeStruct((n_nodes,), jnp.float64)

    def _solve_np(b):
        return K_lu.solve(np.array(b, dtype=np.float64))
        
    
    def _solve_np_T(b):
        return K_lu.solve(np.array(b, dtype=np.float64), trans='T')

    @jax.custom_vjp
    def fem_solve(b: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(_solve_np, shape_struct, b)

    def fem_solve_fwd(b):
        x = fem_solve(b)
        return x, x                  # residual = solution (for bwd)

    def fem_solve_bwd(x, g):
        lam = jax.pure_callback(_solve_np_T, shape_struct, g)
        return (lam,)

    fem_solve.defvjp(fem_solve_fwd, fem_solve_bwd)
    return fem_solve


def _make_matvec(A_np: sp.spmatrix, n_nodes: int):
    """
    JAX-traceable y = A x via pure_callback.
    VJP: dL/dx = A^T (dL/dy).
    """
    shape_struct = jax.ShapeDtypeStruct((n_nodes,), jnp.float64)
    AT_np = A_np.T.tocsr()

    def _fwd_np(x):
        return np.array(A_np @ np.array(x, dtype=np.float64))

    def _bwd_np(g):
        return np.array(AT_np @ np.array(g, dtype=np.float64))

    @jax.custom_vjp
    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(_fwd_np, shape_struct, x)

    def matvec_fwd(x):
        return matvec(x), None

    def matvec_bwd(_, g):
        return (jax.pure_callback(_bwd_np, shape_struct, g),)

    matvec.defvjp(matvec_fwd, matvec_bwd)
    return matvec


# ══════════════════════════════════════════════════════════════════════════════

class DifferentiableForward:
    """
    Differentiable κ → (γ₁, γ₂) forward model.

    All scipy objects cached at construction; public methods are
    JAX-traceable (jax.grad, jax.jit, jax.vmap all work).

    Parameters
    ----------
    ops     : FEMOperators from operators.build_operators
    lam_reg : regularization strength for MAP loss (default 1e-3)
    """

    def __init__(self, ops: FEMOperators, lam_reg: float = 1e-3):
        self.ops      = ops
        self.lam_reg  = lam_reg
        self.n_nodes  = ops.n_nodes
        self.boundary = ops.boundary
        self.interior = ops.interior

        n = ops.n_nodes
        L = build_laplacian(ops)

        # Build all JAX-traceable primitives
        self._fem_solve  = _make_fem_solve(ops.A_coupled_lu, ops.boundary, n)
        self._M_mv       = _make_matvec(ops.M,  n)
        self._S1_mv      = _make_matvec(ops.S1, n)
        self._S2_mv      = _make_matvec(ops.S2, n)
        self._L_mv       = _make_matvec(L,       n)

    # ── core pipeline ──────────────────────────────────────────────────────────

    def rhs_from_kappa(self, kappa: jnp.ndarray) -> jnp.ndarray:
        """F = -2 M κ  (full M, no boundary zeroing — A_coupled handles BCs)."""
        rhs = -2.0 * self._M_mv(kappa)
        idx = int(self.ops.bnd_mesh.node_indices[0])
        return rhs.at[idx].set(0.0)

    def psi_from_kappa(self, kappa: jnp.ndarray) -> jnp.ndarray:
        """Differentiable κ → ψ."""
        return self._fem_solve(self.rhs_from_kappa(kappa))

    def gamma_from_kappa(self, kappa: jnp.ndarray
                         ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Differentiable κ → (γ₁, γ₂). Supports jax.grad/jit/vmap."""
        psi = self.psi_from_kappa(kappa)
        return self._S1_mv(psi), self._S2_mv(psi)

    # ── loss functions ─────────────────────────────────────────────────────────

    def data_fidelity(self, kappa, g1_obs, g2_obs) -> jnp.ndarray:
        g1, g2 = self.gamma_from_kappa(kappa)
        return jnp.sum((g1 - g1_obs)**2) + jnp.sum((g2 - g2_obs)**2)

    def regularizer(self, kappa: jnp.ndarray) -> jnp.ndarray:
        """‖∇κ‖² = κᵀ L κ (FEM H¹ semi-norm)."""
        return jnp.dot(kappa, self._L_mv(kappa))

    def loss_fn(self, kappa, g1_obs, g2_obs) -> jnp.ndarray:
        """MAP loss: ‖γ_pred − γ_obs‖² + λ ‖∇κ‖²"""
        return (self.data_fidelity(kappa, g1_obs, g2_obs)
                + self.lam_reg * self.regularizer(kappa))

    def grad_fn(self, kappa, g1_obs, g2_obs
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(loss, ∂loss/∂κ) in one call."""
        return jax.value_and_grad(self.loss_fn)(kappa, g1_obs, g2_obs)

    # ── validation ─────────────────────────────────────────────────────────────

    def validate_gradients(self, kappa: np.ndarray,
                           g1_obs: np.ndarray, g2_obs: np.ndarray,
                           n_checks: int = 8, eps: float = 1e-5,
                           verbose: bool = True) -> dict:
        """Compare autodiff gradients vs central-difference FD."""
        kj = jnp.array(kappa)
        g1j = jnp.array(g1_obs)
        g2j = jnp.array(g2_obs)

        _, grad_ad = self.grad_fn(kj, g1j, g2j)
        grad_ad = np.array(grad_ad)

        np.random.seed(0)
        interior_idx = np.where(self.interior)[0]
        probe = np.random.choice(interior_idx,
                                 size=min(n_checks, len(interior_idx)),
                                 replace=False)

        if verbose:
            print("=" * 62)
            print("Gradient validation (autodiff vs finite differences)")
            print(f"{'Node':>6}  {'AD':>14}  {'FD':>14}  {'rel err':>10}")
            print("-" * 62)

        rel_errors = []
        for j in probe:
            kp = kappa.copy(); kp[j] += eps
            km = kappa.copy(); km[j] -= eps
            Lp = float(self.loss_fn(jnp.array(kp), g1j, g2j))
            Lm = float(self.loss_fn(jnp.array(km), g1j, g2j))
            g_fd = (Lp - Lm) / (2 * eps)
            g_ad = grad_ad[j]
            rel  = abs(g_ad - g_fd) / (abs(g_fd) + 1e-14)
            rel_errors.append(rel)
            if verbose:
                st = "Y" if rel < 1e-4 else "x"
                print(f"{j:6d}  {g_ad:14.6e}  {g_fd:14.6e}  {rel:10.3e} {st}")

        max_rel = max(rel_errors)
        passed  = max_rel < 1e-4
        if verbose:
            print("-" * 62)
            status = "PASS" if passed else "FAIL"
            print(f"Max relative error: {max_rel:.3e}  →  {status}")
            print("=" * 62)
        return {'max_rel_error': max_rel, 'passed': passed,
                'rel_errors': rel_errors}

    # ── HVP for Newton-CG / Laplace approximation ──────────────────────────────

    def hvp(self, kappa, v, g1_obs, g2_obs) -> jnp.ndarray:
        """Hessian-vector product H·v (O(2 solves), no H formed)."""
        f = lambda k: self.loss_fn(k, g1_obs, g2_obs)
        return jax.jvp(jax.grad(f), (kappa,), (v,))[1]