"""
P3 Lagrange basis functions on the reference triangle.

10 DOFs per element: 3 vertices, 6 edge nodes (t=1/3, t=2/3 per edge),
1 interior bubble. Provides O(h^4) convergence for psi, O(h^2) for shear.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def compute_p3_shape_functions(xi, eta):
    """Evaluate all 10 P3 shape functions at (xi, eta) in the reference triangle."""
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta

    N0 = 0.5 * lam1 * (3.0*lam1 - 1.0) * (3.0*lam1 - 2.0)
    N1 = 0.5 * lam2 * (3.0*lam2 - 1.0) * (3.0*lam2 - 2.0)
    N2 = 0.5 * lam3 * (3.0*lam3 - 1.0) * (3.0*lam3 - 2.0)

    N3 = (9.0/2.0) * lam1 * lam2 * (3.0*lam1 - 1.0)
    N4 = (9.0/2.0) * lam1 * lam2 * (3.0*lam2 - 1.0)
    N5 = (9.0/2.0) * lam2 * lam3 * (3.0*lam2 - 1.0)
    N6 = (9.0/2.0) * lam2 * lam3 * (3.0*lam3 - 1.0)
    N7 = (9.0/2.0) * lam3 * lam1 * (3.0*lam3 - 1.0)
    N8 = (9.0/2.0) * lam3 * lam1 * (3.0*lam1 - 1.0)

    N9 = 27.0 * lam1 * lam2 * lam3

    return jnp.array([N0, N1, N2, N3, N4, N5, N6, N7, N8, N9])


@jit
def compute_p3_shape_gradients_reference(xi, eta):
    """Return (10, 2) array of reference-space gradients dN/d(xi, eta) at (xi, eta)."""
    def Nfun(x):
        return compute_p3_shape_functions(x[0], x[1])
    return jax.jacobian(Nfun)(jnp.array([xi, eta]))


@jit
def compute_jacobian_p3(xi, eta, coords):
    """Return (2, 2) Jacobian of the isoparametric map at (xi, eta)."""
    dN_dref = compute_p3_shape_gradients_reference(xi, eta)
    return jnp.dot(coords.T, dN_dref)


@jit
def compute_p3_shape_gradients_physical(xi, eta, coords):
    """Return (10, 2) physical-space gradients at (xi, eta) given element coords."""
    dN_dref = compute_p3_shape_gradients_reference(xi, eta)
    J = compute_jacobian_p3(xi, eta, coords)
    return jnp.dot(dN_dref, jnp.linalg.inv(J))