"""
P3 finite element assembly for the Poisson equation on triangular meshes.

Assembles the stiffness matrix and RHS using 13-point Dunavant degree-7
quadrature, which is exact for the degree-6 load integrand (cubic * cubic).
"""

import jax
from jax import jit
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .basis import (
    compute_p3_shape_functions,
    compute_p3_shape_gradients_reference,
    compute_p3_shape_gradients_physical,
)


def get_gauss_quadrature_triangle(order=5):
    """
    Quadrature points and weights for the reference triangle.

    Weights sum to 1; the physical integral is approximated as
    area * sum(w_q * f(xi_q, eta_q)).

    order 1: 1-point  (exact degree 1)
    order 2: 3-point  (exact degree 2)
    order 3: 4-point  (exact degree 3)
    order 4: 7-point  (exact degree 5)
    order 5: 13-point (exact degree 7)
    """
    if order == 1:
        points  = jnp.array([[1/3, 1/3]])
        weights = jnp.array([1.0])

    elif order == 2:
        points  = jnp.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        weights = jnp.array([1/3, 1/3, 1/3])

    elif order == 3:
        a = 1/3
        points  = jnp.array([[a, a], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
        weights = jnp.array([-27/48, 25/48, 25/48, 25/48])

    elif order == 4:
        a1 = 0.059715871789770
        a2 = 0.797426985353087
        b1 = 0.470142064105115
        b2 = 0.101286507323456
        w1 = 0.225000000000000
        w2 = 0.132394152788506
        w3 = 0.125939180544827
        points = jnp.array([
            [1/3, 1/3],
            [a1, a1], [a2, a1], [a1, a2],
            [b1, b1], [b2, b1], [b1, b2],
        ])
        weights = jnp.array([w1, w2, w2, w2, w3, w3, w3])

    elif order == 5:
        # Dunavant degree-7 rule (13 points). S111 orbit parameters are
        # genuinely distinct; earlier versions had a degenerate orbit.
        r2 = 0.260345966079040
        r3 = 0.065130102902216
        r4 = 0.048690315425316
        s4 = 0.312865496004875
        t4 = 1.0 - r4 - s4

        w0 = -0.149570044467670
        w1 =  0.175615257433208
        w2 =  0.053347235608839
        w3 =  0.077113760890257

        points = jnp.array([
            [1/3,    1/3   ],
            [r2,     r2    ], [1-2*r2, r2    ], [r2,     1-2*r2],
            [r3,     r3    ], [1-2*r3, r3    ], [r3,     1-2*r3],
            [r4, s4], [s4, r4], [r4, t4], [t4, r4], [s4, t4], [t4, s4],
        ])
        weights = jnp.array([w0, w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3])

    else:
        raise ValueError(f"Quadrature order {order} not implemented")

    return points, weights


@jit
def compute_element_stiffness_p3(coords, quad_points, quad_weights):
    """Return the 10x10 element stiffness matrix for a P3 element."""
    nq = len(quad_weights)
    Ke = jnp.zeros((10, 10))

    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]

    J = jnp.array([[x1 - x0, y1 - y0], [x2 - x0, y2 - y0]])
    detJ  = jnp.linalg.det(J)
    J_inv = jnp.linalg.inv(J)
    area  = jnp.abs(detJ) / 2.0

    for q in range(nq):
        xi, eta = quad_points[q]
        dN_dxi  = compute_p3_shape_gradients_reference(xi, eta)
        dN_dxy  = dN_dxi @ J_inv.T
        for i in range(10):
            for j in range(10):
                Ke = Ke.at[i, j].add(quad_weights[q] * area * jnp.dot(dN_dxy[i], dN_dxy[j]))

    return Ke


@jit
def compute_element_load_p3(coords, source_values, quad_points, quad_weights):
    """Return the 10-element load vector for a P3 element."""
    nq = len(quad_weights)
    Fe = jnp.zeros(10)

    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]

    J    = jnp.array([[x1 - x0, y1 - y0], [x2 - x0, y2 - y0]])
    detJ = jnp.linalg.det(J)
    area = jnp.abs(detJ) / 2.0

    for q in range(nq):
        xi, eta = quad_points[q]
        N       = compute_p3_shape_functions(xi, eta)
        kappa_q = jnp.dot(N, source_values)
        Fe += -2.0 * quad_weights[q] * area * N * kappa_q

    return Fe


def assemble_system_p3(mesh, kappa_values, use_jax=False):
    """
    Assemble the global stiffness matrix K and load vector F.

    Returns K as a sparse CSR matrix and F as a numpy array.
    """
    nodes      = np.array(mesh.nodes)
    elements   = np.array(mesh.elements)
    n_nodes    = len(nodes)
    n_elements = len(elements)

    quad_points, quad_weights = get_gauss_quadrature_triangle(order=5)

    max_entries = n_elements * 100
    I_arr  = np.zeros(max_entries, dtype=np.int32)
    J_arr  = np.zeros(max_entries, dtype=np.int32)
    K_data = np.zeros(max_entries)
    F      = np.zeros(n_nodes)
    entry  = 0

    for elem in elements:
        ec = jnp.array(nodes[elem])
        kv = jnp.array(kappa_values[elem])

        Ke = np.array(compute_element_stiffness_p3(ec, quad_points, quad_weights))
        Fe = np.array(compute_element_load_p3(ec, kv, quad_points, quad_weights))

        for i in range(10):
            gi = elem[i]
            F[gi] += Fe[i]
            for j in range(10):
                I_arr[entry]  = gi
                J_arr[entry]  = elem[j]
                K_data[entry] = Ke[i, j]
                entry += 1

    K = sp.coo_matrix(
        (K_data[:entry], (I_arr[:entry], J_arr[:entry])),
        shape=(n_nodes, n_nodes),
    ).tocsr()

    if use_jax:
        F = jnp.array(F)

    return K, F


def apply_boundary_conditions_p3(K, F, mesh):
    """Apply homogeneous Dirichlet BCs by row replacement."""
    boundary = np.array(mesh.boundary)
    K_bc = K.tolil()
    F_bc = F.copy()
    for node in boundary:
        K_bc[node, :] = 0
        K_bc[node, node] = 1.0
        F_bc[node] = 0.0
    return K_bc.tocsr(), F_bc


def solve_p3_system(K, F, mesh):
    """Solve K psi = F and return psi."""
    return spla.spsolve(K, F)


def solve_poisson_p3(mesh, kappa_values):
    """Full P3 FEM solve for nabla^2 psi = 2*kappa with psi=0 on boundary."""
    K, F = assemble_system_p3(mesh, kappa_values)
    K_bc, F_bc = apply_boundary_conditions_p3(K, F, mesh)
    return solve_p3_system(K_bc, F_bc, mesh)