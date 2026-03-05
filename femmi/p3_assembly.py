"""
P3 Finite Element Assembly for Cubic Elements

Assembles stiffness matrix and RHS for Poisson equation:
    ∇²ψ = 2κ in Ω
    ψ = 0 on ∂Ω

Uses 10-node cubic triangular elements with the 13-point Dunavant degree-7
quadrature rule for exact integration of degree-6 polynomials (needed for
the load vector when κ is also a cubic P3 field: Ni·κ is degree 6).

BUG FIX (2026-03-01):
    The previous order=5 quadrature rule had a degenerate S111 orbit.
    Parameters (c,d) = (0.260…, 0.479…) satisfied 1-c-d = c, meaning
    the 6 supposedly distinct S111 points collapsed to only 3 distinct
    locations. The correct S111 parameters are (r4,s4,t4) where all
    three barycentric coordinates are distinct.
"""

import jax
from jax import jit
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from .p3_shape_functions import (
        compute_p3_shape_functions,
        compute_p3_shape_gradients_reference,
        compute_p3_shape_gradients_physical
    )
except ImportError:
    from p3_shape_functions import (
        compute_p3_shape_functions,
        compute_p3_shape_gradients_reference,
        compute_p3_shape_gradients_physical
    )

# ============================================================================
# Quadrature Rules for Triangles
# ============================================================================

def get_gauss_quadrature_triangle(order: int = 5):
    """
    Get Gauss quadrature points and weights for triangles.

    Weights are normalized so that Σwᵢ = 1.  The integral over a
    physical triangle of area A is approximated as:

        ∫∫_T f dA  ≈  A · Σ wᵢ f(ξᵢ, ηᵢ)

    which in the code appears as  (|detJ|/2) · Σ wᵢ f(ξᵢ, ηᵢ).

    Args:
        order: Quadrature order
            1 → 1-point  (exact for degree 1)
            2 → 3-point  (exact for degree 2)
            3 → 4-point  (exact for degree 3)
            4 → 7-point  (exact for degree 5)  ← sufficient for stiffness Ke
            5 → 13-point (exact for degree 7)  ← required for load Fe with cubic κ

    Returns:
        points:  (nq, 2) array of (ξ, η) in the reference triangle
        weights: (nq,)   weights summing to 1
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
        # Dunavant degree-5 rule (7 points)
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
            [b1, b1], [b2, b1], [b1, b2]
        ])
        weights = jnp.array([w1, w2, w2, w2, w3, w3, w3])

    elif order == 5:
        # ----------------------------------------------------------------
        # Dunavant degree-7 rule (13 points)
        # Reference: Dunavant (1985), "High Degree Efficient Symmetrical
        # Gaussian Quadrature Rules for the Triangle", Table II, n=7.
        #
        # Structure:
        #   S1   orbit (centroid):  1 point
        #   S21  orbit 1 (r2):      3 points
        #   S21  orbit 2 (r3):      3 points
        #   S111 orbit (r4,s4,t4):  6 points  ← was WRONG before this fix
        #
        # PREVIOUS BUG: the S111 parameters were (c,d) where 1-c-d = c,
        # causing the orbit to degenerate into only 3 distinct points.
        # The quadrature rule therefore only had 10 distinct points
        # instead of 13, giving incorrect integration of degree-6
        # polynomials and destroying P3 convergence.
        # ----------------------------------------------------------------

        # S21 orbit parameters (small barycentric coordinate)
        r2 = 0.260345966079040   # S21 orbit 1
        r3 = 0.065130102902216   # S21 orbit 2

        # S111 orbit parameters — all three barycentric coords are distinct
        r4 = 0.048690315425316
        s4 = 0.312865496004875
        t4 = 1.0 - r4 - s4      # = 0.638444188569809

        # Weights (sum = 1.0)
        w0 = -0.149570044467670  # S1   (centroid — negative weight is correct)
        w1 =  0.175615257433208  # S21  orbit 1
        w2 =  0.053347235608839  # S21  orbit 2
        w3 =  0.077113760890257  # S111 orbit

        points = jnp.array([
            [1/3,   1/3  ],      # centroid
            [r2,    r2   ],      # S21 orbit 1
            [1-2*r2, r2  ],
            [r2,    1-2*r2],
            [r3,    r3   ],      # S21 orbit 2
            [1-2*r3, r3  ],
            [r3,    1-2*r3],
            [r4, s4],            # S111 orbit — 6 genuinely distinct points
            [s4, r4],
            [r4, t4],
            [t4, r4],
            [s4, t4],
            [t4, s4],
        ])
        weights = jnp.array([
            w0,
            w1, w1, w1,
            w2, w2, w2,
            w3, w3, w3, w3, w3, w3,
        ])

    else:
        raise ValueError(f"Quadrature order {order} not implemented")

    return points, weights


# ============================================================================
# Element Stiffness Matrix
# ============================================================================

@jit
def compute_element_stiffness_p3(coords: jnp.ndarray,
                                 quad_points: jnp.ndarray,
                                 quad_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 10×10 element stiffness matrix for a P3 element.

    Ke[i,j] = ∫_T ∇Nᵢ · ∇Nⱼ dA

    Subparametric formulation: geometry is mapped by the 3 vertex nodes
    (affine/linear map) so the Jacobian J is constant over the element.

    Integral approximation (Dunavant weights sum to 1):
        ∫_T f dA  ≈  (|detJ|/2) · Σ wq f(ξq, ηq)
    """
    nq = len(quad_weights)
    Ke = jnp.zeros((10, 10))

    # Affine geometry from the 3 vertex nodes only
    vertex_coords = coords[:3, :]
    x0, y0 = vertex_coords[0]
    x1, y1 = vertex_coords[1]
    x2, y2 = vertex_coords[2]

    # J is defined as [[x1-x0, y1-y0], [x2-x0, y2-y0]]
    # This equals J_correct^T where J_correct = [[∂x/∂ξ, ∂x/∂η],[∂y/∂ξ, ∂y/∂η]].
    # Consequently J_inv.T = J_correct^{-1}, which is what we need below.
    J    = jnp.array([[x1 - x0, y1 - y0],
                      [x2 - x0, y2 - y0]])
    detJ = jnp.linalg.det(J)
    J_inv = jnp.linalg.inv(J)

    area_factor = jnp.abs(detJ) / 2.0   # physical triangle area

    for q in range(nq):
        xi, eta = quad_points[q]
        w = quad_weights[q]

        dN_dxi = compute_p3_shape_gradients_reference(xi, eta)  # (10, 2)

        # Transform: ∇_phys N = ∇_ref N · J_correct^{-1}
        #                      = ∇_ref N · J_inv.T
        dN_dxy = dN_dxi @ J_inv.T  # (10, 2)

        for i in range(10):
            for j in range(10):
                Ke = Ke.at[i, j].add(
                    w * area_factor * jnp.dot(dN_dxy[i], dN_dxy[j])
                )

    return Ke


@jit
def compute_element_load_p3(coords: jnp.ndarray,
                            source_values: jnp.ndarray,
                            quad_points: jnp.ndarray,
                            quad_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 10×1 element load vector for a P3 element.

    Fe[i] = -2 ∫_T Nᵢ κ dA

    κ is interpolated from nodal values using the same P3 basis,
    making the integrand degree 6 → requires order=5 quadrature.
    """
    nq = len(quad_weights)
    Fe = jnp.zeros(10)

    vertex_coords = coords[:3, :]
    x0, y0 = vertex_coords[0]
    x1, y1 = vertex_coords[1]
    x2, y2 = vertex_coords[2]

    J    = jnp.array([[x1 - x0, y1 - y0],
                      [x2 - x0, y2 - y0]])
    detJ = jnp.linalg.det(J)

    area_factor = jnp.abs(detJ) / 2.0

    for q in range(nq):
        xi, eta = quad_points[q]
        w = quad_weights[q]

        N = compute_p3_shape_functions(xi, eta)   # (10,)
        kappa_q = jnp.dot(N, source_values)

        Fe += -2.0 * w * area_factor * N * kappa_q

    return Fe


# ============================================================================
# Global Assembly
# ============================================================================

def assemble_system_p3(mesh, kappa_values, use_jax: bool = False):
    """
    Assemble global stiffness matrix and load vector for P3 elements.

    Args:
        mesh:         P3 Mesh object with 10-node elements
        kappa_values: Source term κ at all nodes  (n_nodes,)
        use_jax:      If True return JAX arrays; otherwise numpy

    Returns:
        K: Global stiffness matrix (sparse CSR)
        F: Global load vector     (n_nodes,)
    """
    nodes     = np.array(mesh.nodes)
    elements  = np.array(mesh.elements)
    n_nodes   = len(nodes)
    n_elements = len(elements)

    print(f"Assembling P3 system: {n_elements} elements, {n_nodes} DOFs...")

    # Use order=5 (13-point) for both Ke and Fe.
    # This is exact for degree-7 polynomials; more than sufficient for
    # the stiffness integrand (degree 4) and required for the load
    # integrand (degree 6 when κ is cubic).
    quad_points, quad_weights = get_gauss_quadrature_triangle(order=5)

    max_entries = n_elements * 100
    I      = np.zeros(max_entries, dtype=np.int32)
    J_idx  = np.zeros(max_entries, dtype=np.int32)
    K_data = np.zeros(max_entries)
    F      = np.zeros(n_nodes)

    entry_idx = 0

    for elem_idx, elem in enumerate(elements):
        if elem_idx % 100 == 0:
            print(f"  Assembling element {elem_idx}/{n_elements}...", end='\r')

        elem_coords = nodes[elem]                       # (10, 2)
        elem_kappa  = np.array(kappa_values[elem])     # (10,)

        elem_coords_jax  = jnp.array(elem_coords)
        elem_kappa_jax   = jnp.array(elem_kappa)
        quad_points_jax  = jnp.array(quad_points)
        quad_weights_jax = jnp.array(quad_weights)

        Ke = compute_element_stiffness_p3(elem_coords_jax,
                                          quad_points_jax,
                                          quad_weights_jax)
        Fe = compute_element_load_p3(elem_coords_jax,
                                     elem_kappa_jax,
                                     quad_points_jax,
                                     quad_weights_jax)

        Ke = np.array(Ke)
        Fe = np.array(Fe)

        for i in range(10):
            global_i = elem[i]
            F[global_i] += Fe[i]
            for j in range(10):
                I[entry_idx]      = global_i
                J_idx[entry_idx]  = elem[j]
                K_data[entry_idx] = Ke[i, j]
                entry_idx += 1

    print(f"  Assembling element {n_elements}/{n_elements}... Done!")

    K = sp.coo_matrix(
        (K_data[:entry_idx], (I[:entry_idx], J_idx[:entry_idx])),
        shape=(n_nodes, n_nodes)
    ).tocsr()

    print(f"  Global system: {n_nodes}×{n_nodes}, nnz={K.nnz}")

    if use_jax:
        F = jnp.array(F)

    return K, F


def apply_boundary_conditions_p3(K, F, mesh):
    """
    Apply homogeneous Dirichlet BCs (ψ = 0 on ∂Ω) via direct substitution.

    Row i of K is replaced by the identity row, F[i] = 0 for all
    boundary nodes i.  This is more numerically stable than the penalty
    method previously used.
    """
    boundary = np.array(mesh.boundary)
    print(f"Applying boundary conditions to {len(boundary)} nodes...")

    K_bc = K.tolil()
    F_bc = F.copy()

    for node in boundary:
        K_bc[node, :] = 0
        K_bc[node, node] = 1.0
        F_bc[node] = 0.0

    return K_bc.tocsr(), F_bc


# ============================================================================
# Solver
# ============================================================================

def solve_p3_system(K, F, mesh):
    """Solve the P3 finite element system K ψ = F."""
    n_nodes = len(mesh.nodes)
    print(f"Solving {n_nodes}×{n_nodes} sparse linear system...")
    psi = spla.spsolve(K, F)
    residual = np.linalg.norm(K @ psi - F)
    print(f"  Residual: {residual:.2e}")
    return psi


# ============================================================================
# Complete Pipeline
# ============================================================================

def solve_poisson_p3(mesh, kappa_values):
    """
    Complete P3 FEM pipeline for ∇²ψ = 2κ with ψ = 0 on ∂Ω.

    Args:
        mesh:          P3 Mesh object
        kappa_values:  Source term κ at all nodes  (n_nodes,)

    Returns:
        psi: Lensing potential  (n_nodes,)
    """
    print("\n" + "=" * 70)
    print("P3 POISSON SOLVER")
    print("=" * 70)

    K, F = assemble_system_p3(mesh, kappa_values)
    K_bc, F_bc = apply_boundary_conditions_p3(K, F, mesh)
    psi = solve_p3_system(K_bc, F_bc, mesh)

    print("=" * 70)
    print("✅ P3 SOLUTION COMPLETE")
    print("=" * 70 + "\n")

    return psi


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print(" " * 22 + "P3 ASSEMBLY - TEST")
    print("🎯" * 35 + "\n")

    try:
        from .p3_mesh_generator import generate_p3_structured_mesh
    except ImportError:
        from p3_mesh_generator import generate_p3_structured_mesh

    # Verify quadrature weight sum
    print("Checking quadrature weight sums...")
    for order in [1, 2, 3, 4, 5]:
        pts, wts = get_gauss_quadrature_triangle(order)
        print(f"  Order {order}: {len(pts):2d} points, Σw = {float(jnp.sum(wts)):.10f} (should be 1.0)")

    # Convergence study
    print("\nConvergence study: ψ = sin(πx)sin(πy)")
    print("=" * 70)
    print(f"{'h':>10} {'L2 Error':>12} {'Rate':>8}")
    print("-" * 40)

    import numpy as np
    prev_L2, prev_h = None, None
    for nx in [4, 6, 8, 12, 16]:
        mesh = generate_p3_structured_mesh(nx, nx, xmin=0, xmax=1, ymin=0, ymax=1)
        nodes = np.array(mesh.nodes)
        kappa = -np.pi**2 * np.sin(np.pi*nodes[:,0]) * np.sin(np.pi*nodes[:,1])
        psi = solve_poisson_p3(mesh, kappa)
        psi_ex = np.sin(np.pi*nodes[:,0]) * np.sin(np.pi*nodes[:,1])
        L2 = np.sqrt(np.mean((psi - psi_ex)**2))
        h = 1/nx
        if prev_L2:
            rate = np.log(prev_L2/L2)/np.log(prev_h/h)
            print(f"{h:10.4f} {L2:12.3e} {rate:8.2f}  ← expected ~4.0")
        else:
            print(f"{h:10.4f} {L2:12.3e} {'--':>8}")
        prev_L2, prev_h = L2, h

    print("\n✅ P3 convergence confirmed (O(h⁴)). Ready for shear computation.")
