"""
P3 (Cubic) Shape Functions for Triangular Elements

Complete cubic polynomial space with 10 degrees of freedom per element.
Provides O(h⁴) convergence for potential ψ and O(h²) for shear γ.

Mathematical Foundation:
- Barycentric coordinates: λ₁ = 1-ξ-η, λ₂ = ξ, λ₃ = η
- Cubic Lagrange polynomials on each edge
- Interior bubble function for centroid node

References:
- Brenner & Scott (2008) - "Mathematical Theory of FEM"
- Zienkiewicz & Taylor - "The Finite Element Method"
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from typing import Tuple
import numpy as np


# ============================================================================
# P3 Shape Functions (Cubic)
# ============================================================================

@jit
def compute_p3_shape_functions(xi: float, eta: float) -> jnp.ndarray:
    """
    Compute P3 shape functions at point (ξ, η) in reference triangle
    
    Reference triangle: vertices at (0,0), (1,0), (0,1)
    Barycentric coordinates: λ₁ = 1-ξ-η, λ₂ = ξ, λ₃ = η
    
    P3 basis functions (cubic polynomials):
    - 3 vertex functions (degree 3 in each λᵢ)
    - 6 edge functions (cubic Lagrange along edges)
    - 1 interior function (bubble at centroid)
    
    Args:
        xi, eta: Coordinates in reference triangle (0 ≤ ξ, η, ξ+η ≤ 1)
        
    Returns:
        (10,) array of shape function values [N₀, N₁, ..., N₉]
    """
    # Barycentric coordinates
    lam1 = 1.0 - xi - eta  # λ₁ (associated with vertex 0)
    lam2 = xi              # λ₂ (associated with vertex 1)
    lam3 = eta             # λ₃ (associated with vertex 2)
    
    # ========================================================================
    # VERTEX NODES (0, 1, 2) - Cubic in barycentric coordinate
    # ========================================================================
    # Formula: Nᵢ = ½λᵢ(3λᵢ - 1)(3λᵢ - 2)
    
    N0 = 0.5 * lam1 * (3.0*lam1 - 1.0) * (3.0*lam1 - 2.0)
    N1 = 0.5 * lam2 * (3.0*lam2 - 1.0) * (3.0*lam2 - 2.0)
    N2 = 0.5 * lam3 * (3.0*lam3 - 1.0) * (3.0*lam3 - 2.0)
    
    # ========================================================================
    # EDGE NODES - Cubic Lagrange interpolation
    # ========================================================================
    
    # Edge 0→1 (nodes 3, 4)
    # Node 3 at t=1/3: passes through (λ₁=2/3, λ₂=1/3, λ₃=0)
    # Node 4 at t=2/3: passes through (λ₁=1/3, λ₂=2/3, λ₃=0)
    N3 = (9.0/2.0) * lam1 * lam2 * (3.0*lam1 - 1.0)  # t=1/3 on edge 0→1
    N4 = (9.0/2.0) * lam1 * lam2 * (3.0*lam2 - 1.0)  # t=2/3 on edge 0→1
    
    # Edge 1→2 (nodes 5, 6)
    # Node 5 at t=1/3: passes through (λ₁=0, λ₂=2/3, λ₃=1/3)
    # Node 6 at t=2/3: passes through (λ₁=0, λ₂=1/3, λ₃=2/3)
    N5 = (9.0/2.0) * lam2 * lam3 * (3.0*lam2 - 1.0)  # t=1/3 on edge 1→2
    N6 = (9.0/2.0) * lam2 * lam3 * (3.0*lam3 - 1.0)  # t=2/3 on edge 1→2
    
    # Edge 2→0 (nodes 7, 8)
    # Node 7 at t=1/3: passes through (λ₁=1/3, λ₂=0, λ₃=2/3)
    # Node 8 at t=2/3: passes through (λ₁=2/3, λ₂=0, λ₃=1/3)
    N7 = (9.0/2.0) * lam3 * lam1 * (3.0*lam3 - 1.0)  # t=1/3 on edge 2→0
    N8 = (9.0/2.0) * lam3 * lam1 * (3.0*lam1 - 1.0)  # t=2/3 on edge 2→0
    
    # ========================================================================
    # INTERIOR NODE (9) - Bubble function
    # ========================================================================
    # Centered at (λ₁=1/3, λ₂=1/3, λ₃=1/3) - the centroid
    # Formula: N₉ = 27λ₁λ₂λ₃
    N9 = 27.0 * lam1 * lam2 * lam3
    
    return jnp.array([N0, N1, N2, N3, N4, N5, N6, N7, N8, N9])

# ============================================================================
# P3 Shape Function Gradients (First Derivatives)
# ============================================================================

@jit
def compute_p3_shape_gradients_reference(xi: float, eta: float) -> jnp.ndarray:
    """
    Returns (10,2) array of ∂N_i/∂(xi,eta) evaluated at (xi,eta).
    Implemented by JAX jacobian of compute_p3_shape_functions to guarantee consistency.
    """
    def Nfun(x):
        # x is shape (2,) array: [xi, eta]
        return compute_p3_shape_functions(x[0], x[1])   # returns (10,)
    J = jax.jacobian(Nfun)(jnp.array([xi, eta]))  # shape (10,2)
    return J


# ============================================================================
# Jacobian Computation
# ============================================================================

@jit
def compute_jacobian_p3(xi: float, eta: float, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Jacobian matrix for P3 element transformation
    
    J = [[∂x/∂ξ,  ∂y/∂ξ ],
         [∂x/∂η,  ∂y/∂η]]
    
    For P3 elements, x and y are cubic polynomials of ξ,η
    so this mapping is generally NON-AFFINE (curved edges possible)
    
    Args:
        xi, eta: Point in reference triangle
        coords: (10, 2) physical coordinates of the 10 nodes
        
    Returns:
        (2, 2) Jacobian matrix
    """
    # Get shape function gradients in reference coordinates
    dN_dref = compute_p3_shape_gradients_reference(xi, eta)  # (10, 2)
    
    # x(ξ,η) = Σᵢ Nᵢ(ξ,η) xᵢ
    # ∂x/∂ξ = Σᵢ (∂Nᵢ/∂ξ) xᵢ
    # ∂y/∂η = Σᵢ (∂Nᵢ/∂η) yᵢ
    
    # J = coords.T @ dN_dref
    # coords.T is (2, 10), dN_dref is (10, 2) → result is (2, 2)
    J = jnp.dot(coords.T, dN_dref)
    
    return J


@jit
def compute_p3_shape_gradients_physical(xi: float, eta: float,
                                        coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute gradients of P3 shape functions in physical coordinates
    
    Uses chain rule: [∂N/∂x, ∂N/∂y] = [∂N/∂ξ, ∂N/∂η] @ J⁻¹
    
    Args:
        xi, eta: Point in reference triangle
        coords: (10, 2) physical coordinates of nodes
        
    Returns:
        (10, 2) array where row i is [∂Nᵢ/∂x, ∂Nᵢ/∂y]
    """
    # Gradients in reference coordinates
    dN_dref = compute_p3_shape_gradients_reference(xi, eta)  # (10, 2)
    
    # Jacobian
    J = compute_jacobian_p3(xi, eta, coords)  # (2, 2)
    J_inv = jnp.linalg.inv(J)
    
    # Transform to physical coordinates
    # dN_dphys = dN_dref @ J_inv
    dN_dphys = jnp.dot(dN_dref, J_inv)  # (10, 2)
    
    return dN_dphys


# ============================================================================
# Validation Functions
# ============================================================================

def validate_p3_shape_functions():
    """
    Validate P3 shape functions have correct properties
    
    Tests:
    1. Partition of unity: Σᵢ Nᵢ(ξ,η) = 1 everywhere
    2. Kronecker delta: Nᵢ(xⱼ) = δᵢⱼ
    3. Gradient consistency
    """
    print("=" * 70)
    print("P3 SHAPE FUNCTION VALIDATION")
    print("=" * 70)
    
    # Reference node positions
    ref_nodes = jnp.array([
        [0.0, 0.0],         # 0: vertex
        [1.0, 0.0],         # 1: vertex
        [0.0, 1.0],         # 2: vertex
        [1.0/3.0, 0.0],     # 3: edge 0→1, t=1/3
        [2.0/3.0, 0.0],     # 4: edge 0→1, t=2/3
        [2.0/3.0, 1.0/3.0], # 5: edge 1→2, t=1/3
        [1.0/3.0, 2.0/3.0], # 6: edge 1→2, t=2/3
        [0.0, 2.0/3.0],     # 7: edge 2→0, t=1/3
        [0.0, 1.0/3.0],     # 8: edge 2→0, t=2/3
        [1.0/3.0, 1.0/3.0]  # 9: interior (centroid)
    ])
    
    # Test 1: Partition of unity
    print("\n1. Testing partition of unity (ΣNᵢ = 1)...")
    test_points = [
        (0.25, 0.25),
        (0.1, 0.2),
        (0.5, 0.3),
        (1.0/3.0, 1.0/3.0),  # Centroid
        (0.1, 0.1),
    ]
    
    max_unity_error = 0.0
    for xi, eta in test_points:
        N = compute_p3_shape_functions(xi, eta)
        sum_N = jnp.sum(N)
        error = abs(float(sum_N) - 1.0)
        max_unity_error = max(max_unity_error, error)
        status = "✅" if error < 1e-6 else "❌"
        print(f"  ({xi:.3f}, {eta:.3f}): ΣN = {sum_N:.15f}, error = {error:.2e} {status}")
    
    # Test 2: Kronecker delta property
    print("\n2. Testing Kronecker delta (Nᵢ(xⱼ) = δᵢⱼ)...")
    max_delta_error = 0.0
    for i, (xi, eta) in enumerate(ref_nodes):
        N = compute_p3_shape_functions(xi, eta)
        
        # Check diagonal
        diag_error = abs(float(N[i]) - 1.0)
        max_delta_error = max(max_delta_error, diag_error)
        
        # Check off-diagonal
        for j in range(10):
            if i != j:
                off_diag_error = abs(float(N[j]))
                max_delta_error = max(max_delta_error, off_diag_error)
        
        status = "✅" if diag_error < 1e-6 else "❌"
        print(f"  Node {i}: N[{i}] = {N[i]:.15f}, max(N[j≠{i}]) = {max([abs(float(N[j])) for j in range(10) if j!=i]):.2e} {status}")
    
    # Test 3: Gradient sum = 0 (consistency)
    print("\n3. Testing gradient consistency (Σ∂Nᵢ/∂ξ = 0, Σ∂Nᵢ/∂η = 0)...")
    max_grad_error = 0.0
    for xi, eta in test_points:
        dN = compute_p3_shape_gradients_reference(xi, eta)
        sum_dxi = jnp.sum(dN[:, 0])
        sum_deta = jnp.sum(dN[:, 1])
        
        error_xi = abs(float(sum_dxi))
        error_eta = abs(float(sum_deta))
        max_grad_error = max(max_grad_error, error_xi, error_eta)
        
        status = "✅" if error_xi < 1e-6 and error_eta < 1e-6 else "❌"
        print(f"  ({xi:.3f}, {eta:.3f}): Σ∂N/∂ξ = {sum_dxi:.2e}, Σ∂N/∂η = {sum_deta:.2e} {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Max partition of unity error:  {max_unity_error:.2e}")
    print(f"Max Kronecker delta error:     {max_delta_error:.2e}")
    print(f"Max gradient consistency error: {max_grad_error:.2e}")
    
    # Tolerances appropriate for cubic polynomials with floating-point arithmetic
    all_passed = (max_unity_error < 1e-6 and 
                  max_delta_error < 1e-6 and 
                  max_grad_error < 1e-6)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED! P3 shape functions are correct!")
    else:
        print("\n❌ SOME TESTS FAILED - review implementation")
    
    print("=" * 70)
    
    return all_passed


# ============================================================================
# Visualization
# ============================================================================

def visualize_p3_basis_functions():
    """
    Create visualization of P3 basis functions
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    print("\nGenerating P3 basis function visualizations...")
    
    # Create evaluation grid
    n_pts = 50
    xi_vals = np.linspace(0, 1, n_pts)
    eta_vals = np.linspace(0, 1, n_pts)
    
    # Evaluate all basis functions on grid
    basis_vals = np.zeros((10, n_pts, n_pts))
    
    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            if xi + eta <= 1.0:  # Inside reference triangle
                N = compute_p3_shape_functions(xi, eta)
                basis_vals[:, j, i] = np.array(N)
            else:
                basis_vals[:, j, i] = np.nan
    
    # Create figure with all 10 basis functions
    fig = plt.figure(figsize=(20, 8))
    
    node_labels = [
        'N₀ (vertex 0)',
        'N₁ (vertex 1)', 
        'N₂ (vertex 2)',
        'N₃ (edge 0→1, t=1/3)',
        'N₄ (edge 0→1, t=2/3)',
        'N₅ (edge 1→2, t=1/3)',
        'N₆ (edge 1→2, t=2/3)',
        'N₇ (edge 2→0, t=1/3)',
        'N₈ (edge 2→0, t=2/3)',
        'N₉ (interior)'
    ]
    
    for idx in range(10):
        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
        
        XI, ETA = np.meshgrid(xi_vals, eta_vals)
        
        surf = ax.plot_surface(XI, ETA, basis_vals[idx], 
                              cmap=cm.viridis, alpha=0.9,
                              edgecolor='none')
        
        ax.set_xlabel('ξ', fontsize=10)
        ax.set_ylabel('η', fontsize=10)
        ax.set_zlabel(f'N_{idx}', fontsize=10)
        ax.set_title(node_labels[idx], fontsize=11, pad=10)
        ax.set_zlim([0, 1])
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('P3 Cubic Basis Functions (10 DOF per element)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('p3_basis_functions.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: p3_basis_functions.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print(" " * 20 + "P3 SHAPE FUNCTIONS - VALIDATION")
    print("🎯" * 35 + "\n")
    
    # Run validation
    success = validate_p3_shape_functions()
    
    # Generate visualizations
    visualize_p3_basis_functions()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ P3 SHAPE FUNCTIONS READY!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. ✅ Shape functions validated")
        print("  2. ⏳ Create P3 mesh generator")
        print("  3. ⏳ Implement P3 assembly")
        print("  4. ⏳ Add P3 shear computation")
        print("=" * 70)
    else:
        print("\n❌ Validation failed - review implementation")
