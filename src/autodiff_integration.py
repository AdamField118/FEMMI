"""
Automatic Differentiation Integration for Weak Lensing FEM

Current Status: P1 Elements Only
- ✅ Differentiable forward model: κ → ψ
- ⏳ Shear computation: Awaiting P3 implementation
- ⏳ Inverse problems: Awaiting P3 shear gradients

Future (P3):
- Fully differentiable: κ → ψ → γ
- Gradient-based mass reconstruction
- Bayesian inference with UQ
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from functools import partial
from typing import Tuple, Callable, Dict
import time
import numpy as np
jax.config.update("jax_enable_x64", True)

# ============================================================================
# Differentiable Forward Model (P1 - Potential Only)
# ============================================================================

@jit
def forward_model_potential(kappa: jnp.ndarray, mesh) -> jnp.ndarray:
    """
    Differentiable forward model: κ → ψ (P1 elements)
    
    This is the MINIMAL differentiable version that returns only ψ.
    Use this when you need gradients w.r.t. κ.
    
    Args:
        kappa: (n_nodes,) convergence field
        mesh: Mesh object (P1 triangular elements)
        
    Returns:
        psi: (n_nodes,) lensing potential
    """
    from .fem_solver import assemble_system, apply_dirichlet_bc, conjugate_gradient
    
    # Assemble system
    K, F = assemble_system(mesh, kappa)
    
    # Apply boundary conditions
    K_bc, F_bc = apply_dirichlet_bc(K, F, mesh.boundary)
    
    # Solve (CG is differentiable!)
    psi, _, _ = conjugate_gradient(K_bc, F_bc, tol=1e-6, maxiter=1000)
    
    return psi


# ============================================================================
# Gradient Computation (Potential Only)
# ============================================================================

def compute_gradient_potential(kappa: jnp.ndarray, mesh) -> jnp.ndarray:
    """
    Compute ∂ψ/∂κ using automatic differentiation
    
    Returns Jacobian: (n_nodes, n_nodes) where [i,j] = ∂ψ_i/∂κ_j
    
    Args:
        kappa: (n_nodes,) convergence field
        mesh: Mesh object (P1 elements)
        
    Returns:
        jacobian: (n_nodes, n_nodes) sensitivity matrix
    """
    # Use JAX jacobian for full Jacobian matrix
    jac_fn = jax.jacobian(forward_model_potential, argnums=0)
    
    # This computes ∂ψ/∂κ
    jacobian = jac_fn(kappa, mesh)
    
    return jacobian


def compute_gradient_at_point(kappa: jnp.ndarray, mesh, 
                              node_idx: int) -> jnp.ndarray:
    """
    Compute gradient of ψ at a single node w.r.t. all κ values
    
    More efficient than computing full Jacobian when you only need
    one row: ∂ψ[node_idx]/∂κ
    
    Args:
        kappa: (n_nodes,) convergence field
        mesh: Mesh object
        node_idx: Index of node to compute gradient for
        
    Returns:
        gradient: (n_nodes,) ∂ψ[node_idx]/∂κ
    """
    def psi_at_node(kappa):
        psi = forward_model_potential(kappa, mesh)
        return psi[node_idx]
    
    grad_fn = jax.grad(psi_at_node)
    gradient = grad_fn(kappa)
    
    return gradient


# ============================================================================
# Validation Functions
# ============================================================================

def validate_potential_gradients(kappa: jnp.ndarray, mesh,
                                 n_samples: int = 5,
                                 epsilon: float = 1e-5,
                                 verbose: bool = True) -> Dict:
    """
    Validate autodiff gradients for potential against finite differences
    
    Compares ∂ψ[i]/∂κ[j] from autodiff vs finite differences on random samples
    
    Args:
        kappa: Convergence field
        mesh: Mesh object
        n_samples: Number of gradient components to check
        epsilon: FD step size
        verbose: Print detailed comparison
        
    Returns:
        dict with validation results
    """
    print("=" * 70)
    print("GRADIENT VALIDATION: Potential Gradients (∂ψ/∂κ)")
    print("=" * 70)
    
    # Solve forward model
    psi = forward_model_potential(kappa, mesh)
    n = len(kappa)
    
    # Select random node and kappa indices to test
    np.random.seed(42)
    node_indices = np.random.choice(n, size=min(n_samples, n), replace=False)
    kappa_indices = np.random.choice(n, size=min(n_samples, n), replace=False)
    
    print(f"\nTesting {n_samples} gradient components...")
    print(f"{'Node i':>8} {'κ j':>8} {'Autodiff':>15} {'Finite Diff':>15} {'Rel Error':>12}")
    print("-" * 70)
    
    results = {
        'auto': [],
        'fd': [],
        'rel_error': [],
        'abs_error': []
    }
    
    for node_i, kappa_j in zip(node_indices, kappa_indices):
        # Autodiff gradient: ∂ψ[node_i]/∂κ[kappa_j]
        grad_auto = compute_gradient_at_point(kappa, mesh, node_i)
        g_auto = grad_auto[kappa_j]
        
        # Finite difference
        kappa_pert = kappa.at[kappa_j].add(epsilon)
        psi_pert = forward_model_potential(kappa_pert, mesh)
        g_fd = (psi_pert[node_i] - psi[node_i]) / epsilon
        
        # Compare
        abs_err = float(jnp.abs(g_auto - g_fd))
        rel_err = abs_err / (jnp.abs(g_fd) + 1e-10)
        
        results['auto'].append(float(g_auto))
        results['fd'].append(float(g_fd))
        results['abs_error'].append(abs_err)
        results['rel_error'].append(rel_err)
        
        if verbose:
            print(f"{node_i:8d} {kappa_j:8d} {g_auto:15.6e} {g_fd:15.6e} {rel_err:12.2e}")
    
    # Statistics
    mean_rel_err = np.mean(results['rel_error'])
    max_rel_err = np.max(results['rel_error'])
    
    print("-" * 70)
    print(f"\nValidation Summary:")
    print(f"  Mean relative error: {mean_rel_err:.6e}")
    print(f"  Max relative error:  {max_rel_err:.6e}")
    
    # Pass/fail
    tolerance = 1e-4
    passed = max_rel_err < tolerance
    
    print(f"\n  Result: {'✅ PASS' if passed else '❌ FAIL'} "
          f"(tolerance = {tolerance:.0e})")
    print("=" * 70)
    
    results['mean_rel_error'] = mean_rel_err
    results['max_rel_error'] = max_rel_err
    results['passed'] = passed
    
    return results


# ============================================================================
# Performance Profiling
# ============================================================================

def profile_forward_backward_potential(kappa: jnp.ndarray, mesh,
                                      n_trials: int = 10) -> Dict:
    """
    Profile performance of forward and backward passes for potential
    
    Measures:
    - Forward pass time (κ → ψ)
    - Backward pass time (∂ψ/∂κ for one node)
    
    Args:
        kappa: Convergence field
        mesh: Mesh object
        n_trials: Number of timing trials
        
    Returns:
        dict with timing results
    """
    print("=" * 70)
    print("PERFORMANCE PROFILING: Potential Gradients")
    print("=" * 70)
    
    # Compile everything first
    print("\nWarmup (JIT compilation)...")
    _ = forward_model_potential(kappa, mesh)
    _ = compute_gradient_at_point(kappa, mesh, 0)
    print("✅ Warmup complete\n")
    
    # Forward pass only
    print(f"1. Forward pass (κ → ψ) [{n_trials} trials]...")
    times_fwd = []
    for i in range(n_trials):
        t0 = time.time()
        psi = forward_model_potential(kappa, mesh)
        jax.block_until_ready(psi)  # Wait for GPU
        times_fwd.append(time.time() - t0)
    
    t_fwd = np.median(times_fwd)
    print(f"   Median time: {t_fwd*1000:.2f} ms")
    
    # Backward pass (gradient at one node)
    print(f"\n2. Backward pass (∂ψ[i]/∂κ for single node) [{n_trials} trials]...")
    times_bwd = []
    node_idx = mesh.n_nodes // 2  # Middle node
    for i in range(n_trials):
        t0 = time.time()
        grad = compute_gradient_at_point(kappa, mesh, node_idx)
        jax.block_until_ready(grad)
        times_bwd.append(time.time() - t0)
    
    t_bwd = np.median(times_bwd)
    print(f"   Median time: {t_bwd*1000:.2f} ms")
    
    # Analysis
    print(f"\n3. Analysis:")
    print(f"   Forward:  {t_fwd*1000:6.2f} ms")
    print(f"   Backward: {t_bwd*1000:6.2f} ms  ({t_bwd/t_fwd:.2f}× forward)")
    print(f"   Note: Backward includes forward pass (autodiff)")
    
    print("=" * 70)
    
    return {
        'forward_ms': t_fwd * 1000,
        'backward_ms': t_bwd * 1000,
        'backward_overhead': t_bwd / t_fwd,
    }


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate_autodiff():
    """
    Demonstration of autodiff capabilities (P1 elements - potential only)
    
    NOTE: Full shear→mass reconstruction requires P3 elements.
    This demo shows potential ψ gradients only.
    """
    print("\n" + "🚀" * 35)
    print(" " * 20 + "AUTODIFF DEMONSTRATION (P1)")
    print("🚀" * 35)
    
    # Setup
    print("\nSetup: Creating synthetic problem...")
    from .fem_solver import GaussianLens, solve_lensing_poisson
    from .mesh_generator import generate_structured_mesh
    
    lens = GaussianLens(amplitude=1.0, sigma=0.3)
    mesh = generate_structured_mesh(20, 20, xmin=-1, xmax=1, ymin=-1, ymax=1)
    
    print(f"  Mesh: {mesh.n_nodes} nodes (P1 elements)")
    
    # Generate "true" convergence
    kappa_true = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    
    # Solve forward model (potential only for now)
    print("\nSolving forward model (κ → ψ)...")
    solution = solve_lensing_poisson(mesh, kappa_true, verbose=False)
    
    print(f"  Max |ψ|: {jnp.max(jnp.abs(solution.psi)):.4f}")
    print(f"  Max |α|: {jnp.max(jnp.linalg.norm(solution.alpha, axis=1)):.4f}")
    
    # Validate gradients
    print("\n" + "=" * 70)
    print("Testing autodiff gradients...")
    print("=" * 70)
    validate_potential_gradients(kappa_true, mesh, n_samples=5)
    
    # Profile performance
    print("\n" + "=" * 70)
    print("Profiling performance...")
    print("=" * 70)
    profile_forward_backward_potential(kappa_true, mesh, n_trials=5)
    
    print("\n" + "=" * 70)
    print("✅ Autodiff framework ready!")
    print("=" * 70)
    print("\nCurrent capabilities:")
    print("  1. ✅ Differentiable forward model: κ → ψ")
    print("  2. ✅ Potential gradients: ∂ψ/∂κ")
    print("  3. ✅ Validated against finite differences")
    print("  4. ✅ Performance profiled")
    print("\nLimitations (P1 elements):")
    print("  ⚠️  Shear γ = ∇²ψ not available (P1 → constant ∇²ψ = 0)")
    print("  ⚠️  Cannot do shear→mass reconstruction yet")
    print("\nNext steps:")
    print("  → Implement P3 elements for O(h⁴) potential accuracy")
    print("  → Add P3 shear computation: γ with O(h²) convergence")
    print("  → Complete differentiable shear→mass pipeline")
    print("  → Bayesian inference with Laplace approximation")
    print("=" * 70)


# ============================================================================
# TODO: Shear-based functions (requires P3 implementation)
# ============================================================================

"""
The following functions will be implemented after P3 elements are added:

1. forward_model_shear(kappa, mesh) -> (gamma1, gamma2)
   - Differentiable κ → ψ → γ pipeline
   - Requires P3 second derivatives

2. differentiable_loss(kappa, gamma_obs, mesh) -> loss
   - Loss function: ||γ_pred - γ_obs||²
   - For inverse problem optimization

3. compute_gradient_loss(kappa, gamma_obs, mesh) -> ∂L/∂κ
   - Gradient descent for mass reconstruction
   - Automatic differentiation through full pipeline

4. hessian_vector_product(kappa, v, gamma_obs, mesh) -> Hv
   - For Newton-CG optimization
   - Laplace approximation for UQ

5. compute_fisher_information(kappa, mesh) -> F
   - Fisher information matrix
   - Posterior covariance estimation

See ISP document for complete implementation plan.
"""


if __name__ == "__main__":
    demonstrate_autodiff()
