"""
Simple example: Solving weak lensing Poisson equation with JAX FEM

This demonstrates the basic workflow:
1. Create a mesh
2. Define convergence field  kappa
3. Solve  nabla^2 psi = 2 kappa
4. Extract deflection  alpha and visualize
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fem_solver import solve_lensing_poisson, GaussianLens
from src.mesh_generator import generate_structured_mesh


def simple_cluster_example():
    """
    Simple example: Galaxy cluster modeled as Gaussian
    """
    print("=" * 70)
    print("EXAMPLE: Galaxy Cluster Weak Lensing")
    print("=" * 70)
    
    # Step 1: Create mesh
    print("\nStep 1: Creating mesh...")
    nx, ny = 50, 50
    mesh = generate_structured_mesh(nx, ny, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0)
    print(f"  Created {nx}×{ny} mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Step 2: Define mass distribution (convergence)
    print("\nStep 2: Defining mass distribution...")
    lens = GaussianLens(amplitude=1.5, sigma=0.4)
    
    # Evaluate convergence at mesh nodes
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    print(f"  Max  kappa: {jnp.max(kappa):.4f}")
    print(f"  Total mass ( int kappa): {jnp.sum(kappa) * 16.0 / mesh.n_nodes:.4f}")  # Approximate integral
    
    # Step 3: Solve FEM system
    print("\nStep 3: Solving lensing Poisson equation...")
    solution = solve_lensing_poisson(mesh, kappa, tol=1e-6, maxiter=1000, verbose=True)
    
    # Step 4: Analyze results
    print("\nStep 4: Solution analysis...")
    print(f"  Max | psi|: {jnp.max(jnp.abs(solution.psi)):.6f}")
    
    alpha_mag = jnp.sqrt(jnp.sum(solution.alpha**2, axis=1))
    print(f"  Max | alpha|: {jnp.max(alpha_mag):.6f}")
    print(f"  Mean | alpha|: {jnp.mean(alpha_mag):.6f}")
    
    # Step 5: Visualize
    print("\nStep 5: Creating visualizations...")
    plot_results(mesh, solution)
    
    print("\n" + "=" * 70)
    print("Example complete! Check 'cluster_example.png' for visualization.")
    print("=" * 70)
    
    return mesh, solution


def plot_results(mesh, solution):
    """
    Create a nice 2x2 plot showing all solution components
    """
    from matplotlib.tri import Triangulation
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 14))
    
    # Create triangulation
    triang = Triangulation(
        np.array(mesh.nodes[:, 0]),
        np.array(mesh.nodes[:, 1]),
        np.array(mesh.elements)
    )
    
    # 1. Convergence  kappa (mass distribution)
    ax1 = plt.subplot(2, 2, 1)
    kappa_plot = np.array(solution.convergence)
    levels = np.linspace(0, kappa_plot.max(), 20)
    tcf = ax1.tricontourf(triang, kappa_plot, levels=levels, cmap='hot')
    ax1.triplot(triang, 'w-', alpha=0.05, linewidth=0.2)
    ax1.set_title('Convergence  kappa (Mass Distribution)', 
                  fontsize=16, color='#00ff41', fontweight='bold', pad=15)
    ax1.set_xlabel('x [θ]', fontsize=13)
    ax1.set_ylabel('y [θ]', fontsize=13)
    ax1.set_aspect('equal')
    cbar = plt.colorbar(tcf, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(' kappa', fontsize=13, rotation=0, labelpad=15)
    
    # 2. Lensing potential  psi
    ax2 = plt.subplot(2, 2, 2)
    psi_plot = np.array(solution.psi)
    levels = np.linspace(psi_plot.min(), psi_plot.max(), 20)
    tcf = ax2.tricontourf(triang, psi_plot, levels=levels, cmap='viridis')
    ax2.triplot(triang, 'w-', alpha=0.05, linewidth=0.2)
    ax2.set_title('Lensing Potential  psi', 
                  fontsize=16, color='#00ff41', fontweight='bold', pad=15)
    ax2.set_xlabel('x [θ]', fontsize=13)
    ax2.set_ylabel('y [θ]', fontsize=13)
    ax2.set_aspect('equal')
    cbar = plt.colorbar(tcf, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(' psi', fontsize=13, rotation=0, labelpad=15)
    
    # 3. Deflection magnitude | alpha|
    ax3 = plt.subplot(2, 2, 3)
    alpha_mag = np.sqrt(np.sum(np.array(solution.alpha)**2, axis=1))
    levels = np.linspace(0, alpha_mag.max(), 20)
    tcf = ax3.tricontourf(triang, alpha_mag, levels=levels, cmap='plasma')
    ax3.triplot(triang, 'w-', alpha=0.05, linewidth=0.2)
    ax3.set_title('Deflection Magnitude | alpha|', 
                  fontsize=16, color='#00ff41', fontweight='bold', pad=15)
    ax3.set_xlabel('x [θ]', fontsize=13)
    ax3.set_ylabel('y [θ]', fontsize=13)
    ax3.set_aspect('equal')
    cbar = plt.colorbar(tcf, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('| alpha| [θ]', fontsize=13, rotation=0, labelpad=20)
    
    # 4. Deflection field (quiver plot)
    ax4 = plt.subplot(2, 2, 4)
    
    # Subsample for clearer visualization
    skip = max(1, len(mesh.nodes) // 400)
    nodes_sub = mesh.nodes[::skip]
    alpha_sub = solution.alpha[::skip]
    alpha_mag_sub = alpha_mag[::skip]
    
    # Background: convergence
    tcf = ax4.tricontourf(triang, kappa_plot, levels=20, cmap='hot', alpha=0.3)
    
    # Arrows
    Q = ax4.quiver(
        np.array(nodes_sub[:, 0]),
        np.array(nodes_sub[:, 1]),
        np.array(alpha_sub[:, 0]),
        np.array(alpha_sub[:, 1]),
        alpha_mag_sub,
        cmap='cool',
        scale=8.0,
        width=0.004,
        headwidth=3,
        headlength=4,
        alpha=0.8
    )
    
    ax4.set_title('Deflection Field  alpha =  nabla psi', 
                  fontsize=16, color='#00ff41', fontweight='bold', pad=15)
    ax4.set_xlabel('x [θ]', fontsize=13)
    ax4.set_ylabel('y [θ]', fontsize=13)
    ax4.set_aspect('equal')
    cbar = plt.colorbar(Q, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('| alpha| [θ]', fontsize=13, rotation=0, labelpad=20)
    
    # Overall title
    fig.suptitle('JAX FEM Weak Lensing: Galaxy Cluster Example',
                 fontsize=20, color='#00ff41', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('cluster_example.png', dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    plt.close()


def multi_cluster_example():
    """
    More complex: Multiple galaxy clusters
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Multiple Clusters")
    print("=" * 70)
    
    # Mesh
    nx, ny = 60, 60
    mesh = generate_structured_mesh(nx, ny, xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0)
    
    # Multiple Gaussian clusters at different positions
    clusters = [
        {'center': (-1.0, 0.5), 'amplitude': 1.2, 'sigma': 0.3},
        {'center': (1.2, -0.8), 'amplitude': 1.5, 'sigma': 0.25},
        {'center': (0.3, 1.5), 'amplitude': 0.8, 'sigma': 0.4}
    ]
    
    # Combine convergence from all clusters
    kappa = jnp.zeros(mesh.n_nodes)
    for cluster in clusters:
        cx, cy = cluster['center']
        amp = cluster['amplitude']
        sig = cluster['sigma']
        
        kappa_cluster = jnp.array([
            amp * jnp.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sig**2))
            for x, y in mesh.nodes
        ])
        kappa = kappa + kappa_cluster
    
    print(f"Created {len(clusters)} clusters")
    print(f"Total mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"Max  kappa: {jnp.max(kappa):.4f}")
    
    # Solve
    solution = solve_lensing_poisson(mesh, kappa, verbose=True)
    
    # Plot
    plot_results(mesh, solution)
    print("\nMulti-cluster visualization saved to 'cluster_example.png'")
    
    return mesh, solution


if __name__ == "__main__":
    # Run simple example
    mesh, solution = simple_cluster_example()
    
    # Optionally run multi-cluster example
    # mesh2, solution2 = multi_cluster_example()
