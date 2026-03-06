"""
examples/demo_p3_pipeline.py
============================
Full weak lensing forward model using P3 cubic elements.

    κ → ψ (FEM Poisson solve) → α = ∇ψ → γ₁, γ₂ (via JAX Hessian)

Run from project root:
    bash ./run.sh examples/demo_p3_pipeline.py
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from femmi.p3_mesh_generator  import generate_p3_structured_mesh
from femmi.p3_assembly        import assemble_system_p3, apply_boundary_conditions_p3
from femmi.p3_shape_functions import (
    compute_p3_shape_functions,
    compute_p3_shape_gradients_reference,
)


# ══════════════════════════════════════════════════════════════════════════════
# JAX-AUTODIFF SHEAR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_P3_REF_NODES = np.array([
    [0.0,       0.0      ],
    [1.0,       0.0      ],
    [0.0,       1.0      ],
    [1.0/3.0,   0.0      ],
    [2.0/3.0,   0.0      ],
    [2.0/3.0,   1.0/3.0  ],
    [1.0/3.0,   2.0/3.0  ],
    [0.0,       2.0/3.0  ],
    [0.0,       1.0/3.0  ],
    [1.0/3.0,   1.0/3.0  ],
])


def _build_ref_hessians() -> np.ndarray:
    """Precompute H_ref[eval_pt, shape_fn, i, j] via JAX forward-over-reverse."""
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])
    hess_fn = jax.jacfwd(jax.jacrev(N_vec))
    return np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])


def compute_shear_p3(mesh, psi: np.ndarray, H_ref: np.ndarray):
    """
    Compute shear (γ₁, γ₂) at every mesh node from a P3 solution.

        γ₁ = ½(∂²ψ/∂x² − ∂²ψ/∂y²)
        γ₂ = ∂²ψ/∂x∂y
    """
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes  = len(nodes)
    g1_sum = np.zeros(n_nodes); g2_sum = np.zeros(n_nodes)
    count  = np.zeros(n_nodes, dtype=int)

    for elem in elements:
        x0,y0=nodes[elem[0]]; x1,y1=nodes[elem[1]]; x2,y2=nodes[elem[2]]
        J = np.array([[x1-x0,y1-y0],[x2-x0,y2-y0]])
        A = np.linalg.inv(J).T
        psi_e = psi[elem]

        for local_i in range(10):
            # 'ja,kb,njk->nab' — correct einsum (A[j,a] not A[a,j])
            H_phys = np.einsum('ja,kb,njk->nab', A, A, H_ref[local_i])
            psi_xx = psi_e @ H_phys[:, 0, 0]
            psi_yy = psi_e @ H_phys[:, 1, 1]
            psi_xy = psi_e @ H_phys[:, 0, 1]
            gi = elem[local_i]
            g1_sum[gi] += 0.5 * (psi_xx - psi_yy)
            g2_sum[gi] += psi_xy
            count[gi]  += 1

    count = np.maximum(count, 1)
    return g1_sum / count, g2_sum / count


def compute_deflection_p3(mesh, psi: np.ndarray) -> np.ndarray:
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes  = len(nodes)
    a_sum = np.zeros((n_nodes, 2)); count = np.zeros(n_nodes, dtype=int)

    for elem in elements:
        x0,y0=nodes[elem[0]]; x1,y1=nodes[elem[1]]; x2,y2=nodes[elem[2]]
        J = np.array([[x1-x0,y1-y0],[x2-x0,y2-y0]])
        J_inv = np.linalg.inv(J); psi_e = psi[elem]
        for local_i, (xi, eta) in enumerate(_P3_REF_NODES):
            dN_ref  = np.array(compute_p3_shape_gradients_reference(xi, eta))
            dN_phys = dN_ref @ J_inv.T
            grad_psi = dN_phys.T @ psi_e
            gi = elem[local_i]
            a_sum[gi] += grad_psi; count[gi] += 1

    count = np.maximum(count, 1)
    return a_sum / count[:, None]


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTIC REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_shear_exact(x, y, A, sigma):
    r2     = x**2 + y**2
    kernel = A * np.exp(-r2 / (2*sigma**2))
    return kernel * (y**2 - x**2) / (2*sigma**2), -kernel * x * y / sigma**2


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def solve_pipeline(kappa_fn, nx=30, domain=(-2.5, 2.5, -2.5, 2.5), label=""):
    xmin, xmax, ymin, ymax = domain
    print(f"\n{'═'*55}\n  {label or 'P3 Pipeline'}  |  {nx}×{nx} cells\n{'═'*55}")

    mesh  = generate_p3_structured_mesh(nx, nx, xmin, xmax, ymin, ymax)
    nodes = np.array(mesh.nodes)
    x, y  = nodes[:,0], nodes[:,1]
    kappa = kappa_fn(x, y)

    K, F       = assemble_system_p3(mesh, kappa)
    K_bc, F_bc = apply_boundary_conditions_p3(K, F, mesh)
    psi        = spla.spsolve(K_bc, F_bc)
    print(f"  residual={np.linalg.norm(K_bc@psi-F_bc):.2e}  max|ψ|={np.abs(psi).max():.4f}")

    alpha  = compute_deflection_p3(mesh, psi)
    H_ref  = _build_ref_hessians()
    g1, g2 = compute_shear_p3(mesh, psi, H_ref)
    print(f"  max|α|={np.linalg.norm(alpha,axis=1).max():.4f}  max|γ₁|={np.abs(g1).max():.4f}  max|γ₂|={np.abs(g2).max():.4f}")

    return dict(mesh=mesh, nodes=nodes, kappa=kappa, psi=psi,
                alpha=alpha, gamma1=g1, gamma2=g2,
                gamma_mag=np.sqrt(g1**2+g2**2))


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _tri(res):
    nd = res['nodes']; el = np.array(res['mesh'].elements)[:,:3]
    return mtri.Triangulation(nd[:,0], nd[:,1], triangles=el)


def _panel(fig, ax, tri, data, title, cmap, symmetric=False):
    if symmetric:
        vmax = np.percentile(np.abs(data[np.isfinite(data)]), 98); vmin = -vmax
    else:
        vmin = np.percentile(data[np.isfinite(data)],  1)
        vmax = np.percentile(data[np.isfinite(data)], 99)
    tcf = ax.tricontourf(tri, data, levels=40, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.tricontour(tri, data, levels=8, colors='w', linewidths=0.3, alpha=0.4)
    cb = fig.colorbar(tcf, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=7, colors='#aaa')
    ax.set_title(title, fontsize=10, color='#ddd', pad=5)
    ax.set_aspect('equal'); ax.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x [θ_E]', fontsize=8, color='#aaa')
    ax.set_ylabel('y [θ_E]', fontsize=8, color='#aaa')
    ax.plot(0, 0, '+w', ms=8, mew=1.5)


def plot_main(res, title="P3 FEM Weak Lensing Pipeline", out="p3_pipeline.png"):
    nodes = res['nodes']; tri = _tri(res)
    alpha_mag = np.linalg.norm(res['alpha'], axis=1)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18,11)); fig.patch.set_facecolor('#0e0e0e')
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#eee', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.05, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
    for ax in axes: ax.set_facecolor('#1a1a1a')

    for ax, (data,title_,cmap,sym) in zip(axes, [
        (res['kappa'],     'κ — convergence',         'hot',    False),
        (res['psi'],       'ψ — lensing potential',   'viridis',False),
        (alpha_mag,        '|α| — deflection',        'plasma', False),
        (res['gamma1'],    'γ₁ = ½(ψ_xx − ψ_yy)',    'RdBu_r', True),
        (res['gamma2'],    'γ₂ = ψ_xy',               'RdBu_r', True),
        (res['gamma_mag'], '|γ| = √(γ₁²+γ₂²)',       'YlOrRd', False),
    ]):
        _panel(fig, ax, tri, data, title_, cmap, symmetric=sym)

    # Quiver overlays
    step = max(1, len(nodes)//500); idx = np.arange(0, len(nodes), step)
    axes[2].quiver(nodes[idx,0], nodes[idx,1],
                   res['alpha'][idx,0], res['alpha'][idx,1],
                   alpha_mag[idx], cmap='cool', scale=alpha_mag[idx].max()*12,
                   width=0.003, headwidth=3, alpha=0.75)
    step2 = max(1, len(nodes)//350); idx2 = np.arange(0, len(nodes), step2)
    phi = 0.5*np.arctan2(res['gamma2'][idx2], res['gamma1'][idx2])
    gm  = res['gamma_mag'][idx2]
    axes[5].quiver(nodes[idx2,0], nodes[idx2,1], gm*np.cos(phi), gm*np.sin(phi),
                   color='w', scale=gm.max()*18, width=0.0025,
                   headwidth=0, headlength=0, headaxislength=0, alpha=0.6, pivot='middle')

    plt.savefig(out, dpi=180, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"{out}"); plt.close()


def plot_validation(res, A, sigma, out="p3_shear_validation.png"):
    nodes = res['nodes']; x, y = nodes[:,0], nodes[:,1]; tri = _tri(res)
    g1_ex, g2_ex = gaussian_shear_exact(x, y, A, sigma)
    r1 = res['gamma1'] - g1_ex; r2 = res['gamma2'] - g2_ex
    rms1 = np.sqrt(np.mean(r1**2)); rms2 = np.sqrt(np.mean(r2**2))

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18,11)); fig.patch.set_facecolor('#0e0e0e')
    fig.suptitle("P3 Shear Validation — FEM vs Analytic Gaussian", fontsize=15, color='#eee')
    for ax in axes.flat: ax.set_facecolor('#1a1a1a')

    for row, (fem, exact, resid, lbl, rms) in enumerate([
        (res['gamma1'], g1_ex, r1, 'γ₁', rms1),
        (res['gamma2'], g2_ex, r2, 'γ₂', rms2),
    ]):
        vmax = np.percentile(np.abs(exact), 98); rmax = np.percentile(np.abs(resid), 98)
        for col, (data, ttl, vm, cmap) in enumerate([
            (fem,   f'{lbl} FEM (P3)',                      vmax, 'RdBu_r'),
            (exact, f'{lbl} Analytic',                       vmax, 'RdBu_r'),
            (resid, f'{lbl} FEM−Analytic  (RMS={rms:.3f})', rmax, 'PuOr'),
        ]):
            _panel(fig, axes[row, col], tri, data, ttl, cmap, symmetric=True)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out, dpi=180, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"{out}"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    A, sigma = 1.0, 0.5

    res1 = solve_pipeline(
        lambda x,y: A*np.exp(-(x**2+y**2)/(2*sigma**2)),
        nx=30, domain=(-2.5,2.5,-2.5,2.5),
        label="Single Gaussian (A=1, σ=0.5)",
    )
    plot_main(res1, title="P3 FEM Weak Lensing — Gaussian Cluster",
              out="p3_pipeline_gaussian.png")
    plot_validation(res1, A, sigma)

    res2 = solve_pipeline(
        lambda x,y: (1.2*np.exp(-((x-1.0)**2+(y+0.3)**2)/(2*0.4**2)) +
                     0.9*np.exp(-((x+1.2)**2+(y-0.5)**2)/(2*0.5**2))),
        nx=30, domain=(-3.0,3.0,-3.0,3.0), label="Two Clusters",
    )
    plot_main(res2, title="P3 FEM — Two-Cluster System", out="p3_two_cluster.png")

    print("\nOutput files:")
    print("  p3_pipeline_gaussian.png  — 6-panel κ/ψ/α/γ₁/γ₂/|γ|")
    print("  p3_shear_validation.png   — FEM vs analytic shear")
    print("  p3_two_cluster.png        — two-cluster system")