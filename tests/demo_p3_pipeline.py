"""
demo_p3_pipeline.py
===================
Full weak lensing forward model using P3 cubic elements.

    κ (convergence)
      → solve ∇²ψ = 2κ  (P3 FEM)
      → α = ∇ψ           (deflection angles)
      → γ₁, γ₂ from ∂²ψ  (shear, via JAX autodiff on shape functions)

SHEAR APPROACH
--------------
Rather than deriving Hessian formulas by hand, we let JAX differentiate
the P3 shape functions twice:

    H_ref[i, j, k] = d²Nᵢ/d(ref_j) d(ref_k)

Then apply the constant affine Jacobian J:

    d²Nᵢ/dx_a dx_b = Σ_{j,k} A[a,j] A[b,k] H_ref[i, j, k]

where A = J^{-T}.  Since P3 elements use an *affine* (subparametric)
map, J is constant per element — no correction terms needed.

USAGE
-----
Put this script at your project root and run:

    python demo_p3_pipeline.py

Make sure jax x64 is enabled BEFORE any jnp imports (see line below).
"""

# ── JAX: enable 64-bit BEFORE first use ─────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# ────────────────────────────────────────────────────────────────────────────

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec

# ── Project imports (adjust paths to match your repo layout) ─────────────────
from src.p3_mesh_generator import generate_p3_structured_mesh
from src.p3_assembly import (
    assemble_system_p3,
    apply_boundary_conditions_p3,
)
from src.p3_shape_functions import (
    compute_p3_shape_functions,
    compute_p3_shape_gradients_reference,
)
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# JAX-AUTODIFF SHEAR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Reference coordinates for the 10 P3 DOF nodes
_P3_REF_NODES = np.array([
    [0.0,       0.0      ],   # vertex 0
    [1.0,       0.0      ],   # vertex 1
    [0.0,       1.0      ],   # vertex 2
    [1.0/3.0,   0.0      ],   # edge 0-1, t=1/3
    [2.0/3.0,   0.0      ],   # edge 0-1, t=2/3
    [2.0/3.0,   1.0/3.0  ],   # edge 1-2, t=1/3
    [1.0/3.0,   2.0/3.0  ],   # edge 1-2, t=2/3
    [0.0,       2.0/3.0  ],   # edge 2-0, t=1/3
    [0.0,       1.0/3.0  ],   # edge 2-0, t=2/3
    [1.0/3.0,   1.0/3.0  ],   # interior
])


def _build_ref_hessians() -> np.ndarray:
    """
    Precompute H_ref[eval_pt, shape_fn, i, j] = d²Nⱼ/dξᵢ dξⱼ
    at every reference node, using JAX forward-over-reverse autodiff.

    Shape: (10, 10, 2, 2)
    """
    # JAX-differentiable wrapper around your existing shape function
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])   # → (10,)

    # Hessian of N_vec wrt its 2D input → (10, 2, 2) at one point
    hess_fn = jax.jacfwd(jax.jacrev(N_vec))   # exact, no FD

    H = np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])  # (10, 10, 2, 2)
    return H


def compute_shear_p3(mesh, psi: np.ndarray,
                     H_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute shear components (γ₁, γ₂) at every mesh node from a P3 solution.

        γ₁ = ½(∂²ψ/∂x² − ∂²ψ/∂y²)
        γ₂ = ∂²ψ/∂x∂y

    Args:
        mesh  : P3 mesh with .nodes and .elements
        psi   : (n_nodes,) FEM solution
        H_ref : (10, 10, 2, 2) precomputed reference Hessians from
                _build_ref_hessians()

    Returns:
        gamma1, gamma2 : (n_nodes,) arrays, nodally averaged
    """
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes  = len(nodes)

    g1_sum  = np.zeros(n_nodes)
    g2_sum  = np.zeros(n_nodes)
    count   = np.zeros(n_nodes, dtype=int)

    for elem in elements:
        x0, y0 = nodes[elem[0]]
        x1, y1 = nodes[elem[1]]
        x2, y2 = nodes[elem[2]]

        # Affine Jacobian (constant over the element)
        J     = np.array([[x1-x0, y1-y0],
                           [x2-x0, y2-y0]])
        A     = np.linalg.inv(J).T        # maps ref→physical gradient

        psi_e = psi[elem]                  # (10,) nodal values

        for local_i in range(10):
            # H_ref[local_i]: (10, 2, 2) — all shape-fn Hessians at node local_i
            #
            # Transform reference Hessians to physical coords.
            #
            # The chain rule for second derivatives (affine map, so J constant):
            #
            #   ∂²N/∂x_a ∂x_b = Σ_{j,k} (∂ref_j/∂x_a)(∂ref_k/∂x_b) ∂²N/∂ref_j ∂ref_k
            #
            # A = J_inv.T = J_correct^{-1}, so A[j, a] = ∂ref_j/∂x_a.
            #
            # Correct einsum: H_phys[n,a,b] = Σ_{j,k} A[j,a] A[k,b] H_ref[n,j,k]
            #                               ↑ 'ja,kb' — note index ORDER matters!
            #
            # BUG WAS: 'aj,bk,njk->nab' which uses A[a,j] instead of A[j,a].
            # A is NOT symmetric for upper triangles (skewed J), so this was wrong
            # for half the elements. Diagonal lower triangles have symmetric A,
            # hiding the bug there. The fix is simply swapping the first two
            # subscripts in the einsum string.
            H_phys = np.einsum('ja,kb,njk->nab', A, A, H_ref[local_i])  # (10,2,2)

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
    """
    Compute α = ∇ψ at every mesh node using nodal averaging.

    Returns: (n_nodes, 2) array [αx, αy]
    """
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes  = len(nodes)

    a_sum = np.zeros((n_nodes, 2))
    count = np.zeros(n_nodes, dtype=int)

    for elem in elements:
        x0, y0 = nodes[elem[0]]
        x1, y1 = nodes[elem[1]]
        x2, y2 = nodes[elem[2]]

        J     = np.array([[x1-x0, y1-y0], [x2-x0, y2-y0]])
        J_inv = np.linalg.inv(J)
        psi_e = psi[elem]

        for local_i, (xi, eta) in enumerate(_P3_REF_NODES):
            dN_ref  = np.array(compute_p3_shape_gradients_reference(xi, eta))  # (10,2)
            dN_phys = dN_ref @ J_inv.T                                          # (10,2)
            grad_psi = dN_phys.T @ psi_e                                        # (2,)

            gi = elem[local_i]
            a_sum[gi] += grad_psi
            count[gi] += 1

    count = np.maximum(count, 1)
    return a_sum / count[:, None]


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTIC REFERENCE (Gaussian lens)
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_shear_exact(x, y, A, sigma):
    """
    Analytic shear for a Gaussian convergence κ = A·exp(-r²/2σ²).

    The potential can be written ψ = A·σ²·(1 − exp(−r²/2σ²)), giving:

        γ₁ = A·exp(−r²/2σ²)·(y²−x²) / (2σ²)
        γ₂ = −A·exp(−r²/2σ²)·xy / σ²
    """
    r2     = x**2 + y**2
    kernel = A * np.exp(-r2 / (2*sigma**2))
    gamma1 = kernel * (y**2 - x**2) / (2*sigma**2)
    gamma2 = -kernel * x * y / sigma**2
    return gamma1, gamma2


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def solve_pipeline(kappa_fn, nx=30, domain=(-2.5, 2.5, -2.5, 2.5),
                   label="", verbose=True):
    """
    Run the full pipeline for a given κ(x,y) function.

    Returns a dict with all fields ready for plotting.
    """
    xmin, xmax, ymin, ymax = domain

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  {label or 'P3 Pipeline'}  |  {nx}×{nx} cells")
        print(f"{'═'*55}")

    # 1. Mesh
    if verbose: print("[1/4] Meshing...")
    mesh  = generate_p3_structured_mesh(nx, nx, xmin, xmax, ymin, ymax)
    nodes = np.array(mesh.nodes)
    x, y  = nodes[:, 0], nodes[:, 1]

    # 2. Convergence field
    if verbose: print("[2/4] Computing κ...")
    kappa = kappa_fn(x, y)

    # 3. FEM solve
    if verbose: print("[3/4] FEM solve (∇²ψ = 2κ)...")
    K, F       = assemble_system_p3(mesh, kappa)
    K_bc, F_bc = apply_boundary_conditions_p3(K, F, mesh)
    psi        = spla.spsolve(K_bc, F_bc)
    res        = np.linalg.norm(K_bc @ psi - F_bc)
    if verbose: print(f"      residual = {res:.2e},  max|ψ| = {np.abs(psi).max():.4f}")

    # 4. Derived quantities
    if verbose: print("[4/4] Computing α and γ (JAX autodiff)...")
    alpha = compute_deflection_p3(mesh, psi)

    print("      Precomputing reference Hessians via jax.jacfwd(jax.jacrev(N))...")
    H_ref  = _build_ref_hessians()
    print("      Computing shear at all nodes...")
    g1, g2 = compute_shear_p3(mesh, psi, H_ref)

    if verbose:
        print(f"      max|α|  = {np.linalg.norm(alpha, axis=1).max():.4f}")
        print(f"      max|γ₁| = {np.abs(g1).max():.4f}")
        print(f"      max|γ₂| = {np.abs(g2).max():.4f}")

    return dict(mesh=mesh, nodes=nodes, kappa=kappa, psi=psi,
                alpha=alpha, gamma1=g1, gamma2=g2,
                gamma_mag=np.sqrt(g1**2 + g2**2))


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _tri(res):
    """Build matplotlib Triangulation from result dict."""
    nd = res['nodes']
    el = np.array(res['mesh'].elements)[:, :3]   # vertex-only triangles
    return mtri.Triangulation(nd[:, 0], nd[:, 1], triangles=el)


def _panel(fig, ax, tri, data, title, cmap, symmetric=False, levels=40):
    """Draw a single filled-contour panel."""
    if symmetric:
        vmax = np.percentile(np.abs(data[np.isfinite(data)]), 98)
        vmin = -vmax
    else:
        vmin = np.percentile(data[np.isfinite(data)],  1)
        vmax = np.percentile(data[np.isfinite(data)], 99)
    tcf = ax.tricontourf(tri, data, levels=levels, cmap=cmap,
                         vmin=vmin, vmax=vmax, extend='both')
    ax.tricontour(tri, data, levels=8, colors='w', linewidths=0.3, alpha=0.4)
    cb = fig.colorbar(tcf, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=7, colors='#aaa')
    ax.set_title(title, fontsize=10, color='#ddd', pad=5)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x [θ_E]', fontsize=8, color='#aaa')
    ax.set_ylabel('y [θ_E]', fontsize=8, color='#aaa')
    ax.plot(0, 0, '+w', ms=8, mew=1.5)


def plot_main(res, title="P3 FEM Weak Lensing Pipeline",
              out="p3_pipeline.png"):
    """6-panel figure: κ | ψ | |α| | γ₁ | γ₂ | |γ|"""
    nodes     = res['nodes']
    tri       = _tri(res)
    alpha_mag = np.linalg.norm(res['alpha'], axis=1)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#0e0e0e')
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#eee', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.05, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor('#1a1a1a')

    panels = [
        (res['kappa'],     'κ  –  convergence / surface density',   'hot',    False),
        (res['psi'],       'ψ  –  lensing potential',                'viridis',False),
        (alpha_mag,        '|α|  –  deflection magnitude',           'plasma', False),
        (res['gamma1'],    'γ₁ = ½(ψ_xx − ψ_yy)  [tangential]',     'RdBu_r', True),
        (res['gamma2'],    'γ₂ = ψ_xy  [cross shear]',               'RdBu_r', True),
        (res['gamma_mag'], '|γ| = √(γ₁²+γ₂²)  [shear magnitude]',   'YlOrRd', False),
    ]

    for ax, (data, title_, cmap, sym) in zip(axes, panels):
        _panel(fig, ax, tri, data, title_, cmap, symmetric=sym)

    # Deflection quiver overlay on |α| panel
    ax_a = axes[2]
    step = max(1, len(nodes) // 500)
    idx  = np.arange(0, len(nodes), step)
    amag = alpha_mag[idx]
    ax_a.quiver(nodes[idx, 0], nodes[idx, 1],
                res['alpha'][idx, 0], res['alpha'][idx, 1],
                amag, cmap='cool', scale=amag.max()*12,
                width=0.003, headwidth=3, alpha=0.75)

    # Shear sticks on |γ| panel
    ax_g = axes[5]
    step2 = max(1, len(nodes) // 350)
    idx2  = np.arange(0, len(nodes), step2)
    phi   = 0.5 * np.arctan2(res['gamma2'][idx2], res['gamma1'][idx2])
    gmag  = res['gamma_mag'][idx2]
    ax_g.quiver(nodes[idx2, 0], nodes[idx2, 1],
                gmag*np.cos(phi), gmag*np.sin(phi),
                color='w', scale=gmag.max()*18, width=0.0025,
                headwidth=0, headlength=0, headaxislength=0,
                alpha=0.6, pivot='middle')

    plt.savefig(out, dpi=180, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"✅  {out}")
    plt.close()


def plot_validation(res, A, sigma, out="p3_shear_validation.png"):
    """Compare FEM shear to the analytic Gaussian formula."""
    nodes = res['nodes']
    x, y  = nodes[:, 0], nodes[:, 1]
    tri   = _tri(res)

    g1_ex, g2_ex = gaussian_shear_exact(x, y, A, sigma)
    r1 = res['gamma1'] - g1_ex
    r2 = res['gamma2'] - g2_ex
    rms1 = np.sqrt(np.mean(r1**2));  rms2 = np.sqrt(np.mean(r2**2))

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0e0e0e')
    fig.suptitle("P3 Shear Validation  –  FEM vs Analytic Gaussian",
                 fontsize=15, fontweight='bold', color='#eee')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')

    rows = [
        (res['gamma1'], g1_ex, r1, 'γ₁', rms1),
        (res['gamma2'], g2_ex, r2, 'γ₂', rms2),
    ]

    for row_idx, (fem, exact, resid, lbl, rms) in enumerate(rows):
        vmax  = np.percentile(np.abs(exact), 98)
        rmax  = np.percentile(np.abs(resid), 98)
        for col_idx, (data, ttl, vm, cmap, sym) in enumerate([
            (fem,   f'{lbl}  FEM (P3)',                  vmax, 'RdBu_r', True),
            (exact, f'{lbl}  Analytic',                   vmax, 'RdBu_r', True),
            (resid, f'{lbl}  FEM−Analytic  (RMS={rms:.3f})', rmax, 'PuOr',   True),
        ]):
            ax = axes[row_idx, col_idx]
            _panel(fig, ax, tri, data, ttl, cmap, symmetric=sym)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"✅  {out}")
    plt.close()


def plot_p1_vs_p3(res_p3, nx=30, A=1.0, sigma=0.5,
                  out="p1_vs_p3_shear.png"):
    """
    Show the key motivation for P3:
      P1 → constant gradients per element → shear ≡ 0 everywhere
      P3 → linear gradients per element  → nonzero, converging shear

    P1 shear is identically zero because piecewise-linear ψ has
    zero second derivatives — we show this by plotting zeros for P1.
    """
    nodes_p3 = res_p3['nodes']
    x, y     = nodes_p3[:, 0], nodes_p3[:, 1]
    tri_p3   = _tri(res_p3)

    g1_ex, g2_ex = gaussian_shear_exact(x, y, A, sigma)
    gamma_ex     = np.sqrt(g1_ex**2 + g2_ex**2)
    gamma_p3     = res_p3['gamma_mag']

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0e0e0e')
    fig.suptitle(
        "Why P3?  Shear magnitude |γ| across element orders\n"
        f"Gaussian lens: A={A}, σ={sigma},  {nx}×{nx} mesh",
        fontsize=14, fontweight='bold', color='#eee'
    )
    for ax in axes:
        ax.set_facecolor('#1a1a1a')

    vmax = np.percentile(gamma_ex, 99)

    titles = [
        "P1 elements\n|γ| ≡ 0  (second derivatives vanish)",
        "P3 elements\n|γ| converges ✅",
        "Analytic\n|γ| (ground truth)",
    ]
    datas = [
        np.zeros(len(nodes_p3)),   # P1: exactly zero
        gamma_p3,
        gamma_ex,
    ]
    tris_list = [tri_p3, tri_p3, tri_p3]

    for ax, tri_, data, ttl in zip(axes, tris_list, datas, titles):
        tcf = ax.tricontourf(tri_, data, levels=40, cmap='YlOrRd',
                             vmin=0, vmax=vmax)
        ax.tricontour(tri_, data, levels=8, colors='w', linewidths=0.3, alpha=0.4)
        fig.colorbar(tcf, ax=ax, pad=0.02, fraction=0.046
                     ).ax.tick_params(labelsize=7, colors='#aaa')
        ax.set_title(ttl, fontsize=12, color='#ddd', pad=10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7, colors='#888')
        ax.set_xlabel('x', fontsize=9, color='#aaa')
        ax.set_ylabel('y', fontsize=9, color='#aaa')
        ax.plot(0, 0, '+w', ms=9, mew=2)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out, dpi=180, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"✅  {out}")
    plt.close()


def plot_two_cluster(res, out="p3_two_cluster.png"):
    """Same 6-panel layout for the two-cluster system."""
    plot_main(res, title="P3 FEM  –  Two-Cluster Lensing System", out=out)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Demo 1: Single Gaussian cluster ───────────────────────────────────────
    print("\n" + "★"*55)
    print("  DEMO 1: Single Gaussian Cluster")
    print("★"*55)

    A, sigma = 1.0, 0.5
    domain   = (-2.5, 2.5, -2.5, 2.5)

    def kappa_gaussian(x, y):
        return A * np.exp(-(x**2 + y**2) / (2*sigma**2))

    res1 = solve_pipeline(
        kappa_gaussian, nx=30, domain=domain,
        label="Single Gaussian (A=1, σ=0.5)"
    )

    plot_main(
        res1,
        title="P3 FEM Weak Lensing  –  Gaussian Cluster  (A=1, σ=0.5)",
        out="p3_pipeline_gaussian.png",
    )
    plot_validation(res1, A, sigma, out="p3_shear_validation.png")
    plot_p1_vs_p3(res1, nx=30, A=A, sigma=sigma)

    # ── Demo 2: Two-cluster system ─────────────────────────────────────────
    print("\n" + "★"*55)
    print("  DEMO 2: Two-Cluster System")
    print("★"*55)

    def kappa_two_cluster(x, y):
        c1 = 1.2 * np.exp(-((x-1.0)**2 + (y+0.3)**2) / (2*0.4**2))
        c2 = 0.9 * np.exp(-((x+1.2)**2 + (y-0.5)**2) / (2*0.5**2))
        return c1 + c2

    res2 = solve_pipeline(
        kappa_two_cluster, nx=30,
        domain=(-3.0, 3.0, -3.0, 3.0),
        label="Two Clusters"
    )
    plot_two_cluster(res2, out="p3_two_cluster.png")

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  Output files:")
    print("    p3_pipeline_gaussian.png  — 6-panel κ/ψ/α/γ₁/γ₂/|γ|")
    print("    p3_shear_validation.png   — FEM vs analytic shear")
    print("    p1_vs_p3_shear.png        — why P3 beats P1")
    print("    p3_two_cluster.png        — two-cluster system")
    print("═"*55)