"""
examples/demo_p3_pipeline.py
Full weak lensing forward pipeline using P3 cubic elements.

    kappa -> psi (FEM Poisson solve) -> alpha = grad(psi) -> gamma1, gamma2

Run from project root:
    python examples/demo_p3_pipeline.py
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

from femmi.mesh     import generate_p3_structured_mesh
from femmi.assembly import assemble_system_p3, apply_boundary_conditions_p3
from femmi.basis    import compute_p3_shape_functions, compute_p3_shape_gradients_reference


_P3_REF_NODES = np.array([
    [0.0,      0.0     ], [1.0,      0.0     ], [0.0,      1.0     ],
    [1.0/3.0,  0.0     ], [2.0/3.0,  0.0     ], [2.0/3.0,  1.0/3.0 ],
    [1.0/3.0,  2.0/3.0 ], [0.0,      2.0/3.0 ], [0.0,      1.0/3.0 ],
    [1.0/3.0,  1.0/3.0 ],
])


def _build_ref_hessians():
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])
    hess_fn = jax.jacfwd(jax.jacrev(N_vec))
    return np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])


def compute_shear_p3(mesh, psi, H_ref):
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes  = len(nodes)
    g1_sum   = np.zeros(n_nodes); g2_sum = np.zeros(n_nodes)
    count    = np.zeros(n_nodes, dtype=int)

    for elem in elements:
        x0,y0=nodes[elem[0]]; x1,y1=nodes[elem[1]]; x2,y2=nodes[elem[2]]
        J     = np.array([[x1-x0,y1-y0],[x2-x0,y2-y0]])
        A     = np.linalg.inv(J).T
        psi_e = psi[elem]
        for li in range(10):
            H_phys = np.einsum('ja,kb,njk->nab', A, A, H_ref[li])
            psi_xx = psi_e @ H_phys[:, 0, 0]
            psi_yy = psi_e @ H_phys[:, 1, 1]
            psi_xy = psi_e @ H_phys[:, 0, 1]
            gi = elem[li]
            g1_sum[gi] += 0.5 * (psi_xx - psi_yy)
            g2_sum[gi] += psi_xy
            count[gi]  += 1

    count = np.maximum(count, 1)
    return g1_sum / count, g2_sum / count


def compute_deflection_p3(mesh, psi):
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes  = len(nodes)
    a_sum    = np.zeros((n_nodes, 2)); count = np.zeros(n_nodes, dtype=int)

    for elem in elements:
        x0,y0=nodes[elem[0]]; x1,y1=nodes[elem[1]]; x2,y2=nodes[elem[2]]
        J     = np.array([[x1-x0,y1-y0],[x2-x0,y2-y0]])
        J_inv = np.linalg.inv(J)
        psi_e = psi[elem]
        for li, (xi, eta) in enumerate(_P3_REF_NODES):
            dN_ref  = np.array(compute_p3_shape_gradients_reference(xi, eta))
            dN_phys = dN_ref @ J_inv.T
            gi = elem[li]
            a_sum[gi] += dN_phys.T @ psi_e
            count[gi] += 1

    count = np.maximum(count, 1)
    return a_sum / count[:, None]


def solve_pipeline(kappa_fn, nx=30, domain=(-2.5, 2.5, -2.5, 2.5), label=""):
    xmin, xmax, ymin, ymax = domain
    print(f"\n{label or 'P3 Pipeline'}  |  {nx}x{nx} cells")

    mesh  = generate_p3_structured_mesh(nx, nx, xmin, xmax, ymin, ymax)
    nodes = np.array(mesh.nodes)
    kappa = kappa_fn(nodes[:, 0], nodes[:, 1])

    K, F       = assemble_system_p3(mesh, kappa)
    K_bc, F_bc = apply_boundary_conditions_p3(K, F, mesh)
    psi        = spla.spsolve(K_bc, F_bc)
    print(f"  residual={np.linalg.norm(K_bc@psi-F_bc):.2e}  max|psi|={np.abs(psi).max():.4f}")

    H_ref       = _build_ref_hessians()
    alpha       = compute_deflection_p3(mesh, psi)
    g1, g2      = compute_shear_p3(mesh, psi, H_ref)
    gamma_mag   = np.sqrt(g1**2 + g2**2)
    alpha_mag   = np.linalg.norm(alpha, axis=1)
    print(f"  max|alpha|={alpha_mag.max():.4f}  max|gamma1|={np.abs(g1).max():.4f}  max|gamma2|={np.abs(g2).max():.4f}")

    return dict(mesh=mesh, nodes=nodes, kappa=kappa, psi=psi,
                alpha=alpha, gamma1=g1, gamma2=g2, gamma_mag=gamma_mag)


def _tri(res):
    nd = res['nodes']; el = np.array(res['mesh'].elements)[:, :3]
    return mtri.Triangulation(nd[:, 0], nd[:, 1], triangles=el)


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
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x', fontsize=8, color='#aaa')
    ax.set_ylabel('y', fontsize=8, color='#aaa')
    ax.plot(0, 0, '+w', ms=8, mew=1.5)


def plot_main(res, title="P3 FEM Weak Lensing Pipeline", out="p3_pipeline.png"):
    nodes     = res['nodes']; tri = _tri(res)
    alpha_mag = np.linalg.norm(res['alpha'], axis=1)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 11)); fig.patch.set_facecolor('#0e0e0e')
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#eee', y=0.98)
    gs   = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                    left=0.05, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor('#1a1a1a')

    for ax, (data, ttl, cmap, sym) in zip(axes, [
        (res['kappa'],     'kappa (convergence)',         'hot',    False),
        (res['psi'],       'psi (lensing potential)',     'viridis',False),
        (alpha_mag,        '|alpha| (deflection)',        'plasma', False),
        (res['gamma1'],    'gamma1 = 0.5*(psi_xx-psi_yy)', 'RdBu_r', True),
        (res['gamma2'],    'gamma2 = psi_xy',             'RdBu_r', True),
        (res['gamma_mag'], '|gamma|',                     'YlOrRd', False),
    ]):
        _panel(fig, ax, tri, data, ttl, cmap, symmetric=sym)

    step = max(1, len(nodes)//500); idx = np.arange(0, len(nodes), step)
    axes[2].quiver(nodes[idx, 0], nodes[idx, 1],
                   res['alpha'][idx, 0], res['alpha'][idx, 1],
                   alpha_mag[idx], cmap='cool', scale=alpha_mag[idx].max()*12,
                   width=0.003, headwidth=3, alpha=0.75)

    step2 = max(1, len(nodes)//350); idx2 = np.arange(0, len(nodes), step2)
    phi   = 0.5 * np.arctan2(res['gamma2'][idx2], res['gamma1'][idx2])
    gm    = res['gamma_mag'][idx2]
    axes[5].quiver(nodes[idx2, 0], nodes[idx2, 1], gm*np.cos(phi), gm*np.sin(phi),
                   color='w', scale=gm.max()*18, width=0.0025,
                   headwidth=0, headlength=0, headaxislength=0, alpha=0.6, pivot='middle')

    plt.savefig(out, dpi=180, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    A, sigma = 1.0, 0.5

    res1 = solve_pipeline(
        lambda x, y: A * np.exp(-(x**2+y**2)/(2*sigma**2)),
        nx=30, domain=(-2.5, 2.5, -2.5, 2.5),
        label="Single Gaussian (A=1, sigma=0.5)",
    )
    plot_main(res1, title="P3 FEM Weak Lensing - Gaussian Cluster",
              out="p3_pipeline_gaussian.png")

    res2 = solve_pipeline(
        lambda x, y: (1.2*np.exp(-((x-1.0)**2+(y+0.3)**2)/(2*0.4**2)) +
                      0.9*np.exp(-((x+1.2)**2+(y-0.5)**2)/(2*0.5**2))),
        nx=30, domain=(-3.0, 3.0, -3.0, 3.0), label="Two Clusters",
    )
    plot_main(res2, title="P3 FEM - Two-Cluster System", out="p3_two_cluster.png")