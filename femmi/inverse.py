"""
femmi/inverse.py
MAP mass reconstruction: gamma_obs -> kappa_MAP.

Minimizes ||F kappa - gamma_obs||^2 + lambda * kappa^T R kappa
using L-BFGS with a numpy adjoint gradient.

R is either:
  - H1 prior (default):    R = K
  - Wiener/Matern prior:   R = M + l^2*K
"""

import numpy as np
import scipy.optimize as sopt
import scipy.fft as sfft
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    from .operators import FEMOperators, build_operators, build_operators_adaptive, build_wiener_regularizer
    from .forward   import DifferentiableForward
except ImportError:
    from operators  import FEMOperators, build_operators, build_operators_adaptive, build_wiener_regularizer
    from forward    import DifferentiableForward


@dataclass
class ReconstructionResult:
    kappa_map    : np.ndarray
    psi_map      : np.ndarray
    gamma1_pred  : np.ndarray
    gamma2_pred  : np.ndarray
    loss_history : list
    n_iter       : int
    converged    : bool
    time_s       : float


class MAPReconstructor:
    """
    MAP mass reconstruction using L-BFGS with numpy adjoint.

    Parameters
    ----------
    fwd           : DifferentiableForward
    maxiter       : max L-BFGS iterations
    gtol          : gradient norm tolerance
    callback_every: print progress every N calls (0 = silent)
    wiener_length : if > 0, use R = M + l^2*K instead of R = K
    noise_std     : if set, auto-select lambda via Morozov's principle
    """

    def __init__(self, fwd, maxiter=500, gtol=1e-9, callback_every=50,
                 wiener_length=0.0, noise_std=None):
        self.fwd            = fwd
        self.maxiter        = maxiter
        self.gtol           = gtol
        self.callback_every = callback_every
        self.wiener_length  = wiener_length
        self.noise_std      = noise_std
        self.ops            = fwd.ops

        if wiener_length > 0.0:
            self._R = build_wiener_regularizer(fwd.ops, wiener_length)
        else:
            self._R = fwd.ops.K

    def _make_obj_and_grad(self, gamma1_obs, gamma2_obs):
        ops  = self.ops
        M    = ops.M
        S1   = ops.S1
        S2   = ops.S2
        A_lu = ops.A_coupled_lu
        lam  = self.fwd.lam_reg
        R    = self._R

        loss_history = []
        idx_gauge    = int(ops.bnd_mesh.node_indices[0])

        def obj_grad(kappa_flat):
            kappa = kappa_flat.reshape(-1)

            rhs = -2.0 * M @ kappa
            rhs[idx_gauge] = 0.0
            psi = A_lu.solve(rhs)
            g1  = S1 @ psi
            g2  = S2 @ psi

            r1 = g1 - gamma1_obs
            r2 = g2 - gamma2_obs

            Rk       = R @ kappa
            loss     = float(np.dot(r1, r1) + np.dot(r2, r2)) + float(lam * np.dot(kappa, Rk))
            loss_history.append(loss)

            rhs_adj          = S1.T @ r1 + S2.T @ r2
            rhs_adj[idx_gauge] = 0.0
            adj  = A_lu.solve(rhs_adj, trans='T')
            grad = -4.0 * (M.T @ adj) + 2.0 * lam * Rk

            return loss, grad.astype(np.float64)

        return obj_grad, loss_history

    def reconstruct(self, gamma1_obs, gamma2_obs, kappa_init=None,
                    mask=None, verbose=True):
        """
        Run MAP reconstruction.

        Returns (kappa_map, ReconstructionResult).
        """
        if self.noise_std is not None:
            from .regularization import MorozovSelector
            if verbose:
                print(f"Auto-selecting lambda (noise_std={self.noise_std:.3e})...")
            selector = MorozovSelector(
                self.ops,
                noise_std=self.noise_std,
                wiener_length=self.wiener_length,
                maxiter_inner=min(150, self.maxiter),
                verbose=verbose,
            )
            lam_star = selector.select(gamma1_obs, gamma2_obs)
            if verbose:
                print(f"lambda* = {lam_star:.4e}\n")
            self.fwd.lam_reg = lam_star
            if self.wiener_length > 0.0:
                self._R = build_wiener_regularizer(self.ops, self.wiener_length)
            else:
                self._R = self.ops.K

        ops    = self.ops
        n      = ops.n_nodes
        g1_obs = gamma1_obs.copy()
        g2_obs = gamma2_obs.copy()
        if mask is not None:
            g1_obs[mask] = 0.0
            g2_obs[mask] = 0.0

        kappa0 = np.zeros(n) if kappa_init is None else kappa_init.copy()
        obj_grad, loss_history = self._make_obj_and_grad(g1_obs, g2_obs)

        if verbose:
            prior  = f"Wiener (l={self.wiener_length:.2f})" if self.wiener_length > 0 else "H1"
            loss0, grad0 = obj_grad(kappa0)
            print(f"MAP reconstruction  n={n}  lambda={self.fwd.lam_reg:.2e}  "
                  f"prior={prior}  maxiter={self.maxiter}")
            print(f"  loss(0)={loss0:.4e}  ||grad||(0)={np.linalg.norm(grad0):.4e}")
            loss_history.clear()

        call_count = [0]

        def callback(kappa_flat):
            call_count[0] += 1
            if self.callback_every > 0 and call_count[0] % self.callback_every == 0 and loss_history:
                print(f"  call {call_count[0]:4d}  loss={loss_history[-1]:.6e}")

        t0  = time.perf_counter()
        res = sopt.minimize(
            obj_grad, kappa0, method='L-BFGS-B', jac=True,
            callback=callback,
            options={'maxiter': self.maxiter, 'gtol': self.gtol,
                     'ftol': 1e-30, 'maxcor': 20},
        )
        wall = time.perf_counter() - t0

        kappa_map   = res.x
        psi_map     = ops.psi_from_kappa(kappa_map)
        g1p, g2p    = ops.shear_from_psi(psi_map)

        if verbose:
            print(f"  converged={res.success}  iters={res.nit}  "
                  f"loss={res.fun:.6e}  time={wall:.2f}s")

        result = ReconstructionResult(
            kappa_map=kappa_map, psi_map=psi_map,
            gamma1_pred=g1p, gamma2_pred=g2p,
            loss_history=loss_history, n_iter=res.nit,
            converged=res.success, time_s=wall,
        )
        return kappa_map, result


def kaiser_squires(gamma1, gamma2, nodes, grid_size=64):
    """FFT-based Kaiser-Squires convergence reconstruction on FEM nodes."""
    from scipy.interpolate import griddata

    xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
    ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()

    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    XX, YY = np.meshgrid(xi, yi)

    g1_grid = griddata(nodes, gamma1, (XX, YY), method='linear', fill_value=0.0)
    g2_grid = griddata(nodes, gamma2, (XX, YY), method='linear', fill_value=0.0)

    G1k = sfft.fft2(g1_grid)
    G2k = sfft.fft2(g2_grid)

    kx = sfft.fftfreq(grid_size, d=(xmax - xmin) / grid_size) * 2 * np.pi
    ky = sfft.fftfreq(grid_size, d=(ymax - ymin) / grid_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k2     = KX**2 + KY**2
    k2[0, 0] = 1.0

    kappa_grid  = np.real(sfft.ifft2(
        (KX**2 - KY**2) / k2 * G1k + 2.0 * KX * KY / k2 * G2k
    ))
    kappa_nodes = griddata(
        np.column_stack([XX.ravel(), YY.ravel()]),
        kappa_grid.ravel(), nodes, method='linear', fill_value=0.0,
    )
    return kappa_nodes


def run_comparison(nx=20, noise_level=0.10, lam_reg=1e-2, use_morozov=False,
                   apply_mask=False, mask_center=(0.0, 0.0), mask_radius=0.5,
                   wiener_length=0.0, use_adaptive_mesh=False, refine_factor=3,
                   sigma_lens=0.5, A_lens=1.0,
                   xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5):
    """Benchmark FEM-MAP vs Kaiser-Squires on a synthetic Gaussian lens."""
    print(f"FEM-MAP vs KS: {nx}x{nx} P3, noise={noise_level*100:.0f}%, lambda={lam_reg:.0e}")

    if use_adaptive_mesh and apply_mask:
        ops = build_operators_adaptive(nx, nx, xmin, xmax, ymin, ymax,
                                       mask_center=mask_center,
                                       mask_radius=mask_radius,
                                       refine_factor=refine_factor)
    else:
        ops = build_operators(nx, nx, xmin, xmax, ymin, ymax)

    nodes      = np.array(ops.mesh.nodes)
    fwd        = DifferentiableForward(ops, lam_reg=lam_reg)
    kappa_true = A_lens * np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * sigma_lens**2))

    g1_true, g2_true = ops.forward(kappa_true)
    rng   = np.random.default_rng(42)
    noise = noise_level * np.std(np.sqrt(g1_true**2 + g2_true**2))
    g1_obs = g1_true + rng.normal(0, noise, g1_true.shape)
    g2_obs = g2_true + rng.normal(0, noise, g2_true.shape)

    mask = None
    if apply_mask:
        r    = np.sqrt((nodes[:, 0] - mask_center[0])**2 + (nodes[:, 1] - mask_center[1])**2)
        mask = r < mask_radius
        g1_obs[mask] = 0.0
        g2_obs[mask] = 0.0

    rec = MAPReconstructor(fwd, maxiter=500, gtol=1e-9, callback_every=50,
                           wiener_length=wiener_length,
                           noise_std=(noise if use_morozov else None))
    kappa_map, result = rec.reconstruct(g1_obs, g2_obs)
    kappa_ks          = kaiser_squires(g1_obs, g2_obs, nodes)

    l2_map = float(np.sqrt(np.mean((kappa_map  - kappa_true)**2)))
    l2_ks  = float(np.sqrt(np.mean((kappa_ks   - kappa_true)**2)))
    print(f"  FEM-MAP L2={l2_map:.4f}  KS L2={l2_ks:.4f}  "
          f"improvement={(l2_ks-l2_map)/l2_ks*100:+.1f}%")

    _plot_comparison(nodes, kappa_true, kappa_map, kappa_ks,
                     result, l2_map, l2_ks, noise_level, apply_mask, mask)
    return kappa_map, kappa_ks, kappa_true, result


def _plot_comparison(nodes, kappa_true, kappa_map, kappa_ks,
                     result, l2_map, l2_ks, noise_level, apply_mask, mask):
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    tag    = f"noise={noise_level*100:.0f}%" + (" + mask" if apply_mask else "")

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor='#1a1a1a')
    fig.suptitle(f"MAP reconstruction  |  {tag}", color='white', fontsize=14, y=1.02)

    panels = [
        (kappa_true,             "kappa truth",                      'hot',    None),
        (kappa_map,              f"FEM-MAP  L2={l2_map:.3f}",        'hot',    None),
        (kappa_ks,               f"Kaiser-Squires  L2={l2_ks:.3f}", 'RdYlGn', None),
        (kappa_map - kappa_true, "MAP residual",                      'RdBu_r', 0.35),
    ]

    for ax, (data, title, cmap, sym) in zip(axes[:4], panels):
        ax.set_facecolor('#1a1a1a')
        vmax = sym if sym else np.percentile(np.abs(data), 99)
        vmin = -vmax if sym else 0
        tc   = ax.tripcolor(triang, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        if apply_mask and mask is not None:
            ax.scatter(nodes[mask, 0], nodes[mask, 1], c='cyan', s=1, alpha=0.3)
        ax.set_title(title, color='white', fontsize=10)
        ax.set_aspect('equal')

    ax5 = axes[4]
    ax5.set_facecolor('#1a1a1a')
    if result.loss_history:
        ax5.semilogy(result.loss_history, color='#00e676', lw=1.5)
    ax5.set_title('Convergence', color='white', fontsize=10)
    ax5.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('map_reconstruction.png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()