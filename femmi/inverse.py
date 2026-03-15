"""
inverse.py
==========
MAP mass reconstruction: gamma_obs -> kappa_MAP.

Solves:
    kappa_MAP = argmin  ||gamma_pred(kappa) - gamma_obs||^2
                kappa     + lambda * kappa^T R kappa

where R is either:
  - H1 prior (default):     R = K           (penalises ||grad kappa||^2)
  - Wiener/Matern prior:    R = M + l^2*K   (penalises at correlation length l)

The Wiener prior is activated by passing wiener_length=l to MAPReconstructor
or run_comparison. Setting l ~ sigma_lens (the lens scale) gives a prior
that matches the expected spatial structure of kappa.

Uses L-BFGS with a pure numpy adjoint gradient (no JAX JIT in the loop).
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


# =============================================================================
# MAP Reconstructor
# =============================================================================

@dataclass
class ReconstructionResult:
    """Output from MAP reconstruction."""
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
    MAP mass reconstruction using L-BFGS with explicit numpy adjoint.

    Parameters
    ----------
    fwd           : DifferentiableForward  (carries ops and lam_reg)
    maxiter       : maximum L-BFGS iterations
    gtol          : gradient-norm stopping tolerance
    callback_every: print progress every N function calls (0 = silent)
    wiener_length : if > 0, use Matern prior R = M + l^2*K instead of R = K.
                    Recommended: l = sigma_lens ~ 0.5 for a Gaussian lens.
    """

    def __init__(self,
                 fwd: DifferentiableForward,
                 maxiter: int = 500,
                 gtol: float = 1e-9,
                 callback_every: int = 50,
                 wiener_length: float = 0.0,
                 noise_std: Optional[float] = None):
        self.fwd            = fwd
        self.maxiter        = maxiter
        self.gtol           = gtol
        self.callback_every = callback_every
        self.wiener_length  = wiener_length
        self.noise_std      = noise_std
        self.ops            = fwd.ops

        # Build regularizer matrix R once
        if wiener_length > 0.0:
            self._R = build_wiener_regularizer(fwd.ops, wiener_length)
        else:
            self._R = fwd.ops.K   # plain H1 prior

    # -------------------------------------------------------------------------

    def _make_obj_and_grad(self,
                           gamma1_obs: np.ndarray,
                           gamma2_obs: np.ndarray):
        """
        Returns a callable (kappa -> loss, grad) using the explicit adjoint.

        Derivation
        ----------
        Forward:     psi = K^{-1}(-2 M kappa),   gamma = S psi
        Residual:    r   = gamma_pred - gamma_obs
        Loss:        L   = ||r||^2 + lambda * kappa^T R kappa
        Adjoint:     adj = K^{-1}(S1^T r1 + S2^T r2)
        Gradient:    dL/dkappa = -4 M^T adj + 2 lambda R kappa
        """
        ops   = self.ops
        M     = ops.M
        S1    = ops.S1
        S2    = ops.S2
        A_lu  = ops.A_coupled_lu   # FEM-BEM coupled solver (replaces K_lu)
        lam   = self.fwd.lam_reg
        R     = self._R
 
        loss_history = []
        
        idx_gauge = int(ops.bnd_mesh.node_indices[0])
 
        def obj_grad(kappa_flat):
            kappa = kappa_flat.reshape(-1)
 
            # Forward pass — A_coupled ψ = −2Mκ  (MATH.md §6.2)
            # Full M (no boundary zeroing); A_coupled encodes BCs via BEM.
            rhs = -2.0 * M @ kappa
            rhs[idx_gauge] = 0.0
            psi = A_lu.solve(rhs)
            g1  = S1 @ psi
            g2  = S2 @ psi
 
            # Residuals
            r1 = g1 - gamma1_obs
            r2 = g2 - gamma2_obs
 
            # Data loss
            data_loss = float(np.dot(r1, r1) + np.dot(r2, r2))
 
            # Regularisation loss
            Rk       = R @ kappa
            reg_loss = float(lam * np.dot(kappa, Rk))
 
            loss = data_loss + reg_loss
            loss_history.append(loss)
 
            # Adjoint solve — A_coupled φ = S1ᵀr1 + S2ᵀr2  (MATH.md §12.2)
            rhs_adj = S1.T @ r1 + S2.T @ r2
            rhs_adj[idx_gauge] = 0.0
            adj = A_lu.solve(rhs_adj, trans='T')
 
            grad = -4.0 * (M.T @ adj) + 2.0 * lam * Rk
 
            return loss, grad.astype(np.float64)

        return obj_grad, loss_history

    # -------------------------------------------------------------------------

    def reconstruct(self,
                    gamma1_obs: np.ndarray,
                    gamma2_obs: np.ndarray,
                    kappa_init: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None,
                    verbose: bool = True) -> Tuple[np.ndarray, ReconstructionResult]:
        """
        Run MAP reconstruction.

        Args:
            gamma1_obs, gamma2_obs : observed shear (n_nodes,)
            kappa_init             : initial guess (zeros if None)
            mask                   : boolean array; masked nodes set to 0 in obs
            verbose                : print header and convergence summary

        Returns:
            kappa_map : (n_nodes,) MAP estimate
            result    : ReconstructionResult dataclass
        """
        # ── Phase 3.4: auto-select λ via Morozov if noise_std provided ───
        if self.noise_std is not None:
            from .regularization import MorozovSelector
            if verbose:
                print(f"[Morozov] Auto-selecting λ (noise_std={self.noise_std:.3e})...")
            selector = MorozovSelector(
                self.ops,
                noise_std=self.noise_std,
                wiener_length=self.wiener_length,
                maxiter_inner=min(150, self.maxiter),
                verbose=verbose,
            )
            lam_star = selector.select(gamma1_obs, gamma2_obs)
            if verbose:
                print(f"[Morozov] λ* = {lam_star:.4e}\n")
            self.fwd.lam_reg = lam_star
            # Rebuild regularizer at new λ
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
            prior_name = (f"Wiener (l={self.wiener_length:.2f})"
                          if self.wiener_length > 0 else "H1 (K)")
            loss0, grad0 = obj_grad(kappa0)
            print("=" * 60)
            print(f"MAP Reconstruction  (L-BFGS, numpy adjoint)")
            print(f"  n_nodes   = {n}")
            print(f"  lambda    = {self.fwd.lam_reg:.2e}")
            print(f"  prior     = {prior_name}")
            print(f"  maxiter   = {self.maxiter}  |  gtol = {self.gtol:.0e}")
            print(f"  loss(k=0) = {loss0:.4e}  |  ||grad||(k=0) = {np.linalg.norm(grad0):.4e}")
            print("=" * 60)
            loss_history.clear()

        call_counter = [0]
        cb_every = self.callback_every

        def callback(kappa_flat):
            call_counter[0] += 1
            if cb_every > 0 and call_counter[0] % cb_every == 0 and loss_history:
                print(f"  call {call_counter[0]:4d}  loss = {loss_history[-1]:.6e}")

        t0  = time.perf_counter()
        res = sopt.minimize(
            obj_grad, kappa0,
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options={
                'maxiter' : self.maxiter,
                'gtol'    : self.gtol,
                'ftol'    : 1e-30,
                'maxcor'  : 20,
                'disp'    : False,
            }
        )
        wall = time.perf_counter() - t0

        kappa_map = res.x
        psi_map   = ops.psi_from_kappa(kappa_map)
        g1p, g2p  = ops.shear_from_psi(psi_map)

        if verbose:
            print(f"\n  Converged : {res.success}")
            print(f"  Message   : {res.message}")
            print(f"  Iterations: {res.nit}")
            print(f"  Fcn calls : {res.nfev}")
            print(f"  Final loss: {res.fun:.6e}")
            print(f"  Wall time : {wall:.2f} s")
            print(f"  max|kappa|: {np.abs(kappa_map).max():.4f}")
            print("=" * 60)

        result = ReconstructionResult(
            kappa_map   = kappa_map,
            psi_map     = psi_map,
            gamma1_pred = g1p,
            gamma2_pred = g2p,
            loss_history= loss_history,
            n_iter      = res.nit,
            converged   = res.success,
            time_s      = wall,
        )
        return kappa_map, result


# =============================================================================
# Kaiser-Squires reference
# =============================================================================

def kaiser_squires(gamma1: np.ndarray, gamma2: np.ndarray,
                   nodes: np.ndarray,
                   grid_size: int = 64) -> np.ndarray:
    """
    FFT-based Kaiser-Squires convergence reconstruction.

    Interpolates the irregular FEM-node shear onto a uniform grid, applies
    the KS Fourier kernel, then interpolates back to FEM nodes.
    """
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
    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0

    Dk = (KX**2 - KY**2) / k2
    Ok = 2.0 * KX * KY / k2

    Kappak = Dk * G1k + Ok * G2k
    kappa_grid = np.real(sfft.ifft2(Kappak))

    kappa_nodes = griddata(
        np.column_stack([XX.ravel(), YY.ravel()]),
        kappa_grid.ravel(),
        nodes,
        method='linear',
        fill_value=0.0,
    )
    return kappa_nodes


# =============================================================================
# Benchmark: FEM-MAP vs Kaiser-Squires
# =============================================================================

def run_comparison(nx: int = 20,
                   noise_level: float = 0.10,
                   lam_reg: float = 1e-2,
                   use_morozov: bool = False,
                   apply_mask: bool = False,
                   mask_center: Tuple[float, float] = (0.0, 0.0),
                   mask_radius: float = 0.5,
                   wiener_length: float = 0.0,
                   use_adaptive_mesh: bool = False,
                   refine_factor: int = 3,
                   sigma_lens: float = 0.5,
                   A_lens: float = 1.0,
                   xmin: float = -2.5, xmax: float = 2.5,
                   ymin: float = -2.5, ymax: float = 2.5):
    """
    Benchmark FEM-MAP vs Kaiser-Squires on a synthetic Gaussian lens survey.

    Returns
    -------
    kappa_map   : (n_nodes,) FEM-MAP reconstruction
    kappa_ks    : (n_nodes,) Kaiser-Squires reconstruction
    kappa_true  : (n_nodes,) ground truth
    result      : ReconstructionResult
    """
    prior_tag = f"Wiener(l={wiener_length})" if wiener_length > 0 else "H1"
    mesh_tag  = f"adaptive(x{refine_factor})" if use_adaptive_mesh else "structured"

    print("=" * 60)
    print(f"FEM-MAP vs Kaiser-Squires Benchmark")
    print(f"  {nx}x{nx} P3 {mesh_tag}  |  noise={noise_level*100:.0f}%  "
          f"|  lambda={lam_reg:.0e}  |  prior={prior_tag}")
    print("=" * 60)

    if use_adaptive_mesh and apply_mask:
        ops = build_operators_adaptive(
            nx, nx, xmin, xmax, ymin, ymax,
            mask_center=mask_center,
            mask_radius=mask_radius,
            refine_factor=refine_factor,
            verbose=True,
        )
    else:
        ops = build_operators(nx, nx, xmin, xmax, ymin, ymax, verbose=True)

    nodes = np.array(ops.mesh.nodes)
    fwd   = DifferentiableForward(ops, lam_reg=lam_reg)

    r2         = nodes[:, 0]**2 + nodes[:, 1]**2
    kappa_true = A_lens * np.exp(-r2 / (2 * sigma_lens**2))

    g1_true, g2_true = ops.forward(kappa_true)

    rng   = np.random.default_rng(42)
    noise = noise_level * np.std(np.sqrt(g1_true**2 + g2_true**2))
    g1_obs = g1_true + rng.normal(0, noise, g1_true.shape)
    g2_obs = g2_true + rng.normal(0, noise, g2_true.shape)

    mask = None
    if apply_mask:
        r    = np.sqrt((nodes[:, 0] - mask_center[0])**2 +
                       (nodes[:, 1] - mask_center[1])**2)
        mask = r < mask_radius
        g1_obs[mask] = 0.0
        g2_obs[mask] = 0.0
        print(f"\n  Masked {mask.sum()} nodes ({100*mask.mean():.1f}%)")

    noise_std_morozov = (noise if use_morozov else None)
    rec = MAPReconstructor(fwd, maxiter=500, gtol=1e-9, callback_every=50,
                           wiener_length=wiener_length,
                           noise_std=noise_std_morozov)
    kappa_map, result = rec.reconstruct(g1_obs, g2_obs, verbose=True)

    kappa_ks = kaiser_squires(g1_obs, g2_obs, nodes)

    l2_map = float(np.sqrt(np.mean((kappa_map  - kappa_true)**2)))
    l2_ks  = float(np.sqrt(np.mean((kappa_ks   - kappa_true)**2)))
    improv = 100.0 * (l2_ks - l2_map) / l2_ks

    print(f"\n{'─'*60}")
    print(f"Reconstruction quality  (L2 error vs truth)")
    print(f"  FEM-MAP  : {l2_map:.4f}")
    print(f"  K-S      : {l2_ks:.4f}")
    print(f"  Improvement: {improv:+.1f}%  (+ = FEM-MAP better)")
    print(f"{'─'*60}\n")

    _plot_comparison(nodes, kappa_true, kappa_map, kappa_ks,
                     result, l2_map, l2_ks,
                     noise_level, apply_mask, mask)

    return kappa_map, kappa_ks, kappa_true, result


def _plot_comparison(nodes, kappa_true, kappa_map, kappa_ks,
                     result, l2_map, l2_ks,
                     noise_level, apply_mask, mask):
    """5-panel comparison figure."""
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1])

    tag = f"noise={noise_level*100:.0f}%"
    if apply_mask:
        tag += " + mask"

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor='#1a1a1a')
    fig.suptitle(f"MAP Mass Reconstruction  |  {tag}",
                 color='white', fontsize=14, y=1.02)

    panels = [
        (kappa_true,            "kappa truth",                          'hot',    None),
        (kappa_map,             f"kappa FEM-MAP\nL2={l2_map:.3f}",     'hot',    None),
        (kappa_ks,              f"kappa Kaiser-Squires\nL2={l2_ks:.3f}",'RdYlGn',None),
        (kappa_map - kappa_true,"MAP residual",                         'RdBu_r', 0.35),
    ]

    for ax, (data, title, cmap, sym) in zip(axes[:4], panels):
        ax.set_facecolor('#1a1a1a')
        vmax = sym if sym else np.percentile(np.abs(data), 99)
        vmin = -vmax if sym else 0
        tc = ax.tripcolor(triang, data, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading='gouraud')
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        if apply_mask and mask is not None:
            mx = nodes[mask, 0]; my = nodes[mask, 1]
            ax.scatter(mx, my, c='cyan', s=1, alpha=0.3)
        ax.set_title(title, color='white', fontsize=10)
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    ax5 = axes[4]
    ax5.set_facecolor('#1a1a1a')
    if result.loss_history:
        ax5.semilogy(result.loss_history, color='#00e676', lw=1.5)
    ax5.set_xlabel('L-BFGS call', color='white', fontsize=9)
    ax5.set_ylabel('Loss', color='white', fontsize=9)
    ax5.set_title('Convergence', color='white', fontsize=10)
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.2)
    for sp in ax5.spines.values():
        sp.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig('map_reconstruction.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a')
    print("Saved: map_reconstruction.png")
    plt.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_comparison(
        nx=20,
        noise_level=0.10,
        lam_reg=2e-2,
        apply_mask=True,
        mask_center=(0.0, 0.0),
        mask_radius=0.6,
        wiener_length=0.5,
        use_adaptive_mesh=True,
        refine_factor=3,
    )