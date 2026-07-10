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


@dataclass
class BModeDiagnostics:
    """
    Quality diagnostics derived from the B-mode channel.

    A real lensing signal has zero B-mode, so the B-mode is used two ways:
    (1) as an independent noise-level estimate to cross-check the MAD delta fed
    to Morozov, and (2) as a systematics flag via the coherent (reconstructable)
    B-mode shear power, which should sit at or below the noise floor.

    Fields (all shear quantities are per-component RMS, matching Morozov's
    delta convention):
      flag              : 'clean' | 'marginal' | 'contaminated'
      emode_shear_rms   : RMS of the fitted E-mode shear  F @ kappa_E
      bmode_shear_rms   : RMS of the coherent B-mode shear F @ kappa_B (~0 ideal)
      bmode_to_emode    : bmode_shear_rms / emode_shear_rms  (contamination)
      bmode_snr         : bmode_shear_rms / delta_noise  (drives the flag)
      delta_mad         : per-component noise from MAD on the raw shear (biased
                          high by the E-mode signal it contains)
      delta_noise       : per-component noise from the doubly-reduced residual
                          RMS(gamma_obs - F@kappa_E - rot^-1(F@kappa_B)), i.e.
                          the data with BOTH coherent modes removed. This is the
                          clean, signal- and systematics-free noise floor, and
                          the delta to feed back to Morozov.
      delta_consistency : delta_noise / delta_mad  (<< 1 confirms MAD's signal
                          bias; a good MAD estimate on signal-free data -> ~1)
      kappa_e_rms       : RMS of kappa_E over the reported region
      kappa_b_rms       : RMS of kappa_B over the reported region
    """
    flag              : str
    emode_shear_rms   : float
    bmode_shear_rms   : float
    bmode_to_emode    : float
    bmode_snr         : float
    delta_mad         : float
    delta_noise       : float
    delta_consistency : float
    kappa_e_rms       : float
    kappa_b_rms       : float

    def summary(self) -> str:
        return (
            f"B-mode diagnostics: [{self.flag.upper()}]\n"
            f"  coherent B/E shear   = {self.bmode_to_emode:.3f}\n"
            f"  B-mode SNR (B/noise) = {self.bmode_snr:.2f}\n"
            f"  delta (MAD)          = {self.delta_mad:.4e}\n"
            f"  delta (noise floor)  = {self.delta_noise:.4e}  "
            f"(ratio {self.delta_consistency:.2f})\n"
            f"  kappa RMS  E={self.kappa_e_rms:.4e}  B={self.kappa_b_rms:.4e}"
        )


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
        lam  = self.fwd.lam_reg
        R    = self._R

        loss_history = []

        def obj_grad(kappa_flat):
            kappa = kappa_flat.reshape(-1)

            psi = ops._solve_psi(-2.0 * M @ kappa)
            g1  = S1 @ psi
            g2  = S2 @ psi

            r1 = g1 - gamma1_obs
            r2 = g2 - gamma2_obs

            Rk   = R @ kappa
            loss = float(np.dot(r1, r1) + np.dot(r2, r2)) + float(lam * np.dot(kappa, Rk))
            loss_history.append(loss)

            adj  = ops._solve_adjoint(S1.T @ r1 + S2.T @ r2)
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

    def reconstruct_eb(self, gamma1_obs, gamma2_obs, kappa_init=None,
                       mask=None, verbose=True, shared_lambda=True):
        """
        Reconstruct both the E-mode (physical convergence) and the B-mode
        (systematics null-test) convergence maps.

        A real lensing potential produces only E-mode shear, so the forward
        operator F reconstructs the E-mode convergence directly. The B-mode is
        the same estimator applied to the shear rotated by 45 degrees,
        (g1, g2) -> (g2, -g1): the spin-2 rotation g -> g*exp(-2i*pi/4) = -i*g
        maps E onto B. For a pure gravitational-lensing signal kappa_B is
        consistent with zero; spatially coherent kappa_B flags residual
        systematics (PSF leakage, intrinsic alignments, additive shear bias).

        Parameters
        ----------
        shared_lambda : if True (default), the B-mode solve reuses the
            regularisation strength selected for the E-mode solve, so the two
            maps are directly comparable at matched smoothing. If False, lambda
            is selected independently for each mode.

        Returns
        -------
        (kappa_E, kappa_B, result_E, result_B)
        """
        if verbose:
            print("=== E-mode reconstruction ===")
        kappa_E, result_E = self.reconstruct(
            gamma1_obs, gamma2_obs, kappa_init=kappa_init,
            mask=mask, verbose=verbose,
        )

        # 45-degree spin-2 rotation: g -> -i*g  =>  (g1, g2) -> (g2, -g1)
        g1_b =  np.asarray(gamma2_obs, dtype=np.float64)
        g2_b = -np.asarray(gamma1_obs, dtype=np.float64)

        saved_noise = self.noise_std
        if shared_lambda:
            self.noise_std = None  # reuse lam_reg fixed by the E-mode solve
        try:
            if verbose:
                print("=== B-mode reconstruction (45-deg rotated shear) ===")
            kappa_B, result_B = self.reconstruct(
                g1_b, g2_b, kappa_init=kappa_init,
                mask=mask, verbose=verbose,
            )
        finally:
            self.noise_std = saved_noise

        if verbose:
            e_rms = float(np.sqrt(np.mean(kappa_E**2)))
            b_rms = float(np.sqrt(np.mean(kappa_B**2)))
            ratio = b_rms / (e_rms + 1e-30)
            print(f"E-mode RMS={e_rms:.4e}  B-mode RMS={b_rms:.4e}  "
                  f"B/E={ratio:.3f}  (small B/E => clean null test)")

        return kappa_E, kappa_B, result_E, result_B

    def bmode_diagnostics(self, gamma1_obs, gamma2_obs, mask=None,
                          region=None, clean_snr=1.0, contam_snr=2.0,
                          verbose=True):
        """
        Reconstruct E/B and return B-mode-based quality diagnostics.

        Since a real lensing signal is pure E-mode, the B-mode channel gives an
        independent handle on both the noise level and residual systematics:

          * delta_noise = RMS of the data with BOTH coherent modes removed,
            gamma_obs - F@kappa_E - rot^-1(F@kappa_B), estimates the
            per-component shear noise free of the E signal AND of any coherent
            B systematics, and can be fed back to Morozov as noise_std or used
            to cross-check the MAD estimate (which is biased high by the signal).
          * bmode_shear_rms = RMS(F @ kappa_B) is the coherent, reconstructable
            B-mode power. Theory says it is zero; a value rising above the noise
            floor (bmode_snr = bmode_shear_rms / delta_noise) flags systematics
            (PSF leakage, intrinsic alignments, additive shear bias).

        Why not the naive residuals: RMS(rotated_shear - F@kappa_B) and even
        RMS(gamma_obs - F@kappa_E) are both inflated by coherent B-mode shear,
        which F cannot fit and which therefore leaks into any single-mode
        residual. Removing both fits isolates the incoherent noise.

        NOTE: this is a *diagnostic only*. Do not add the B-mode to the loss --
        nulling it by construction destroys its value as an independent test.

        Parameters
        ----------
        region    : optional boolean node mask over which kappa RMS is reported
                    (e.g. a signal-free annulus); defaults to all nodes.
        clean_snr : bmode_snr below this => 'clean'.
        contam_snr: bmode_snr above this => 'contaminated'; between the two
                    thresholds => 'marginal'.

        Returns
        -------
        (BModeDiagnostics, kappa_E, kappa_B)
        """
        kappa_E, kappa_B, _, _ = self.reconstruct_eb(
            gamma1_obs, gamma2_obs, mask=mask, verbose=verbose)

        g1 = np.asarray(gamma1_obs, dtype=np.float64)
        g2 = np.asarray(gamma2_obs, dtype=np.float64)
        n2 = g1.size + g2.size

        def _rms2(a, b):
            return float(np.sqrt((np.dot(a, a) + np.dot(b, b)) / n2))

        # Fitted shear for each mode.
        e1, e2 = (np.asarray(a, dtype=np.float64) for a in self.ops.forward(kappa_E))
        b1, b2 = (np.asarray(a, dtype=np.float64) for a in self.ops.forward(kappa_B))

        emode_shear_rms = _rms2(e1, e2)
        bmode_shear_rms = _rms2(b1, b2)

        # Noise floor: remove BOTH coherent modes. The B-mode fit F@kappa_B lives
        # in the 45-deg-rotated frame; rotate it back (rot^-1: (a,b)->(-b,a)) to
        # subtract it in the original frame. What remains is incoherent noise.
        delta_noise = _rms2(g1 - e1 - (-b2), g2 - e2 - b1)
        from .regularization import estimate_noise_level
        delta_mad = estimate_noise_level(np.concatenate([g1, g2]), method='mad')

        bmode_snr = bmode_shear_rms / (delta_noise + 1e-30)
        if bmode_snr < clean_snr:
            flag = 'clean'
        elif bmode_snr < contam_snr:
            flag = 'marginal'
        else:
            flag = 'contaminated'

        sel = slice(None) if region is None else region
        diag = BModeDiagnostics(
            flag=flag,
            emode_shear_rms=emode_shear_rms,
            bmode_shear_rms=bmode_shear_rms,
            bmode_to_emode=bmode_shear_rms / (emode_shear_rms + 1e-30),
            bmode_snr=bmode_snr,
            delta_mad=delta_mad,
            delta_noise=delta_noise,
            delta_consistency=delta_noise / (delta_mad + 1e-30),
            kappa_e_rms=float(np.sqrt(np.mean(np.asarray(kappa_E)[sel]**2))),
            kappa_b_rms=float(np.sqrt(np.mean(np.asarray(kappa_B)[sel]**2))),
        )
        if verbose:
            print(diag.summary())
        return diag, kappa_E, kappa_B


def kaiser_squires(gamma1, gamma2, nodes, grid_size=64, return_bmode=False):
    """
    FFT-based Kaiser-Squires convergence reconstruction on FEM nodes.

    If return_bmode is True, returns (kappa_E, kappa_B) where kappa_B is the
    B-mode map (the estimator applied to the 45-deg-rotated shear); otherwise
    returns kappa_E only.
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
    k2     = KX**2 + KY**2
    k2[0, 0] = 1.0

    cos2 = (KX**2 - KY**2) / k2
    sin2 = 2.0 * KX * KY / k2

    kappa_grid   = np.real(sfft.ifft2(cos2 * G1k + sin2 * G2k))  # E-mode
    kappa_pts    = np.column_stack([XX.ravel(), YY.ravel()])
    kappa_E_nodes = griddata(kappa_pts, kappa_grid.ravel(), nodes,
                             method='linear', fill_value=0.0)
    if not return_bmode:
        return kappa_E_nodes

    kappa_b_grid  = np.real(sfft.ifft2(cos2 * G2k - sin2 * G1k))  # B-mode
    kappa_B_nodes = griddata(kappa_pts, kappa_b_grid.ravel(), nodes,
                             method='linear', fill_value=0.0)
    return kappa_E_nodes, kappa_B_nodes


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
