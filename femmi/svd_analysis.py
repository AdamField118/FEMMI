"""
femmi/svd_analysis.py
======================
Phase 4: SVD of the forward operator F, Picard diagnostics, and
inverse scattering support-recovery indicators.

The forward operator F: L²(Ω) → L²(Ω)²  maps  κ ↦ (γ₁, γ₂).
Since F is compact, it admits a singular value decomposition
    F = Σᵢ σᵢ uᵢ ⊗ vᵢ*,    σ₁ ≥ σ₂ ≥ ··· → 0
where uᵢ are left singular functions (shear patterns) and vᵢ are right
singular functions (mass patterns).  See MATH.md §15.

Public API
----------
    SVDResult                           – named result container
    compute_svd(ops, n_singular, ...)   → SVDResult
    picard_plot(ops, gamma_obs, ...)    – three-panel Picard diagnostic
    FactorizationIndicator              – Kirsch support recovery
    LinearSamplingIndicator             – LSM support recovery

References
----------
  C&K §6.2 Thm 6.15  (factorization method)
  C&K §5.5            (linear sampling method)
  C&K §10.1           (SVD, Picard condition)
  MATH.md §15–17
"""

import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Optional
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVDResult:
    """
    Truncated SVD of the forward operator F.

    Attributes
    ----------
    sigma   : (n_singular,)     singular values in descending order
    U       : (2*n_nodes, n_singular)  left singular vectors (stacked γ₁, γ₂)
    V       : (n_nodes, n_singular)    right singular vectors (κ patterns)
    residuals: (n_singular,)    ‖F vᵢ − σᵢ uᵢ‖ / σᵢ  (relative residuals)
    n_nodes : int
    """
    sigma    : np.ndarray
    U        : np.ndarray
    V        : np.ndarray
    residuals: np.ndarray
    n_nodes  : int


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Truncated SVD of F
# ─────────────────────────────────────────────────────────────────────────────

def compute_svd(ops,
                n_singular: int = 40,
                method: str = "lanczos",
                tol: float = 1e-10,
                maxiter: Optional[int] = None) -> SVDResult:
    """
    Compute leading n_singular singular triplets (σᵢ, uᵢ, vᵢ) of F.

    F: L²(Ω) → L²(Ω)²,   F κ = (γ₁, γ₂) = S · A_coupled⁻¹ · (−2M) κ

    Strategy
    --------
    SVD via the self-adjoint eigenvalue problem on F*F:
        F*F vᵢ = σᵢ² vᵢ,    then uᵢ = F vᵢ / σᵢ

    F*F matvec (one forward + one adjoint per call):
        x ↦ F*(F x) = −2Mᵀ A_coupled⁻ᵀ (S₁ᵀ(S₁ψ) + S₂ᵀ(S₂ψ))
        where ψ = A_coupled⁻¹(−2Mx)

    method='lanczos': scipy ARPACK eigsh on the F*F LinearOperator.
    method='dense':   form F explicitly; only for small meshes (n < 500).

    Parameters
    ----------
    ops        : FEMOperators
    n_singular : number of singular triplets to compute
    method     : 'lanczos' (default) or 'dense'
    tol        : ARPACK convergence tolerance
    maxiter    : maximum ARPACK iterations (None = ARPACK default)

    Returns
    -------
    SVDResult
    """
    n = ops.n_nodes
    M  = ops.M
    S1 = ops.S1
    S2 = ops.S2
    A_lu = ops.A_coupled_lu
    idx_gauge = int(ops.bnd_mesh.node_indices[0])

    def _forward(kappa: np.ndarray) -> np.ndarray:
        """κ → (γ₁, γ₂) stacked as (2n,)"""
        rhs = -2.0 * M @ kappa
        rhs[idx_gauge] = 0.0
        psi = A_lu.solve(rhs)
        return np.concatenate([S1 @ psi, S2 @ psi])

    def _adjoint(g: np.ndarray) -> np.ndarray:
        """(γ₁, γ₂) stacked as (2n,) → κ"""
        g1, g2 = g[:n], g[n:]
        rhs = S1.T @ g1 + S2.T @ g2
        rhs[idx_gauge] = 0.0
        phi = A_lu.solve(rhs)
        return -2.0 * (M.T @ phi)

    def _FstarF(x: np.ndarray) -> np.ndarray:
        return _adjoint(_forward(x))

    if method == "dense":
        if n > 1000:
            warnings.warn(
                f"compute_svd method='dense' with n={n} > 1000. "
                "This will be slow. Use method='lanczos'.")
        F_dense = np.zeros((2 * n, n))
        e = np.zeros(n)
        for i in range(n):
            e[:] = 0.0; e[i] = 1.0
            F_dense[:, i] = _forward(e)
        U_full, sigma_full, Vt_full = np.linalg.svd(F_dense, full_matrices=False)
        k = min(n_singular, len(sigma_full))
        sigma = sigma_full[:k]
        U     = U_full[:, :k]
        V     = Vt_full[:k, :].T

    else:  # lanczos via ARPACK
        op = spla.LinearOperator((n, n), matvec=_FstarF, dtype=np.float64)
        k  = min(n_singular, n - 2)
        eigenvalues, V = spla.eigsh(op, k=k, which='LM',
                                    tol=tol, maxiter=maxiter)
        # eigsh may return negative eigenvalues (numerical noise near zero)
        # sort descending, clip negatives
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V           = V[:, idx]
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sigma       = np.sqrt(eigenvalues)

        # Compute left singular vectors u_i = F v_i / σ_i
        U = np.zeros((2 * n, k))
        for i in range(k):
            if sigma[i] > 1e-14:
                U[:, i] = _forward(V[:, i]) / sigma[i]
            else:
                U[:, i] = 0.0

    # Relative residuals ‖F vᵢ − σᵢ uᵢ‖ / σᵢ
    residuals = np.zeros(len(sigma))
    for i in range(len(sigma)):
        if sigma[i] > 1e-14:
            fv = _forward(V[:, i])
            residuals[i] = np.linalg.norm(fv - sigma[i] * U[:, i]) / sigma[i]

    return SVDResult(sigma=sigma, U=U, V=V, residuals=residuals, n_nodes=n)


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Picard plot
# ─────────────────────────────────────────────────────────────────────────────

def picard_plot(ops,
                gamma_obs: np.ndarray,
                noise_std: float,
                n_singular: int = 40,
                svd_result: Optional[SVDResult] = None,
                save: Optional[str] = "picard.pdf",
                show: bool = False) -> dict:
    """
    Three-panel Picard diagnostic plot.

    Panel 1: log σᵢ vs i           (singular value decay)
    Panel 2: log|⟨γ_obs, uᵢ⟩| vs i (Fourier coefficients of data)
    Panel 3: log(|⟨γ_obs, uᵢ⟩|/σᵢ) vs i (amplified noise)

    The Picard condition holds if panel 2 decays faster than panel 1.
    The crossover in panel 3 (where the ratio starts rising) gives the
    effective regularisation cutoff.

    Parameters
    ----------
    ops        : FEMOperators
    gamma_obs  : (2, n_nodes) or (2*n_nodes,) observed shear
    noise_std  : per-component noise standard deviation δ
    n_singular : number of singular values (ignored if svd_result given)
    svd_result : precomputed SVDResult (reuse to avoid recomputation)
    save       : filename to save figure (None = don't save)
    show       : call plt.show() after plotting

    Returns
    -------
    dict with keys:
        'svd'        : SVDResult used
        'coeffs'     : |⟨γ_obs, uᵢ⟩|
        'ratio'      : |⟨γ_obs, uᵢ⟩| / σᵢ
        'picard_ok'  : bool, True if coefficients decay faster than σᵢ
        'cutoff_idx' : index where ratio starts rising (noise floor crossing)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if svd_result is None:
        print(f"Computing SVD (n_singular={n_singular})...")
        svd_result = compute_svd(ops, n_singular=n_singular)

    svd = svd_result
    n   = ops.n_nodes

    # Flatten gamma_obs to (2n,)
    g_flat = np.asarray(gamma_obs, dtype=np.float64).ravel()
    if g_flat.shape[0] == 2 * n:
        pass
    elif g_flat.shape[0] == n:
        # single component provided — duplicate (shouldn't happen in normal use)
        g_flat = np.concatenate([g_flat, np.zeros(n)])
    else:
        raise ValueError(f"gamma_obs shape mismatch: {gamma_obs.shape}, n_nodes={n}")

    # Fourier coefficients |⟨γ_obs, uᵢ⟩|
    coeffs = np.abs(svd.U.T @ g_flat)          # (k,)

    # Amplified noise ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(svd.sigma > 1e-14, coeffs / svd.sigma, np.nan)

    # Picard condition: check if coefficients decay faster than sigma
    # over the central 60% of modes
    k = len(svd.sigma)
    mid = slice(k // 5, 4 * k // 5)
    coeff_rate = np.polyfit(np.arange(k)[mid],
                            np.log(np.maximum(coeffs[mid], 1e-20)), 1)[0]
    sigma_rate = np.polyfit(np.arange(k)[mid],
                            np.log(np.maximum(svd.sigma[mid], 1e-20)), 1)[0]
    picard_ok = coeff_rate < sigma_rate  # both negative; coeff steeper = faster decay

    # Noise floor crossing: where ratio first starts rising consistently
    ratio_finite = np.where(np.isfinite(ratio), ratio, np.nan)
    cutoff_idx = k - 1
    for i in range(k - 3):
        if (np.nanmean(ratio_finite[i:i+3]) < np.nanmean(ratio_finite[i+3:i+6])
                if i + 6 <= k else False):
            cutoff_idx = i
            break

    # ── Plot ─────────────────────────────────────────────────────────────────
    idx = np.arange(1, k + 1)
    noise_line = noise_std * np.sqrt(2 * n)   # expected noise contribution

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Picard Diagnostic Plot", fontsize=13, fontweight='bold')

    axes[0].semilogy(idx, svd.sigma, 'b.-', lw=1.5, ms=4)
    axes[0].set_xlabel("Mode index i"); axes[0].set_ylabel("σᵢ")
    axes[0].set_title("Singular values")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(idx, coeffs, 'r.-', lw=1.5, ms=4, label="|⟨γ_obs, uᵢ⟩|")
    axes[1].semilogy(idx, svd.sigma, 'b--', lw=1, alpha=0.5, label="σᵢ")
    axes[1].axhline(noise_line, color='gray', lw=1, ls=':', label=f"δ√(2n)")
    axes[1].set_xlabel("Mode index i"); axes[1].set_ylabel("Coefficient")
    axes[1].set_title(f"Fourier coefficients  (Picard: {'✓' if picard_ok else '✗'})")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(idx, ratio, 'g.-', lw=1.5, ms=4)
    axes[2].axvline(cutoff_idx + 1, color='red', lw=1.5, ls='--',
                    label=f"Cutoff i={cutoff_idx+1}")
    axes[2].axhline(noise_line, color='gray', lw=1, ls=':', label=f"δ√(2n)")
    axes[2].set_xlabel("Mode index i"); axes[2].set_ylabel("|⟨γ_obs, uᵢ⟩| / σᵢ")
    axes[2].set_title("Amplified noise ratio")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Picard plot saved: {save}")
    if show:
        plt.show()
    plt.close(fig)

    return {
        'svd'        : svd,
        'coeffs'     : coeffs,
        'ratio'      : ratio,
        'picard_ok'  : picard_ok,
        'cutoff_idx' : cutoff_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe function: shear of point mass at z
# ─────────────────────────────────────────────────────────────────────────────

def _probe_function(ops, z: np.ndarray) -> np.ndarray:
    """
    Φ_z = F δ_z = shear pattern of a unit point mass at z.

    Discretised: δ_z is approximated as a P3 nodal basis function
    contribution at the nearest node to z (scaled by 1/area to give
    unit integrated mass).

    Returns (2*n_nodes,) stacked (γ₁, γ₂).
    """
    nodes = np.array(ops.mesh.nodes)
    n     = ops.n_nodes
    M     = ops.M
    S1    = ops.S1
    S2    = ops.S2
    A_lu  = ops.A_coupled_lu
    idx_gauge = int(ops.bnd_mesh.node_indices[0])

    # Find nearest node to z
    dists = np.sum((nodes - np.asarray(z)[None, :2])**2, axis=1)
    j     = int(np.argmin(dists))

    # κ = e_j / M[j,j]  so that ∫ N_j κ dA = 1 (unit mass)
    kappa_delta      = np.zeros(n)
    m_jj             = float(M[j, j])
    kappa_delta[j]   = 1.0 / m_jj if m_jj > 1e-20 else 1.0

    rhs = -2.0 * M @ kappa_delta
    rhs[idx_gauge] = 0.0
    psi = A_lu.solve(rhs)
    return np.concatenate([S1 @ psi, S2 @ psi])


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  Factorization method indicator
# ─────────────────────────────────────────────────────────────────────────────

class FactorizationIndicator:
    """
    Support recovery via the Kirsch factorization method [C&K Thm 6.15].

        W(z)⁻¹ = Σ_{σᵢ > δ}  |⟨Φ_z, uᵢ⟩|² / σᵢ

    W(z) is large inside support(κ) and small outside.

    Usage
    -----
        fi = FactorizationIndicator(ops, n_singular=40)
        W  = fi.indicator_map(test_points)   # (n_test,) — large inside support
        fi.plot(mesh_or_grid)

    Parameters
    ----------
    ops          : FEMOperators
    n_singular   : number of SVD modes to use
    noise_floor  : threshold below which σᵢ are considered noise (default: auto)
    svd_result   : precomputed SVDResult (avoids recomputation)
    """

    def __init__(self,
                 ops,
                 n_singular: int = 40,
                 noise_floor: Optional[float] = None,
                 svd_result: Optional[SVDResult] = None):
        self.ops   = ops
        self.noise_floor = noise_floor
        if svd_result is not None:
            self.svd = svd_result
        else:
            print(f"[FactorizationIndicator] Computing SVD (n={n_singular})...")
            self.svd = compute_svd(ops, n_singular=n_singular)
        self._setup_active_modes()

    def _setup_active_modes(self):
        svd = self.svd
        if self.noise_floor is None:
            # Auto: use modes with σ > 1% of σ_max
            self.noise_floor = 0.01 * svd.sigma[0]
        active = svd.sigma > self.noise_floor
        self._sigma_active = svd.sigma[active]
        self._U_active     = svd.U[:, active]   # (2n, k_active)
        self.n_active      = int(active.sum())
        print(f"[FactorizationIndicator] {self.n_active}/{len(svd.sigma)} "
              f"active modes (noise_floor={self.noise_floor:.2e})")

    def probe_function(self, z: np.ndarray) -> np.ndarray:
        """Φ_z = shear of a unit point mass at z. Returns (2*n_nodes,)."""
        return _probe_function(self.ops, z)

    def indicator_map(self, test_points: np.ndarray) -> np.ndarray:
        """
        Evaluate W at all test points.

        Parameters
        ----------
        test_points : (n_test, 2) array of (x, y) locations

        Returns
        -------
        W : (n_test,) indicator — large inside support(κ), small outside
        """
        test_points = np.asarray(test_points, dtype=np.float64)
        if test_points.ndim == 1:
            test_points = test_points[None, :]
        n_test = test_points.shape[0]
        W_inv  = np.zeros(n_test)

        for i, z in enumerate(test_points):
            phi_z  = self.probe_function(z)                   # (2n,)
            coeffs = self._U_active.T @ phi_z                 # (k_active,)
            W_inv[i] = float(np.sum(coeffs**2 / self._sigma_active))

        W = W_inv.copy()
        w_max = W.max()
        if w_max > 0:
            W /= w_max
        return W

    def plot(self, grid_size: int = 64,
             domain: Optional[tuple] = None,
             kappa_true: Optional[np.ndarray] = None,
             save: Optional[str] = "factorization_indicator.pdf",
             show: bool = False) -> np.ndarray:
        """
        Plot the indicator map on a regular grid.

        Parameters
        ----------
        grid_size  : number of grid points per axis
        domain     : (xmin, xmax, ymin, ymax) — defaults to ops mesh domain
        kappa_true : optional true κ for overlay contour
        save       : save path
        show       : call plt.show()

        Returns
        -------
        W_grid : (grid_size, grid_size) indicator values
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata as scipy_griddata

        nodes = np.array(self.ops.mesh.nodes)
        if domain is None:
            xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
            ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
        else:
            xmin, xmax, ymin, ymax = domain

        xi = np.linspace(xmin, xmax, grid_size)
        yi = np.linspace(ymin, ymax, grid_size)
        XX, YY = np.meshgrid(xi, yi)
        test_pts = np.column_stack([XX.ravel(), YY.ravel()])

        print(f"[FactorizationIndicator.plot] Evaluating {len(test_pts)} grid points...")
        W_flat = self.indicator_map(test_pts)
        W_grid = W_flat.reshape(grid_size, grid_size)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(W_grid, origin='lower',
                       extent=[xmin, xmax, ymin, ymax],
                       cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='W(z) (normalised)')

        if kappa_true is not None:
            kappa_grid = scipy_griddata(nodes, kappa_true, (XX, YY), method='linear')
            ax.contour(XX, YY, kappa_grid,
                       levels=[0.3 * np.nanmax(kappa_grid)],
                       colors='cyan', linewidths=1.5, linestyles='--')

        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Factorization Method Indicator W(z)')
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved: {save}")
        if show:
            plt.show()
        plt.close(fig)
        return W_grid


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Linear sampling method indicator
# ─────────────────────────────────────────────────────────────────────────────

class LinearSamplingIndicator:
    """
    Support recovery via the linear sampling method [C&K §5.5].

        I(z) = 1 / ‖g_z^α‖

    where g_z^α = (FᵀF + α I)⁻¹ Fᵀ Φ_z is the Tikhonov solution to F g = Φ_z.

    In SVD form:
        ‖g_z^α‖² = Σᵢ (σᵢ / (σᵢ² + α))² |⟨Φ_z, uᵢ⟩|²

    I(z) is large inside support(κ) and small outside.
    More numerically stable than the factorization method near corners/edges.

    Parameters
    ----------
    ops          : FEMOperators
    n_singular   : number of SVD modes
    alpha        : Tikhonov parameter (None = auto: 1% of σ_max²)
    svd_result   : precomputed SVDResult
    """

    def __init__(self,
                 ops,
                 n_singular: int = 40,
                 alpha: Optional[float] = None,
                 svd_result: Optional[SVDResult] = None):
        self.ops   = ops
        self.alpha = alpha
        if svd_result is not None:
            self.svd = svd_result
        else:
            print(f"[LinearSamplingIndicator] Computing SVD (n={n_singular})...")
            self.svd = compute_svd(ops, n_singular=n_singular)

        if self.alpha is None:
            self.alpha = (0.01 * self.svd.sigma[0])**2
        print(f"[LinearSamplingIndicator] α = {self.alpha:.2e}")

    def probe_function(self, z: np.ndarray) -> np.ndarray:
        return _probe_function(self.ops, z)

    def indicator_map(self, test_points: np.ndarray) -> np.ndarray:
        """
        Evaluate I(z) = 1/‖g_z^α‖ at all test points.

        Returns
        -------
        I : (n_test,) indicator — large inside support(κ), small outside
        """
        test_points = np.asarray(test_points, dtype=np.float64)
        if test_points.ndim == 1:
            test_points = test_points[None, :]
        n_test = test_points.shape[0]
        I_vals = np.zeros(n_test)

        sigma = self.svd.sigma
        U     = self.svd.U
        alpha = self.alpha

        # Tikhonov filter: σᵢ / (σᵢ² + α)
        filters = sigma / (sigma**2 + alpha)   # (k,)

        for i, z in enumerate(test_points):
            phi_z      = self.probe_function(z)          # (2n,)
            coeffs     = U.T @ phi_z                     # (k,)
            g_norm_sq  = float(np.sum((filters * coeffs)**2))
            I_vals[i] = np.sqrt(g_norm_sq)

        # Normalise to [0, 1]
        i_max = I_vals.max()
        if i_max > 0:
            I_vals /= i_max
        return I_vals

    def plot(self, grid_size: int = 64,
             domain: Optional[tuple] = None,
             kappa_true: Optional[np.ndarray] = None,
             save: Optional[str] = "lsm_indicator.pdf",
             show: bool = False) -> np.ndarray:
        """Plot the LSM indicator map on a regular grid."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata as scipy_griddata

        nodes = np.array(self.ops.mesh.nodes)
        if domain is None:
            xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
            ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
        else:
            xmin, xmax, ymin, ymax = domain

        xi = np.linspace(xmin, xmax, grid_size)
        yi = np.linspace(ymin, ymax, grid_size)
        XX, YY = np.meshgrid(xi, yi)
        test_pts = np.column_stack([XX.ravel(), YY.ravel()])

        print(f"[LinearSamplingIndicator.plot] Evaluating {len(test_pts)} grid points...")
        I_flat = self.indicator_map(test_pts)
        I_grid = I_flat.reshape(grid_size, grid_size)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(I_grid, origin='lower',
                       extent=[xmin, xmax, ymin, ymax],
                       cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='I(z) (normalised)')

        if kappa_true is not None:
            kappa_grid = scipy_griddata(nodes, kappa_true, (XX, YY), method='linear')
            ax.contour(XX, YY, kappa_grid,
                       levels=[0.3 * np.nanmax(kappa_grid)],
                       colors='cyan', linewidths=1.5, linestyles='--')

        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Linear Sampling Method Indicator I(z)')
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved: {save}")
        if show:
            plt.show()
        plt.close(fig)
        return I_grid