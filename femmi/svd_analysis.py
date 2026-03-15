"""
femmi/svd_analysis.py
SVD of the forward operator F, Picard diagnostics, and inverse scattering
support-recovery indicators (factorization method and linear sampling method).

Reference: MATH.md sections 15-17, C&K chapters 5-6, 10.
"""

import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class SVDResult:
    """Truncated SVD of F. U shape (2*n_nodes, k), V shape (n_nodes, k)."""
    sigma    : np.ndarray
    U        : np.ndarray
    V        : np.ndarray
    residuals: np.ndarray
    n_nodes  : int


def compute_svd(ops, n_singular=40, method='lanczos', tol=1e-10, maxiter=None):
    """
    Compute leading n_singular singular triplets of F: L^2(Omega) -> L^2(Omega)^2.

    method='lanczos': ARPACK eigsh on F*F (default, scales to large meshes)
    method='dense':   form F explicitly (only for small meshes, n < 500)
    """
    n    = ops.n_nodes
    M    = ops.M
    S1   = ops.S1
    S2   = ops.S2
    A_lu = ops.A_coupled_lu
    idx_gauge = int(ops.bnd_mesh.node_indices[0])

    def _forward(kappa):
        rhs = -2.0 * M @ kappa
        rhs[idx_gauge] = 0.0
        psi = A_lu.solve(rhs)
        return np.concatenate([S1 @ psi, S2 @ psi])

    def _adjoint(g):
        g1, g2 = g[:n], g[n:]
        rhs = S1.T @ g1 + S2.T @ g2
        rhs[idx_gauge] = 0.0
        phi = A_lu.solve(rhs)
        return -2.0 * (M.T @ phi)

    if method == 'dense':
        if n > 1000:
            warnings.warn(f"compute_svd method='dense' with n={n} > 1000. Use method='lanczos'.")
        F_dense = np.zeros((2 * n, n))
        e = np.zeros(n)
        for i in range(n):
            e[:] = 0.0; e[i] = 1.0
            F_dense[:, i] = _forward(e)
        U_full, sigma_full, Vt_full = np.linalg.svd(F_dense, full_matrices=False)
        k     = min(n_singular, len(sigma_full))
        sigma = sigma_full[:k]
        U     = U_full[:, :k]
        V     = Vt_full[:k, :].T
    else:
        op  = spla.LinearOperator((n, n), matvec=lambda x: _adjoint(_forward(x)), dtype=np.float64)
        k   = min(n_singular, n - 2)
        eigenvalues, V = spla.eigsh(op, k=k, which='LM', tol=tol, maxiter=maxiter)
        idx            = np.argsort(eigenvalues)[::-1]
        eigenvalues    = np.maximum(eigenvalues[idx], 0.0)
        V              = V[:, idx]
        sigma          = np.sqrt(eigenvalues)

        U = np.zeros((2 * n, k))
        for i in range(k):
            if sigma[i] > 1e-14:
                U[:, i] = _forward(V[:, i]) / sigma[i]

    residuals = np.zeros(len(sigma))
    for i in range(len(sigma)):
        if sigma[i] > 1e-14:
            fv = _forward(V[:, i])
            residuals[i] = np.linalg.norm(fv - sigma[i] * U[:, i]) / sigma[i]

    return SVDResult(sigma=sigma, U=U, V=V, residuals=residuals, n_nodes=n)


def picard_plot(ops, gamma_obs, noise_std, n_singular=40, svd_result=None,
                save='picard.pdf', show=False):
    """
    Three-panel Picard diagnostic plot.

    Panel 1: log sigma_i vs i
    Panel 2: log |<gamma_obs, u_i>| vs i
    Panel 3: log(|<gamma_obs, u_i>| / sigma_i) vs i

    Returns dict with keys: svd, coeffs, ratio, picard_ok, cutoff_idx.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if svd_result is None:
        svd_result = compute_svd(ops, n_singular=n_singular)

    svd = svd_result
    n   = ops.n_nodes

    g_flat = np.asarray(gamma_obs, dtype=np.float64).ravel()
    if g_flat.shape[0] == n:
        g_flat = np.concatenate([g_flat, np.zeros(n)])
    elif g_flat.shape[0] != 2 * n:
        raise ValueError(f"gamma_obs shape mismatch: {gamma_obs.shape}, n_nodes={n}")

    coeffs = np.abs(svd.U.T @ g_flat)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(svd.sigma > 1e-14, coeffs / svd.sigma, np.nan)

    k   = len(svd.sigma)
    mid = slice(k // 5, 4 * k // 5)
    coeff_rate = np.polyfit(np.arange(k)[mid], np.log(np.maximum(coeffs[mid], 1e-20)), 1)[0]
    sigma_rate = np.polyfit(np.arange(k)[mid], np.log(np.maximum(svd.sigma[mid], 1e-20)), 1)[0]
    picard_ok  = coeff_rate < sigma_rate

    cutoff_idx = k - 1
    ratio_finite = np.where(np.isfinite(ratio), ratio, np.nan)
    for i in range(k - 3):
        if i + 6 <= k and (np.nanmean(ratio_finite[i:i+3]) < np.nanmean(ratio_finite[i+3:i+6])):
            cutoff_idx = i
            break

    idx         = np.arange(1, k + 1)
    noise_line  = noise_std * np.sqrt(2 * n)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Picard Diagnostic Plot", fontsize=13, fontweight='bold')

    axes[0].semilogy(idx, svd.sigma, 'b.-', lw=1.5, ms=4)
    axes[0].set_xlabel("Mode i"); axes[0].set_ylabel("sigma_i")
    axes[0].set_title("Singular values"); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(idx, coeffs, 'r.-', lw=1.5, ms=4, label="|<gamma_obs, u_i>|")
    axes[1].semilogy(idx, svd.sigma, 'b--', lw=1, alpha=0.5, label="sigma_i")
    axes[1].axhline(noise_line, color='gray', lw=1, ls=':', label="delta*sqrt(2n)")
    axes[1].set_xlabel("Mode i")
    axes[1].set_title(f"Fourier coefficients (Picard: {'ok' if picard_ok else 'FAIL'})")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(idx, ratio, 'g.-', lw=1.5, ms=4)
    axes[2].axvline(cutoff_idx + 1, color='red', lw=1.5, ls='--',
                    label=f"cutoff i={cutoff_idx+1}")
    axes[2].axhline(noise_line, color='gray', lw=1, ls=':')
    axes[2].set_xlabel("Mode i"); axes[2].set_ylabel("|<gamma_obs, u_i>| / sigma_i")
    axes[2].set_title("Amplified noise ratio")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    if show:
        plt.show()
    plt.close(fig)

    return {'svd': svd, 'coeffs': coeffs, 'ratio': ratio,
            'picard_ok': picard_ok, 'cutoff_idx': cutoff_idx}


def _probe_function(ops, z):
    """
    Compute Phi_z = F delta_z (shear pattern of a unit point mass at z).

    Returns (2*n_nodes,) stacked (gamma1, gamma2).
    """
    nodes     = np.array(ops.mesh.nodes)
    n         = ops.n_nodes
    M, S1, S2 = ops.M, ops.S1, ops.S2
    A_lu      = ops.A_coupled_lu
    idx_gauge = int(ops.bnd_mesh.node_indices[0])

    j   = int(np.argmin(np.sum((nodes - np.asarray(z)[None, :2])**2, axis=1)))
    m_jj = float(M[j, j])

    kappa_delta    = np.zeros(n)
    kappa_delta[j] = 1.0 / m_jj if m_jj > 1e-20 else 1.0

    rhs = -2.0 * M @ kappa_delta
    rhs[idx_gauge] = 0.0
    psi = A_lu.solve(rhs)
    return np.concatenate([S1 @ psi, S2 @ psi])


class FactorizationIndicator:
    """
    Support recovery via the Kirsch factorization method (C&K Thm 6.15).

    W(z)^{-1} = sum_{sigma_i > delta} |<Phi_z, u_i>|^2 / sigma_i

    W(z) is large inside support(kappa), small outside.
    """

    def __init__(self, ops, n_singular=40, noise_floor=None, svd_result=None):
        self.ops         = ops
        self.noise_floor = noise_floor
        if svd_result is not None:
            self.svd = svd_result
        else:
            print(f"Computing SVD (n={n_singular})...")
            self.svd = compute_svd(ops, n_singular=n_singular)
        self._setup_active_modes()

    def _setup_active_modes(self):
        if self.noise_floor is None:
            self.noise_floor = 0.01 * self.svd.sigma[0]
        active = self.svd.sigma > self.noise_floor
        self._sigma_active = self.svd.sigma[active]
        self._U_active     = self.svd.U[:, active]
        self.n_active      = int(active.sum())
        print(f"  {self.n_active}/{len(self.svd.sigma)} active modes "
              f"(noise_floor={self.noise_floor:.2e})")

    def probe_function(self, z):
        return _probe_function(self.ops, z)

    def indicator_map(self, test_points):
        """Evaluate W at all test_points (n_test, 2). Returns (n_test,) indicator."""
        test_points = np.asarray(test_points, dtype=np.float64)
        if test_points.ndim == 1:
            test_points = test_points[None, :]
        n_test = test_points.shape[0]
        W_inv  = np.zeros(n_test)

        for i, z in enumerate(test_points):
            phi_z    = self.probe_function(z)
            coeffs   = self._U_active.T @ phi_z
            W_inv[i] = float(np.sum(coeffs**2 / self._sigma_active))

        W = W_inv.copy()
        w_max = W.max()
        if w_max > 0:
            W /= w_max
        return W

    def plot(self, grid_size=64, domain=None, kappa_true=None,
             save='factorization_indicator.pdf', show=False):
        """Plot indicator map on a regular grid. Returns (grid_size, grid_size) array."""
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
        XX, YY   = np.meshgrid(xi, yi)
        test_pts = np.column_stack([XX.ravel(), YY.ravel()])

        print(f"Evaluating {len(test_pts)} grid points...")
        W_grid = self.indicator_map(test_pts).reshape(grid_size, grid_size)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(W_grid, origin='lower', extent=[xmin, xmax, ymin, ymax],
                       cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='W(z) (normalised)')
        if kappa_true is not None:
            kg = scipy_griddata(nodes, kappa_true, (XX, YY), method='linear')
            ax.contour(XX, YY, kg, levels=[0.3 * np.nanmax(kg)],
                       colors='cyan', linewidths=1.5, linestyles='--')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Factorization Method Indicator W(z)')
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight'); print(f"Saved: {save}")
        if show:
            plt.show()
        plt.close(fig)
        return W_grid


class LinearSamplingIndicator:
    """
    Support recovery via the linear sampling method (C&K section 5.5).

    I(z) = 1 / ||g_z^alpha|| where g_z^alpha = (F*F + alpha*I)^{-1} F* Phi_z.

    I(z) is large inside support(kappa), more stable near corners than
    the factorization method.
    """

    def __init__(self, ops, n_singular=40, alpha=None, svd_result=None):
        self.ops   = ops
        self.alpha = alpha
        if svd_result is not None:
            self.svd = svd_result
        else:
            print(f"Computing SVD (n={n_singular})...")
            self.svd = compute_svd(ops, n_singular=n_singular)
        if self.alpha is None:
            self.alpha = (0.01 * self.svd.sigma[0])**2
        print(f"  alpha = {self.alpha:.2e}")

    def probe_function(self, z):
        return _probe_function(self.ops, z)

    def indicator_map(self, test_points):
        """Evaluate I(z) = 1/||g_z^alpha|| at all test_points. Returns (n_test,)."""
        test_points = np.asarray(test_points, dtype=np.float64)
        if test_points.ndim == 1:
            test_points = test_points[None, :]
        n_test  = test_points.shape[0]
        I_vals  = np.zeros(n_test)
        sigma   = self.svd.sigma
        U       = self.svd.U
        filters = sigma / (sigma**2 + self.alpha)

        for i, z in enumerate(test_points):
            phi_z     = self.probe_function(z)
            coeffs    = U.T @ phi_z
            I_vals[i] = np.sqrt(float(np.sum((filters * coeffs)**2)))

        i_max = I_vals.max()
        if i_max > 0:
            I_vals /= i_max
        return I_vals

    def plot(self, grid_size=64, domain=None, kappa_true=None,
             save='lsm_indicator.pdf', show=False):
        """Plot LSM indicator map on a regular grid. Returns (grid_size, grid_size) array."""
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
        XX, YY   = np.meshgrid(xi, yi)
        test_pts = np.column_stack([XX.ravel(), YY.ravel()])

        print(f"Evaluating {len(test_pts)} grid points...")
        I_grid = self.indicator_map(test_pts).reshape(grid_size, grid_size)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(I_grid, origin='lower', extent=[xmin, xmax, ymin, ymax],
                       cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='I(z) (normalised)')
        if kappa_true is not None:
            kg = scipy_griddata(nodes, kappa_true, (XX, YY), method='linear')
            ax.contour(XX, YY, kg, levels=[0.3 * np.nanmax(kg)],
                       colors='cyan', linewidths=1.5, linestyles='--')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Linear Sampling Method Indicator I(z)')
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight'); print(f"Saved: {save}")
        if show:
            plt.show()
        plt.close(fig)
        return I_grid