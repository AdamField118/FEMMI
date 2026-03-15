"""
femmi/regularization.py
Automatic regularization parameter selection via Morozov's discrepancy principle.

Selects lambda such that ||F kappa_lambda - gamma_obs|| = c * delta,
where delta is the noise level and c ~ 1.

Reference: MATH.md section 13, C&K Thm 10.4.
"""

import numpy as np
import scipy.optimize as sopt
import time
from typing import Optional

from .operators import FEMOperators, build_wiener_regularizer


def estimate_noise_level(gamma_obs, method='mad'):
    """
    Estimate per-component noise std from observed shear.

    method='mad': 1.4826 * median(|gamma - median(gamma)|)  (robust)
    method='std': direct standard deviation
    """
    g = np.asarray(gamma_obs, dtype=np.float64).ravel()
    if method == 'mad':
        med = np.median(g)
        return 1.4826 * float(np.median(np.abs(g - med)))
    elif method == 'std':
        return float(np.std(g))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mad' or 'std'.")


def discrepancy(lam, ops, gamma1_obs, gamma2_obs, delta, c=1.0,
                maxiter_inner=150, wiener_length=0.5, gtol_inner=1e-6):
    """
    Compute D(lambda) = ||F kappa_lambda - gamma_obs|| - c * delta.

    D(lambda) is strictly monotone decreasing:
      - large lambda -> over-smoothed -> D > 0
      - small lambda -> over-fitted   -> D < 0

    The Morozov parameter lambda* is the unique root D(lambda*) = 0.
    """
    from .inverse import MAPReconstructor
    from .forward import DifferentiableForward

    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(fwd, maxiter=maxiter_inner, gtol=gtol_inner,
                           callback_every=0, wiener_length=wiener_length)
    kappa_lam, _ = rec.reconstruct(gamma1_obs, gamma2_obs, verbose=False)

    g1_pred, g2_pred = ops.forward(kappa_lam)
    r1     = g1_pred - gamma1_obs
    r2     = g2_pred - gamma2_obs
    n_data = len(gamma1_obs) + len(gamma2_obs)
    return float(np.sqrt((np.dot(r1, r1) + np.dot(r2, r2)) / n_data)) - c * delta


class MorozovSelector:
    """
    Select lambda by Morozov's discrepancy principle using Brent's method.

    Typical cost: 15-25 MAP solves (each at maxiter_inner iterations).

    Usage:
        selector = MorozovSelector(ops, noise_std=0.02)
        lam_star = selector.select(gamma1_obs, gamma2_obs)
    """

    def __init__(self, ops, noise_std=None, c=1.0, lam_min=1e-8, lam_max=10.0,
                 wiener_length=0.5, maxiter_inner=150, verbose=True):
        self.ops           = ops
        self.noise_std     = noise_std
        self.c             = c
        self.lam_min       = lam_min
        self.lam_max       = lam_max
        self.wiener_length = wiener_length
        self.maxiter_inner = maxiter_inner
        self.verbose       = verbose

    def _D(self, lam, gamma1_obs, gamma2_obs, delta):
        t0  = time.perf_counter()
        val = discrepancy(lam, self.ops, gamma1_obs, gamma2_obs,
                          delta=delta, c=self.c,
                          maxiter_inner=self.maxiter_inner,
                          wiener_length=self.wiener_length)
        if self.verbose:
            print(f"    lambda={lam:.3e}  D={val:+.4f}  ({time.perf_counter()-t0:.1f}s)")
        return val

    def select(self, gamma1_obs, gamma2_obs, noise_std=None):
        """
        Find lambda* via Brent's method on D(lambda) = 0.

        Returns the Morozov regularization parameter.
        """
        delta = noise_std or self.noise_std
        if delta is None:
            g_all = np.concatenate([gamma1_obs, gamma2_obs])
            delta = estimate_noise_level(g_all, method='mad')
            if self.verbose:
                print(f"  Estimated noise level delta={delta:.4e} (MAD)")

        if self.verbose:
            print(f"MorozovSelector: bracket=[{self.lam_min:.0e}, {self.lam_max:.0e}]  "
                  f"delta={delta:.4e}")

        t_total = time.perf_counter()
        D_lo    = self._D(self.lam_min, gamma1_obs, gamma2_obs, delta)
        D_hi    = self._D(self.lam_max, gamma1_obs, gamma2_obs, delta)

        if D_lo > 0:
            if self.verbose:
                print("  D(lam_lo) > 0 - returning lam_min")
            return self.lam_min
        if D_hi < 0:
            if self.verbose:
                print("  D(lam_hi) < 0 - returning lam_max")
            return self.lam_max

        lam_star = sopt.brentq(
            lambda lam: self._D(lam, gamma1_obs, gamma2_obs, delta),
            self.lam_min, self.lam_max,
            xtol=1e-8, rtol=1e-6, maxiter=30,
        )

        if self.verbose:
            D_star = self._D(lam_star, gamma1_obs, gamma2_obs, delta)
            print(f"  lambda* = {lam_star:.6e}  D(lambda*) = {D_star:+.2e}  "
                  f"total={time.perf_counter()-t_total:.1f}s")

        return lam_star

    def lcurve(self, gamma1_obs, gamma2_obs, n_points=20, noise_std=None):
        """
        Compute L-curve and discrepancy curve over n_points log-spaced lambda values.

        Returns dict with keys: lam, residual_norm, kappa_norm, discrepancy, delta.
        """
        from .inverse import MAPReconstructor
        from .forward import DifferentiableForward

        delta = noise_std or self.noise_std
        if delta is None:
            g_all = np.concatenate([gamma1_obs, gamma2_obs])
            delta = estimate_noise_level(g_all, method='mad')

        lam_vals  = np.logspace(np.log10(self.lam_min), np.log10(self.lam_max), n_points)
        res_norms = np.zeros(n_points)
        kap_norms = np.zeros(n_points)
        disc_vals = np.zeros(n_points)

        for i, lam in enumerate(lam_vals):
            fwd = DifferentiableForward(self.ops, lam_reg=lam)
            rec = MAPReconstructor(fwd, maxiter=self.maxiter_inner, gtol=1e-6,
                                   callback_every=0, wiener_length=self.wiener_length)
            kappa_lam, _ = rec.reconstruct(gamma1_obs, gamma2_obs, verbose=False)

            g1p, g2p = self.ops.forward(kappa_lam)
            r1 = g1p - gamma1_obs
            r2 = g2p - gamma2_obs
            n_data = len(gamma1_obs) + len(gamma2_obs)
            rn     = float(np.sqrt((np.dot(r1, r1) + np.dot(r2, r2)) / n_data))

            res_norms[i] = rn
            kap_norms[i] = float(np.linalg.norm(kappa_lam))
            disc_vals[i] = rn - self.c * delta

            print(f"  [{i+1:2d}/{n_points}] lambda={lam:.2e}  "
                  f"res={rn:.4f}  ||kappa||={kap_norms[i]:.4f}")

        return {
            'lam': lam_vals, 'residual_norm': res_norms,
            'kappa_norm': kap_norms, 'discrepancy': disc_vals, 'delta': delta,
        }