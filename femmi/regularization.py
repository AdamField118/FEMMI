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
                maxiter_inner=150, wiener_length=0.5, gtol_inner=1e-6,
                data_weight=None, prior=None):
    """
    Compute D(lambda) = ||F kappa_lambda - gamma_obs|| - c * delta.

    D(lambda) is monotone INCREASING in lambda:
      - large lambda -> over-smoothed -> residual large -> D > 0
      - small lambda -> over-fitted   -> residual small -> D < 0

    The Morozov parameter lambda* is the unique root D(lambda*) = 0.

    prior : optional non-Gaussian Prior (femmi.priors). NOTHING here needs the
    prior to be quadratic -- the discrepancy is evaluated by actually solving the
    MAP problem at each lambda and measuring the residual, which works for TV,
    sparsity, max-entropy or a learned score prior just as well as for Wiener.
    (Before this was threaded through, custom priors silently ran at a fixed
    lam_reg and were badly mis-scaled: TV/sparse/maxent scored 2.7-3.8 in shape
    L2 against Wiener's 0.31 in the benchmark grid.)

    data_weight : optional per-node weight; when given (e.g. a binary galaxy
    selection for a catalog-native mesh), the residual RMS is taken over the
    active (nonzero-weight) nodes only, so delta stays a per-component shear std.
    """
    from .inverse import MAPReconstructor
    from .forward import DifferentiableForward

    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(fwd, maxiter=maxiter_inner, gtol=gtol_inner,
                           callback_every=0, wiener_length=wiener_length,
                           data_weight=data_weight, prior=prior)
    kappa_lam, _ = rec.reconstruct(gamma1_obs, gamma2_obs, verbose=False)

    g1_pred, g2_pred = ops.forward(kappa_lam)
    r1 = g1_pred - gamma1_obs
    r2 = g2_pred - gamma2_obs
    if data_weight is None:
        num    = np.dot(r1, r1) + np.dot(r2, r2)
        n_data = len(gamma1_obs) + len(gamma2_obs)
    else:
        w      = np.asarray(data_weight, dtype=np.float64)
        num    = np.dot(w * r1, r1) + np.dot(w * r2, r2)
        n_data = 2 * int(np.count_nonzero(w))
    return float(np.sqrt(num / max(n_data, 1))) - c * delta


def lcurve_lambda(ops, gamma1_obs, gamma2_obs, lam_grid=None, wiener_length=0.5,
                  maxiter_inner=150, data_weight=None, prior=None, verbose=False):
    """Select lambda by the L-curve corner -- the maximum-curvature point of
    (log residual, log solution norm) as lambda varies.

    This is the fallback for when the discrepancy principle is INAPPLICABLE: if
    the smallest lambda in the bracket already leaves a residual above the assumed
    noise level, the model cannot explain the data to that tolerance and there is
    no root to find. Morozov's own guard then returns lam_min, which is the worst
    possible answer -- an essentially unregularised fit that amplifies noise. On a
    tapered log-normal field this measured 3.4x worse than the best lambda, and
    2x worse than simply taking lam_max.

    The L-curve needs no bracket and no reliable noise estimate; it just costs one
    MAP solve per grid point.
    """
    from .inverse import MAPReconstructor
    from .forward import DifferentiableForward

    lam_grid = np.asarray(lam_grid if lam_grid is not None
                          else np.logspace(-6, 1, 12), float)
    res, sol = [], []
    for lam in lam_grid:
        fwd = DifferentiableForward(ops, lam_reg=float(lam))
        rec = MAPReconstructor(fwd, maxiter=maxiter_inner, gtol=1e-6,
                               callback_every=0, wiener_length=wiener_length,
                               data_weight=data_weight, prior=prior)
        k, _ = rec.reconstruct(gamma1_obs, gamma2_obs, verbose=False)
        g1, g2 = ops.forward(k)
        r1 = g1 - gamma1_obs; r2 = g2 - gamma2_obs
        if data_weight is not None:
            w = np.asarray(data_weight, float)
            r = np.sqrt(np.dot(w * r1, r1) + np.dot(w * r2, r2))
        else:
            r = np.sqrt(np.dot(r1, r1) + np.dot(r2, r2))
        res.append(max(r, 1e-300)); sol.append(max(np.linalg.norm(k), 1e-300))

    x = np.log(np.asarray(res)); y = np.log(np.asarray(sol))
    # discrete curvature of the (x, y) trace; endpoints cannot be corners
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    curv = np.abs((dx * ddy - dy * ddx) / np.power(dx**2 + dy**2 + 1e-300, 1.5))
    curv[0] = curv[-1] = -np.inf          # endpoints are not corners (after abs)
    lam_star = float(lam_grid[int(np.argmax(curv))])
    if verbose:
        print(f"  L-curve corner: lambda* = {lam_star:.4e}")
    return lam_star


class MorozovSelector:
    """
    Select lambda by Morozov's discrepancy principle using Brent's method.

    Typical cost: 15-25 MAP solves (each at maxiter_inner iterations).

    Usage:
        selector = MorozovSelector(ops, noise_std=0.02)
        lam_star = selector.select(gamma1_obs, gamma2_obs)
    """

    def __init__(self, ops, noise_std=None, c=1.0, lam_min=1e-8, lam_max=10.0,
                 wiener_length=0.5, maxiter_inner=150, verbose=True,
                 data_weight=None, prior=None):
        self.ops           = ops
        self.noise_std     = noise_std
        self.c             = c
        self.lam_min       = lam_min
        self.lam_max       = lam_max
        self.wiener_length = wiener_length
        self.maxiter_inner = maxiter_inner
        self.verbose       = verbose
        self.data_weight   = data_weight
        self.prior         = prior

    def _D(self, lam, gamma1_obs, gamma2_obs, delta):
        t0  = time.perf_counter()
        val = discrepancy(lam, self.ops, gamma1_obs, gamma2_obs,
                          delta=delta, c=self.c,
                          maxiter_inner=self.maxiter_inner,
                          wiener_length=self.wiener_length,
                          data_weight=self.data_weight, prior=self.prior)
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
            # No root: even the least-regularised solution leaves a residual above
            # the assumed noise, i.e. MODEL error dominates and the discrepancy
            # principle does not apply. Returning lam_min here (the old behaviour)
            # is the worst available answer -- it hands back an essentially
            # unregularised fit that amplifies noise. Measured on a tapered
            # log-normal field: lam_min gave 1.55 shape L2 against 0.46 at the
            # best lambda, and even lam_max would have given 0.77. Fall back to
            # the L-curve corner, which needs no bracket.
            if self.verbose:
                print("  D(lam_lo) > 0: discrepancy target unreachable (model "
                      "error > assumed noise); falling back to the L-curve")
            return lcurve_lambda(
                self.ops, gamma1_obs, gamma2_obs, wiener_length=self.wiener_length,
                maxiter_inner=self.maxiter_inner, data_weight=self.data_weight,
                prior=self.prior, verbose=self.verbose)
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