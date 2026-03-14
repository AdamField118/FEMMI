"""
femmi/regularization.py
========================
Automatic regularization parameter selection via Morozov's discrepancy
principle.

The Morozov discrepancy principle selects λ such that the reconstruction
residual matches the noise level:

    ‖F κ_λ − γ_obs‖ = c δ

where δ is the noise level and c = O(1) (typically 1).  This is
provably optimal: as δ → 0, ‖κ_{λ*} − κ_true‖ → 0 [C&K Thm 10.4].

Public API
----------
    estimate_noise_level(gamma_obs, method='mad') → float
    discrepancy(lam, ops, fwd, gamma1_obs, gamma2_obs, delta,
                c=1.0, maxiter_inner=150, wiener_length=0.5) → float
    MorozovSelector                     – Brent root-finding on D(λ)
        .select(gamma1_obs, gamma2_obs) → float
        .lcurve(gamma1_obs, gamma2_obs, n_points=30) → dict
"""

import numpy as np
import scipy.optimize as sopt
import time
from typing import Optional, Tuple

from .operators import FEMOperators, build_wiener_regularizer


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Noise level estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_noise_level(gamma_obs: np.ndarray,
                         method: str = "mad") -> float:
    """
    Estimate per-component noise standard deviation δ from observed shear.

    Parameters
    ----------
    gamma_obs : (n_nodes,) or (2, n_nodes) array
        Observed shear component(s).  If 2D, both components are pooled.
    method : str
        'mad'  – Median Absolute Deviation (robust, default):
                 δ = 1.4826 × median(|γ − median(γ)|)
                 The factor 1.4826 = 1/Φ⁻¹(0.75) makes it a consistent
                 estimator of σ for Gaussian noise.
        'std'  – Direct standard deviation (sensitive to outliers).

    Returns
    -------
    float
        Estimated noise standard deviation δ ≥ 0.

    Notes
    -----
    For weak-lensing catalogs, MAD is preferred because the shear field
    contains real structure that inflates the standard deviation.  MAD
    measures the noise floor more robustly.

    If γ_obs = γ_true + η where η ~ N(0, σ²), then MAD → σ as N → ∞.
    """
    g = np.asarray(gamma_obs, dtype=np.float64).ravel()

    if method == "mad":
        med = np.median(g)
        return 1.4826 * float(np.median(np.abs(g - med)))
    elif method == "std":
        return float(np.std(g))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mad' or 'std'.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Discrepancy functional D(λ)
# ─────────────────────────────────────────────────────────────────────────────

def discrepancy(lam: float,
                ops: FEMOperators,
                gamma1_obs: np.ndarray,
                gamma2_obs: np.ndarray,
                delta: float,
                c: float = 1.0,
                maxiter_inner: int = 150,
                wiener_length: float = 0.5,
                gtol_inner: float = 1e-6) -> float:
    """
    Compute D(λ) = ‖F κ_λ − γ_obs‖ − c δ.

    κ_λ is the MAP solution at regularisation parameter λ:
        κ_λ = argmin_κ ‖Fκ − γ_obs‖² + λ κᵀR κ

    Properties
    ----------
    D(λ) is strictly monotone decreasing in λ (C&K §13):
      - Large λ → over-smoothed κ_λ → large residual → D > 0
      - Small λ → over-fitted κ_λ → small residual → D < 0

    The Morozov regularisation parameter λ* is the unique root D(λ*) = 0.

    Parameters
    ----------
    lam          : regularisation parameter (> 0)
    ops          : FEMOperators (assembled A_coupled, M, S1, S2)
    gamma1_obs   : (n_nodes,) observed γ₁
    gamma2_obs   : (n_nodes,) observed γ₂
    delta        : noise level estimate
    c            : discrepancy constant (default 1.0)
    maxiter_inner: L-BFGS iterations for the inner MAP solve.
                   150 is sufficient for λ-selection; use 500 for the
                   final reconstruction after λ* is found.
    wiener_length: correlation length for Matérn prior R = M + l²K.
    gtol_inner   : gradient tolerance for inner L-BFGS solve.

    Returns
    -------
    float
        D(λ) = ‖F κ_λ − γ_obs‖ − c δ.
        Negative if λ too small (over-fitting), positive if too large.
    """
    from .inverse import MAPReconstructor
    from .forward import DifferentiableForward

    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(
        fwd,
        maxiter=maxiter_inner,
        gtol=gtol_inner,
        callback_every=0,
        wiener_length=wiener_length,
    )
    kappa_lam, _ = rec.reconstruct(gamma1_obs, gamma2_obs, verbose=False)

    # Compute forward residual ‖F κ_λ − γ_obs‖
    g1_pred, g2_pred = ops.forward(kappa_lam)
    r1 = g1_pred - gamma1_obs
    r2 = g2_pred - gamma2_obs
    n_data = len(gamma1_obs) + len(gamma2_obs)
    residual_norm = float(np.sqrt((np.dot(r1, r1) + np.dot(r2, r2)) / n_data))

    return residual_norm - c * delta


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  MorozovSelector
# ─────────────────────────────────────────────────────────────────────────────

class MorozovSelector:
    """
    Select the regularisation parameter λ by Morozov's discrepancy
    principle using Brent's root-finding algorithm.

    Algorithm
    ---------
    1. Evaluate D(λ) at a bracket [λ_lo, λ_hi] such that:
           D(λ_lo) < 0   (small λ → over-fitting)
           D(λ_hi) > 0   (large λ → over-smoothing)
    2. Apply scipy.optimize.brentq to find λ* with D(λ*) ≈ 0.
    3. Typical cost: 15–25 MAP solves (each with maxiter_inner iterations).

    Usage
    -----
        selector = MorozovSelector(ops, noise_std=0.02)
        lam_star = selector.select(gamma1_obs, gamma2_obs)
        # then run full MAP at lam_star:
        rec = MAPReconstructor(DifferentiableForward(ops, lam_reg=lam_star), ...)

    Parameters
    ----------
    ops           : FEMOperators
    noise_std     : noise level δ.  If None, estimated via MAD.
    c             : discrepancy constant (default 1.0; see MATH.md §13.1)
    lam_min       : lower bound for λ bracket (default 1e-8)
    lam_max       : upper bound for λ bracket (default 1.0)
    wiener_length : Matérn prior correlation length (default 0.5)
    maxiter_inner : L-BFGS iterations per λ trial (default 150)
    verbose       : print progress (default True)
    """

    def __init__(self,
                 ops: FEMOperators,
                 noise_std: Optional[float] = None,
                 c: float = 1.0,
                 lam_min: float = 1e-8,
                 lam_max: float = 10.0,
                 wiener_length: float = 0.5,
                 maxiter_inner: int = 150,
                 verbose: bool = True):
        self.ops           = ops
        self.noise_std     = noise_std
        self.c             = c
        self.lam_min       = lam_min
        self.lam_max       = lam_max
        self.wiener_length = wiener_length
        self.maxiter_inner = maxiter_inner
        self.verbose       = verbose

    # ─────────────────────────────────────────────────────────────────────────

    def _D(self, lam: float,
           gamma1_obs: np.ndarray,
           gamma2_obs: np.ndarray,
           delta: float) -> float:
        """Evaluate D(λ) with progress logging."""
        t0 = time.perf_counter()
        val = discrepancy(
            lam, self.ops, gamma1_obs, gamma2_obs,
            delta=delta, c=self.c,
            maxiter_inner=self.maxiter_inner,
            wiener_length=self.wiener_length,
        )
        if self.verbose:
            print(f"    λ={lam:.3e}  D(λ)={val:+.4f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
        return val

    # ─────────────────────────────────────────────────────────────────────────

    def select(self,
               gamma1_obs: np.ndarray,
               gamma2_obs: np.ndarray,
               noise_std: Optional[float] = None) -> float:
        """
        Find λ* by Brent's method on D(λ) = ‖Fκ_λ − γ_obs‖ − c δ.

        Parameters
        ----------
        gamma1_obs, gamma2_obs : (n_nodes,) observed shear components
        noise_std : override noise level (uses self.noise_std if None)

        Returns
        -------
        lam_star : float
            The Morozov regularisation parameter satisfying D(λ*) ≈ 0.
        """
        # Noise level
        delta = noise_std or self.noise_std
        if delta is None:
            g_all = np.concatenate([gamma1_obs, gamma2_obs])
            delta = estimate_noise_level(g_all, method="mad")
            if self.verbose:
                print(f"  Estimated noise level δ = {delta:.4e}  (MAD)")

        if self.verbose:
            print("=" * 55)
            print("MorozovSelector: finding λ* via Brent's method")
            print(f"  δ = {delta:.4e},  c = {self.c}")
            print(f"  bracket [{self.lam_min:.0e}, {self.lam_max:.0e}]")
            print(f"  maxiter_inner = {self.maxiter_inner}")
            print("=" * 55)

        t_total = time.perf_counter()

        # Check bracket signs
        D_lo = self._D(self.lam_min, gamma1_obs, gamma2_obs, delta)
        D_hi = self._D(self.lam_max, gamma1_obs, gamma2_obs, delta)

        if self.verbose:
            print(f"  D(λ_lo={self.lam_min:.0e}) = {D_lo:+.4f}")
            print(f"  D(λ_hi={self.lam_max:.0e}) = {D_hi:+.4f}")

        # Handle degenerate cases
        if D_lo > 0:
            if self.verbose:
                print("  WARNING: D(λ_lo) > 0 — data residual too large even "
                      "at smallest λ. Returning λ_min.")
            return self.lam_min

        if D_hi < 0:
            if self.verbose:
                print("  WARNING: D(λ_hi) < 0 — noise level may be "
                      "overestimated or λ_max too small. Returning λ_max.")
            return self.lam_max

        # Brent's method
        if self.verbose:
            print("\n  Brent root finding:")
        lam_star = sopt.brentq(
            lambda lam: self._D(lam, gamma1_obs, gamma2_obs, delta),
            self.lam_min, self.lam_max,
            xtol=1e-8,
            rtol=1e-6,
            maxiter=30,
        )

        if self.verbose:
            D_star = self._D(lam_star, gamma1_obs, gamma2_obs, delta)
            print(f"\n  λ* = {lam_star:.6e}")
            print(f"  D(λ*) = {D_star:+.2e}  (should be ≈ 0)")
            print(f"  Total time: {time.perf_counter()-t_total:.1f}s")
            print("=" * 55)

        return lam_star

    # ─────────────────────────────────────────────────────────────────────────

    def lcurve(self,
               gamma1_obs: np.ndarray,
               gamma2_obs: np.ndarray,
               n_points: int = 20,
               noise_std: Optional[float] = None) -> dict:
        """
        Compute L-curve and discrepancy curve for diagnostic plotting.

        Evaluates MAP reconstruction at n_points logarithmically spaced
        values of λ ∈ [lam_min, lam_max].

        Parameters
        ----------
        gamma1_obs, gamma2_obs : observed shear
        n_points : number of λ values (default 20)
        noise_std : noise level override

        Returns
        -------
        dict with keys:
            'lam'          : (n_points,) λ values
            'residual_norm': (n_points,) ‖Fκ_λ − γ_obs‖
            'kappa_norm'   : (n_points,) ‖κ_λ‖
            'discrepancy'  : (n_points,) D(λ) values
            'delta'        : noise level used
        """
        from .inverse import MAPReconstructor
        from .forward import DifferentiableForward

        delta = noise_std or self.noise_std
        if delta is None:
            g_all = np.concatenate([gamma1_obs, gamma2_obs])
            delta = estimate_noise_level(g_all, method="mad")

        lam_vals  = np.logspace(np.log10(self.lam_min),
                                np.log10(self.lam_max), n_points)
        res_norms = np.zeros(n_points)
        kap_norms = np.zeros(n_points)
        disc_vals = np.zeros(n_points)

        print(f"L-curve: evaluating {n_points} λ values...")
        for i, lam in enumerate(lam_vals):
            fwd = DifferentiableForward(self.ops, lam_reg=lam)
            rec = MAPReconstructor(
                fwd, maxiter=self.maxiter_inner,
                gtol=1e-6, callback_every=0,
                wiener_length=self.wiener_length,
            )
            kappa_lam, _ = rec.reconstruct(
                gamma1_obs, gamma2_obs, verbose=False)

            g1p, g2p = self.ops.forward(kappa_lam)
            r1 = g1p - gamma1_obs
            r2 = g2p - gamma2_obs
            n_data = len(gamma1_obs) + len(gamma2_obs)
            rn = float(np.sqrt((np.dot(r1, r1) + np.dot(r2, r2)) / n_data))

            res_norms[i] = rn
            kap_norms[i] = float(np.linalg.norm(kappa_lam))
            disc_vals[i] = rn - self.c * delta

            print(f"  [{i+1:2d}/{n_points}] λ={lam:.2e}  "
                  f"‖res‖={rn:.4f}  ‖κ‖={kap_norms[i]:.4f}  "
                  f"D={disc_vals[i]:+.4f}")

        return {
            'lam'          : lam_vals,
            'residual_norm': res_norms,
            'kappa_norm'   : kap_norms,
            'discrepancy'  : disc_vals,
            'delta'        : delta,
        }