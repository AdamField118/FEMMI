"""Legacy per-catalogue C1 regularization selectors (Morozov/L-curve and CV).

The historical constants are retained for API reproducibility. Current fair
comparisons use independent held-out joint calibration of lambda and physical
length for every FEM kind (femmi.calibration). Earlier selector win counts were
measured before the shared-noise and convergence fixes and are not current
performance claims. Use femmi.calibration for current paired comparisons.
"""
from __future__ import annotations
import numpy as np

DEFAULT_LAM_GRID = np.logspace(-2.0, 1.0, 7)
MOROZOV_C = 0.9119       # historical calibration, not universal
CALIBRATED_LAM = 1.2111  # historical Argyris-only setting


def _penalty(rec, kappa):
    """The regularisation functional actually being minimised.

    Not ||kappa||. The C^1 DOF vector mixes values with first and second
    derivatives, which carry different units, so its Euclidean norm is not a
    meaningful size for an L-curve. The quantity that trades off against the
    residual is the penalty term itself: kappa^T R kappa for the Wiener prior,
    phi(kappa) for a non-Gaussian one.
    """
    if rec.prior is None:
        return float(np.sqrt(max(np.dot(kappa, rec.R @ kappa), 0.0)))
    return float(rec.prior.value_grad(kappa)[0])


def _residual_rms(rec, kappa, g1_obs, g2_obs):
    """Weighted RMS shear residual, per component, over ACTIVE nodes only.

    Ring vertices carry zero data weight, so counting them would divide by a
    node count the data never covered and make the residual look artificially
    small -- delta must stay comparable to a per-component shear std.
    """
    from .observations import prepare_observations, weighted_rms
    g1_obs, g2_obs, w = prepare_observations(
        g1_obs, g2_obs, rec.space.n_vertices, rec.w)
    psi = rec.psi_of(kappa)
    r1 = rec.S1 @ psi - g1_obs
    r2 = rec.S2 @ psi - g2_obs
    return weighted_rms(r1, r2, w)


def c1_lambda_path(rec, g1_obs, g2_obs, lam_grid=None, warm_start=True,
                   maxiter=None, verbose=False):
    """Residual and penalty along a lambda path, reusing one reconstructor.

    Mutates and restores `rec.lam` (and `rec.maxiter` if `maxiter` is given).
    Returns lam / resid / penalty ordered by INCREASING lambda, whatever order
    the solve ran in.
    """
    lam_grid = np.asarray(DEFAULT_LAM_GRID if lam_grid is None else lam_grid, float)
    if lam_grid.ndim != 1 or len(lam_grid) < 3:
        raise ValueError("lam_grid must be 1-D with at least 3 points")
    if np.any(lam_grid <= 0):
        raise ValueError("lam_grid must be positive (it is used on a log scale)")

    lam0, maxiter0 = rec.lam, rec.maxiter
    order = np.argsort(lam_grid)[::-1]          # descending: the homotopy direction
    resid = np.empty(len(lam_grid))
    pen = np.empty(len(lam_grid))
    kappas = [None] * len(lam_grid)
    k_prev = None
    try:
        if maxiter is not None:
            rec.maxiter = int(maxiter)
        for i in order:
            rec.lam = float(lam_grid[i])
            k, _ = rec.reconstruct(g1_obs, g2_obs,
                                   kappa_init=k_prev if warm_start else None)
            resid[i] = _residual_rms(rec, k, g1_obs, g2_obs)
            pen[i] = _penalty(rec, k)
            kappas[i] = k
            if warm_start:
                k_prev = k
            if verbose:
                print(f"    lam={lam_grid[i]:.3e}  resid={resid[i]:.5f}  "
                      f"penalty={pen[i]:.5f}", flush=True)
    finally:
        rec.lam, rec.maxiter = lam0, maxiter0

    asc = np.argsort(lam_grid)
    return dict(lam=lam_grid[asc], resid=resid[asc], penalty=pen[asc],
                kappa=[kappas[i] for i in asc])


def lcurve_corner(lam, resid, penalty):
    """Maximum-curvature point of the (log resid, log penalty) trace.

    Endpoints are excluded AFTER taking the absolute value. Doing it the other
    way round is a live bug this project has already hit once: setting
    curv[0] = -inf and then applying np.abs turns it into +inf, so the corner
    search returns an endpoint every time.
    """
    x = np.log(np.maximum(resid, 1e-300))
    y = np.log(np.maximum(penalty, 1e-300))
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curv = np.abs((dx * ddy - dy * ddx) / np.power(dx**2 + dy**2 + 1e-300, 1.5))
    curv[0] = curv[-1] = -np.inf
    return float(lam[int(np.argmax(curv))])


def morozov_root(lam, resid, delta, c=1.0):
    """Interpolated root of rms_resid(lam) = c*delta on a log-log grid.

    Returns nan when the target is not bracketed -- the caller decides what that
    means, because the two ways of failing need opposite responses: unreachable
    at the small-lambda end means fall back to the L-curve, while still
    over-fitting at the large-lambda end means take lam_max.
    """
    target = float(c) * float(delta)
    d = np.asarray(resid, float) - target
    if d[0] > 0 or d[-1] < 0:
        return np.nan
    j = int(np.argmax(d >= 0))                  # first grid point at/above target
    if j == 0:
        return float(lam[0])
    lo, hi = j - 1, j
    # linear in (log lam, log resid); the residual curve is smooth and monotone
    # here, and a log-scale interpolation cannot leave the bracket
    xl, xh = np.log(resid[lo]), np.log(resid[hi])
    if not np.isfinite(xl) or not np.isfinite(xh) or xh == xl:
        return float(lam[hi])
    t = (np.log(target) - xl) / (xh - xl)
    t = min(max(t, 0.0), 1.0)
    return float(np.exp(np.log(lam[lo]) + t * (np.log(lam[hi]) - np.log(lam[lo]))))


def cv_lambda(rec, g1_obs, g2_obs, lam_grid=None, n_folds=5, seed=0,
              warm_start=True, maxiter=None, verbose=False):
    """K-fold cross-validated lambda: predictive error on HELD-OUT galaxies.

    Select the strength by predicting shear excluded from the training
    likelihood. The geometry is held fixed across folds.

    The fold structure is the natural one here: a fold is a subset of the
    DATA-CARRYING vertices, dropped from the data weight rather than from the
    mesh. Removing them from the mesh instead would change the discretisation
    between folds and compare different function spaces, which is not a
    cross-validation of lambda.

    Cost is n_folds x len(lam_grid) MAP solves, warm-started within folds.
    Its benefit relative to joint calibration needs independent evaluation;
    the earlier catalogue win counts are historical, not current evidence.

    Returns (lam, info) with info["cv"] the mean held-out RMS per lambda.
    """
    lam_grid = np.asarray(DEFAULT_LAM_GRID if lam_grid is None else lam_grid, float)
    lam_grid = np.sort(lam_grid)
    w_full = np.asarray(rec.w, float)
    active = np.flatnonzero(w_full)
    if len(active) < n_folds:
        raise ValueError(f"only {len(active)} data-carrying vertices for "
                         f"{n_folds} folds")

    rng = np.random.default_rng(seed)
    order = rng.permutation(active)
    folds = np.array_split(order, int(n_folds))

    lam0, maxiter0, w0 = rec.lam, rec.maxiter, rec.w
    err = np.zeros((int(n_folds), len(lam_grid)))
    try:
        if maxiter is not None:
            rec.maxiter = int(maxiter)
        for f, hold in enumerate(folds):
            w = w_full.copy()
            w[hold] = 0.0
            rec.w = w
            k_prev = None
            for i in range(len(lam_grid) - 1, -1, -1):     # descending homotopy
                rec.lam = float(lam_grid[i])
                k, _ = rec.reconstruct(g1_obs, g2_obs,
                                       kappa_init=k_prev if warm_start else None)
                k_prev = k
                psi = rec.psi_of(k)
                r1 = (rec.S1 @ psi - g1_obs)[hold]
                r2 = (rec.S2 @ psi - g2_obs)[hold]
                err[f, i] = np.sqrt(np.sum(w_full[hold] * (r1*r1 + r2*r2))
                                    / max(2 * len(hold), 1))
            if verbose:
                print(f"    fold {f}: {np.array2string(err[f], precision=4)}",
                      flush=True)
    finally:
        rec.lam, rec.maxiter, rec.w = lam0, maxiter0, w0

    cv = err.mean(axis=0)
    lam_star = float(lam_grid[int(np.argmin(cv))])
    if verbose:
        print(f"  CV lambda* = {lam_star:.4e}", flush=True)
    return lam_star, dict(method="cv", lam=lam_grid, cv=cv, per_fold=err,
                          n_folds=int(n_folds))


def select_c1_lambda(rec, g1_obs, g2_obs, noise_std=None, lam_grid=None,
                     c=MOROZOV_C, warm_start=True, maxiter=None, verbose=False):
    """Per-catalog lambda for a C^1 MAP reconstruction: Morozov, else L-curve.

    noise_std : per-component shear noise. Required -- there is no MAD fallback
        here on purpose. On a catalog-native C^1 mesh the shear lives at
        vertices that include a zero-weight guard ring, so a MAD over the raw
        vector estimates the noise of a vector that is partly not data. The
        density experiment knows its own noise level; anything that does not
        should measure it before calling.

    c : discrepancy constant, defaulting to the held-out-calibrated MOROZOV_C
        rather than the textbook 1.0. Pass c=1.0 to get the uncalibrated rule.

    Returns (lam, info). `info["method"]` is 'morozov', 'lcurve' or 'lam_max',
    and `info` carries the whole path so a caller can plot or re-check it.
    """
    if noise_std is None:
        raise ValueError("noise_std is required: a MAD estimate over a C^1 DOF "
                         "vector with a zero-weight guard ring is not a noise level")

    path = c1_lambda_path(rec, g1_obs, g2_obs, lam_grid=lam_grid,
                          warm_start=warm_start, maxiter=maxiter, verbose=verbose)
    lam, resid = path["lam"], path["resid"]

    root = morozov_root(lam, resid, noise_std, c=c)
    if np.isfinite(root):
        method = "morozov"
        lam_star = root
    elif resid[0] - c * noise_std > 0:
        # even the least-regularised fit leaves a residual above the assumed
        # noise: model error dominates and the principle does not apply
        method = "lcurve"
        lam_star = lcurve_corner(lam, resid, path["penalty"])
    else:
        method = "lam_max"
        lam_star = float(lam[-1])

    if verbose:
        print(f"  lambda* = {lam_star:.4e}  ({method})", flush=True)
    info = dict(method=method, delta=float(noise_std), **path)
    return lam_star, info
