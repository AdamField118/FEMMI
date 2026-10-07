"""
femmi/c1_lambda.py
Per-catalog regularisation weight for the C^1 (Argyris) inverse path.

WHY THIS EXISTS
---------------
The density experiment was comparing a TUNED method against an UNTUNED one.
`catalog.reconstruct_catalog` runs with `use_morozov=True` by default, so the P3
arm of every density sweep picked its own lambda for each catalog; the Argyris
arm was pinned at lam=0.3 for all densities and all realisations. Any accuracy
gap measured that way is a lower bound on the element's contribution, because
part of it is P3's tuning advantage.

The visible symptom was a flat spot: Argyris's error barely moved between
n_eff = 10 and 20 gal/arcmin^2 (0.5159 -> 0.5224, inside the seed scatter) while
both baselines kept improving. A fixed lambda over-regularises exactly as the
catalog gets denser -- more sources need LESS smoothing, and a constant prior
weight does not know that.

WHAT IT DOES
------------
The same two-stage rule as the P3 path (femmi.regularization): Morozov's
discrepancy principle where it applies, L-curve corner where it does not.

    D(lam) = rms_resid(lam) - c * delta

is increasing in lam, so the root is bracketed on a log grid and interpolated.
If D > 0 even at the smallest lam on the grid, the model cannot explain the data
to the assumed noise and there is no root -- returning lam_min there is the worst
available answer (an essentially unregularised fit), so it falls back to the
L-curve corner, exactly as MorozovSelector does.

WHY A GRID AND NOT BRENT
------------------------
Brent needs 15-25 MAP solves. Here one MAP solve at n_eff = 30 is ~30 s and the
sweep is (4 densities) x (6 seeds), so Brent would cost days. A 9-point log grid
with WARM STARTS along the lambda path costs ~9 solves' worth of operator reuse
and a small fraction of that in iterations, because the coupled operator, its LU
factorisation, the mass matrix and the prior are all lambda-independent and are
built once. Descending lambda is the right homotopy direction: the large-lambda
problem is well conditioned and each solution is a good initial guess for the
next.

The interpolated root is as accurate as the residual curve is smooth, which is
far more precision than the choice deserves -- lambda enters the result through a
log-scale trade-off, not a threshold.

    rec = C1MAPReconstructor(space, lam=0.3, data_weight=w)
    lam, info = select_c1_lambda(rec, g1, g2, noise_std=0.05)
    rec.lam = lam
    kappa, _ = rec.reconstruct(g1, g2)
"""

from __future__ import annotations
import numpy as np


# The oracle study (MATH.md 18.3j) put the best lambda between 0.3 and 1 across
# every density and seed, and nothing below 1e-2 was ever selected, so the grid
# starts there -- seven points instead of nine for the same coverage, which
# matters because every point is a MAP solve.
DEFAULT_LAM_GRID = np.logspace(-2.0, 1.0, 7)

# Morozov's constant, MEASURED rather than assumed.
#
# The textbook value is c = 1: stop when the residual equals the noise. That
# over-regularises here by 3-4x in lambda, and measurably: at n_eff = 5 it scored
# 0.84 against the pinned lambda's 0.67. The reason is not subtle. This problem
# fits ~6x more unknowns than observations (an Argyris vertex carries 6 DOFs and
# supplies 2 shear components), so the model can drive the residual below delta
# legitimately, and the degrees-of-freedom-corrected target is
#
#     delta * sqrt(1 - p/n)   with p the effective number of fitted parameters.
#
# Calibrating that single scalar against the oracle optimum on HELD-OUT catalogs
# -- seeds 100, 101, 102, disjoint from every seed used in any reported result --
# over four densities gives
#
#     c = 0.9119 +/- 0.0157   (n = 12, spread 0.791-0.980)
#
# i.e. p/n ~ 0.17, and it is stable across density (0.87, 0.92, 0.94, 0.92 at
# n_eff = 5, 10, 20, 30). Calibrating on the reported seeds instead would be
# tuning on the test set; that is why the calibration seeds are held out and the
# constant is frozen here rather than re-fitted per run.
MOROZOV_C = 0.9119

# The lambda the density experiment actually uses -- and the reason is a NEGATIVE
# result about the rule above.
#
# Run blind on 24 catalogs, per-catalog Morozov with the calibrated c beats a
# pinned lambda in only 10 of them and adds variance. The cause is measurable:
# the residual is nearly flat in lambda near the optimum (0.0421 -> 0.0493 as
# lambda goes 1 -> 3.16), so d log lambda / d log resid ~ 7 and the ~10%
# catalog-to-catalog spread in the correct c becomes a factor ~2 in the selected
# lambda. The discrepancy signal simply does not locate lambda on this problem.
#
# What DOES transfer is the typical value. The geometric mean of the per-catalog
# optima on the same held-out seeds is 1.2111, and applying that single frozen
# number blind to the 24 reported catalogs beats the old pinned 0.3 in 19 of 24
# (mean shape L2 0.4764 against 0.5358) and matches the per-catalog ORACLE to
# within noise for n_eff >= 20. Per-catalog adaptation is worth nothing here;
# getting the constant right is worth ~11% overall and ~21% at survey densities.
#
# See MATH.md 18.3j. Task #52 (cross-validation on held-out galaxies) is the
# route to the remaining headroom at low density, since it does not go through
# the flat residual curve.
CALIBRATED_LAM = 1.2111


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

    WHY THIS AND NOT MOROZOV. The discrepancy principle fails on this problem for
    a measurable reason (MATH.md 18.3j): the fitting residual is nearly flat in
    lambda near the optimum, so d log lambda / d log resid ~ 7 and any error in
    the target maps to a factor ~2 in lambda. Cross-validation does not go
    through that curve at all -- it measures how well the reconstruction predicts
    shear it never saw, which is directly the quantity being optimised and is
    steep in lambda on both sides of the optimum.

    The fold structure is the natural one here: a fold is a subset of the
    DATA-CARRYING vertices, dropped from the data weight rather than from the
    mesh. Removing them from the mesh instead would change the discretisation
    between folds and compare different function spaces, which is not a
    cross-validation of lambda.

    MEASURED (MATH.md 18.3n). Blind on the 24 reported catalogs it beats the
    frozen CALIBRATED_LAM in 20 of them -- against Morozov's 10 of 24 on the same
    data, same grid, same solver. It pays where the constant is weak: at
    n_eff = 5 it recovers 77% of the remaining oracle gap, and at n_eff >= 20 it
    adds nothing because the constant is already at the oracle there.

    Cost is n_folds x len(lam_grid) MAP solves, warm-started along lambda within
    each fold -- ~35 against one, for 2.4% overall. That is why it is not the
    default; use it on sparse catalogs, where it buys the most and the solves are
    cheapest.

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
