"""
tests/test_c1_lambda.py
Per-catalog lambda selection on the C^1 path (femmi.c1_lambda).

The reason this module exists is a FAIRNESS bug, not a feature request:
For the experimental C1 discrepancy-principle selector, the
P3 arm of every density sweep tuned its own lambda per catalog while the Argyris
arm was pinned at 0.3. These tests cover the pieces that decide the number, and
in particular the two failure modes the P3 selector already hit once each:

  * the L-curve corner search returning an endpoint, because `curv[0] = -inf`
    followed by np.abs() becomes +inf;
  * Morozov being applied when it does not apply -- no root in the bracket --
    where returning lam_min is the worst available answer.

The expensive end-to-end path (a real Argyris solve per grid point) is covered
by one small case; the arithmetic is tested directly.

Run:
    python -m pytest tests/test_c1_lambda.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.c1_lambda import (DEFAULT_LAM_GRID, MOROZOV_C, CALIBRATED_LAM,
                             morozov_root, lcurve_corner, c1_lambda_path,
                             select_c1_lambda, _residual_rms)


# ------------------------------------------------------- calibrated values ---

def test_calibrated_constants_are_the_held_out_measured_ones():
    """Both constants were fitted on catalogs (seeds 100-102) disjoint from every
    seed reported in MATH.md. Changing either silently would turn a held-out
    calibration into tuning on the test set, so they are pinned here."""
    assert MOROZOV_C == pytest.approx(0.9119, abs=1e-4)
    assert CALIBRATED_LAM == pytest.approx(1.2111, abs=1e-4)


def test_calibrated_lambda_is_inside_the_default_grid():
    """It is the geometric mean of per-catalog optima found on that grid, so a
    value outside it would mean the grid no longer brackets the optimum."""
    assert DEFAULT_LAM_GRID[0] < CALIBRATED_LAM < DEFAULT_LAM_GRID[-1]


def test_morozov_c_is_below_one_for_the_stated_reason():
    """c < 1 is the degrees-of-freedom correction: ~6x more unknowns than
    observations means the residual can drop below delta legitimately. A value
    at or above 1 would mean that reasoning no longer applies."""
    assert 0.5 < MOROZOV_C < 1.0


# ---------------------------------------------------------------- morozov ---

def test_morozov_root_recovers_a_known_crossing():
    """Residual is increasing in lambda; the root is where it meets c*delta."""
    lam = np.logspace(-3, 1, 9)
    resid = 0.01 * (lam ** 0.5)              # monotone, crosses 0.02 at lam=4
    r = morozov_root(lam, resid, delta=0.02)
    assert np.isfinite(r)
    assert abs(r - 4.0) < 0.5                # grid-interpolated, not exact
    assert lam[0] <= r <= lam[-1]


def test_morozov_root_is_nan_when_unbracketed_at_either_end():
    """Both ends fail, and they need OPPOSITE responses from the caller, so the
    root finder must not paper over either by clamping."""
    lam = np.logspace(-3, 1, 9)
    hi = np.full(9, 0.5)                     # residual above target everywhere
    assert np.isnan(morozov_root(lam, hi, delta=0.02))
    lo = np.full(9, 0.001)                   # residual below target everywhere
    assert np.isnan(morozov_root(lam, lo, delta=0.02))


def test_morozov_root_stays_inside_the_bracket():
    lam = np.logspace(-3, 1, 9)
    rng = np.random.default_rng(0)
    for _ in range(20):
        resid = np.sort(rng.uniform(0.001, 0.2, 9))
        target = float(rng.uniform(resid[0], resid[-1]))
        r = morozov_root(lam, resid, delta=target)
        assert np.isfinite(r) and lam[0] <= r <= lam[-1]


# ---------------------------------------------------------------- l-curve ---

def test_lcurve_corner_finds_an_interior_corner():
    """A synthetic L: penalty falls steeply, turns, then flattens."""
    lam = np.logspace(-3, 1, 9)
    resid = np.linspace(0.01, 0.10, 9)
    penalty = np.concatenate([np.linspace(10.0, 2.0, 4), np.linspace(1.6, 1.4, 5)])
    c = lcurve_corner(lam, resid, penalty)
    assert c in lam[1:-1]


def test_lcurve_corner_never_returns_an_endpoint():
    """The live bug this guards: setting curv[0] = -inf and THEN taking np.abs
    turns it into +inf, so every corner search returns an endpoint. Random
    curves must never select one."""
    lam = np.logspace(-3, 1, 9)
    rng = np.random.default_rng(1)
    for _ in range(50):
        resid = np.sort(rng.uniform(0.001, 0.2, 9))
        penalty = np.sort(rng.uniform(0.1, 10.0, 9))[::-1]
        c = lcurve_corner(lam, resid, penalty)
        assert c != lam[0] and c != lam[-1]


# ------------------------------------------------------------- validation ---

def test_noise_std_is_required():
    """A MAD estimate over a C^1 DOF vector that includes a zero-weight guard
    ring is not a noise level, so there is deliberately no fallback."""
    with pytest.raises(ValueError, match="noise_std is required"):
        select_c1_lambda(object(), np.zeros(3), np.zeros(3), noise_std=None)


def test_lam_grid_validation():
    class _Rec:
        lam = 1.0
        maxiter = 10
    with pytest.raises(ValueError, match="at least 3"):
        c1_lambda_path(_Rec(), np.zeros(3), np.zeros(3), lam_grid=[1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        c1_lambda_path(_Rec(), np.zeros(3), np.zeros(3), lam_grid=[0.0, 1.0, 2.0])


def test_default_grid_spans_the_useful_range():
    """1e-2 to 10. The lower end was raised from 1e-3 once the oracle study
    showed nothing below 1e-2 is ever selected -- every grid point costs a full
    MAP solve, so unused coverage is not free."""
    assert DEFAULT_LAM_GRID[0] == pytest.approx(1e-2)
    assert DEFAULT_LAM_GRID[-1] == pytest.approx(10.0)
    assert np.all(np.diff(DEFAULT_LAM_GRID) > 0)


# ------------------------------------------------------------- end to end ---

@pytest.fixture(scope="module")
def small_problem():
    pytest.importorskip("galsim", reason="scores against the GalSim NFW truth")
    from femmi.elements import C1Space, catalog_triangulation
    from femmi.c1_inverse import C1MAPReconstructor
    from femmi.density import sample_catalog
    from femmi.truth import galsim_nfw_truth

    x, y = sample_catalog(120, radius=3.0, seed=0)
    v, t, ring, _ = catalog_triangulation(x, y)
    S = C1Space(v, t, kind="argyris")
    kt, g1t, g2t = galsim_nfw_truth(v, halos=((2.0e14, 4.0, (0.0, 0.0)),))
    rng = np.random.default_rng(1)
    g1 = g1t + rng.normal(0, 0.05, len(g1t))
    g2 = g2t + rng.normal(0, 0.05, len(g2t))
    rec = C1MAPReconstructor(S, lam=1.0, wiener_length=1.0,
                             data_weight=(~ring).astype(float), maxiter=60)
    return rec, g1, g2


def test_path_restores_reconstructor_state(small_problem):
    """The path mutates rec.lam and rec.maxiter to avoid rebuilding the coupled
    operator, so it must put them back -- a caller that reused the object
    afterwards would otherwise silently solve at the wrong lambda."""
    rec, g1, g2 = small_problem
    lam0, mi0 = rec.lam, rec.maxiter
    c1_lambda_path(rec, g1, g2, lam_grid=np.logspace(-2, 0, 3), maxiter=20)
    assert rec.lam == lam0 and rec.maxiter == mi0


def test_residual_increases_with_lambda(small_problem):
    """The property Morozov's root-finding depends on. If this ever fails the
    discrepancy principle is not applicable and the root is meaningless."""
    rec, g1, g2 = small_problem
    p = c1_lambda_path(rec, g1, g2, lam_grid=np.logspace(-2, 1, 4), maxiter=60)
    assert np.all(np.diff(p["resid"]) > 0), p["resid"]
    assert np.all(np.diff(p["penalty"]) < 0), p["penalty"]   # and penalty falls


def test_residual_rms_ignores_zero_weight_nodes(small_problem):
    """Ring vertices carry no data. Counting them would divide by nodes the data
    never covered and make the residual look artificially small, which would
    then drag the Morozov root."""
    rec, g1, g2 = small_problem
    k = np.zeros(rec.space.n_dofs)
    got = _residual_rms(rec, k, g1, g2)
    w = rec.w
    psi = rec.psi_of(k)
    r1 = rec.S1 @ psi - g1
    r2 = rec.S2 @ psi - g2
    want = np.sqrt((np.dot(w * r1, r1) + np.dot(w * r2, r2))
                   / (2 * np.count_nonzero(w)))
    assert got == pytest.approx(want)
    assert np.count_nonzero(w) < len(w)      # the ring really is excluded


def test_select_returns_a_grid_bounded_lambda_and_a_named_method(small_problem):
    rec, g1, g2 = small_problem
    grid = np.logspace(-2, 1, 4)
    lam, info = select_c1_lambda(rec, g1, g2, noise_std=0.05, lam_grid=grid,
                                 maxiter=60)
    assert info["method"] in ("morozov", "lcurve", "lam_max")
    assert grid[0] <= lam <= grid[-1]
    assert info["delta"] == 0.05
    assert len(info["resid"]) == len(grid)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
