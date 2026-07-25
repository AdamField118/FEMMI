"""
tests/test_lambda_selection.py
Regularisation-parameter selection (femmi.regularization).

Two defects motivated these, both surfaced by the benchmark grid rather than by
any unit test:

  1. Morozov selection was skipped entirely for non-quadratic priors, leaving
     TV/sparsity/max-entropy at a fixed lam_reg and badly mis-scaled.
  2. When the discrepancy target is UNREACHABLE (model error above the assumed
     noise, so D(lam_min) > 0) the selector returned lam_min -- the least
     regularised solution, i.e. maximal noise amplification. Measured on a
     tapered log-normal field that was 3.4x worse in shape L2 than the best
     lambda, and 2x worse than simply taking lam_max.

Run:
    python -m pytest tests/test_lambda_selection.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.experiments import square_ops
from femmi.regularization import MorozovSelector, lcurve_lambda, discrepancy
from femmi.priors import make_prior
from femmi.truth import lognormal_truth

pytest.importorskip("galsim", reason="independent truth generators")


def _noisy(ops, source_fn, noise=0.02, seed=0):
    nodes = np.array(ops.mesh.nodes)
    kt, g1, g2 = source_fn(nodes)
    rng = np.random.default_rng(seed)
    return kt, g1 + rng.normal(0, noise, len(g1)), g2 + rng.normal(0, noise, len(g2))


def test_discrepancy_accepts_a_non_quadratic_prior():
    """The discrepancy is evaluated by actually solving the MAP problem, so it is
    defined for any prior -- nothing requires quadratic structure."""
    from femmi.truth import galsim_nfw_truth
    ops = square_ops(8, 2.5)
    _, g1, g2 = _noisy(ops, lambda n: galsim_nfw_truth(n, halos=((2e14, 4.0, (0., 0.)),)))
    tv = make_prior("tv", ops)
    d = discrepancy(1e-2, ops, g1, g2, delta=0.02, maxiter_inner=30,
                    wiener_length=1.0, prior=tv)
    assert np.isfinite(d)


def test_lcurve_returns_an_interior_grid_point():
    """The corner is a maximum-curvature point; endpoints are excluded, and a bug
    where abs() was applied after setting them to -inf made argmax pick them."""
    ops = square_ops(8, 2.5)
    _, g1, g2 = _noisy(ops, lambda n: lognormal_truth(n, 2.5, seed=0))
    grid = np.logspace(-6, 1, 8)
    lam = lcurve_lambda(ops, g1, g2, lam_grid=grid, wiener_length=1.0,
                        maxiter_inner=40, data_weight=np.ones(ops.n_nodes))
    assert grid[0] < lam < grid[-1]


def test_unreachable_discrepancy_does_not_return_lam_min():
    """The core regression. On a field where the model cannot reach the assumed
    noise level, the selector must NOT hand back the unregularised extreme."""
    ops = square_ops(10, 2.5)
    _, g1, g2 = _noisy(ops, lambda n: lognormal_truth(n, 2.5, seed=0))
    sel = MorozovSelector(ops, noise_std=0.02, wiener_length=1.0, verbose=False,
                          maxiter_inner=40, data_weight=np.ones(ops.n_nodes))
    assert sel._D(sel.lam_min, g1, g2, 0.02) > 0        # target genuinely unreachable
    lam = sel.select(g1, g2)
    assert lam > 100 * sel.lam_min


def test_bracketed_case_still_finds_the_root():
    """The normal path must be untouched: where a root exists, Morozov finds it."""
    from femmi.truth import galsim_nfw_truth
    ops = square_ops(10, 2.5)
    _, g1, g2 = _noisy(ops, lambda n: galsim_nfw_truth(n, halos=((2e14, 4.0, (0., 0.)),)))
    sel = MorozovSelector(ops, noise_std=0.02, wiener_length=1.0, verbose=False,
                          maxiter_inner=60, data_weight=np.ones(ops.n_nodes))
    assert sel._D(sel.lam_min, g1, g2, 0.02) < 0        # a root exists
    lam = sel.select(g1, g2)
    assert sel.lam_min < lam < sel.lam_max


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
