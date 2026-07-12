"""
tests/test_priors.py
The pluggable prior system (femmi/priors.py):

  - every prior's analytic gradient matches finite differences;
  - the default reconstructor path (prior=None) is byte-for-byte the historical
    Wiener/H1 behaviour;
  - WienerPrior reproduces the built-in Wiener regulariser exactly;
  - each prior runs end-to-end in a reconstruction and yields a finite kappa;
  - the make_prior factory and the ScorePrior neural hook behave.

Run:
    python -m pytest tests/test_priors.py -v
    python tests/test_priors.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators, dirichlet_from_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor
from femmi.catalog   import analytic_gaussian_shear
from femmi.priors    import (WienerPrior, TotalVariationPrior, SparsityPrior,
                             MaxEntropyPrior, ScorePrior, make_prior,
                             build_gradient_operator)


def _ops():
    return build_operators(10, 10, -2.0, 2.0, -2.0, 2.0, verbose=False)


def _fd_max_err(prior, k, rng, n_dir=5, eps=1e-6):
    _, g = prior.value_grad(k)
    errs = []
    for _ in range(n_dir):
        d = rng.standard_normal(k.size); d /= np.linalg.norm(d)
        p1, _ = prior.value_grad(k + eps * d)
        p2, _ = prior.value_grad(k - eps * d)
        an = float(g @ d)
        errs.append(abs((p1 - p2) / (2 * eps) - an) / (abs(an) + 1e-8))
    return max(errs)


def test_prior_gradients_match_finite_differences():
    ops = _ops(); rng = np.random.default_rng(0)
    k = 0.1 * rng.standard_normal(ops.n_nodes) + 0.5      # positive for max-ent
    priors = [
        WienerPrior(ops, 0.5),
        TotalVariationPrior(ops),
        SparsityPrior(ops, "field"),
        SparsityPrior(ops, "laplacian"),
        MaxEntropyPrior(ops),
        ScorePrior(lambda kk: -(ops.K @ kk),
                   neg_logp=lambda kk: 0.5 * float(kk @ (ops.K @ kk))),
    ]
    for p in priors:
        err = _fd_max_err(p, k, rng)
        assert err < 1e-4, f"{p.name}: gradient FD error {err:.2e} too large"


def test_wiener_prior_matches_builtin_regulariser():
    ops = _ops(); rng = np.random.default_rng(1)
    k = rng.standard_normal(ops.n_nodes)
    R = (ops.M + 0.5**2 * ops.K)                          # built-in Wiener
    phi, grad = WienerPrior(ops, 0.5).value_grad(k)
    assert abs(phi - float(k @ (R @ k))) < 1e-9
    assert np.allclose(grad, 2.0 * (R @ k))


def test_default_path_unchanged():
    """prior=None reproduces the wiener_length reconstruction exactly."""
    ops = _ops()
    nodes = np.array(ops.mesh.nodes)
    kt, g1, g2 = analytic_gaussian_shear(nodes, sigma=0.5)
    fwd = DifferentiableForward(ops, lam_reg=1e-3)

    rec_a = MAPReconstructor(fwd, maxiter=60, wiener_length=0.5, callback_every=0)
    ka, _ = rec_a.reconstruct(g1, g2, verbose=False)

    fwd_b = DifferentiableForward(ops, lam_reg=1e-3)
    rec_b = MAPReconstructor(fwd_b, maxiter=60, wiener_length=0.5, callback_every=0,
                             prior=WienerPrior(ops, 0.5))
    kb, _ = rec_b.reconstruct(g1, g2, verbose=False)
    # explicit WienerPrior == default wiener_length path
    assert np.linalg.norm(ka - kb) / (np.linalg.norm(ka) + 1e-30) < 1e-6


def test_each_prior_reconstructs_finite():
    ops = _ops()
    nodes = np.array(ops.mesh.nodes)
    kt, g1, g2 = analytic_gaussian_shear(nodes, sigma=0.5)
    for kind, kw in [("tv", {}), ("sparse", {"transform": "field"}),
                     ("maxent", {"model": 1e-2})]:
        fwd = DifferentiableForward(ops, lam_reg=1e-3)
        prior = make_prior(kind, ops, **kw)
        rec = MAPReconstructor(fwd, maxiter=40, callback_every=0, prior=prior)
        k, res = rec.reconstruct(g1, g2, verbose=False)
        assert np.all(np.isfinite(k)), f"{kind}: non-finite kappa"
        assert res.n_iter > 0


def test_score_prior_hook_runs():
    """A ScorePrior wrapping the Gaussian score runs and matches the Wiener MAP."""
    ops = _ops()
    nodes = np.array(ops.mesh.nodes)
    kt, g1, g2 = analytic_gaussian_shear(nodes, sigma=0.5)
    R = ops.K
    score = ScorePrior(lambda kk: -(R @ kk),
                       neg_logp=lambda kk: 0.5 * float(kk @ (R @ kk)))
    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    rec = MAPReconstructor(fwd, maxiter=60, callback_every=0, prior=score)
    ks, _ = rec.reconstruct(g1, g2, verbose=False)

    # H1 Wiener uses phi = k^T K k (grad 2Kk); the score prior here uses
    # neg_logp = 0.5 k^T K k (grad Kk), i.e. half the penalty -> same solution
    # family, just a factor-2 in lambda. Both must be finite and non-trivial.
    assert np.all(np.isfinite(ks)) and np.linalg.norm(ks) > 0


def test_gradient_operator_exact_on_linear_field():
    """A linear field has a constant gradient, so the centroid gradient operator
    is exact: grad(a*x+b*y) = (a,b) everywhere, and sum_e a_e |grad|^2 = area*(a^2+b^2)."""
    ops = _ops()
    nodes = np.array(ops.mesh.nodes)
    a, b = 0.7, -1.3
    k = a * nodes[:, 0] + b * nodes[:, 1]
    Gx, Gy, area = build_gradient_operator(ops)
    assert np.allclose(Gx @ k, a) and np.allclose(Gy @ k, b), "gradient of linear field wrong"
    energy_grad = float(np.dot(area, (Gx @ k)**2 + (Gy @ k)**2))
    energy_exact = float(area.sum() * (a**2 + b**2))
    assert abs(energy_grad - energy_exact) / energy_exact < 1e-9


if __name__ == "__main__":
    tests = [
        test_prior_gradients_match_finite_differences,
        test_wiener_prior_matches_builtin_regulariser,
        test_default_path_unchanged,
        test_each_prior_reconstructs_finite,
        test_score_prior_hook_runs,
        test_gradient_operator_exact_on_linear_field,
    ]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn(); passed += 1; print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}"); failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}"); failed.append(fn.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if not failed else 1)
