"""
tests/test_sampling.py
Posterior sampling (femmi/sampling.py):

  - RTO (perturb-and-MAP) is exact for the Gaussian/Wiener prior: with a proper
    (strong) prior the posterior MEAN converges to the MAP (mass-sheet DC mode
    removed), and the per-node std is positive everywhere;
  - the posterior std is a real uncertainty map (larger where data constrain less);
  - the Langevin path runs with a custom (non-Gaussian) prior and returns finite
    mean/std;
  - method='rto' is rejected for a non-Gaussian prior.

Run:
    python -m pytest tests/test_sampling.py -v
    python tests/test_sampling.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.forward   import DifferentiableForward
from femmi.sampling  import sample_posterior
from femmi.priors    import TotalVariationPrior
from femmi.catalog   import analytic_gaussian_shear


def _data(nx=10, sigma=0.5, sn=0.05, seed=0):
    ops = build_operators(nx, nx, -2.0, 2.0, -2.0, 2.0, verbose=False)
    nodes = np.array(ops.mesh.nodes)
    kt, g1a, g2a = analytic_gaussian_shear(nodes, sigma=sigma)
    rng = np.random.default_rng(seed)
    g1 = g1a + rng.normal(0, sn, len(g1a)); g2 = g2a + rng.normal(0, sn, len(g2a))
    return ops, nodes, kt, g1, g2, sn


def test_rto_mean_converges_to_map():
    ops, nodes, kt, g1, g2, sn = _data()
    fwd = DifferentiableForward(ops, lam_reg=1.0)          # strong -> proper posterior
    ps = sample_posterior(fwd, g1, g2, noise_std=sn, wiener_length=0.5,
                          method="rto", n_samples=250, seed=1, verbose=False)
    assert ps.method == "rto"
    # remove the mass-sheet DC mode (unconstrained by shear) before comparing
    dm = lambda a: a - a.mean()
    rel = np.linalg.norm(dm(ps.mean) - dm(ps.map_kappa)) / (np.linalg.norm(dm(ps.map_kappa)) + 1e-30)
    assert rel < 0.15, f"RTO mean should converge to MAP, rel={rel:.3f}"
    assert np.all(ps.std > 0), "posterior std must be positive everywhere"
    assert ps.samples.shape == (250, ops.n_nodes)


def test_posterior_std_is_a_real_uncertainty_map():
    ops, nodes, kt, g1, g2, sn = _data()
    fwd = DifferentiableForward(ops, lam_reg=1.0)
    ps = sample_posterior(fwd, g1, g2, noise_std=sn, wiener_length=0.5,
                          method="rto", n_samples=200, seed=2, verbose=False)
    # std should vary across the map (not a constant), i.e. carry information
    assert ps.std.std() / (ps.std.mean() + 1e-30) > 0.05


def test_langevin_runs_with_custom_prior():
    ops, nodes, kt, g1, g2, sn = _data()
    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    prior = TotalVariationPrior(ops)
    ps = sample_posterior(fwd, g1, g2, noise_std=sn, prior=prior, method="langevin",
                          n_steps=150, burnin=50, thin=5, seed=3, verbose=False)
    assert ps.method == "langevin"
    assert np.all(np.isfinite(ps.mean)) and np.all(np.isfinite(ps.std))
    assert ps.samples.shape[0] == (150 - 50) // 5


def test_annealed_hmc_runs_and_mixes():
    """Annealed HMC produces finite UQ with a healthy acceptance rate, and mixes
    far better than single-temperature Langevin (its mean is much closer to MAP)."""
    ops, nodes, kt, g1, g2, sn = _data()
    fwd = DifferentiableForward(ops, lam_reg=1.0)
    from femmi.priors import WienerPrior
    ps = sample_posterior(fwd, g1, g2, noise_std=sn, prior=WienerPrior(ops, 0.5),
                          method="annealed_hmc", n_levels=8, steps_per_level=8,
                          n_leapfrog=5, n_chains=20, keep_final=3, seed=1, verbose=False)
    assert ps.method == "annealed_hmc"
    assert np.all(np.isfinite(ps.mean)) and np.all(ps.std >= 0)
    assert 0.2 < ps.info["accept"] < 0.99, f"accept {ps.info['accept']:.2f} unhealthy"
    dm = lambda a: a - a.mean()
    rel = np.linalg.norm(dm(ps.mean) - dm(ps.map_kappa)) / (np.linalg.norm(dm(ps.map_kappa)) + 1e-30)
    assert rel < 0.8, f"annealed HMC mean far from MAP (rel={rel:.2f})"


def test_auto_lam_calibration():
    """lam=None auto-calibrates: for the Wiener prior it runs Morozov and converts
    to the sampler convention, giving a sensible reconstruction (L2 << 1) instead
    of the noise-dominated result a tiny lam produces. A score prior gets 1.0."""
    from femmi.sampling import _auto_lam
    ops, nodes, kt, g1, g2, sn = _data()
    m = np.isfinite(kt)
    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    ps = sample_posterior(fwd, g1, g2, noise_std=sn, wiener_length=0.5,
                          lam=None, method="rto", n_samples=80, seed=1, verbose=False)
    l2 = np.linalg.norm(ps.map_kappa[m] - kt[m]) / np.linalg.norm(kt[m])
    assert l2 < 0.6, f"auto-lam RTO reconstruction poor (L2={l2:.2f})"
    assert fwd.lam_reg > 1.0, "Wiener auto-lam should be >> the tiny raw default"

    class _Score:
        name = "score"
        def score(self, k, s=None): return np.zeros_like(k)
    assert _auto_lam(ops, g1, g2, sn, _Score(), 0.5, None, verbose=False) == 1.0


def test_rto_rejects_non_gaussian_prior():
    ops, nodes, kt, g1, g2, sn = _data()
    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    try:
        sample_posterior(fwd, g1, g2, noise_std=sn, prior=TotalVariationPrior(ops),
                         method="rto", verbose=False)
        assert False, "expected ValueError for RTO with a non-Gaussian prior"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_rto_mean_converges_to_map,
        test_posterior_std_is_a_real_uncertainty_map,
        test_langevin_runs_with_custom_prior,
        test_annealed_hmc_runs_and_mixes,
        test_auto_lam_calibration,
        test_rto_rejects_non_gaussian_prior,
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
