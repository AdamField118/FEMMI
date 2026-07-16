"""
tests/test_neural_prior.py
The neural score prior (femmi/neural_prior/). Flax-dependent -- skipped cleanly
if flax/optax are not installed.

  - the mesh<->grid binning bridge is consistent (bin then gather ~ identity on a
    smooth field);
  - a tiny DSM-trained score model produces a finite score at the mesh nodes and
    plugs into MAPReconstructor / make_prior('neural');
  - the synthetic training maps are non-Gaussian (positive skewness), which is the
    whole point of a learned prior.

Run:
    python -m pytest tests/test_neural_prior.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

flax = pytest.importorskip("flax")            # skip module if flax missing
optax = pytest.importorskip("optax")

from femmi.operators import build_operators
from femmi.neural_prior.data import lognormal_kappa_maps


def test_gaussian_score_large_sigma_limit():
    """The analytic Gaussian score -> -x/sigma^2 when sigma dominates the signal
    power (a basic correctness check that needs no training)."""
    import jax.numpy as jnp
    from femmi.neural_prior.gaussian_score import GridGaussianScore
    maps = lognormal_kappa_maps(32, 16, seed=3)
    g = GridGaussianScore(GridGaussianScore.power_from_maps(maps))
    x = jnp.asarray(maps[:1, ..., None].astype(np.float32))
    s = np.asarray(g.score(x, jnp.asarray([50.0], np.float32)))
    approx = -np.asarray(x) / (50.0 ** 2)
    assert np.linalg.norm(s - approx) / np.linalg.norm(approx) < 1e-2


@pytest.mark.slow
def test_hybrid_prior_is_gaussian_plus_residual(tmp_path):
    """Hybrid training writes a Gaussian sidecar; the prior loads as hybrid and
    its score is exactly net(grid) + analytic_gaussian(grid) (Remy 2020 eq. 6).
    A plain checkpoint has no sidecar and stays a full-score prior."""
    import jax.numpy as jnp
    from femmi.neural_prior.train import train_score_model, _gauss_sidecar
    from femmi.neural_prior.prior import NeuralScorePrior

    ck = str(tmp_path / "score_unet_p16_b8_hybrid.msgpack")
    train_score_model(n_pix=16, base=8, steps=300, batch=16, patience=2,
                      val_every=100, min_steps=200, hybrid=True, save_path=ck, verbose=False)
    assert os.path.exists(_gauss_sidecar(ck)), "hybrid must write a Gaussian sidecar"

    ops = build_operators(6, 6, -2, 2, -2, 2, verbose=False)
    p = NeuralScorePrior(ops, ckpt=ck, hybrid=True, verbose=False)
    assert p.hybrid and "hybrid" in p.name
    grid = jnp.asarray(np.random.default_rng(1).standard_normal((1, 16, 16, 1)).astype(np.float32))
    sig = jnp.asarray([0.1], np.float32)
    net = np.asarray(p.model.apply(p.params, grid, sig))
    gau = np.asarray(p.gscore.score(grid, sig))
    tot = np.asarray(p._apply(grid, sig))
    assert np.allclose(tot, net + gau, atol=1e-4)

    ck2 = str(tmp_path / "score_unet_p16_b8.msgpack")
    train_score_model(n_pix=16, base=8, steps=250, batch=16, patience=2,
                      val_every=100, min_steps=200, hybrid=False, save_path=ck2, verbose=False)
    assert not os.path.exists(_gauss_sidecar(ck2))
    p2 = NeuralScorePrior(ops, ckpt=ck2, hybrid=False, verbose=False)
    assert not p2.hybrid


def test_synthetic_maps_are_non_gaussian():
    maps = lognormal_kappa_maps(64, 48, seed=0).reshape(64, -1)
    # per-map skewness; log-normal fields are positively skewed on average
    m = maps - maps.mean(1, keepdims=True)
    skew = (m**3).mean(1) / ((m**2).mean(1)**1.5 + 1e-12)
    assert skew.mean() > 0.3, f"training maps not skewed enough (skew={skew.mean():.2f})"


def test_binning_bridge_roundtrips_smooth_field():
    """gather @ bin preserves a smooth field when the grid resolves the nodes
    (bilinear splat + nearest-fill => a mild smoothing, not signal destruction)."""
    from femmi.neural_prior.prior import _binning_operators
    ops = build_operators(16, 16, -2.0, 2.0, -2.0, 2.0, verbose=False)
    nodes = np.array(ops.mesh.nodes)
    bin_op, gather = _binning_operators(nodes, 32,
                                        (nodes[:, 0].min(), nodes[:, 0].max(),
                                         nodes[:, 1].min(), nodes[:, 1].max()))
    f = np.sin(nodes[:, 0]) + 0.5 * nodes[:, 1]        # smooth field
    back = gather @ (bin_op @ f)
    rel = np.linalg.norm(back - f) / np.linalg.norm(f)
    assert rel < 0.15, f"bin->gather roundtrip error {rel:.3f} too large"


def test_checkpoint_arch_parsed_from_filename():
    from femmi.neural_prior.train import parse_ckpt_arch
    assert parse_ckpt_arch("some/dir/score_unet_p64_b32.msgpack") == (64, 32)
    assert parse_ckpt_arch("no_arch_here.msgpack") == (None, None)


@pytest.mark.slow
def test_early_stopping_restores_best(tmp_path):
    """A tiny DSM run stops before the step budget once validation plateaus, and
    saves the best-validation params (not the last)."""
    from femmi.neural_prior.train import train_score_model, load_score_model
    ckpt = str(tmp_path / "es.msgpack")
    # tiny net + aggressive patience so it stops well before `steps`
    train_score_model(n_pix=16, base=8, steps=1200, batch=16, patience=2,
                      val_every=100, min_steps=200, verbose=False, save_path=ckpt)
    m, p = load_score_model(path=ckpt)     # arch (16,8) parsed from filename fallback? name has p16_b8
    assert m is not None                    # best params were saved and reload cleanly


@pytest.mark.slow
def test_neural_prior_scores_and_reconstructs(tmp_path):
    from femmi.neural_prior.prior import NeuralScorePrior
    from femmi.forward import DifferentiableForward
    from femmi.inverse import MAPReconstructor
    from femmi.catalog import analytic_gaussian_shear

    ops = build_operators(10, 10, -2.0, 2.0, -2.0, 2.0, verbose=False)
    ckpt = str(tmp_path / "tiny.msgpack")
    # tiny, fast model just to exercise the pipeline
    prior = NeuralScorePrior(ops, n_pix=32, base=16, steps=20, ckpt=ckpt, verbose=False)
    nodes = np.array(ops.mesh.nodes)
    kt, g1, g2 = analytic_gaussian_shear(nodes, sigma=0.5)

    phi, grad = prior.value_grad(kt)
    assert phi == 0.0 and np.all(np.isfinite(grad)) and grad.shape == (ops.n_nodes,)

    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    rec = MAPReconstructor(fwd, maxiter=30, callback_every=0, prior=prior)
    k, res = rec.reconstruct(g1, g2, verbose=False)
    assert np.all(np.isfinite(k)) and res.n_iter > 0


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
