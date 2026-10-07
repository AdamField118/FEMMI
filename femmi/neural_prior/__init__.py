"""
femmi.neural_prior

A FEMMI-native, gradient-exploiting implementation of the neural score-matching
prior of Remy et al. 2020 (arXiv:2011.08271; code at
https://github.com/b-remy/score-estimation-comparison, branch lensing-recon).
Their work is cited, not vendored. See README.md in this directory -- including
"What is the training data?".

One flag away: `reconstruct_catalog(..., prior='neural')` or
`make_prior('neural', ops)` trains a small default score model on self-contained
synthetic non-Gaussian maps on first use and caches it. Because FEMMI's forward
is differentiable, the same learned score drives posterior sampling
(`femmi.sampling.sample_posterior(method='langevin')`), not just MAP.
Requires flax + optax (`pip install femmi[neural]`).

WHY THE IMPORTS ARE LAZY
------------------------
`data.lognormal_kappa_maps` is pure numpy and has no JAX dependency, but it lives
in this package -- and importing anything from a package runs its __init__. When
that __init__ eagerly imported `prior` and `train` (which need flax/optax), a
`from .neural_prior.data import lognormal_kappa_maps` in `femmi.truth` raised
ModuleNotFoundError for anyone without the optional `neural` extra.

That is not a neural-prior problem, it is a TRUTH problem: `truth.lognormal_truth`
generates the non-Gaussian field that MATH.md 18.3.12 uses to bound the scope of
the density claim, and it was unusable without an unrelated extra installed. The
module-level `__getattr__` below defers the heavy imports until a name that
actually needs them is requested, so the numpy-only path stays importable while
`from femmi.neural_prior import NeuralScorePrior` behaves exactly as before.
"""

_LAZY = {
    "ScorePrior": ("..priors", "ScorePrior"),
    "NeuralScorePrior": (".prior", "NeuralScorePrior"),
    "train_score_model": (".train", "train_score_model"),
    "load_score_model": (".train", "load_score_model"),
    "get_or_train": (".train", "get_or_train"),
    "lognormal_kappa_maps": (".data", "lognormal_kappa_maps"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Import on first use, so the numpy-only members do not drag in flax/optax."""
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module
    return getattr(import_module(module, __package__), attr)


def __dir__():
    return sorted(__all__)
