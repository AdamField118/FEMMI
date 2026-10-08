"""
femmi.neural_prior

A FEMMI-native, gradient-exploiting implementation of the neural score-matching
prior of Remy et al. 2020 (arXiv:2011.08271; code at
https://github.com/b-remy/score-estimation-comparison, branch lensing-recon).
Their work is cited, not vendored. See README.md in this directory -- including
"What is the training data?".

In the experimental pipeline,
`make_prior('neural', ops)` trains a small default score model on self-contained
synthetic non-Gaussian maps on first use and caches it. Because FEMMI's forward
is differentiable, the same learned score drives posterior sampling
(`femmi.sampling.sample_posterior(method='langevin')`). Score-only priors
are rejected by MAP because they do not supply a consistent energy.
Requires flax + optax (`pip install femmi[neural]`).

Imports are lazy so NumPy-only truth generation remains available without
Flax or Optax. Accessing a neural model or trainer loads its optional dependencies.
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
