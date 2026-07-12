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
"""

from ..priors import ScorePrior
from .prior import NeuralScorePrior
from .train import train_score_model, load_score_model, get_or_train
from .data import lognormal_kappa_maps

__all__ = [
    "ScorePrior",
    "NeuralScorePrior",
    "train_score_model",
    "load_score_model",
    "get_or_train",
    "lognormal_kappa_maps",
]
