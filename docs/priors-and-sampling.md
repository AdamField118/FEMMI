# Research priors and sampling

The production `FEMMapper` has a quadratic mass-plus-gradient penalty. The
separate `femmi run` research pipeline exposes other priors and posterior
samplers. Its configuration is described in [Configuration and CLI](configuration.md).

| Prior | Penalty or score | Interface |
|---|---|---|
| `wiener` | Quadratic FE precision | Energy-based MAP and Gaussian sampling |
| `tv` | Smoothed total variation | Energy-based MAP and approximate sampling |
| `sparse` | Smoothed L1 on values or a Laplacian | Energy-based MAP and approximate sampling |
| `maxent` | Positive-field entropy penalty | Energy-based MAP and approximate sampling |
| `neural` | Learned score | Experimental score sampling |

```python
from femmi.priors import make_prior
prior = make_prior('tv', ops, eps=1e-3)
```

A custom energy prior must provide a consistent value and gradient. A `ScorePrior`
without `neg_logp` cannot be used for energy-based MAP. The neural score does not
supply a matching energy, so select `inverse.method=sample`.

## Samplers

`sample_posterior` returns retained samples, a sample mean, and a per-coefficient
standard deviation. For score-only priors, `map_kappa` is an initialization,
identified by `info['map_kind']`, rather than a fitted posterior mode.

- `rto` uses Gaussian perturb-and-solve draws for a Wiener prior. Its Gaussian
  interpretation requires positive-definite posterior precision; finite solve
  tolerance and factorization jitter limit exactness.
- `langevin` uses an unadjusted finite-step diffusion and has step-size bias.
- `annealed_hmc` uses an annealing schedule and finite chains. For score-only
  priors, a numerical line integral does not guarantee an exact Metropolis
  correction when the score is nonconservative. A nonzero final noise level
  also changes the likelihood. These outputs are approximate.

`method=auto` chooses RTO for Wiener and annealed HMC otherwise. Inspect the
returned `info` and convergence diagnostics. More draws improve estimation of
posterior spread; they do not by themselves narrow that posterior.

## Strength and noise normalization

MAP uses `J_data + lam_MAP*phi`, while sampling uses
`J_data/(2*noise_std**2) + lam_sample*phi`. With the same weights and prior,
matching modes requires `lam_sample=lam_MAP/(2*noise_std**2)`.

With lambda unset, the Wiener sampler uses discrepancy-selected MAP strength
and converts it to the sampling convention. Other priors default to 1.0; this
is a coefficient convention, not an empirical calibration. Tune them for the
intended observations and inspect sensitivity to that choice.

At positive Wiener length, `R=M+ell**2*K`. At zero length, the research prior
uses `K`; the production mapper uses `M`. Keep this distinction explicit when
comparing runs. See the [observation model](observation-model.md).
