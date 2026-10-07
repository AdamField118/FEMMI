# Priors & sampling

See [the observation model](observation-model.md) for mask/weight conventions,
MAP-to-sampling normalization, score-only restrictions, and sampler limitations.
Only Gaussian RTO has the exact Gaussian interpretation (up to numerical solve
error); finite-step Langevin and annealed score HMC remain approximate.


## The prior menu

Set `prior.kind` in the config (or `prior=...` in the API). Every prior exposes
the same interface — a value and gradient — so they are interchangeable in both
MAP reconstruction and posterior sampling.

| `kind` | prior | good for |
|---|---|---|
| `wiener` | Gaussian / Matérn, $R = M + \ell^2 K$ (default) | smooth fields; exact Gaussian UQ |
| `tv` | total variation (smoothed) | piecewise-smooth structure, edges |
| `sparse` | smoothed-$L_1$ on the field or its Laplacian | compact / peaked mass |
| `maxent` | maximum entropy (Marshall 2002) | positive, high-dynamic-range maps |
| `neural` | learned score prior (Remy et al. 2020) | realistic non-Gaussian mass maps |

```python
from femmi.priors import make_prior
prior = make_prior("tv", ops, eps=1e-3)
```

## Posterior sampling (UQ)

`sample_posterior` returns the posterior mean, a per-node uncertainty map, and the
retained samples, exploiting the differentiable forward:

- **`rto`** (perturb-and-MAP / Randomize-Then-Optimize) — *exact* independent
  samples for the Gaussian/Wiener posterior, one linear solve per draw. No step
  tuning.
- **`annealed_hmc`** — the paper's tempered HMC for non-Gaussian / neural priors:
  anneal the noise level $\sigma_{\max}\to\sigma_{\min}$, running HMC at each level
  with a noise-conditional score and a score-integral Metropolis correction.
- **`langevin`** — single-temperature mass-preconditioned ULA, a lightweight
  fallback.

`method: auto` picks `rto` for the Gaussian/Wiener prior and `annealed_hmc`
otherwise.

## Auto-calibrated regularization

!!! important "Leave `inverse.lam: null`"
    The sampler's data term carries weight $1/\sigma_n^2$ (often $10^2$–$10^3$), so
    a small hand-set `lam` lets the likelihood swamp the prior and the posterior
    collapses to noise (relative $L_2 > 1$, uncertainty ≈ the signal amplitude).

With `lam` unset, FEMMI calibrates it automatically:

- **Wiener** — runs the same Morozov discrepancy selection the MAP uses to get
  $\lambda_{\text{MAP}}$, then converts to the sampler's noise-normalised
  convention, $\lambda = \lambda_{\text{MAP}} / (2\sigma_n^2)$. The RTO posterior
  then reproduces the MAP reconstruction.
- **Other priors** — default to coefficient 1.0. This is a convention, not a
  measured calibration of neural, TV, sparse or entropy priors. Tune and validate
  those strengths independently for the intended data.

You can still pass an explicit `lam` to override.

!!! tip "More chains ≠ smaller std"
    The posterior std is the genuine width of the posterior, set by the data and
    prior. Increasing `sampler.n_chains` sharpens the *estimate* of that width, it
    does not shrink it — to reduce uncertainty you need a stronger/better-matched
    prior or better data.
