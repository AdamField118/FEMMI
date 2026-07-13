# Neural non-Gaussian prior for FEMMI

A FEMMI-native, gradient-exploiting implementation of the score-based
non-Gaussian prior of

> B. Remy, F. Lanusse, Z. Ramzi, J. Liu, N. Jeffrey, J.-L. Starck,
> *Probabilistic Mapping of Dark Matter by Neural Score Matching*,
> Third Workshop on Machine Learning and the Physical Sciences, NeurIPS 2020.
> arXiv:2011.08271. Code: https://github.com/b-remy/score-estimation-comparison
> (branch `lensing-recon`).

Their work is **cited, not vendored** — this is an independent implementation on
FEMMI's own differentiable FEM-BEM forward. `prior='neural'` is the only flag you
need.

## What is the training data? (start here)

The prior is a small neural network that learns *what a convergence (κ) map
typically looks like*. To learn that, it needs example κ maps — and the honest
answer to "where do they come from" is: **FEMMI generates them itself, no
download.**

- The maps are **synthetic shifted-log-normal random fields** (`data.py`,
  `lognormal_kappa_maps`). We draw a Gaussian random field with a power-law power
  spectrum, then exponentiate it. Exponentiating turns a symmetric Gaussian field
  into a **skewed, peaky** field — bright compact peaks on a near-empty
  background — which is the standard simple model for weak-lensing convergence
  (Clerkin et al. 2017; Hilbert et al. 2011). That skew/peakiness is exactly the
  *non-Gaussian* structure a plain Gaussian (Wiener) prior cannot represent, and
  therefore exactly what the network is there to learn.
- They are generated **fresh on every training batch** (an infinite stream), so
  there is no dataset file to manage and nothing to overfit.
- This makes the shipped prior a faithful *demonstration of the mechanism*, not a
  calibrated cosmological prior. **To do science**, retrain on real simulation κ
  maps (e.g. MassiveNu, as Remy et al. use, or your GalSim NFW fields) — point
  `train_score_model` at them instead of the synthetic generator. The interface,
  the reconstructor, and the sampler are all unchanged; only the training maps
  differ.

## How it is trained (Denoising Score Matching)

We never need the probability `p(κ)` itself — only its **score** `∇log p(κ)`,
which is all a gradient-based reconstructor or sampler uses. DSM learns it
cheaply (Remy et al. eq. 3; Vincent 2011):

1. Take a clean map `κ`, add Gaussian noise: `κ' = κ + σ u`, `u ~ N(0, I)`, with
   `σ` drawn over a range.
2. Train a noise-conditional U-Net `r_θ(κ', σ)` to minimise
   `E‖u + σ r_θ(κ', σ)‖²`.
3. At the optimum, `r_θ(κ', σ) = ∇log p_σ(κ')` — the score we want.

`train.py` runs this loop; `denoiser.py` is the Flax U-Net; a small default model
trains in a couple of minutes on CPU and is cached under `checkpoints/`. Training
tracks a **held-out validation DSM loss** and **early-stops with patience**
(keeping the best-validation params), so a large `steps` budget is an upper bound,
not wasted compute — it stops once the model has converged.

**Referencing a trained model.** Checkpoints are named
`score_unet_p{n_pix}_b{base}.msgpack` and the architecture is read back from that
name, so you point at a run by file and nothing else is needed:

```python
prior = make_prior('neural', ops, ckpt='path/to/score_unet_p64_b32.msgpack')
```

## Why FEMMI is a natural fit

- FEMMI's MAP gradient already contains the exact **analytic Gaussian score**:
  the Wiener/Matérn term `∇φ = 2λRκ` (`femmi/priors.py::WienerPrior`) is precisely
  the Gaussian `−∇log p_th` of Remy et al.'s eq. 6. So the network only supplies
  the **non-Gaussian residual** on top — the minimal-reliance split they advocate.
- FEMMI's forward `κ → γ` is **differentiable** (`forward.py`, JAX `custom_vjp`),
  so the likelihood score is exact. The same learned score therefore drives both
  MAP (`MAPReconstructor(prior=...)`) and posterior **sampling**
  (`femmi.sampling.sample_posterior(method='langevin')`), not just point
  estimation — the gradient-preserving quality this exploits.

## Usage

```python
from femmi.catalog import reconstruct_catalog
rec = reconstruct_catalog(x, y, g1, g2, prior='neural')      # one flag; trains+caches on first use

# or explicitly, with the sampler for uncertainty quantification
from femmi.priors import make_prior
from femmi.sampling import sample_posterior
prior = make_prior('neural', ops)
ps = sample_posterior(fwd, g1, g2, noise_std=0.05, prior=prior, method='langevin')
mean, std = ps.mean, ps.std                                  # posterior mean + uncertainty map
```

## Files

- `denoiser.py` — Flax noise-conditional U-Net (the score network `r_θ`).
- `data.py` — synthetic non-Gaussian training maps (swap for real sims here).
- `train.py` — DSM training loop + checkpoint save/load + `get_or_train`.
- `prior.py` — `NeuralScorePrior`: the mesh↔grid bridge; a `Prior` you can drop
  into the reconstructor or sampler.
