# Neural score prior

This package provides a noise-conditioned U-Net and denoising score-matching
training for the research sampler. It follows the score-prior approach of
[Remy et al. (2020)](https://arxiv.org/abs/2011.08271) with FEMMI's own forward
operator and grid-to-mesh coupling.

Install `.[neural]`, then use `prior.kind=neural` and `inverse.method=sample`.
A learned score alone does not provide the consistent energy required by MAP.
The score sampler is approximate; see the
[sampling guide](../../docs/priors-and-sampling.md) for its limitations.

Default training generates shifted-lognormal synthetic maps. They provide a
self-contained demonstration, not a calibrated cosmological prior. The trainer
also accepts external simulation maps. Keep training and evaluation lines of
sight independent and record the training distribution with each checkpoint.

| Module | Purpose |
|---|---|
| `data.py` | Synthetic training fields |
| `massivenus.py` | External map loading and patch sampling |
| `denoiser.py` | Flax noise-conditioned U-Net |
| `train.py` | Training, validation-loss stopping, and checkpoint management |
| `prior.py` | Score evaluation and mesh/grid interpolation |

The default learns the full score. Hybrid mode learns a residual above a Gaussian
power-spectrum score; keep its `.gauss.npy` sidecar with the checkpoint.
Architecture parameters are encoded in checkpoint filenames. For commands,
data settings, and checkpoint examples, see the
[neural-prior guide](../../docs/neural-prior.md).
