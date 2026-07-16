# Neural prior

FEMMI includes a learned **score prior** — a network
$r_\theta(\kappa,\sigma)\approx\nabla\log p_\sigma(\kappa)$ trained by Denoising
Score Matching (Remy et al. 2020). It models the *non-Gaussian* structure of
realistic mass maps on top of the Gaussian score FEMMI already has, and because
the forward operator is differentiable, the same learned score drives posterior
sampling (annealed HMC).

Install the extra:

```bash
pip install -e ".[neural]"
```

## One flag away

```bash
femmi run --config configs/default.yaml \
    --set inverse.method=sample --set prior.kind=neural
```

On first use with no checkpoint, FEMMI trains a small default model on synthetic
non-Gaussian (shifted-log-normal) maps and caches it — no external data required.
The sampler auto-selects the neural prior weight ($\lambda = 1.0$; see
[Priors & sampling](priors-and-sampling.md)).

## Training your own

```bash
femmi train-prior --config configs/default.yaml \
    --set prior.neural.n_pix=64 --set prior.neural.base=32 --set prior.neural.steps=20000
```

The architecture (`n_pix`, `base`) and step budget come from the config's
`prior.neural` section (override with `--set`). Training uses validation-loss early
stopping (patience), so it stops when it stops improving instead of always running
every step, and caches to `checkpoints/score_unet_p<n_pix>_b<base>.msgpack`. The
checkpoint filename encodes
the architecture (`score_unet_p64_b32.msgpack`), so you can point a run at a model
by name and the grid resolution follows automatically:

```yaml
prior:
  kind: neural
  neural:
    ckpt: checkpoints/score_unet_p64_b32.msgpack   # arch (p64,b32) read from the name
```

or leave `ckpt: null` and FEMMI loads the default cached model matching
`n_pix`/`base`.

## Hybrid mode (1-to-1 with the paper)

By default the network learns the **full** score. Set `hybrid: true` to instead
learn only the **non-Gaussian residual** on top of an analytic Gaussian prior —
exactly the decomposition in Remy et al. 2020 (eq. 6):

$$\nabla\log p(\kappa) = \nabla\log p_{\rm th}(\kappa) + r_\theta(\kappa,\sigma).$$

```yaml
prior:
  kind: neural
  neural: {n_pix: 64, base: 32, hybrid: true}
```

`p_th` is a stationary Gaussian whose power spectrum is estimated from the
training field, so the net only has to model what the Gaussian prior misses. Train
a hybrid model with the same flag:

```bash
femmi train-prior --config configs/default.yaml --set prior.neural.hybrid=true
```

The Gaussian power spectrum is saved as a `.gauss.npy` sidecar next to the
checkpoint; its presence is what marks a model "hybrid," so plain (full-score)
checkpoints stay fully backward compatible. The full-score default is unchanged.

## Choosing the test field

The learned prior encodes the morphology of its **training data** (log-normal
random fields). A smooth single-halo synthetic truth mismatches that prior — a
smoothness prior (Wiener) will legitimately beat it there. To see the neural prior
help, apply it to genuinely non-Gaussian fields: realistic cluster catalogs
(`data.source: frontier`) or a log-normal synthetic. This is the regime the method
is designed for and the fair comparison for reproducing the paper.

## Performance

The score net is called many thousands of times inside the sampler. FEMMI JITs the
whole U-Net forward (compiled once as fused kernels, not one op-dispatch per layer)
and runs it in float32 — the FEM forward stays in float64 where it needs the
precision. This is what keeps the neural sampler from stalling on the GPU.
