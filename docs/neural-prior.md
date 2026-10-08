# Neural prior

> Score-only neural priors support experimental sampling, not energy-based
> L-BFGS MAP. Set `inverse.method: sample`. A valid MAP objective requires a
> matching prior value and gradient; the score alone is insufficient.
> See [the observation model](observation-model.md) for approximation limits.


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

## Run score sampling

```bash
femmi run --config configs/default.yaml \
    --set inverse.method=sample --set prior.kind=neural
```

On first use with no checkpoint, FEMMI trains a small default model on synthetic
non-Gaussian (shifted-log-normal) maps and caches it — no external data required.
The sampler uses the default neural prior weight ($\lambda = 1.0$; see
[Priors & sampling](priors-and-sampling.md)).

## Training your own

```bash
femmi train-prior --config configs/default.yaml \
    --set prior.neural.n_pix=64 --set prior.neural.base=32 --set prior.neural.steps=20000
```

The architecture (`n_pix`, `base`) and step budget come from the config's
`prior.neural` section (override with `--set`). Training uses validation-loss early
stopping (patience), so it stops when it stops improving instead of always running
every step, and caches to `femmi/neural_prior/checkpoints/score_unet_p<n_pix>_b<base>.msgpack`. The
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

## Training on MassiveNuS

The shipped default trains on synthetic shifted-log-normal maps. To use simulation-based training, supply **MassiveNuS** convergence maps
(Liu et al. 2018). Matching a simulation suite alone does not reproduce a
published experiment; the data split, processing, network, and sampler also matter.

Download the fiducial galaxy-lensing maps from the
[Columbia Lensing group](http://columbialensing.org) — the file
`convergence_gal_mnv0.00000_om0.30000_As2.1000.tar` — and extract one source
redshift (`Maps10/` = z_s=1, the paper's choice; the tar also has z=0.5/1.5/2/2.5):

```bash
tar xf convergence_gal_mnv0.00000_om0.30000_As2.1000.tar Maps10   # 512x512 FITS maps
pip install -e ".[neural,paper]"        # paper extra: galsim + astropy
femmi train-prior --config configs/paper_artifacts.yaml \
    --set prior.neural.train_data=massivenus \
    --set prior.neural.data_dir=/abs/path/to/Maps10 \
    --set prior.neural.hybrid=true
```

The loader reads `.npy` / `.npz` / `.fits` maps and serves random `n_pix` patches,
so nothing else about training changes. A redshift folder holds ~10,000 maps, so to
stay memory-bounded it globs once and holds a fixed random pool of
`prior.neural.pool_maps` maps (default 512 ≈ 0.5 GB) in RAM. If you extract several
redshifts into one folder, `prior.neural.map_glob='*z1.00*'` trains on just one.

## Hybrid mode

By default the network learns the **full** score. Set `hybrid: true` to instead
learn only the **non-Gaussian residual** on top of an analytic Gaussian prior —
following the decomposition in Remy et al. 2020 (eq. 6):

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
checkpoint; its presence is what marks a model "hybrid," and must be kept with it. A checkpoint without this sidecar is interpreted as
a full-score model.

## Choosing evaluation data

The learned prior reflects its training distribution. Evaluate on independent
maps and keep simulation lines of sight separate across training, calibration,
and evaluation. A shifted-lognormal synthetic field is useful for controlled
tests, but agreement with a matched training distribution does not establish
performance on a survey catalogue.

`configs/lognormal.yaml` provides a synthetic research configuration. Record
which forward operator generated its shear: a manufactured FEM input tests
internal consistency, while independent truth tests model mismatch as well.
Compare priors with the same observations and report sensitivity to prior strength.

The network executes in float32 through JAX; FEM solves remain float64. Training
and repeated score evaluation can use a suitable JAX accelerator installation.
Profile the whole sampling workflow before choosing resources.
