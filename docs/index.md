# FEMMI

**Finite Element Mass Map Inversion** — a P3 finite-element / boundary-element
pipeline for reconstructing the projected mass (convergence, $\kappa$) of a lens
from weak-lensing shear, with a differentiable forward operator that powers both
MAP reconstruction and full posterior uncertainty quantification.

## Why FEMMI

- **Catalog-native.** FEM nodes are placed at galaxy positions, so the data term
  is evaluated exactly where you have measurements — no gridding/binning of the
  shear before inversion.
- **Symmetric FEM–BEM coupling.** The exterior mass-sheet mode is handled with a
  Steinbach Steklov–Poincaré coupling; the forward runs in float64 where it needs
  the precision to converge.
- **A menu of priors.** Wiener (default), total-variation, sparsity, maximum
  entropy, and a learned **neural score prior** (Remy et al. 2020) — all behind
  one interface.
- **Uncertainty, not just a point estimate.** Exact perturb-and-MAP for the
  Gaussian posterior and the paper's annealed HMC for non-Gaussian/neural priors,
  exploiting the differentiable forward.
- **One config, one command.** `femmi run --config my_run.yaml` describes the whole
  pipeline (forward, data, inverse, prior, sampler, output).

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md) — reconstruct a map in a dozen lines
- [Configuration & CLI](configuration.md) — the `femmi run` workflow
- [Priors & sampling](priors-and-sampling.md) — choosing a prior, auto-calibrated UQ
- [Neural prior](neural-prior.md) — training and using the learned score prior

The mathematical foundations (weak-lensing forward model, FEM–BEM coupling,
regularization, sampling) are documented in
[`MATH.md`](https://github.com/AdamField118/FEMMI/blob/main/MATH.md).
