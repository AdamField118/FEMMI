# Observation model and numerical checks

FEMMI currently reconstructs convergence from **linear shear** on a flat tangent
plane. Inputs must already have a consistent component convention, calibration,
angular unit and source-efficiency normalization. The estimator does not fit
individual source redshifts or the nonlinear reduced shear `g = gamma/(1-kappa)`.
Using measured galaxy shapes as shear is a weak-lensing approximation, not an
internal conversion. No unpublished catalogue results are bundled here.

## Data selection and noise

For active positions, the diagonal noise model is

\[
\operatorname{Var}(n_{a,i})=\sigma_n^2/w_i,\qquad
J_{\rm data}=\sum_{a=1}^2\sum_{i:w_i>0}w_i(F_a\kappa-d_a)_i^2.
\]

- `data_weight` is a finite nonnegative vector, shared by both components.
  Zero means no observation. Negative/nonfinite weights and empty observations
  are errors. A full correlated covariance is not supported by this interface.
- `mask=True` means missing data. The mask is applied before regularization
  selection and is combined with the weights. Missing shear placeholders may
  be NaN; active observations must be finite. Input arrays are not modified.
- `noise_std` is the reference component noise at unit weight. Morozov measures
  `sqrt(J_data / (2*n_active))`. For heterogeneous weights, automatic MAD and
  B-mode residual diagnostics use whitened components `sqrt(w)*gamma`.
  Signal and fitted degrees of freedom can bias these noise estimates; they
  are diagnostics, not independent noise calibrations.
- In `reconstruct_catalog(..., use_weights=True)`, supplied galaxy weights are
  divided by the mean of **positive retained** weights. Supply `noise_std` in
  that normalized convention. `CatalogReconstruction.data_weight` records the
  actual weights. Multiplying precision weights changes the required noise
  scale and regularization strength; it is not a neutral operation by itself.
- Catalogue guard/boundary nodes carry no observations. The structured pipeline
  also excludes P3 boundary rows, whose shear predictions are intentionally zero.
  Low-level array APIs default to all supplied rows: pass explicit weights when
  some rows are placeholders or unsupported boundary outputs.

The MAP loss is `J_data + lam_MAP * phi`. The sampling negative log density is
`J_data/(2*noise_std**2) + lam_sample * phi`, so matched modes require
`lam_sample = lam_MAP/(2*noise_std**2)`. This equality presumes the same prior,
weights and noise model. Positive Wiener length gives `R=M+ell**2*K`; zero
length retains the historical gradient penalty `R=K` in both element paths.
This finite-dimensional precision is not automatically a continuum Matérn-1/2
covariance in two dimensions.

## Forward and transpose

Let `Q` zero prescribed load entries. The discrete forward is
`F = -2*S*A^{-1}*Q*M`; its Euclidean transpose is
`F.T = -2*M.T*Q*A^{-T}*S.T`. The projection acts **after** the transpose solve.
This is shared by reconstruction, the NumPy adjoint, JAX VJP, SVD and sampling.
The public quadratic Hessian action uses this composition directly; it does
not require unsupported differentiation through a JAX host callback.

P3 shear uses averaged element Hessians. Argyris selects shared vertex Hessian
DOFs. HCT is C1, not C2: vertex Hessians are recovered by averaging all incident
subtriangle traces with physical area weights. This is a defined recovery
operator, not a unique pointwise Hessian. Its mass, stiffness and load integrals
use quadrature on each of its three polynomial pieces. See the
[HCT element definition](https://defelement.org/elements/hsieh-clough-tocher.html).

## Priors and posterior sampling

Energy-based MAP requires a consistent penalty value and gradient. `ScorePrior`
requires `neg_logp` for MAP. `NeuralScorePrior` supplies only a score; passing a
zero surrogate value with that score to L-BFGS was inconsistent and is now
rejected. Choose a score sampler explicitly instead. Score-only sampling starts
from zero; its returned `map_kappa` is an initialization, identified by
`info['map_kind'] == 'initialization'`, not a claimed posterior mode.

Gaussian RTO uses perturbations with covariance equal to the posterior precision.
In particular, data perturbations contribute `F.T*W*F/noise_std**2`, not a
squared-weight covariance. CG nonconvergence raises an error. Its Gaussian
interpretation requires a positive definite posterior precision; residual
solver tolerance and the tiny numerical prior factorization jitter limit exactness.

Langevin has finite-step bias. Annealed HMC has finite-chain and annealing error;
a nonzero final sigma still changes the likelihood. Learned scores and the
grid/mesh interpolation need not form a conservative vector field. A finite
score line integral is not an exact Metropolis energy difference in general.
These methods remain experimental approximate samplers, marked in `info`.
A custom prior with a real energy uses that energy and gradient in HMC.

## Boundary assumptions and claims

BEM models a harmonic exterior under an isolated-source assumption. For
nonzero integrated convergence, the two-dimensional potential has a logarithmic
far-field term, not a zero-at-infinity condition. The implemented normalization
and node pin specify a discrete boundary model; neither proves shear inverse
uniqueness. See [Squires & Kaiser (1996)](https://arxiv.org/abs/astro-ph/9512094)
for finite-field reconstruction and reference-level issues.

The nodal SVD is not a noise-weighted, finite-element-mass-normalized information
spectrum. Legacy `FactorizationIndicator` and `LinearSamplingIndicator` use no
observed shear and cannot recover unknown mass support. Their names remain for
compatibility, with corrected descriptions of their geometry-only outputs.

## Verification and benchmark status

Run the complete suite, including optional scientific dependencies:

```bash
pip install -e '.[dev,neural,galsim,io]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FEMMI_REQUIRE_OPTIONAL=1 python -m pytest -q
```

The regression suite includes explicit small-matrix transpose/precision checks,
JAX parity, missing-data invariance through regularization selection, non-binary
weights, a dense-reference posterior covariance check, HCT polynomial recovery,
triangle-order invariance, and quadrature convergence. Older diagnostic scripts
now expose their checked invariants to pytest instead of only printing failures.

Historical benchmark tables predate these corrections. Recalibrate P3, Argyris,
HCT and KS on held-out catalogues before regenerating numerical comparisons.
Correctness checks alone do not establish publication-level reconstruction
quality or a speed advantage. Numba optimization is a separate next step.
