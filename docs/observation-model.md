# Observation model

FEMMI models linear shear on a flat field at one effective source plane.
Observed reduced shear is not internally converted to shear. Source redshifts
can be retained and selected, but do not change the lensing efficiency of each
row. Calibrate input shapes and establish component signs before reconstruction.

## Likelihood and prior

For convergence coefficients `k`, the production mapper minimizes

$$J(k)=\sum_i w_i[(F_1k-d_1)_i^2+(F_2k-d_2)_i^2]
       +\lambda k^T(M+\ell^2K)k.$$

`M` and `K` are the finite-element mass and stiffness matrices. `length=0`
therefore gives a mass penalty in `FEMMapper`. Weights are shared by the two
components and used as supplied. If they are relative inverse variances,
`Var(n_ai)=sigma_n^2/w_i`; the prior strength must use the same normalization.
Scaling all weights and lambda together leaves the map unchanged. The production
interface does not model correlated component or inter-source noise.

Only source rows enter the likelihood. Mesh boundary vertices are not additional
observations. Missing regions contain prior-dependent predictions. In the
research array interface, `mask=True` removes a row and combines with zero
weights; active shear values must be finite. The catalogue mapper requires
explicit selection of finite inputs before constructing its mesh.

## Forward operator and adjoint

With prescribed-load projection `Q`, coupled matrix `A`, and observation
recovery `S`, the discrete operator is

$$F=-2SA^{-1}QM,\qquad F^T=-2M^TQA^{-T}S^T.$$

The transpose projection follows the transpose solve. P3 recovers shear from
averaged element Hessians. Argyris uses shared vertex Hessian degrees of freedom.
HCT is C1: its vertex Hessian is an area-weighted recovery from incident
subtriangle traces, not a uniquely defined second derivative. Assembly integrates
each of HCT's three polynomial pieces.

The production solver applies `F` and `F.T` in a quadratic normal equation.
The research JAX interface supplies a custom adjoint for its host solve. Numba
accelerates geometry-dependent assembly; it does not replace that adjoint or
make `FEMMapper` JAX-traceable.

## Exterior and reference level

BEM represents a harmonic exterior under an isolated-source assumption. A
nonzero integrated convergence has a logarithmic two-dimensional far field;
its potential cannot also be assumed to vanish at infinity. The discrete gauge
fixes the potential reference. Boundary normalization and the prior can restrict
mass-sheet-like directions; this is a model restriction, not evidence that the
catalogue alone fixes an absolute mass sheet.

A finite-rank calculation does not prove uniqueness in a continuum function
space. Nodal singular values also depend on coefficient scaling and cannot be
read as physically normalized information without a noise metric and an FE
mass metric.

## Interpreting rotated-shear B maps

`mode='B'` fits `(gamma2, -gamma1)` with the same scalar inverse as E. It is
not an orthogonal E/B projector on a finite, irregular catalogue. Even data
manufactured by the discrete E operator can produce a B map because

$$F_1^TWF_2-F_2^TWF_1$$

need not vanish. A smaller normal-equation residual does not remove this
cross-response. Finite-field E/B ambiguity is discussed by
[Bunn et al. (2003)](https://arxiv.org/abs/astro-ph/0207338); the particular
numerical response of this estimator must still be measured.

`mapper.diagnose_b()` fits rotated E-predicted shear and rotated residual shear
separately. Their sum should reproduce the raw B fit within solver error.
The decomposition is conditional on the E fit and prior. Its residual is neither
a pure-B estimator nor a calibrated detection statistic. Set
`b_diagnostics=True` in `map_mass`, with B requested, to save both diagnostic maps.

## Research priors and sampling

The research `WienerPrior` uses `M+ell^2 K` at positive length and a gradient
penalty `K` at zero length. This zero-length convention differs from the
production mapper; use explicit positive lengths when comparing the interfaces.
Neither precision is automatically a continuum Matérn covariance in two dimensions.

Research MAP minimizes `J_data + lam_MAP * phi`; sampling uses
`J_data/(2*sigma_n^2) + lam_sample * phi`. Matched modes require
`lam_sample=lam_MAP/(2*sigma_n^2)` with otherwise identical assumptions.
See [priors and sampling](priors-and-sampling.md) for energy requirements and
numerical approximation limits.
