# Mathematical reference

The user-facing [observation model](docs/observation-model.md) describes units,
selection, shear conventions, and boundary assumptions. This reference gives
the discrete equations used by the implementation.

## Potential and shear

With the sign convention used by the coupled load, convergence and the lensing
potential satisfy `Delta psi = 2 kappa`. Shear is recovered from second
potential derivatives using the recovery matrices for `gamma1=(psi_xx-psi_yy)/2` and `gamma2=psi_xy`. The
implementation defines the complete sign through `F=-2 S A^{-1} Q M`;
input catalogue sign conversions occur before applying this operator.

The mesh contains source locations and unobserved boundary nodes. Cubic
Lagrange, Argyris, and HCT spaces have different coefficients and Hessian
recovery rules; their coefficient vectors are not interchangeable. HCT is
piecewise cubic and C1, so it requires an explicit recovery for vertex Hessians.
Its volume integrals are evaluated on each macroelement subtriangle.

## Coupled solve

Let `A` denote the assembled coupled FEM/BEM system, including the discrete
gauge, and `M` the volume mass matrix. Extend the volume load with zero boundary
unknown loads. With `Q` the prescribed-load projection,

$$u=-2A^{-1}QMk,\qquad d=Su=Fk.$$

The transpose with respect to coefficient Euclidean inner products is

$$F^T=-2M^TQA^{-T}S^T.$$

`Q` acts after the transpose solve. Any equilibration used for sparse LU must
appear with matching transpose scalings. Adjoint tests compare both inner
products for independent vectors; gradient checks then exercise the full loss.

The exterior is harmonic under an isolated-source model. Nonzero net mass
permits logarithmic potential growth. Gauge fixing, the chosen boundary model,
and regularization do not establish continuum inverse uniqueness.

## Quadratic estimator

For `W=diag(w,w)` and `R=M+ell^2 K`, the production estimator solves

$$Hk=F^TWd,\qquad H=F^TWF+\lambda R.$$

The data term is the sum of both weighted component residual squares; there is
no hidden factor of one half in the reported objective. Positive lambda and the
mass term regularize the finite-dimensional problem. Prior-preconditioned CG
uses an internal relative target and a separately recomputed residual
acceptance threshold. A converged solve establishes numerical solution of this
model, not its adequacy for a given survey.

For a fixed mesh, weights, and prior, the estimator is linear in the data.
Superposition of single-halo fits must agree with a fit of their summed shear.
Changing the source sample or retuning the prior changes the operator and does
not define a superposition test.

The physical norm of a coefficient perturbation is

$$\|k\|_{L^2(\Omega)}^2=k^TMk.$$

Use this norm when comparing C1 coefficient errors: value and derivative degrees
of freedom carry different units. A data-information spectrum would solve
`F.T C_n^-1 F v = sigma^2 M v`; raw nodal singular values do not have that normalization.

## Rotated-shear response

Write `L=H^-1 F.T W` and `J(d1,d2)=(d2,-d1)`. The displayed B map is `L J d`.
For a discrete E input `d=F k`, its right-hand side is

$$F^TWJFk=(F_1^TWF_2-F_2^TWF_1)k.$$

This skew cross-operator need not be zero. For the fitted E model `k_E=Ld`,
linearity gives the diagnostic identity

$$LJd=LJFk_E+LJ(d-Fk_E).$$

The two terms are the predicted E leakage and residual response. They depend
on the fitted model; the second term is not an independent pure-B reconstruction.

## Aperture mass

For normalized radius `x=r/R <= 1`, the polynomial filters paired with SMPy's
Schneider shear aperture are

$$U(r)=\frac{l+2}{\pi R^2}(1-x^2)^l[1-(l+2)x^2],$$
$$Q(r)=\frac{(l+1)(l+2)}{\pi R^2}x^2(1-x^2)^l.$$

Their continuous aperture integrals agree for compatible convergence and shear
on complete support. The comparison uses `l=3` and a fixed physical radius.
The sampled U kernel is corrected to have zero sum, so a constant sheet gives
zero discrete aperture mass. SMPy's sampled Q is left unchanged. A dense,
noiseless Q-versus-U control records their finite-grid discrepancy separately.
Apertures crossing the scoring field boundary are excluded; observed catalogue
holes remain part of the reconstruction challenge.

## Probability normalization

For independent noise with variance `sigma_n^2/w_i`, the negative log likelihood
is `J_data/(2 sigma_n^2)`. A sampling prior coefficient therefore differs from
the unnormalized MAP coefficient by `1/(2 sigma_n^2)`. The research zero-length
Wiener penalty uses `K`, whereas the production mapper uses `M`; configurations
must state which estimator is intended.

Gaussian perturb-and-solve sampling requires a positive-definite posterior
precision and perturbations with that same precision covariance. Finite solve
error and numerical factorization jitter limit exactness. Score-only priors
lack the energy required by energy-based MAP. Finite-step Langevin and annealed
score sampling have additional approximation errors; see the
[sampling guide](docs/priors-and-sampling.md).
