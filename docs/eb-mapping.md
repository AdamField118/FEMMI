# Joint E/B mapping

Use `FEMMapper.reconstruct_eb` when both fields should enter the likelihood.
The ordinary `reconstruct` method fits an E model. Rotating the observed shear
and calling that method again gives a useful diagnostic, but the two fits do
not account for each other's predicted shear.

## Model and call

Let `F` be the existing shear forward operator and let
`J(g1, g2) = (-g2, g1)`. The joint model is

\[
d = F\kappa_E + JF\kappa_B + n.
\]

It minimizes

\[
\|W^{1/2}(F\kappa_E+JF\kappa_B-d)\|^2
+\lambda_E\kappa_E^T R_E\kappa_E
+\lambda_B\kappa_B^T R_B\kappa_B,
\qquad R_X=M+\ell_X^2K.
\]

```python
joint = mapper.reconstruct_eb(
    lam_e=0.3, length_e=0.6,
    lam_b=0.3, length_b=0.6,
)
e, b = joint.evaluate(points)  # points.shape == (n, 2), in arcminutes
joint.save("joint-map.npz")
```

Omitted priors use `mapper.config.lam` and `mapper.config.length` for each
component. Both strengths must be positive, and lengths must be nonnegative.
These example values demonstrate the interface; choose priors for the intended
data and scientific task. Alternative shear arrays may be supplied as the
first two arguments, using the same positions and weights.

The solver applies rotation inside every joint normal-operator action and uses
the existing forward and transpose solves. It reuses the geometry and coupled
factorization. The two prior factorizations are built for each joint call.
There is no dense forward matrix in the joint solve. A fresh residual check
must meet `MapperConfig.residual_tolerance`; a failed solve raises.
This SciPy MAP interface does not add a JAX differentiation rule through the
joint solution. The existing differentiable forward operator is unchanged.

`JointMassMap.kappa_e` and `kappa_b` contain values at source positions;
`e_coefficients` and `b_coefficients` contain the full FE vectors.
`predicted_g1` and `predicted_g2` are the **summed** E+B prediction. Diagnostics
include both priors, the joint objective, residuals, iteration count, and time.
`evaluate` returns NaN outside the computational mesh. `save` writes these
arrays, mesh, observations, and JSON metadata to NPZ without Python pickles.

## FITS products

Add `joint_eb: true` under the selected `methods.p3`, `methods.argyris`, or
`methods.hct` YAML section, or pass it to `map_mass`. Optional
`joint_b_lam` and `joint_b_length` override the B prior; E uses `lam` and
`length`. This adds products alongside the ordinary maps:

- `*_joint_e_mode.fits` and `*_joint_b_mode.fits`, with the same WCS and coverage.
- Joint predictions and residuals in `*_sources.fits`.
- Both priors and joint convergence diagnostics in the run JSON.

The return dictionary contains `joint_maps` and `joint_reconstruction`.
Headers identify these products as joint MAP and record both priors.
Ordinary SNR products still refer to the ordinary fits. No joint uncertainty or
joint SNR is computed. Do not divide a joint map by the rotated-fit noise map.

## What a joint B map means

A joint fit assigns part of the measured shear to each field under two stated
priors. It is **not** a pure-mode projection. Strengthening the B penalty can
make B smaller without improving separation. Assess E-only, B-only, and mixed
injections, vary the prior ratio, and retain the summed shear residual.

Finite-field E/B ambiguity and sampling leakage are distinct from an incorrect
adjoint or a failed linear solve. In particular, a sufficiently flexible E
model can span the entire sampled observation space. Projecting out that whole
space leaves no data for a B measurement. Making the solver tolerance smaller
cannot change this rank condition.

For a small catalogue, inspect the sampled spaces with:

```python
from femmi.eb import sampled_eb_overlap
report = sampled_eb_overlap(mapper, rtol=1e-9, max_entries=2_000_000)
```

This diagnostic forms the weighted sampled forward matrix, computes its rank,
and compares the E and rotated-E subspaces. It excludes zero-weight sources.
Dense storage is capped; it is not a survey-scale operation. Repeat at nearby
rank tolerances before interpreting small singular values. Its ranks describe
the chosen discrete operator, not continuum pure-mode counts.

A future pure-mode estimator needs an explicit resolvable field space or
boundary construction, with transfer-function and noise tests. A small
posterior B amplitude is not a substitute. For the underlying distinction,
see [Bunn et al. (2003)](https://arxiv.org/abs/astro-ph/0207338). The connection
between Wiener/MAP inference and purification is developed by
[Bunn & Wandelt (2017)](https://arxiv.org/abs/1610.03345); ordinary finite-prior
joint MAP does not implement their pure-mode limit.

## Injection controls

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/diagnostics/joint_eb.py --seeds 728 729 \
  --sources 60 --output results/joint-eb.json
```

The command generates independent analytic Gaussian shear, includes a masked
field and a mixed E+B input, and varies the E/B prior ratio. It writes raw
metrics and solver diagnostics outside the source tree. These controls are
for checking interpretation and numerical behavior, not for selecting a
publication winner.
