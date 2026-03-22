# FEMMI - Finite Element Mass Map Inversion

Weak gravitational lensing mass reconstruction via P3 FEM-BEM coupled
boundary value problems, Morozov-regularised Tikhonov inversion, and
inverse scattering support recovery.

---

## Overview

FEMMI reconstructs the projected mass density $\kappa(\theta)$ of a gravitational lens
from observed weak-lensing shear maps $(\gamma_1, \gamma_2)$. The lensing potential $\psi$ satisfies

$$\nabla^2 \psi = 2\kappa \quad \text{in } \mathbb{R}^2, \qquad \psi(\theta) \to 0 \text{ as } |\theta| \to \infty,$$

with shear components

$$\gamma_1 = \tfrac{1}{2}\left(\frac{\partial^2\psi}{\partial\theta_1^2} - \frac{\partial^2\psi}{\partial\theta_2^2}\right), \qquad \gamma_2 = \frac{\partial^2\psi}{\partial\theta_1 \partial\theta_2}.$$

The inverse problem (recovering $\kappa$ from $(\gamma_1, \gamma_2)$) is ill-posed and
requires regularisation and careful treatment of the unbounded domain.

| Feature | Kaiser-Squires (1993) | FEMMI |
|---|---|---|
| Boundary condition | Periodic / Dirichlet | Exact exterior via BEM |
| Regularisation | Manual smoothing | Morozov discrepancy principle |
| Mass-sheet degeneracy | Present | Resolved ($F$ injective) |
| Inverse method | Direct linear | MAP + SVD support recovery |

---

## Mathematical Foundations

Full derivations are in [`MATH.md`](MATH.md). Key ideas:

**FEM-BEM coupling.** FEMMI couples a P3 FEM interior to a boundary element
method on $\partial\Omega$ encoding $\psi \to 0$ at infinity. The coupled stiffness matrix is

$$A_{\mathrm{coupled}} = K + P^\top C P, \qquad C = V_h^{-1}\left(\tfrac{1}{2}M_b + K_h\right)$$

where $K$ is the Neumann stiffness (no Dirichlet row modification), $V_h$ the
single-layer BEM matrix, $K_h$ the double-layer matrix, $M_b$ the boundary mass
matrix, and $P$ the DOF restriction to $\partial\Omega$. This makes the forward operator $F$
injective, resolving the mass-sheet degeneracy present in Kaiser-Squires.

**Shear operators.** Physical shear is computed from P3 reference-element
Hessians via the covariant transform $H_{\mathrm{phys}} = A^\top H_{\mathrm{ref}} A$ (where $A = J^{-T}$), giving
$O(h^2)$ shear convergence vs $O(h^0)$ for P2 and zero for P1.

**Morozov regularisation.** The MAP estimate minimises
$\|F\kappa - \gamma_{\mathrm{obs}}\|^2 + \lambda\|\kappa\|_R^2$ with $R = M + \mathcal{l}^2 K$ (Matern-Wiener prior).
$\lambda$ is selected automatically by Brent's method on the discrepancy functional
$D(\lambda) = \|F\kappa_\lambda - \gamma_{\mathrm{obs}}\| - c\delta$ (C\&K Thm 10.4).

**Inverse scattering.** The forward operator $F$ is structurally equivalent
to the Born-approximation far-field operator in acoustic scattering. FEMMI
implements the factorization method (C\&K Thm 6.15) and linear sampling
method (C\&K \S5.5) for parameter-free support recovery.

---

## Codebase Structure

```
femmi/
|-- __init__.py
|-- types.py             # Mesh namedtuple
|-- mesh.py              # Structured and adaptive P3 mesh generation
|-- basis.py             # P3 Lagrange basis functions (10 DOF/element)
|-- assembly.py          # P3 element stiffness/mass assembly; Poisson solve
|-- bem.py               # BEM: V_h, K_h, M_b; Calderon operator
|-- operators.py         # K, M, S1, S2, A_coupled; FEMOperators dataclass
|-- forward.py           # DifferentiableForward (JAX custom_vjp)
|-- inverse.py           # MAPReconstructor, kaiser_squires
|-- regularization.py    # MorozovSelector, estimate_noise_level
`-- svd_analysis.py      # SVD of F, Picard diagnostic, FactorizationIndicator, LSM

tests/
|-- test_fem_bem_coupling.py   # BEM matrices (V_h, K_h, M_b, Calderon)
|-- test_coupled_pipeline.py   # FEM-BEM pipeline invariants
|-- test_morozov.py            # Morozov lambda selection, monotonicity
|-- test_factorization.py      # SVD, Picard, support recovery indicators
|-- test_convergence_p3.py     # O(h^4) L2 Poisson convergence
|-- test_convergence.py        # Forward operator gamma convergence
`-- test_regression.py         # End-to-end NFW reconstruction

examples/
|-- smpy_comparison.py               # FEMMI vs Kaiser-Squires benchmark
|-- generate_presentation_figures.py # Figures (for final presentation in class)
`-- visualize_results.py             # SVD modes, Picard, convergence plots
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires: JAX >= 0.4, SciPy >= 1.11, NumPy >= 1.25, matplotlib.

**64-bit arithmetic is mandatory.** FEMMI enforces this at import time.
For a $20 \times 20$ mesh $\kappa(A_{\mathrm{coupled}}) = O(1600)$; in 32-bit the solve error
$O(\kappa\,\varepsilon_{32}) \approx 2 \times 10^{-5}$ dominates the P3 discretisation error
$h^4 \approx 6 \times 10^{-6}$.

---

## Basic Usage

```python
import numpy as np
from femmi.operators import build_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor
from femmi.regularization import estimate_noise_level

# Build mesh and operators (20x20 structured P3 mesh on [-2.5, 2.5]^2)
ops = build_operators(nx=20, ny=20, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5)

# Forward model: kappa -> (gamma1, gamma2)
nodes = np.array(ops.mesh.nodes)
kappa_true = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * 0.5**2))
g1, g2 = ops.forward(kappa_true)

# MAP reconstruction with automatic lambda (Morozov)
noise_std = estimate_noise_level(np.concatenate([g1, g2]), method='mad')
fwd = DifferentiableForward(ops, lam_reg=1e-3)
rec = MAPReconstructor(fwd, noise_std=noise_std, wiener_length=0.5)
kappa_map, result = rec.reconstruct(g1_obs, g2_obs)
```

```python
# Adaptive mesh near a circular mask
from femmi.operators import build_operators_adaptive
ops_a = build_operators_adaptive(
    nx=20, ny=20, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
    mask_center=(0., 0.), mask_radius=0.6, refine_factor=3,
)
```

```python
# SVD and support recovery
from femmi.svd_analysis import compute_svd, FactorizationIndicator

svd = compute_svd(ops, n_singular=40)
fi  = FactorizationIndicator(ops, svd_result=svd)
test_grid = np.column_stack([XX.ravel(), YY.ravel()])  # shape (n_test, 2)
W = fi.indicator_map(test_grid)  # large inside support(kappa)
```

---

## Algorithm Summary

**Forward solve** (two solves per MAP iteration):

$$f = -2M\kappa, \qquad A_{\mathrm{coupled}}\psi = f, \qquad \gamma_1 = S_1\psi, \quad \gamma_2 = S_2\psi.$$

**Adjoint gradient** (for L-BFGS):

$$r = (\gamma_1 - \gamma_{1,\mathrm{obs}}, \gamma_2 - \gamma_{2,\mathrm{obs}}), \qquad A_{\mathrm{coupled}}^\top \phi = S_1^\top r_1 + S_2^\top r_2, \qquad \nabla\mathcal{L} = -4M\phi + 2\lambda R\kappa.$$

**Morozov $\lambda$ selection:** Brent root-finding on $D(\lambda) = \|F\kappa_\lambda - \gamma_{\mathrm{obs}}\|_{\mathrm{RMS}} - c\delta$,
typically 15--25 forward solves.

**BEM assembly:** Diagonal blocks of $V_h$ use Gauss-Jacobi quadrature with
weight $w(t) = -\ln(t)$ (25 points, relative error $< 10^{-12}$). Off-diagonal
blocks use standard Gauss-Legendre (8 points).

---

## Convergence

Forward operator on mesh sequence $8 \to 32$ ($\sigma = 1.5$, deep interior):

| Mesh | $\gamma$ rate | Theory |
|------|--------|--------|
| $8 \to 10$ | $\approx 2.0$ | $O(h^2)$ |
| $10 \to 14$ | $\approx 2.0$ | $O(h^2)$ |
| $14 \to 32$ | $\approx 2.0$ | $O(h^2)$ |

Poisson solve (P3, smooth RHS, unit square):

| Mesh | $L^2$ rate | Theory |
|------|---------|--------|
| $4 \to 8$  | 3.86 | $O(h^4)$ |
| $8 \to 16$ | 3.90 | $O(h^4)$ |
| $16 \to 32$| 3.97 | $O(h^4)$ |

---

## References

1. Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer.
2. Steinbach, O. (2008). *Numerical Approximation Methods for Elliptic Boundary Value Problems*. Springer.
3. Sauter, S. & Schwab, C. (2011). *Boundary Element Methods*. Springer.
4. Kirsch, A. (1998). Characterization of the shape of a scattering obstacle. *Inverse Problems*, 14, 1489.
5. Colton, D. & Kirsch, A. (1996). A simple method for solving inverse scattering problems. *Inverse Problems*, 12, 383.
6. Morozov, V. A. (1966). On the solution of functional equations by the method of regularization. *Soviet Math. Doklady*, 7, 414.
7. Kaiser, N. & Squires, G. (1993). Mapping the dark matter with weak gravitational lensing. *ApJ*, 404, 441.
8. Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *IJNME*, 21(6), 1129.
9. Brenner, S. & Scott, R. (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer.
