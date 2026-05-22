# FEMMI — Finite Element Mass Map Inversion

Weak gravitational lensing mass reconstruction via P3 FEM-BEM coupled boundary value problems, with automatic Morozov-regularised MAP inversion and inverse-scattering support recovery.

---

## Overview

FEMMI reconstructs the projected mass density $\kappa(\boldsymbol{\theta})$ of a gravitational lens from observed weak-lensing shear $(\gamma_1, \gamma_2)$. The lensing potential $\psi$ satisfies

$$\nabla^2\psi = 2\kappa \quad \text{in } \mathbb{R}^2, \qquad \psi(\boldsymbol{\theta}) \to 0 \text{ as } |\boldsymbol{\theta}| \to \infty,$$

with shear components

$$\gamma_1 = \tfrac{1}{2}\!\left(\frac{\partial^2\psi}{\partial\theta_1^2} - \frac{\partial^2\psi}{\partial\theta_2^2}\right), \qquad \gamma_2 = \frac{\partial^2\psi}{\partial\theta_1 \,\partial\theta_2}.$$

The central methodological claim is that the standard practice of truncating this problem to a finite domain with Dirichlet boundary conditions ($\psi = 0$ on $\partial\Omega$) encodes the wrong continuous operator: the true $\psi$ decays only logarithmically and is nonzero at any finite boundary. The resulting systematic error propagates throughout the interior by the maximum principle. FEMMI replaces this with an exact exterior representation via boundary elements, enforcing the correct far-field condition without approximation.

| Feature | Kaiser-Squires (1993) | FEMMI |
|---|---|---|
| Far-field boundary condition | Periodic / Dirichlet (wrong) | Exact exterior via BEM |
| Regularisation parameter | Manual smoothing kernel | Morozov discrepancy principle |
| Mass-sheet degeneracy | Present in $F$ | Resolved ($F$ injective) |
| Inverse method | Direct FFT | MAP + L-BFGS, Matérn prior |
| Masked / missing data | Unreliable near mask | Inpainting via prior covariance |
| Source positions | Binned grid | Catalog-native (raw galaxy positions) |
| Element order | N/A | P3 cubic (required for $\nabla^2\psi$) |

---

## Mathematical Foundations

Full derivations are in [`MATH.md`](MATH.md). The key ideas:

**FEM-BEM coupling.** A P3 FEM interior solves $\nabla^2\psi = 2\kappa$ in $\Omega$ while a boundary element method encodes $\nabla^2\psi = 0$ in the exterior and $\psi \to 0$ at infinity. The coupled stiffness matrix is assembled via Schur complement reduction:

$$A_{\mathrm{coupled}} = K + P^\top C\, P, \qquad C = V_h^{-1}\!\left(\tfrac{1}{2}M_b + K_h\right),$$

where $K$ is the Neumann stiffness (no Dirichlet row modification), $V_h$ the single-layer BEM matrix, $K_h$ the double-layer matrix, $M_b$ the boundary mass matrix, and $P$ the DOF restriction to $\partial\Omega$. This is the Johnson-Nédélec monolithic coupling. The Calderon preconditioner $C$ is assembled once and stored; each forward solve requires two SuperLU triangular solves (forward and adjoint).

**Why P3 elements.** Shear is the Hessian of $\psi$: $\gamma_1 = \frac{1}{2}(\partial^2\psi/\partial x^2 - \partial^2\psi/\partial y^2)$, $\gamma_2 = \partial^2\psi/\partial x\,\partial y$. P1 elements give identically zero second derivatives; P2 gives piecewise-constant second derivatives with no convergence. P3 gives piecewise-linear second derivatives and $O(h^2)$ shear convergence. The 10-node P3 Lagrange element used here achieves $O(h^4)$ in $L^2$ for the Poisson solve.

**MAP reconstruction.** The estimate minimises

$$\mathcal{L}(\kappa) = \|F\kappa - \gamma_{\mathrm{obs}}\|^2 + \lambda\,\kappa^\top R\,\kappa, \qquad R = M + \ell^2 K \;\text{(Matérn-}\tfrac{1}{2}\text{ prior)}.$$

$\lambda$ is selected automatically by Brent's method on the discrepancy functional $D(\lambda) = \|F\kappa_\lambda - \gamma_{\mathrm{obs}}\|_{\mathrm{RMS}} - c\delta$ (Morozov 1966; C&K Thm 10.4), using 15–25 MAP solves. The gradient is computed via the adjoint: $\partial\mathcal{L}/\partial\kappa = -4M A_{\mathrm{coupled}}^{-T}(S_1^\top r_1 + S_2^\top r_2) + 2\lambda R\kappa$.

**Injectivity and the mass-sheet degeneracy.** The BEM far-field normalization fixes the $\kappa \to \kappa + c$ degeneracy present in all FFT-based methods: the forward operator $F$ is injective, so the MAP problem has a unique solution. A single-node gauge condition removes the remaining scalar null space (the additive constant in $\psi$).

**Inverse scattering connection.** The forward operator $F: L^2(\Omega) \to L^2(\Omega)^2$ is compact, placing the lensing problem in the same mathematical framework as the Born approximation in acoustic inverse scattering (C&K Ch. 8, 10). FEMMI implements the Kirsch factorization method and linear sampling method for parameter-free support recovery from the truncated SVD of $F$.

---

## Preliminary Results

Synthetic benchmarks on a Gaussian convergence field ($\sigma_\kappa = 0.5$, $20\times20$ P3 mesh on $[-2.5, 2.5]^2$). MAP uses automatic $\lambda$ selection via Morozov's principle with a Matérn prior ($\ell = \sigma_\kappa$). Kaiser-Squires uses the standard FFT inversion. Figures generated by `examples/generate_figures.py`.

### Reconstruction at 10% noise (single realisation, seed=42)

![Reconstruction at 10% noise](outputs/fig_reconstruction.png)

### Masked field ($r < 0.6$), adaptive mesh

The Matérn prior propagates $\kappa$ correlation structure into the masked region; KS-FFT collapses near the mask boundary.

![Masked reconstruction](outputs/fig_masked.png)

### Forward operator convergence

$O(h^2)$ shear convergence validated on mesh sequence $n_x \in \{8, 10, 14, 18, 24, 32\}$ against a reference mesh ($n_x = 40$), with $\sigma = 1.5$ to ensure $\sigma/h \geq 1$ on all test meshes.

![Forward convergence](outputs/fig_convergence.png)

| Mesh transition | $\|\gamma_h - \gamma_{\mathrm{ref}}\|$ rate | Theory |
|---|---|---|
| $8 \to 14$ | $\approx 2.0$ | $O(h^2)$ |
| $14 \to 32$ | $\approx 2.0$ | $O(h^2)$ |

---

## Codebase Structure

```
femmi/
├── __init__.py
├── types.py             # Mesh namedtuple
├── mesh.py              # Structured and adaptive P3 mesh generation
├── basis.py             # P3 Lagrange basis (10 DOF/element)
├── assembly.py          # P3 element stiffness/mass assembly
├── bem.py               # BEM: V_h, K_h, M_b, Calderon operator
├── operators.py         # K, M, S1, S2, A_coupled; FEMOperators dataclass
├── forward.py           # DifferentiableForward (JAX custom_vjp)
├── inverse.py           # MAPReconstructor, kaiser_squires
├── regularization.py    # MorozovSelector, estimate_noise_level
└── svd_analysis.py      # SVD of F, Picard diagnostic, FactorizationIndicator, LSM

tests/
├── test_fem_bem_coupling.py    # BEM matrices (V_h, K_h, M_b, Calderon)
├── test_coupled_pipeline.py    # FEM-BEM pipeline invariants
├── test_morozov.py             # Morozov lambda selection, monotonicity
├── test_factorization.py       # SVD, Picard, support recovery
├── test_convergence_p3.py      # O(h^4) L2 Poisson convergence
├── test_convergence.py         # Forward operator gamma convergence
└── test_regression.py          # End-to-end NFW reconstruction

examples/
├── generate_figures.py         # Preliminary results figures (self-contained)
├── smpy_comparison.py          # Full Monte Carlo benchmark vs SMPy KS
└── visualize_results.py        # SVD modes, Picard, convergence diagnostics
```

---

## Installation

```bash
git clone https://github.com/AdamField118/FEMMI.git
cd FEMMI
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, JAX >= 0.4, SciPy >= 1.11, NumPy >= 1.25, matplotlib.

**64-bit arithmetic is mandatory.** FEMMI enforces this at import time via `jax.config.update("jax_enable_x64", True)`. For a $20\times20$ mesh, $\kappa(A_{\mathrm{coupled}}) = O(1600)$; in 32-bit the solve error $O(\kappa\,\varepsilon_{32}) \approx 2\times10^{-5}$ dominates the P3 discretisation error $h^4 \approx 6\times10^{-6}$.

---

## Quick Start

```python
import numpy as np
from femmi.operators      import build_operators
from femmi.forward        import DifferentiableForward
from femmi.inverse        import MAPReconstructor
from femmi.regularization import estimate_noise_level

# Build mesh and operators (20x20 P3 mesh on [-2.5, 2.5]^2)
ops = build_operators(nx=20, ny=20, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5)

# Synthetic convergence field and noiseless forward model
nodes      = np.array(ops.mesh.nodes)
kappa_true = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * 0.5**2))
g1, g2     = ops.forward(kappa_true)

# Add 10% noise
noise = 0.10 * np.std(np.hypot(g1, g2))
rng   = np.random.default_rng(42)
g1_obs = g1 + rng.normal(0, noise, g1.shape)
g2_obs = g2 + rng.normal(0, noise, g2.shape)

# MAP reconstruction with automatic lambda (Morozov)
noise_std = estimate_noise_level(np.concatenate([g1_obs, g2_obs]), method='mad')
fwd = DifferentiableForward(ops, lam_reg=1e-3)
rec = MAPReconstructor(fwd, noise_std=noise_std, wiener_length=0.5)
kappa_map, result = rec.reconstruct(g1_obs, g2_obs)
```

```python
# Locally refined mesh near a circular mask (e.g. bright cluster core)
from femmi.operators import build_operators_adaptive
ops = build_operators_adaptive(
    nx=20, ny=20, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
    mask_center=(0., 0.), mask_radius=0.6, refine_factor=3,
)
```

```python
# SVD and support recovery (Kirsch factorization method)
from femmi.svd_analysis import compute_svd, FactorizationIndicator

svd = compute_svd(ops, n_singular=40)
fi  = FactorizationIndicator(ops, svd_result=svd)

import numpy as np
XX, YY    = np.meshgrid(np.linspace(-2.5, 2.5, 64), np.linspace(-2.5, 2.5, 64))
test_pts  = np.column_stack([XX.ravel(), YY.ravel()])
W         = fi.indicator_map(test_pts).reshape(64, 64)  # large inside supp(kappa)
```

---

## Algorithm Summary

**Forward solve** (two SuperLU solves per MAP iteration):

$$\mathbf{f} = -2M\kappa, \qquad A_{\mathrm{coupled}}\,\psi = \mathbf{f}, \qquad \gamma_1 = S_1\psi, \quad \gamma_2 = S_2\psi.$$

**Adjoint gradient** (for L-BFGS):

$$\mathbf{r} = (\gamma_1 - \gamma_{1,\mathrm{obs}},\; \gamma_2 - \gamma_{2,\mathrm{obs}}), \qquad A_{\mathrm{coupled}}^\top \phi = S_1^\top r_1 + S_2^\top r_2, \qquad \nabla\mathcal{L} = -4M\phi + 2\lambda R\kappa.$$

**Morozov $\lambda$ selection:** Brent root-finding on $D(\lambda) = \|F\kappa_\lambda - \gamma_{\mathrm{obs}}\|_{\mathrm{RMS}} - c\delta$, typically 15–25 forward solves.

**BEM assembly:** Diagonal blocks of $V_h$ use Gauss-Jacobi quadrature with weight $w(t) = -\ln t$ (25 points, relative error $< 10^{-12}$) via Duffy decomposition. Off-diagonal blocks use 25-point Gauss-Legendre.

---

## Convergence Validation

**Poisson solve** (P3, smooth manufactured solution, unit square):

| Mesh | $L^2$ rate | Theory |
|---|---|---|
| $4 \to 8$ | 3.86 | $O(h^4)$ |
| $8 \to 16$ | 3.90 | $O(h^4)$ |
| $16 \to 32$ | 3.97 | $O(h^4)$ |

**Forward operator** ($\gamma$, deep interior, $\sigma=1.5$, ref $n_x=40$):

| Mesh | $\|\gamma_h - \gamma_{\mathrm{ref}}\|$ rate | Theory |
|---|---|---|
| $8 \to 14$ | $\approx 2.0$ | $O(h^2)$ |
| $14 \to 32$ | $\approx 2.0$ | $O(h^2)$ |

The $\psi$ convergence rate is capped at $O(h^{5/3})$ on square domains due to reentrant corner singularities in the exterior BEM solution (singularity exponent $2/3$ from the $270^\circ$ exterior angle). Since $\psi$ is never directly observed — only the shear $\gamma = \nabla^2\psi$ enters the data — this cap does not degrade reconstruction quality.

---

## References

1. Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer.
2. Steinbach, O. (2008). *Numerical Approximation Methods for Elliptic Boundary Value Problems*. Springer.
3. Sauter, S. & Schwab, C. (2011). *Boundary Element Methods*. Springer.
4. Kirsch, A. (1998). Characterization of the shape of a scattering obstacle using the spectral data of the far-field operator. *Inverse Problems*, 14, 1489–1512.
5. Colton, D. & Kirsch, A. (1996). A simple method for solving inverse scattering problems in the resonance region. *Inverse Problems*, 12, 383–393.
6. Morozov, V. A. (1966). On the solution of functional equations by the method of regularization. *Soviet Math. Doklady*, 7, 414–417.
7. Kaiser, N. & Squires, G. (1993). Mapping the dark matter with weak gravitational lensing. *ApJ*, 404, 441–450.
8. Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *IJNME*, 21(6), 1129–1148.
9. Brenner, S. & Scott, R. (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer.