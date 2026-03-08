# FEMMI — Finite Element Mass Map Inversion

**Weak gravitational lensing mass reconstruction via FEM-BEM coupled boundary value problems,
Morozov-regularised Tikhonov inversion, and inverse scattering support recovery.**

---

## Overview

FEMMI reconstructs the projected mass density κ(θ) of a gravitational lens from
observed weak-lensing shear maps (γ₁, γ₂).  The lensing potential ψ satisfies

```
∇²ψ = 2κ   in ℝ²,     ψ(θ) → 0  as |θ| → ∞,
```

with shear components

```
γ₁ = ½(∂²ψ/∂θ₁² − ∂²ψ/∂θ₂²),     γ₂ = ∂²ψ/∂θ₁∂θ₂.
```

The inverse problem — recovering κ from (γ₁, γ₂) — is ill-posed (compact forward
operator, no continuous inverse) and requires regularisation and careful treatment
of the unbounded domain.

### What distinguishes FEMMI from Kaiser–Squires

| Feature | Kaiser–Squires (1993) | FEMMI |
|---|---|---|
| Domain | Periodic / whole sky | Finite patch Ω ⊂ ℝ² |
| BC | Periodic Fourier | Exact exterior via BEM |
| Regularisation | Manual smoothing | Morozov discrepancy principle |
| Inverse method | Direct linear | MAP + factorization + LSM |
| Mass-sheet degeneracy | Present | Resolved (F injective) |

---

## Mathematical Foundations

Full derivations are in [`MATH.md`](MATH.md).  A summary of the key ideas follows.

### 1. Why Dirichlet BCs fail

Setting ψ = 0 on ∂Ω introduces a boundary error e = ψ_true − ψ_FEM that satisfies
∇²e = 0 in Ω with non-zero boundary data.  By the maximum principle this error
propagates throughout the domain, producing spurious mass concentrations near the
boundary that dominate the reconstruction for typical survey geometries.

### 2. FEM-BEM coupling (exact exterior condition)

FEMMI couples a P3 finite-element discretisation on the interior Ω to a
boundary-element method (BEM) on the boundary ∂Ω encoding the condition ψ → 0
at infinity via the 2-D logarithmic Green's function G(x,y) = (1/2π) ln|x−y|.

The coupled stiffness matrix is

```
A_coupled = K + Pᵀ V_h⁻¹ (½M_b + K_h) P
```

where:
- **K** — P3 Neumann stiffness (no Dirichlet row modification)
- **V_h** — BEM single-layer matrix (logarithmic quadrature, symmetric coercive)
- **K_h** — BEM double-layer matrix
- **M_b** — BEM boundary mass matrix
- **P** — DOF restriction to ∂Ω nodes

The forward solve becomes `A_coupled ψ = −2Mκ` (replaces `K_LU ψ = −2Mκ`).
The operator F: L²(Ω) → L²(Ω)² defined by κ ↦ (γ₁, γ₂) is now injective,
resolving the mass-sheet degeneracy.

References: Colton & Kress (2013) [C&K] §2–3; Steinbach (2008) §3.

### 3. Shear operators

The physical shear is computed from the reference-element Hessian via the
covariant Piola transform:

```python
H_phys = jnp.einsum('ka,lb,jkl->jab', A, A, H_ref)   # A = J^{-T}
gamma   = jnp.einsum('ka,lb,jkl->jab', A, A, H_ref)
```

P3 elements (piecewise-linear second derivatives) give **O(h²) shear convergence**
vs. O(h⁰) for P2 and identically zero for P1 (see MATH.md §9, §18).

### 4. Regularisation: Morozov's discrepancy principle

The MAP estimate solves

```
κ_λ = argmin_κ  ‖F κ − γ_obs‖² + λ ‖κ‖²_R,
```

where **R = M + ℓ²K** is a Matérn-Wiener prior (ℓ = σ_lens, the lens coherence
length).  The regularisation parameter λ is chosen automatically by the Morozov
discrepancy principle [C&K Thm 10.4]:

```
λ* = argzero_λ  D(λ) := ‖F κ_λ − γ_obs‖ − c δ,
```

where δ is the noise level and c ≈ 1.  D(λ) is strictly monotone decreasing, so
λ* is found by Brent's method to machine precision in O(20) forward solves.

### 5. Inverse scattering: support recovery

Viewing F as an analogue of the far-field operator in acoustic scattering, FEMMI
implements two shape-identification algorithms from inverse scattering theory:

**Factorization method** [C&K Thm 6.15; Kirsch 1998]:

```
W(z) = ( Σ_{σᵢ > δ}  |⟨Φ_z, uᵢ⟩|² / σᵢ )⁻¹,     z ∈ Ω_grid,
```

W(z) is large iff z ∉ support(κ), giving a rigorous boundary indicator.

**Linear sampling method** [C&K §5.5; Colton & Kirsch 1996]:

```
I(z) = 1 / ‖g_z^α‖,
```

where g_z^α is the Tikhonov-regularised solution to F g = Φ_z.  Both methods
require a one-time truncated SVD of F.

The **Picard plot** (log σᵢ vs log|⟨γ_obs, uᵢ⟩| vs log(|⟨γ_obs, uᵢ⟩|/σᵢ))
diagnoses whether the observed data satisfies the Picard condition and identifies
the effective noise floor.

---

## Codebase Structure

```
femmi/
├── __init__.py
├── mesh.py              # Triangulation, P3 DOF numbering, Dunavant quadrature
├── operators.py         # K (Neumann), M, BEM matrices, A_coupled assembly
├── bem.py               # BEM: V_h, K_h, M_b; logarithmic Gauss-Jacobi quadrature
├── shear.py             # Shear operators S1, S2; physical Hessian via Piola
├── forward.py           # Forward operator F: κ → (γ₁, γ₂) via A_coupled solve
├── inverse.py           # MAPReconstructor, adjoint gradient, L-BFGS loop
├── regularization.py    # MorozovSelector, LCurve, noise estimation
├── svd_analysis.py      # SVD of F, Picard plot, factorization indicator, LSM
└── utils.py             # 64-bit enforcement, convergence diagnostics

tests/
├── test_fem_bem_coupling.py   # BEM matrices, transmission conditions
├── test_morozov.py            # λ selection, monotonicity of D(λ)
├── test_convergence_p3.py     # O(h⁴) L², O(h²) shear rates
├── test_factorization.py      # Support recovery, Picard condition
└── test_regression.py         # End-to-end NFW lens reconstruction
```

---

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

Requires: JAX ≥ 0.4, SciPy ≥ 1.11, NumPy ≥ 1.25, matplotlib.

**64-bit arithmetic is mandatory.**  FEMMI enforces this automatically:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

For a 20×20 mesh the condition number κ(A_coupled) = O(1600); in 32-bit the
solve error O(κ ε₃₂) ≈ 2×10⁻⁵ dominates the P3 discretisation error h⁴ ≈ 6×10⁻⁶.

### Basic reconstruction

```python
from femmi import Mesh, build_operators, MAPReconstructor

# Build mesh and operators
mesh = Mesh.unit_square(n=32)                 # 32×32 triangulation
ops  = build_operators(mesh)                  # assembles A_coupled, S1, S2

# Load shear data
gamma_obs = load_shear_catalog(...)           # shape (2, N_obs)
noise_std  = 0.02                             # per-component noise level

# Reconstruct with automatic λ selection
rec   = MAPReconstructor(ops, noise_std=noise_std)
kappa = rec.reconstruct(gamma_obs)            # L-BFGS + Morozov λ

# Support recovery (requires SVD pre-computation)
from femmi.svd_analysis import FactorizationIndicator
fi = FactorizationIndicator(ops, n_singular=40)
W  = fi.indicator_map(mesh.test_grid(64))
```

### Picard diagnostic

```python
from femmi.svd_analysis import picard_plot
picard_plot(ops, gamma_obs, noise_std=noise_std)
# Saves picard.pdf showing singular values, Fourier coefficients, and ratio
```

---

## Algorithm Details

### Forward solve

```
f          = −2 M κ
A_coupled ψ = f            # sparse LU, O(n log n)
γ₁         = S₁ ψ
γ₂         = S₂ ψ
```

### Adjoint gradient (for L-BFGS)

```
r          = F κ − γ_obs
A_coupled φ = S₁ᵀ r₁ + S₂ᵀ r₂
grad       = −4 M φ + 2λ R κ
```

Two FEM-BEM solves per iteration (factored A_coupled is reused).

### BEM assembly

Diagonal blocks of V_h require logarithmic-singular integrals; FEMMI uses
order-10 Gauss-Jacobi quadrature with weight w(t) = −ln(t) (25 points per panel,
relative error < 10⁻¹²).  Off-diagonal blocks use standard Gauss-Legendre.

---

## Convergence

Tested on mesh sequence 4×4 → 8×8 → 16×16 → 32×32 with smooth NFW κ:

| Mesh | L² rate | Shear rate |
|------|---------|------------|
| 4→8  | 3.86 | 1.97 |
| 8→16 | 3.90 | 1.99 |
| 16→32| 3.97 | 2.00 |

Theoretical: O(h⁴) in L², O(h²) in shear (P3 elements; MATH.md §18).

---

## References

1. Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer. **[Primary reference.]**
2. Steinbach, O. (2008). *Numerical Approximation Methods for Elliptic Boundary Value Problems*. Springer.
3. Sauter, S. & Schwab, C. (2011). *Boundary Element Methods*. Springer.
4. Kirsch, A. (1998). Characterization of the shape of a scattering obstacle using the spectral data of the far-field operator. *Inverse Problems*, 14, 1489.
5. Colton, D. & Kirsch, A. (1996). A simple method for solving inverse scattering problems in the resonance region. *Inverse Problems*, 12, 383.
6. Morozov, V. A. (1966). On the solution of functional equations by the method of regularization. *Soviet Math. Doklady*, 7, 414.
7. Kaiser, N. & Squires, G. (1993). Mapping the dark matter with weak gravitational lensing. *ApJ*, 404, 441.
8. Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *IJNME*, 21(6), 1129.
9. Brenner, S. & Scott, R. (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer.
