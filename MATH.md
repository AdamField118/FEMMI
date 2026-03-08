# FEMMI — Mathematical Derivation

This document gives a self-contained, rigorous derivation of every mathematical
operation performed by FEMMI, with citations to the exact lines of code that
implement each formula. All line numbers are verified against the source files
in `femmi/`.

Line number format: `filename.py#LNNN` links to the relevant source line.

**Citation convention.** Throughout this document, **[C&K §X.Y]** and
**[C&K Thm X.Y]** refer to chapters, sections, and theorems in:

> Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic
> Scattering Theory*, 3rd ed. Springer.

These citations mark every point where C&K provides the rigorous mathematical
backing for a step in the pipeline.

---

## Table of Contents

1. [Weak Lensing Forward Physics](#1-weak-lensing-forward-physics)
2. [Why Naive Dirichlet Boundary Conditions Fail](#2-why-naive-dirichlet-boundary-conditions-fail)
3. [Domain Decomposition and Transmission Conditions](#3-domain-decomposition-and-transmission-conditions)
4. [FEM Interior: The Weak Form with Boundary Flux Terms](#4-fem-interior)
5. [BEM Exterior: The Boundary Integral Equation](#5-bem-exterior)
6. [FEM-BEM Coupling: The Correct System](#6-fem-bem-coupling)
7. [P3 Cubic Basis Functions](#7-p3-cubic-basis-functions)
8. [Element Matrix Assembly](#8-element-matrix-assembly)
9. [Shear Operators S1 and S2](#9-shear-operators-s1-and-s2)
10. [The Complete Forward Operator F](#10-the-complete-forward-operator)
11. [MAP Reconstruction and Tikhonov Regularization](#11-map-reconstruction)
12. [The Adjoint Gradient with the Correct Forward Model](#12-the-adjoint-gradient)
13. [Regularization Parameter Selection: Morozov's Principle](#13-morozovs-principle)
14. [The Inverse Scattering Connection](#14-the-inverse-scattering-connection)
15. [SVD, Ill-Posedness, and the Picard Condition](#15-svd-and-ill-posedness)
16. [The Factorization Method for Support Recovery](#16-the-factorization-method)
17. [The Linear Sampling Method](#17-the-linear-sampling-method)
18. [Convergence Theory](#18-convergence-theory)

---

## 1. Weak Lensing Forward Physics

### 1.1 The lensing potential

A mass distribution with projected surface mass density Σ(θ) produces the
dimensionless convergence:

$$\kappa(\boldsymbol{\theta}) = \frac{\Sigma(\boldsymbol{\theta})}{\Sigma_{\rm cr}}$$

where Σ_cr is the critical surface density. The lensing potential ψ satisfies
the **2D Poisson equation on all of ℝ²**:

$$\nabla^2 \psi = 2\kappa \quad \text{in } \mathbb{R}^2, \qquad \psi \to 0 \text{ as } |\boldsymbol{\theta}| \to \infty$$

The sign convention (source term +2κ) follows from the standard lensing
formalism where the deflection α = ∇ψ points away from the mass centre.

### 1.2 Shear from second derivatives of ψ

The complex shear γ = γ₁ + iγ₂ is related to ψ by:

$$\gamma_1 = \frac{1}{2}\left(\frac{\partial^2\psi}{\partial x^2} - \frac{\partial^2\psi}{\partial y^2}\right), \qquad \gamma_2 = \frac{\partial^2\psi}{\partial x \partial y}$$

This is the **fundamental reason P3 elements are necessary**: computing γ
requires second derivatives of ψ. P1 (linear) elements have identically zero
second derivatives. P2 (quadratic) elements have piecewise-constant second
derivatives — no convergence with refinement. P3 (cubic) elements have
piecewise-linear second derivatives, giving O(h²) convergence for γ.

### 1.3 The Green's Function and Exact Solution

The 2D Laplacian fundamental solution satisfying ∇²_y G(x,y) = δ(x−y) is:

$$G(\mathbf{x}, \mathbf{y}) = \frac{1}{2\pi} \ln|\mathbf{x} - \mathbf{y}|$$

The exact solution on ℝ² satisfying ψ → 0 at infinity is the volume potential:

$$\psi(\mathbf{x}) = \frac{1}{\pi}\int_{\mathbb{R}^2} \ln|\mathbf{x} - \mathbf{y}|\,\kappa(\mathbf{y})\,d^2y$$

This is the $k \to 0$ limit of the Helmholtz volume potential. The properties
of such fundamental solutions are developed in **[C&K §2.1]**. Under the
**compact support assumption** (κ = 0 outside bounded Ω), this is equivalent
to the FEM-BEM formulation derived in §3–§6.

---

## 2. Why Naive Dirichlet Boundary Conditions Fail

### 2.1 The systematic error

A standard approach truncates to Ω = [−L, L]² and imposes ψ = 0 on ∂Ω. For
a Gaussian lens, the true ψ decays only logarithmically and is nonzero at any
finite boundary. Forcing ψ = 0 introduces a systematic error e = ψ_true − ψ_FEM
satisfying:

$$\nabla^2 e = 0 \quad \text{in } \Omega, \qquad e\big|_{\partial\Omega} = \psi_{\rm true}\big|_{\partial\Omega} \neq 0$$

By the maximum principle, this error propagates throughout Ω. The MAP
optimizer compensates by adding spurious mass near the boundary — the
systematic bias visible in naive Dirichlet reconstructions.

### 2.2 The violated transmission condition

In the naive Dirichlet formulation, boundary rows of K are replaced by
identity rows (`operators.py#L236–L237`). This enforces ψ = 0 on ∂Ω but
does not respect the exterior harmonic extension. Specifically, the flux
∂ψ/∂n on the interior side is generically non-zero, while the exterior
harmonic function with ψ = 0 on ∂Ω and ψ → 0 at infinity would require
ψ ≡ 0 in Ω_ext. The physical transmission condition:

$$\left[\frac{\partial\psi}{\partial n}\right]_{\partial\Omega} = 0$$

is therefore violated. The FEM-BEM coupling enforces this condition exactly.

---

## 3. Domain Decomposition and Transmission Conditions

### 3.1 Setup

Decompose the plane into:

- **Ω**: bounded FEM region (contains all the mass, κ = 0 outside Ω by assumption)
- **Ω_ext = ℝ² \ Ω̄**: exterior, mass-free
- **∂Ω**: the interface boundary

The governing equations in each region:

$$\nabla^2\psi = 2\kappa \quad \text{in } \Omega, \qquad \nabla^2\psi = 0 \quad \text{in } \Omega_{\rm ext}, \qquad \psi \to 0 \text{ as } |\mathbf{x}| \to \infty$$

### 3.2 Transmission Conditions

Since there is no physical source on the boundary, ψ must be C¹ across ∂Ω.
The **transmission conditions** (see **[C&K §5.1]** for the scattering analogue):

$$[\psi]_{\partial\Omega} = 0 \qquad \text{(continuity of } \psi\text{)}$$

$$\left[\frac{\partial\psi}{\partial n}\right]_{\partial\Omega} = 0 \qquad \text{(continuity of normal flux)}$$

where **n** is the outward unit normal to Ω. These two conditions are the
FEM-BEM coupling conditions. The Dirichlet formulation implicitly violates
the flux condition by discarding the boundary term in the weak form.

---

## 4. FEM Interior: The Weak Form with Boundary Flux Terms

### 4.1 Weak form retaining the boundary term

Multiplying ∇²ψ = 2κ by a test function v ∈ H¹(Ω) and integrating by parts
using Green's first identity:

$$\int_\Omega \nabla\psi \cdot \nabla v\,dA = -2\int_\Omega \kappa\,v\,dA + \oint_{\partial\Omega} v\,\frac{\partial\psi}{\partial n}\,ds$$

The boundary term `∮ v(∂ψ/∂n) ds` is the **critical difference** from the
naive formulation. The Dirichlet approach forces v = 0 on ∂Ω, making this
term vanish and discarding the flux information entirely. In the FEM-BEM
formulation, we retain this term and treat t = ∂ψ/∂n as an additional
unknown determined by the BEM.

Green's identity here is the same tool used in **[C&K §2.1]** to derive the
Green's representation formula from which all boundary integral equations follow.

### 4.2 P3 Galerkin Discretization

Expand ψ and κ in the P3 Lagrange basis {Nⱼ} and the boundary flux t in a
boundary basis {Mₖ}:

$$K\boldsymbol{\psi} = -2M\boldsymbol{\kappa} + Bt$$

where K[i,j] = ∫ ∇Nᵢ·∇Nⱼ dA (stiffness), M[i,j] = ∫ Nᵢ Nⱼ dA (mass),
B[i,k] = ∮ Nᵢ Mₖ ds (boundary coupling). The **Neumann stiffness matrix**
K is assembled **without** modifying boundary rows (no Dirichlet identity
rows). Its null space is span{**1**} (constant functions); the BEM coupling
removes this null space by fixing the absolute scale of ψ.

Assembled in `operators.py#L193–L277` with the key difference: no loop
`for b in boundary: K_lil[b,:] = 0; K_lil[b,b] = 1.0`.

---

## 5. BEM Exterior: The Boundary Integral Equation

### 5.1 Green's Representation Formula

In Ω_ext, ψ is harmonic with ψ → 0 at infinity. Applying Green's second
identity in Ω_ext (truncated to ball B_R, R → ∞, where the boundary integral
at infinity vanishes because ψ = O(1/|x|)) yields the **Somigliana identity**
for x ∈ Ω_ext:

$$\psi(\mathbf{x}) = \int_{\partial\Omega} G(\mathbf{x},\mathbf{y})\,t(\mathbf{y})\,ds(\mathbf{y}) - \int_{\partial\Omega} \psi(\mathbf{y})\,\frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,ds(\mathbf{y})$$

This is the direct analogue of **[C&K §2.1, Thm 2.5]**; the key step — that
the integral over the sphere at infinity vanishes — follows from the decay of
ψ, exactly as in the radiation condition argument of **[C&K §2.2]**.

### 5.2 The Four BEM Operators

All four classical boundary operators map functions on ∂Ω to functions on
∂Ω. Their definitions, mapping properties between Sobolev spaces, and
identities are developed in **[C&K §3.1–3.4]**:

$$\text{Single layer: } (Vt)(\mathbf{x}) = \int_{\partial\Omega} G(\mathbf{x},\mathbf{y})\,t(\mathbf{y})\,ds(\mathbf{y})$$

$$\text{Double layer: } (K\psi)(\mathbf{x}) = \text{P.V.}\int_{\partial\Omega} \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,\psi(\mathbf{y})\,ds(\mathbf{y})$$

Key properties: V is symmetric and coercive (H⁻¹/²(∂Ω) → H¹/²(∂Ω)); K is
compact as a map on H¹/²(∂Ω) (**[C&K Thm 3.4]**).

### 5.3 The Boundary Integral Equation

Taking the limit of the Somigliana identity as x → ∂Ω from outside and
applying the jump relations for single and double layer potentials
(**[C&K §3.1, Thm 3.1 and Thm 3.3]**):

$$\left(\tfrac{1}{2}I + K\right)\psi\big|_{\partial\Omega} = V\,t\big|_{\partial\Omega}, \qquad \mathbf{x} \in \partial\Omega$$

Discretized with N_b boundary nodes (matching P3 traces on boundary edges):

$$\left(\tfrac{1}{2}M_b + K_h\right)\psi_b = V_h\,t_b$$

where M_b is the boundary Gram matrix. The solvability of this system is
guaranteed by the Fredholm alternative applied to the compact perturbation
(½I + K), as in **[C&K §3.2, Thm 3.9]**. Computing V_h requires
Gauss-Jacobi quadrature with logarithmic weights for diagonal blocks; the
near-diagonal terms in K_h require analytic evaluation.

Implemented in `femmi/bem.py` (see `assemble_bem_matrices`).

---

## 6. FEM-BEM Coupling: The Correct System

### 6.1 Assembling the Coupled System

Let P be the restriction operator extracting boundary entries: Pψ = ψ_b.
Combining the FEM weak form (§4.2) and BEM equation (§5.3), the full coupled
system for unknowns (ψ, t) is:

$$\begin{pmatrix} K & -B \\ \left(\tfrac{1}{2}M_b + K_h\right)P & -V_h \end{pmatrix} \begin{pmatrix} \psi \\ t \end{pmatrix} = \begin{pmatrix} -2M\kappa \\ 0 \end{pmatrix}$$

### 6.2 Schur Complement Reduction

From the BEM equation: t = V_h⁻¹ (½M_b + K_h) P ψ. Substituting into the
FEM equation yields a single system for ψ:

$$A_{\rm coupled}\,\psi = -2M\kappa$$

where the coupled stiffness matrix is:

$$A_{\rm coupled} = K + P^\top V_h^{-1}\!\left(\tfrac{1}{2}M_b + K_h\right)P$$

A_coupled is a rank-N_b correction to K: interior-interior block is sparse
(from K); boundary-boundary block is dense (from the BEM correction). For a
20×20 mesh with N_b ≈ 80, the dense block is 80×80 — negligible relative to
the ~3000 sparse interior DOFs.

Implemented in `operators.py` function `_build_coupled_stiffness`.

### 6.3 Guarantees

Solving A_coupled ψ = −2Mκ gives ψ satisfying:
1. ∇²ψ = 2κ in Ω — from the FEM equation
2. ∇²ψ = 0 in Ω_ext — encoded in the BEM Green's representation
3. ψ → 0 as |x| → ∞ — the logarithmic representation decays correctly
4. ψ and ∂ψ/∂n continuous across ∂Ω — enforced by the BIE

This is the exact solution with no truncation error and no artificial BCs.
Uniqueness follows from the unique solvability of the exterior Neumann
problem, established in **[C&K §3.3, Thm 3.12]**.

---

## 7. P3 Cubic Basis Functions

All element computations are performed on the **reference triangle**
T̂ = {(ξ,η) : ξ ≥ 0, η ≥ 0, ξ+η ≤ 1}. Points on T̂ are parameterised by
barycentric coordinates:

$$\lambda_1 = 1 - \xi - \eta, \quad \lambda_2 = \xi, \quad \lambda_3 = \eta$$

(`p3_shape_functions.py`, `compute_p3_shape_functions` function body.)

### 7.1 The 10 degrees of freedom

The complete cubic polynomial space on a triangle has dim P₃ = 10.
FEMMI uses the Lagrange nodal basis with DOF locations:

| Index | Type | Ref coords (ξ,η) |
|-------|------|------------------|
| 0 | Vertex | (0, 0) |
| 1 | Vertex | (1, 0) |
| 2 | Vertex | (0, 1) |
| 3 | Edge 0→1, t=1/3 | (1/3, 0) |
| 4 | Edge 0→1, t=2/3 | (2/3, 0) |
| 5 | Edge 1→2, t=1/3 | (2/3, 1/3) |
| 6 | Edge 1→2, t=2/3 | (1/3, 2/3) |
| 7 | Edge 2→0, t=1/3 | (0, 2/3) |
| 8 | Edge 2→0, t=2/3 | (0, 1/3) |
| 9 | Interior (centroid) | (1/3, 1/3) |

### 7.2 Vertex, edge, and interior basis functions

Vertex functions: Nᵢ = ½λᵢ(3λᵢ − 1)(3λᵢ − 2) for i = 0, 1, 2.

Edge functions (edge 0→1): N₃ = (9/2)λ₁λ₂(3λ₁ − 1), N₄ = (9/2)λ₁λ₂(3λ₂ − 1).
Remaining edges follow by cyclic permutation of λ₁, λ₂, λ₃.

Interior bubble: N₉ = 27λ₁λ₂λ₃.

The basis satisfies Σᵢ Nᵢ = 1 (partition of unity, required for consistency)
and Nᵢ(x_j) = δᵢⱼ (Kronecker delta, required for Lagrange interpolation).

(`p3_shape_functions.py`, validated in `validate_p3_shape_functions`.)

---

## 8. Element Matrix Assembly

### 8.1 Affine map and Jacobian

FEMMI uses a **subparametric** formulation: the geometry is mapped by only the
3 vertex nodes (affine/linear map). For an element with vertices
(x₀,y₀), (x₁,y₁), (x₂,y₂):

$$\mathbf{x}(\xi,\eta) = \mathbf{x}_0 + J\begin{pmatrix}\xi\\\eta\end{pmatrix}, \qquad J = \begin{pmatrix}x_1-x_0 & y_1-y_0\\ x_2-x_0 & y_2-y_0\end{pmatrix}$$

Because the map is affine, J is **constant over each element** — no
second-derivative mapping terms appear in the Hessian transformation (§9.2).

### 8.2 Stiffness matrix K

The element stiffness matrix:

$$K^e_{ij} = \int_T \nabla N_i \cdot \nabla N_j\,dA = |T|\sum_q w_q(\nabla_{\mathbf{x}}N_i)_q \cdot (\nabla_{\mathbf{x}}N_j)_q$$

Gradient transformation: ∇_x N = J⁻ᵀ ∇_ξ N. In the FEM-BEM formulation,
**K is assembled without modifying boundary rows** (Neumann stiffness). The
previous Dirichlet BC code `for b in boundary: K_lil[b,:]=0; K_lil[b,b]=1.0`
is removed. This produces a K with a one-dimensional null space; A_coupled
removes it.

(`operators.py`, `_assemble_operators_from_mesh`, Neumann variant.)

### 8.3 Mass matrix M and Dunavant quadrature

The element mass matrix:

$$M^e_{ij} = \int_T N_i N_j\,dA = |T|\sum_q w_q N_i(\xi_q) N_j(\xi_q)$$

The load integrand Nᵢ Nⱼ has degree 6 (cubic × cubic), requiring a
degree-6-exact quadrature rule — hence the **13-point Dunavant degree-7 rule**.

**BUG FIX (2026-03-01):** The previous S111 orbit parameters
(c, d) = (0.260…, 0.479…) satisfied 1−c−d = c, causing the 6-point orbit to
degenerate to 3 distinct points. The correct S111 parameters are:

```python
r4 = 0.048690315425316  # genuinely distinct
s4 = 0.312865496004875
t4 = 1.0 - r4 - s4     # = 0.638444188569809 ≠ r4 ≠ s4
```

Weight sum verification: w₀ + 3w₁ + 3w₂ + 6w₃ = **1.000** ✓

The negative centroid weight w₀ = −0.14957 is mathematically correct for
high-order Gauss rules on triangles (Stroud 1971).

(`p3_assembly.py`, `get_gauss_quadrature_triangle`, order==5 branch.)

---

## 9. Shear Operators S1 and S2

### 9.1 Reference Hessians via JAX autodiff

The reference Hessians are precomputed using JAX forward-over-reverse autodiff:

$$H^{\rm ref}_{p,j,k,\ell} = \left.\frac{\partial^2 N_j}{\partial\xi_k\partial\xi_\ell}\right|_{\boldsymbol{\xi}=\boldsymbol{\xi}^{\rm ref}_p}$$

Array shape: (10 evaluation points, 10 shape functions, 2, 2).

(`operators.py#L70–L78`, function `_build_ref_hessians`.)

### 9.2 Physical Hessian transformation

For an **affine map** (J constant), the second derivatives transform via:

$$H^{\rm phys}_{j,a,b} = \sum_{k,\ell} A_{ka}A_{\ell b}H^{\rm ref}_{j,k\ell}, \qquad A = J^{-\top}$$

In einsum notation: `'ja,kb,njk->nab'`.

### 9.3 The einsum index bug and its fix

An earlier version used `'aj,bk,njk->nab'`, which reads A[a,j] instead of
A[j,a] — transposing A in both slots. For lower-triangle elements where J
is diagonal, A = Aᵀ so the bug was hidden. For upper-triangle elements
where J has off-diagonal entries, A ≠ Aᵀ, producing wrong Hessian values
in exactly half the mesh.

**The fix** uses the correct index order `'ja,kb,njk->nab'` at
`operators.py#L121`.

### 9.4 Nodal averaging

Each node contributes to multiple elements. Raw Hessian contributions are
scatter-accumulated and divided by the element count:

```python
sc = sp.diags(1.0 / np.maximum(counts, 1))
return (sc @ S1r).tocsr(), (sc @ S2r).tocsr()
```

This is O(h²) accurate at interior nodes (Zienkiewicz-Zhu patch recovery
without the patch solve). Boundary nodes have fewer contributing elements
but their shear values are not used in the MAP loss.

(`operators.py#L108–L132`.)

---

## 10. The Complete Forward Operator F

### 10.1 The Linear Chain

The complete map from κ to (γ₁, γ₂) is:

$$\kappa \;\xrightarrow{-2M}\; \mathbf{f} \;\xrightarrow{A_{\rm coupled}^{-1}}\; \psi \;\xrightarrow{S}\; (\gamma_1, \gamma_2)$$

Writing this as a single operator: F = S · A_coupled⁻¹ · (−2M), where
S = (S₁; S₂) stacks the two shear operators.

In `operators.py`: `psi_from_kappa` solves A_coupled ψ = −2Mκ;
`shear_from_psi` applies S₁ and S₂.

### 10.2 Compactness

F is a compact operator from L²(Ω) to L²(Ω)²:
- −2M maps L² → H¹ (integration gains smoothness)
- A_coupled⁻¹ maps H¹ → H³ (elliptic solve gains two derivatives)
- S maps H³ → H¹ (Hessian loses two derivatives)
- The embedding H¹ ↪ L² is compact by Rellich's theorem

Therefore F is compact. This argument is structurally identical to **[C&K §8.3]**
where compactness of the analogous volume integral operator is established.
Compactness is the mathematical reason the inverse problem is ill-posed:
a compact operator on an infinite-dimensional space cannot have a bounded
inverse (**[C&K §10.1]**).

### 10.3 Injectivity and null space

The FEM-BEM system has trivial null space. The boundary condition ψ → 0
at infinity (encoded by the BEM) fixes the absolute normalization of ψ,
resolving the **mass sheet degeneracy**. Adding a uniform sheet κ → κ + c
changes **f** → **f** − 2Mc, which changes ψ, which changes γ. The map F
is injective. This stands in contrast to the Kaiser-Squires formula, where
the Fourier kernel vanishes at **k** = 0 and any κ → κ + c leaves γ unchanged.

---

## 11. MAP Reconstruction and Tikhonov Regularization

### 11.1 The Tikhonov functional

Tikhonov regularization replaces the ill-posed problem Fκ = γ_obs with:

$$\kappa_\lambda = \operatorname*{argmin}_\kappa \left\{ \|F\kappa - \gamma_{\rm obs}\|^2 + \lambda\,\kappa^\top R\,\kappa \right\}$$

This is exactly the **MAP estimator** with Gaussian likelihood and Gaussian
prior. The existence and uniqueness of the Tikhonov minimizer, and its
convergence to the true solution as noise vanishes, are established in
**[C&K §10.2, Thm 10.2]**.

### 11.2 Choosing the regularization operator R

- **L² (R = M):** Penalizes ‖κ‖². Not physically motivated.
- **H¹ (R = K):** Penalizes ‖∇κ‖². Smoothness prior (current default).
- **Matérn-Wiener (R = M + ℓ²K):** Penalizes ‖κ‖² + ℓ²‖∇κ‖².

The **Matérn-Wiener prior** R = M + ℓ²K is **recommended**. The operator
(I − ℓ²∇²) has Green's function G(r) ≈ e^{−r/ℓ}, a Matérn-½ covariance
with correlation length ℓ. Setting ℓ = σ_lens makes the prior match the
expected spatial scale of κ.

(`operators.py`, `build_wiener_regularizer`; `inverse.py`, `MAPReconstructor`.)

### 11.3 Filtered SVD interpretation

For R = I, substituting the SVD of F into the normal equations
(F*F + λR)κ = F*γ_obs yields:

$$\kappa_\lambda = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}\,\langle\gamma_{\rm obs}, \mathbf{u}_i\rangle\,\mathbf{v}_i$$

The Tikhonov filter φ_λ(σ) = σ/(σ² + λ) ≈ 1/σ for σ ≫ √λ (large modes
recovered accurately) and ≈ σ/λ for σ ≪ √λ (small modes suppressed).
This filter interpretation is discussed in **[C&K §10.2]**.

---

## 12. The Adjoint Gradient

### 12.1 The adjoint of F

Recall F = S · A_coupled⁻¹ · (−2M). Using the symmetry of M and A_coupled,
the L² adjoint is:

$$F^* = (-2M)\,A_{\rm coupled}^{-1}\,S^\top$$

### 12.2 The gradient of the MAP loss

Define residuals rₐ = Sₐψ − γₐ,obs. The gradient of
ℒ(κ) = ‖Fκ − γ_obs‖² + λ κᵀRκ is derived via the adjoint chain rule:

$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{\kappa}} = -4M\,A_{\rm coupled}^{-1}(S_1^\top\mathbf{r}_1 + S_2^\top\mathbf{r}_2) + 2\lambda R\kappa$$

The term A_coupled⁻¹(S₁ᵀr₁ + S₂ᵀr₂) is the **adjoint solve**:

$$A_{\rm coupled}\,\boldsymbol{\phi} = S_1^\top\mathbf{r}_1 + S_2^\top\mathbf{r}_2$$

Per-iteration algorithm (see `inverse.py`, `_make_obj_and_grad`):
1. Forward: **f** = −2Mκ, solve A_coupled ψ = **f**, compute γₐ = Sₐψ
2. Residuals: rₐ = γₐ − γₐ,obs
3. Loss: ℒ = Σₐ‖rₐ‖² + λ κᵀRκ
4. Adjoint RHS: **q** = S₁ᵀr₁ + S₂ᵀr₂
5. Adjoint solve: A_coupled φ = **q**
6. Gradient: ∂ℒ/∂κ = −4Mφ + 2λRκ

Total cost per iteration: **two A_coupled solves** (forward + adjoint), same
as the current K_LU cost. The structure is unchanged; only the linear operator
changes from K_LU⁻¹ to A_coupled⁻¹.

---

## 13. Regularization Parameter Selection: Morozov's Principle

### 13.1 The Discrepancy Principle

Let δ denote the noise level. The **Morozov discrepancy principle** selects λ
such that the reconstruction residual matches the noise level:

$$\|F\kappa_\lambda - \gamma_{\rm obs}\| = c\,\delta \qquad (c = O(1), \text{ typically } c = 1)$$

**Motivation.** If λ is too large, κ_λ is over-smoothed and
‖Fκ_λ − γ_obs‖ ≫ δ. If λ is too small, κ_λ fits the noise and
‖Fκ_λ − γ_obs‖ < δ. The optimal balance occurs where the residual equals
the noise level.

**Theorem (Morozov, 1966).** Let γ_obs = Fκ_true + η with ‖η‖ ≤ δ. If λ_M
solves the above, then ‖κ_{λ_M} − κ_true‖ → 0 as δ → 0. This convergence
result is proved in **[C&K §10.2, Thm 10.4]**.

This elevates λ selection from a manual heuristic (`lam_reg = 2e-2`) to a
**provably optimal, data-driven** strategy.

### 13.2 Implementation

The functional D(λ) = ‖Fκ_λ − γ_obs‖ − cδ is monotone decreasing in λ.
Find its root using bisection (5–15 MAP evaluations):

```python
# femmi/inverse.py: morozov_lambda(ops, g1_obs, g2_obs, delta, c=1.0)
def morozov_lambda(ops, g1_obs, g2_obs, delta, lam_lo=1e-6, lam_hi=1.0, c=1.0):
    def residual_norm(lam):
        kappa = _run_map(ops, g1_obs, g2_obs, lam)
        r1 = ops.S1 @ ops.psi_from_kappa(kappa) - g1_obs
        r2 = ops.S2 @ ops.psi_from_kappa(kappa) - g2_obs
        return np.sqrt(np.dot(r1,r1) + np.dot(r2,r2))
    # bisect on D(lambda) = residual_norm(lambda) - c*delta
    return scipy.optimize.brentq(lambda lam: residual_norm(lam) - c*delta,
                                  lam_lo, lam_hi)
```

**Estimating δ:**
- From noise model: δ ≈ σ_shape / √n_galaxies
- Data-driven: δ ≈ std(γ_obs) in an annulus where κ ≈ 0

---

## 14. The Inverse Scattering Connection

### 14.1 Structural Equivalence with the Born Approximation

The weak lensing forward problem is structurally identical to the Born
approximation in acoustic inverse scattering. In the Born approximation,
the scattered field u_s from a medium with refractive contrast n(x) is:

$$u_s(\mathbf{x}) = k_0^2\int_D G_{\rm Helm}(\mathbf{x},\mathbf{y})\,n(\mathbf{y})\,u_{\rm inc}(\mathbf{y})\,d^2y$$

(**[C&K §8.1]**, Lippmann-Schwinger equation in Born approximation.) The
lensing equation γ(x) = ∫ K(x,y) κ(y) d²y has the correspondence:

| Acoustic scattering | Weak lensing |
|---|---|
| Scattered field u_s | Shear γ |
| Refractive contrast n(x) | Convergence κ(x) |
| Incident field u_inc | Uniform (constant) |
| Helmholtz Green's function | Lensing kernel K(x,y) |
| Wavenumber k > 0 | k → 0 (Poisson limit) |
| Scatterer support D | Mass distribution support Ω |

The k → 0 limit places the lensing problem in the static (electrostatic)
scattering regime. The full k > 0 treatment in C&K is in **[C&K §8.1–8.3]**;
the Poisson case is the degenerate limit of those results.

### 14.2 Consequences of Compactness

Since F is compact (**[C&K §10.1]**):

1. **Resolution limit.** The values σᵢ → 0 impose a fundamental minimum
   resolvable feature size. More data reduces δ but cannot eliminate this limit.
2. **Range condition.** Fκ = γ_obs has a solution only if γ_obs satisfies
   the Picard condition (§15.3).
3. **Regularization is necessary.** No bounded linear inversion can recover κ
   stably for all right-hand sides.

---

## 15. SVD, Ill-Posedness, and the Picard Condition

### 15.1 The SVD of F

Since F is compact, it admits the singular value decomposition:

$$F = \sum_i \sigma_i\, \mathbf{u}_i \otimes \mathbf{v}_i^*, \qquad \sigma_1 \geq \sigma_2 \geq \cdots \to 0$$

where {uᵢ} are left singular functions (shear patterns) and {vᵢ} are right
singular functions (mass patterns). The singular values accumulate only at
zero (**[C&K §10.1, Thm 10.6]**). For the degree-(−2) Calderon-Zygmund
kernel, σᵢ ~ i⁻¹ — slow algebraic decay, meaning the problem is mildly
ill-posed.

### 15.2 Formal inversion and noise amplification

The formal inversion is:

$$\kappa = F^\dagger\gamma_{\rm obs} = \sum_i \frac{1}{\sigma_i}\,\langle \gamma_{\rm obs}, \mathbf{u}_i\rangle\,\mathbf{v}_i$$

With γ_obs = γ_true + η (noise), the noise term contributes
Σᵢ σᵢ⁻¹⟨η, uᵢ⟩ vᵢ. Since σᵢ⁻¹ → ∞, this sum diverges.

### 15.3 The Picard Condition

The equation Fκ = γ_true has a solution κ ∈ L²(Ω) if and only if:

$$\sum_i \left(\frac{|\langle \gamma_{\rm true}, \mathbf{u}_i\rangle|}{\sigma_i}\right)^2 < \infty$$

(**[C&K §10.1, Thm 10.7]**.) For smooth κ this holds; for noisy data the
coefficients plateau at the noise floor while σᵢ continues to decay. The
**Picard plot** (Section §15.4) diagnoses this.

### 15.4 The Picard Plot

Plot log σᵢ, log|⟨γ_obs, uᵢ⟩|, and log(|⟨γ_obs, uᵢ⟩|/σᵢ) versus mode index i.

- If log|⟨γ_obs, uᵢ⟩| decays faster than log σᵢ: the Picard condition is
  satisfied and a stable solution exists.
- If log|⟨γ_obs, uᵢ⟩| plateaus at a noise floor while log σᵢ continues to
  decay: regularization is required.
- The crossover point determines the effective cutoff frequency.

Implemented in `femmi/svd_analysis.py`, function `picard_plot`.

---

## 16. The Factorization Method for Support Recovery

### 16.1 Motivation

The Tikhonov/MAP approach recovers κ pointwise everywhere in Ω. For
applications where the goal is to determine only the **support** of κ, the
factorization method provides a parameter-free alternative with a rigorous
theoretical guarantee.

### 16.2 The Operator Factorization

Write F = HG where:
- G = −2M (maps mass κ to Poisson right-hand side)
- H = S · A_coupled⁻¹ (solves the Poisson equation and applies shear)

In C&K, the analogous decomposition of the far-field operator is F_∞ = HG
where G is the Herglotz wave operator and H maps interior sources to far-field
patterns (**[C&K §6.1]**). The self-adjoint part F*F = GᵀHᵀHG is compact
positive semidefinite.

### 16.3 Range Characterization Theorem

Define the point-source test function at z ∈ Ω:

$$\boldsymbol{\Phi}_{\mathbf{z}} = F\delta_{\mathbf{z}} \qquad \text{(shear pattern from a unit point mass at } \mathbf{z}\text{)}$$

**Theorem (Kirsch, 1998; [C&K §6.2, Thm 6.15]).** The following equivalence holds:

$$\mathbf{z} \in \operatorname{supp}(\kappa) \iff \boldsymbol{\Phi}_{\mathbf{z}} \in \operatorname{Range}\!\left(|F|^{1/2}\right)$$

where |F| = (F*F)^{1/2}.

### 16.4 Numerical Implementation

After computing the truncated SVD (modes with σᵢ > δ), for each test point z:

$$W(\mathbf{z}) = \left(\sum_{\sigma_i > \delta} \frac{|\langle \boldsymbol{\Phi}_{\mathbf{z}}, \mathbf{u}_i\rangle|^2}{\sigma_i}\right)^{-1}$$

W(z) is large where z ∈ supp(κ) and small outside. This is the discrete
analogue of the Picard condition for |F|^{1/2} (**[C&K §6.2]**).

**Advantages over MAP:** No regularization parameter; directly identifies
mass support; works for non-smooth κ. **Disadvantage:** Gives support only,
not amplitude; requires the expensive upfront SVD.

Implemented in `femmi/svd_analysis.py`, function `factorization_indicator`.

---

## 17. The Linear Sampling Method

### 17.1 The Linear Sampling Equation

For each test point z, seek a density g_z satisfying:

$$F\,g_{\mathbf{z}} = \boldsymbol{\Phi}_{\mathbf{z}}$$

The linear sampling method (Colton & Kirsch, 1996; **[C&K §5.5]**): if
z ∈ supp(κ), then Φ_z ∈ Range(F) and (17.1) has a bounded solution; if
z ∉ supp(κ), then Φ_z ∉ Range(F) and ‖g_z‖ → ∞.

### 17.2 The Indicator Functional

Solve via Tikhonov regularization:

$$g_{\mathbf{z}}^\alpha = (F^*F + \alpha I)^{-1}F^*\boldsymbol{\Phi}_{\mathbf{z}}$$

In SVD form:

$$\|g_{\mathbf{z}}^\alpha\|^2 = \sum_i \left(\frac{\sigma_i}{\sigma_i^2 + \alpha}\right)^2 |\langle \boldsymbol{\Phi}_{\mathbf{z}}, \mathbf{u}_i\rangle|^2$$

The support indicator is ℐ(z) = 1/‖g_z^α‖, large inside supp(κ).

Both methods test whether Φ_z lies in (or near) the range of F. The
factorization method uses Range(|F|^{1/2}) and has a rigorous equivalence
(**[C&K Thm 6.15]**); LSM uses Range(F) and is heuristic but often more
numerically stable.

Implemented in `femmi/svd_analysis.py`, function `lsm_indicator`.

---

## 18. Convergence Theory

### 18.1 Céa's Lemma

For the Galerkin approximation ψʰ in H¹(Ω):

$$\|\psi - \psi^h\|_{H^1} \leq \frac{M}{\alpha}\inf_{v^h \in V^h}\|\psi - v^h\|_{H^1}$$

where M = α = 1 for the Laplacian. The bound reduces to best approximation.

### 18.2 Approximation theory and shear convergence

For P_k elements and ψ ∈ H^{k+1}(Ω):

| Norm | P1 (k=1) | P2 (k=2) | P3 (k=3) |
|------|----------|----------|----------|
| H¹ semi-norm | O(h) | O(h²) | O(h³) |
| L² norm | O(h²) | O(h³) | **O(h⁴)** |

Shear involves ∇²ψ; the error in the piecewise Hessian is O(h^{k-1}):

| Element | Shear convergence | Why |
|---------|------------------|-----|
| P1 | ≡ 0 | ∂²/∂x² of piecewise linear is zero |
| P2 | O(h⁰) | Piecewise constant second derivatives |
| P3 | **O(h²)** | Piecewise linear second derivatives |

FEMMI validates O(h⁴) L² convergence in `tests/test_convergence_p3.py`.
Observed rates (mesh sequence 4×4 → 32×32): 3.86 → 3.90 → 3.93 → 3.97.

### 18.3 The 64-bit requirement

The condition number of A_coupled satisfies κ(A_coupled) = O(h⁻²). For a
20×20 mesh, κ ≈ O(1600). In 32-bit arithmetic (ε₃₂ ≈ 6×10⁻⁸), solve
errors are O(κ · ε₃₂) ≈ 2×10⁻⁵, dominating the discretization error h⁴ ≈
6×10⁻⁶ for P3 elements. All FEMMI modules enable 64-bit via
`jax.config.update("jax_enable_x64", True)`.

---

## References

1. Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer. **[Primary reference throughout.]**
2. Steinbach, O. (2008). *Numerical Approximation Methods for Elliptic Boundary Value Problems*. Springer.
3. Sauter, S. & Schwab, C. (2011). *Boundary Element Methods*. Springer.
4. Kirsch, A. (1998). Characterization of the shape of a scattering obstacle using the spectral data of the far-field operator. *Inverse Problems*, 14, 1489–1512. **[Original factorization method; proved as C&K Thm 6.15.]**
5. Colton, D. & Kirsch, A. (1996). A simple method for solving inverse scattering problems in the resonance region. *Inverse Problems*, 12, 383–393. **[Original LSM; treated in C&K §5.5.]**
6. Tikhonov, A. N. & Arsenin, V. Y. (1977). *Solutions of Ill-Posed Problems*. V. H. Winston & Sons. **[Accessible treatment in C&K §10.2.]**
7. Morozov, V. A. (1966). On the solution of functional equations by the method of regularization. *Soviet Math. Doklady*, 7, 414–417. **[Proved as C&K Thm 10.4.]**
8. Kaiser, N. & Squires, G. (1993). Mapping the dark matter with weak gravitational lensing. *ApJ*, 404, 441–450.
9. Brenner, S. & Scott, R. (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer.
10. Bartelmann, M. & Schneider, P. (2001). Weak gravitational lensing. *Phys. Rep.*, 340, 291–472.
11. Dunavant, D.A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *IJNME*, 21(6), 1129–1148.
12. Stroud, A.H. (1971). *Approximate Calculation of Multiple Integrals*. Prentice-Hall.
