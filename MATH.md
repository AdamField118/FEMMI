# FEMMI --- Mathematical Derivation

This document gives a self-contained, rigorous derivation of every mathematical
operation performed by FEMMI, with references to the functions that implement
each formula.

**Citation convention.** Throughout this document, **[C\&K \S X.Y]** and
**[C\&K Thm X.Y]** refer to:

> Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic
> Scattering Theory*, 3rd ed. Springer.

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

A mass distribution with projected surface mass density $\Sigma(\theta)$ produces the
dimensionless convergence:

$$\kappa(\boldsymbol{\theta}) = \frac{\Sigma(\boldsymbol{\theta})}{\Sigma_{\rm cr}}$$

where $\Sigma_{\rm cr}$ is the critical surface density. The lensing potential $\psi$ satisfies
the **2D Poisson equation on all of $\mathbb{R}^2$**:

$$\nabla^2 \psi = 2\kappa \quad \text{in } \mathbb{R}^2, \qquad \psi \to 0 \text{ as } |\boldsymbol{\theta}| \to \infty$$

### 1.2 Shear from second derivatives of $\psi$

The complex shear $\gamma = \gamma_1 + i\gamma_2$ is related to $\psi$ by:

$$\gamma_1 = \frac{1}{2}\left(\frac{\partial^2\psi}{\partial x^2} - \frac{\partial^2\psi}{\partial y^2}\right), \qquad \gamma_2 = \frac{\partial^2\psi}{\partial x \partial y}$$

This is the **fundamental reason P3 elements are necessary**: computing $\gamma$
requires second derivatives of $\psi$. P1 (linear) elements have identically zero
second derivatives. P2 (quadratic) elements have piecewise-constant second
derivatives --- no convergence with refinement. P3 (cubic) elements have
piecewise-linear second derivatives, giving $O(h^2)$ convergence for $\gamma$.

### 1.3 The Green's Function and Exact Solution

The 2D Laplacian fundamental solution satisfying $\nabla^2_y G(x,y) = \delta(x-y)$ is:

$$G(\mathbf{x}, \mathbf{y}) = \frac{1}{2\pi} \ln|\mathbf{x} - \mathbf{y}|$$

The exact solution on $\mathbb{R}^2$ satisfying $\psi \to 0$ at infinity is the volume potential:

$$\psi(\mathbf{x}) = \frac{1}{\pi}\int_{\mathbb{R}^2} \ln|\mathbf{x} - \mathbf{y}|\,\kappa(\mathbf{y})\,d^2y$$

The properties of such fundamental solutions are developed in **[C\&K \S2.1]**.
Under the **compact support assumption** ($\kappa = 0$ outside bounded $\Omega$), this is
equivalent to the FEM-BEM formulation derived in Sections 3--6.

---

## 2. Why Naive Dirichlet Boundary Conditions Fail

### 2.1 The systematic error

A standard approach truncates to $\Omega = [-L, L]^2$ and imposes $\psi = 0$ on $\partial\Omega$. For
a Gaussian lens, the true $\psi$ decays only logarithmically and is nonzero at any
finite boundary. Forcing $\psi = 0$ introduces a systematic error $e = \psi_{\rm true} - \psi_{\rm FEM}$
satisfying:

$$\nabla^2 e = 0 \quad \text{in } \Omega, \qquad e\big|_{\partial\Omega} = \psi_{\rm true}\big|_{\partial\Omega} \neq 0$$

By the maximum principle, this error propagates throughout $\Omega$. The MAP
optimizer compensates by adding spurious mass near the boundary.

### 2.2 The violated transmission condition

In a naive Dirichlet formulation, boundary rows of $K$ are replaced by identity
rows (enforcing $\psi = 0$ on $\partial\Omega$). This does not respect the exterior harmonic
extension. Specifically, the flux $\partial\psi/\partial n$ on the interior side is generically
non-zero, while the exterior harmonic function with $\psi = 0$ on $\partial\Omega$ and $\psi \to 0$
at infinity would require $\psi \equiv 0$ in $\Omega_{\rm ext}$. The physical transmission condition:

$$\left[\frac{\partial\psi}{\partial n}\right]_{\partial\Omega} = 0$$

is therefore violated. FEMMI's FEM-BEM coupling enforces this condition exactly
by retaining the Neumann stiffness (no boundary row modification) and coupling
to the exterior via BEM.

---

## 3. Domain Decomposition and Transmission Conditions

### 3.1 Setup

Decompose the plane into:

- $\Omega$: bounded FEM region (contains all the mass, $\kappa = 0$ outside $\Omega$ by assumption)
- $\Omega_{\rm ext} = \mathbb{R}^2 \setminus \bar{\Omega}$: exterior, mass-free
- $\partial\Omega$: the interface boundary

The governing equations in each region:

$$\nabla^2\psi = 2\kappa \quad \text{in } \Omega, \qquad \nabla^2\psi = 0 \quad \text{in } \Omega_{\rm ext}, \qquad \psi \to 0 \text{ as } |\mathbf{x}| \to \infty$$

### 3.2 Transmission Conditions

Since there is no physical source on the boundary, $\psi$ must be $C^1$ across $\partial\Omega$.
The **transmission conditions** (see **[C\&K \S5.1]** for the scattering analogue):

$$[\psi]_{\partial\Omega} = 0 \qquad \text{(continuity of } \psi\text{)}$$

$$\left[\frac{\partial\psi}{\partial n}\right]_{\partial\Omega} = 0 \qquad \text{(continuity of normal flux)}$$

where $\mathbf{n}$ is the outward unit normal to $\Omega$.

---

## 4. FEM Interior: The Weak Form with Boundary Flux Terms

### 4.1 Weak form retaining the boundary term

Multiplying $\nabla^2\psi = 2\kappa$ by a test function $v \in H^1(\Omega)$ and integrating by parts
using Green's first identity:

$$\int_\Omega \nabla\psi \cdot \nabla vdA = -2\int_\Omega \kappa vdA + \oint_{\partial\Omega} v\frac{\partial\psi}{\partial n}ds$$

The boundary term $\oint_{\partial\Omega} v(\partial\psi/\partial n)ds$ is the **critical difference** from the
naive formulation. The Dirichlet approach forces $v = 0$ on $\partial\Omega$, making this
term vanish and discarding the flux information entirely. In the FEM-BEM
formulation, we retain this term and treat $t = \partial\psi/\partial n$ as an additional
unknown determined by the BEM.

### 4.2 P3 Galerkin Discretization

Expand $\psi$ and $\kappa$ in the P3 Lagrange basis $\{N_j\}$ and the boundary flux $t$ in a
boundary basis $\{M_k\}$:

$$K\boldsymbol{\psi} = -2M\boldsymbol{\kappa} + Bt$$

where $K_{ij} = \int \nabla N_i \cdot \nabla N_j\,dA$ (stiffness), $M_{ij} = \int N_i N_j\,dA$ (mass),
$B_{ik} = \oint N_i M_k\,ds$ (boundary coupling). The **Neumann stiffness matrix**
$K$ is assembled **without modifying boundary rows**. Its null space is
$\mathrm{span}\{\mathbf{1}\}$ (constant functions); the BEM coupling and gauge fix remove this.

Assembled in `operators.py`, function `_assemble_operators_from_mesh`.

---

## 5. BEM Exterior: The Boundary Integral Equation

### 5.1 Green's Representation Formula

In $\Omega_{\rm ext}$, $\psi$ is harmonic with $\psi \to 0$ at infinity. Applying Green's second
identity in $\Omega_{\rm ext}$ yields the **Somigliana identity** for $\mathbf{x} \in \Omega_{\rm ext}$:

$$\psi(\mathbf{x}) = \int_{\partial\Omega} G(\mathbf{x},\mathbf{y})t(\mathbf{y})ds(\mathbf{y}) - \int_{\partial\Omega} \psi(\mathbf{y})\frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})ds(\mathbf{y})$$

This is the direct analogue of **[C\&K \S2.1, Thm 2.5]**.

### 5.2 The Four BEM Operators

All four classical boundary operators map functions on $\partial\Omega$ to functions on
$\partial\Omega$. Their definitions and properties are developed in **[C\&K \S3.1--3.4]**:

$$\text{Single layer: } (Vt)(\mathbf{x}) = \int_{\partial\Omega} G(\mathbf{x},\mathbf{y})t(\mathbf{y})ds(\mathbf{y})$$

$$\text{Double layer: } (K\psi)(\mathbf{x}) = \mathrm{P.V.}\int_{\partial\Omega} \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\psi(\mathbf{y})ds(\mathbf{y})$$

Key properties: $V$ is symmetric; on the unit square (logarithmic capacity
$\approx 0.59 < 1$) $V$ is negative-definite, but remains invertible. $K$ is compact
(**[C\&K Thm 3.4]**). Implemented in `bem.py` functions `assemble_single_layer`
and `assemble_double_layer`.

### 5.3 The Boundary Integral Equation

Taking the limit of the Somigliana identity as $\mathbf{x} \to \partial\Omega$ and applying the jump
relations (**[C\&K \S3.1, Thm 3.1 and Thm 3.3]**):

$$\left(\tfrac{1}{2}I + K\right)\psi\big|_{\partial\Omega} = Vt\big|_{\partial\Omega}, \qquad \mathbf{x} \in \partial\Omega$$

Discretized with $N_b$ boundary nodes (P3 traces on boundary edges):

$$\left(\tfrac{1}{2}M_b + K_h\right)\psi_b = V_ht_b$$

where $M_b$ is the boundary Gram matrix assembled in `bem.py`,
`assemble_boundary_mass`. The solvability follows from the Fredholm alternative
applied to the compact perturbation, as in **[C\&K \S3.2, Thm 3.9]**.

Diagonal blocks of $V_h$ require logarithmic-singular integrals; FEMMI uses
Gauss-Jacobi quadrature with weight $w(t) = -\ln(t)$ via `log_gauss_jacobi_points`
in `bem.py` (25 points, relative error $< 10^{-12}$).

---

## 6. FEM-BEM Coupling: The Correct System

### 6.1 Assembling the Coupled System

Let $P$ be the restriction operator extracting boundary entries: $P\psi = \psi_b$.
The full coupled system for unknowns $(\psi, t)$ is:

$$\begin{pmatrix} K & -B \\\ \left(\tfrac{1}{2}M_b + K_h\right)P & -V_h \end{pmatrix} \begin{pmatrix} \psi \\ t \end{pmatrix} = \begin{pmatrix} -2M\kappa \\\ 0 \end{pmatrix}$$

### 6.2 Schur Complement Reduction

From the BEM equation: $t = V_h^{-1} (\tfrac{1}{2}M_b + K_h) P \psi$. Substituting yields:

$$A_{\rm coupled}\psi = -2M\kappa$$

where:

$$A_{\rm coupled} = K + P^\top V_h^{-1}\left(\tfrac{1}{2}M_b + K_h\right)P$$

Implemented in `operators.py`, function `_assemble_operators_from_mesh`.
The dense Calderon matrix $C = V_h^{-1}(\tfrac{1}{2}M_b + K_h)$ is stored in
`FEMOperators.C_dense` (shape $N_b \times N_b$); the boundary-block update
`K[bnd_idx, bnd_idx] += C_dense` produces $A_{\rm coupled}$ as a sparse CSR matrix.
A gauge fix pins one boundary DOF to remove the constant null space.

$A_{\rm coupled}$ is then factorised once by SuperLU and stored in
`FEMOperators.A_coupled_lu`. All forward and adjoint solves reuse this
factorisation.

### 6.3 Guarantees

Solving $A_{\rm coupled}\,\psi = -2M\kappa$ gives $\psi$ satisfying:
1. $\nabla^2\psi = 2\kappa$ in $\Omega$
2. $\nabla^2\psi = 0$ in $\Omega_{\rm ext}$ (encoded in the BEM Green's representation)
3. $\psi \to 0$ as $|\mathbf{x}| \to \infty$ (the logarithmic representation decays correctly)
4. $\psi$ and $\partial\psi/\partial n$ continuous across $\partial\Omega$

Uniqueness follows from **[C\&K \S3.3, Thm 3.12]**.

---

## 7. P3 Cubic Basis Functions

All element computations are performed on the **reference triangle**
$\hat{T} = \{(\xi,\eta) : \xi \geq 0, \eta \geq 0, \xi+\eta \leq 1\}$. Points on $\hat{T}$ are parameterised by
barycentric coordinates:

$$\lambda_1 = 1 - \xi - \eta, \quad \lambda_2 = \xi, \quad \lambda_3 = \eta$$

Implemented in `basis.py`, `compute_p3_shape_functions`.

### 7.1 The 10 degrees of freedom

The complete cubic polynomial space on a triangle has $\dim P_3 = 10$.
FEMMI uses the Lagrange nodal basis with DOF locations:

| Index | Type | Ref coords $(\xi,\eta)$ |
|-------|------|------------------|
| 0 | Vertex | $(0, 0)$ |
| 1 | Vertex | $(1, 0)$ |
| 2 | Vertex | $(0, 1)$ |
| 3 | Edge $0 \to 1$, $t=1/3$ | $(1/3, 0)$ |
| 4 | Edge $0 \to 1$, $t=2/3$ | $(2/3, 0)$ |
| 5 | Edge $1 \to 2$, $t=1/3$ | $(2/3, 1/3)$ |
| 6 | Edge $1 \to 2$, $t=2/3$ | $(1/3, 2/3)$ |
| 7 | Edge $2 \to 0$, $t=1/3$ | $(0, 2/3)$ |
| 8 | Edge $2 \to 0$, $t=2/3$ | $(0, 1/3)$ |
| 9 | Interior (centroid) | $(1/3, 1/3)$ |

### 7.2 Vertex, edge, and interior basis functions

Vertex functions: $N_i = \tfrac{1}{2}\lambda_i(3\lambda_i - 1)(3\lambda_i - 2)$ for $i = 0, 1, 2$.

Edge functions (edge $0 \to 1$): $N_3 = \tfrac{9}{2}\lambda_1\lambda_2(3\lambda_1 - 1)$,
$N_4 = \tfrac{9}{2}\lambda_1\lambda_2(3\lambda_2 - 1)$.
Remaining edges follow by cyclic permutation of $\lambda_1, \lambda_2, \lambda_3$.

Interior bubble: $N_9 = 27\lambda_1\lambda_2\lambda_3$.

The basis satisfies $\sum_i N_i = 1$ (partition of unity) and $N_i(\mathbf{x}_j) = \delta_{ij}$
(Kronecker delta). Validated in `tests/test_convergence_p3.py`.

---

## 8. Element Matrix Assembly

### 8.1 Affine map and Jacobian

FEMMI uses a **subparametric** formulation: the geometry is mapped by only the
3 vertex nodes (affine/linear map). For an element with vertices
$(x_0,y_0)$, $(x_1,y_1)$, $(x_2,y_2)$:

$$\mathbf{x}(\xi,\eta) = \mathbf{x}_0 + J\begin{pmatrix}\xi\\\ \eta\end{pmatrix}, \qquad J = \begin{pmatrix}x_1-x_0 & y_1-y_0\\\ x_2-x_0 & y_2-y_0\end{pmatrix}$$

Because the map is affine, $J$ is **constant over each element**.

### 8.2 Stiffness matrix $K$

The element stiffness matrix:

$$K^e_{ij} = \int_T \nabla N_i \cdot \nabla N_j\,dA = |T|\sum_q w_q(\nabla_{\mathbf{x}}N_i)_q \cdot (\nabla_{\mathbf{x}}N_j)_q$$

Gradient transformation: $\nabla_x N = J^{-T} \nabla_\xi N$. $K$ is assembled **without
modifying boundary rows** (Neumann stiffness). The previous Dirichlet BC
approach (zeroing boundary rows and setting the diagonal to 1) is not
applied; that null space is removed by the BEM coupling and gauge fix.

Assembled in `operators.py`, `_assemble_operators_from_mesh`.

### 8.3 Mass matrix $M$ and Dunavant quadrature

The element mass matrix:

$$M^e_{ij} = \int_T N_i N_j dA = |T|\sum_q w_q N_i(\xi_q) N_j(\xi_q)$$

The load integrand $N_i N_j$ has degree 6 (cubic $\times$ cubic), requiring a
degree-6-exact quadrature rule, hence the **13-point Dunavant degree-7 rule**
in `assembly.py`, `get_gauss_quadrature_triangle(order=5)`.

---

## 9. Shear Operators $S_1$ and $S_2$

### 9.1 Reference Hessians via JAX autodiff

The reference Hessians are precomputed using JAX forward-over-reverse autodiff:

$$H^{\rm ref}_{p,j,k,\ell} = \left.\frac{\partial^2 N_j}{\partial\xi_k\partial\xi_\ell}\right|_{\boldsymbol{\xi}=\boldsymbol{\xi}^{\rm ref}_p}$$

Array shape: (10 evaluation points, 10 shape functions, 2, 2).

Implemented in `operators.py`, `_build_ref_hessians`.

### 9.2 Physical Hessian transformation

For an **affine map** ($J$ constant), the second derivatives transform via:

$$H^{\rm phys}_{j,a,b} = \sum_{k,\ell} A_{ka}A_{\ell b}H^{\rm ref}_{j,k\ell}, \qquad A = J^{-T}$$

In einsum notation: `'ja,kb,njk->nab'`.

### 9.3 The einsum index order

An earlier version used `'aj,bk,njk->nab'`, transposing $A$ in both slots.
For lower-triangle elements where $J$ is diagonal, $A = A^\top$ so the bug was
hidden. For upper-triangle elements, $A \neq A^\top$, producing wrong Hessians in
exactly half the mesh. The correct index order `'ja,kb,njk->nab'` is
implemented in `operators.py`, `_assemble_shear_ops`.

### 9.4 Nodal averaging

Each node contributes to multiple elements. Raw Hessian contributions are
scatter-accumulated and divided by the element count:

```python
sc = sp.diags(1.0 / np.maximum(counts, 1))
return (sc @ S1r).tocsr(), (sc @ S2r).tocsr()
```

This is $O(h^2)$ accurate at interior nodes. Boundary nodes have fewer
contributing elements; their shear values are zeroed:

```python
S1_lil[boundary, :] = 0;  S2_lil[boundary, :] = 0
```

Both implemented in `operators.py`, `_assemble_shear_ops` and
`_assemble_operators_from_mesh`.

---

## 10. The Complete Forward Operator $F$

### 10.1 The Linear Chain

The complete map from $\kappa$ to $(\gamma_1, \gamma_2)$ is:

$$\kappa \xrightarrow{-2M} \mathbf{f} \xrightarrow{A_{\rm coupled}^{-1}} \psi \xrightarrow{S} (\gamma_1, \gamma_2)$$

Writing this as a single operator: $F = S \cdot A_{\rm coupled}^{-1} \cdot (-2M)$, where
$S = (S_1; S_2)$ stacks the two shear operators.

In `operators.py`: `FEMOperators.psi_from_kappa` solves $A_{\rm coupled}\,\psi = -2M\kappa$
(with gauge fix); `FEMOperators.shear_from_psi` applies $S_1$ and $S_2$.
`FEMOperators.forward` chains both. The JAX-differentiable wrapper lives
in `forward.py`, `DifferentiableForward`.

### 10.2 Compactness

$F$ is a compact operator from $L^2(\Omega)$ to $L^2(\Omega)^2$:
- $-2M$ maps $L^2 \to H^1$ (integration gains smoothness)
- $A_{\rm coupled}^{-1}$ maps $H^{-1} \to H^1$ (elliptic solve gains two derivatives)
- $S$ maps $H^1 \to L^2$ (Hessian)
- The embedding $H^1 \hookrightarrow L^2$ is compact by Rellich's theorem

Compactness is the mathematical reason the inverse problem is ill-posed
(**[C\&K \S10.1]**).

### 10.3 Injectivity and null space

The FEM-BEM system has trivial null space. The boundary condition $\psi \to 0$
at infinity (encoded by the BEM) fixes the absolute normalization of $\psi$,
resolving the **mass sheet degeneracy**. Adding a uniform sheet $\kappa \to \kappa + c$
changes $\mathbf{f} \to \mathbf{f} - 2Mc$, which changes $\psi$, which changes $\gamma$. The map $F$
is injective in contrast to Kaiser-Squires, where the Fourier kernel
vanishes at $\mathbf{k} = \mathbf{0}$.

---

## 11. MAP Reconstruction and Tikhonov Regularization

### 11.1 The Tikhonov functional

Tikhonov regularization replaces the ill-posed problem $F\kappa = \gamma_{\rm obs}$ with:

$$\kappa_\lambda = \underset{\kappa}{\text{argmin}} \left\{ \|F\kappa - \gamma_{\rm obs}\|^2 + \lambda\kappa^\top R\kappa \right\}$$

This is exactly the **MAP estimator** with Gaussian likelihood and Gaussian
prior. Existence, uniqueness, and convergence are established in
**[C\&K \S10.2, Thm 10.2]**. Implemented in `inverse.py`, `MAPReconstructor`.

### 11.2 Choosing the regularization operator $R$

- **$H^1$ ($R = K$):** Penalizes $\|\nabla\kappa\|^2$. Smoothness prior.
- **Matern-Wiener ($R = M + \ell^2 K$):** Penalizes $\|\kappa\|^2 + \ell^2\|\nabla\kappa\|^2$. **Recommended.**

The **Matern-Wiener prior** $R = M + \ell^2 K$ has Green's function
$G(r) \approx e^{-r/\ell}$, a Matern-$\frac{1}{2}$ covariance with correlation length $\ell$.
Setting $\ell = \sigma_{\rm lens}$ matches the prior to the expected spatial scale of $\kappa$.

Assembled in `operators.py`, `build_wiener_regularizer`. Selected by
`wiener_length` parameter in `MAPReconstructor`.

### 11.3 Filtered SVD interpretation

For $R = I$, the Tikhonov filter is $\phi_\lambda(\sigma) = \sigma/(\sigma^2 + \lambda)$:
$\approx 1/\sigma$ for $\sigma \gg \sqrt{\lambda}$
(large modes recovered accurately) and $\approx \sigma/\lambda$ for $\sigma \ll \sqrt{\lambda}$ (small modes
suppressed). This filter interpretation is discussed in **[C\&K \S10.2]**.

---

## 12. The Adjoint Gradient

### 12.1 The adjoint of $F$

Recall $F = S \cdot A_{\rm coupled}^{-1} \cdot (-2M)$. Using the symmetry of $M$ and $A_{\rm coupled}$,
the $L^2$ adjoint is:

$$F^* = (-2M)A_{\rm coupled}^{-T}S^\top$$

### 12.2 The gradient of the MAP loss

Define residuals $r_a = S_a\psi - \gamma_{a,\rm obs}$. The gradient of
$\mathcal{L}(\kappa) = \|F\kappa - \gamma_{\rm obs}\|^2 + \lambda\kappa^\top R\kappa$ is:

$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{\kappa}} = -4MA_{\rm coupled}^{-T}(S_1^\top\mathbf{r}_1 + S_2^\top\mathbf{r}_2) + 2\lambda R\kappa$$

The term $A_{\rm coupled}^{-T}(S_1^\top r_1 + S_2^\top r_2)$ is the **adjoint solve** using
`trans='T'` in the SuperLU factorisation.

Per-iteration algorithm in `inverse.py`, `MAPReconstructor._make_obj_and_grad`:

1. Forward: $\mathbf{f} = -2M\kappa$ (gauge fix applied), solve $A_{\rm coupled}\psi = \mathbf{f}$, compute $\gamma_a = S_a\psi$
2. Residuals: $r_a = \gamma_a - \gamma_{a,\rm obs}$
3. Loss: $\mathcal{L} = \sum_a\|r_a\|^2 + \lambda\kappa^\top R\kappa$
4. Adjoint RHS: $\mathbf{q} = S_1^\top r_1 + S_2^\top r_2$ (gauge fix applied)
5. Adjoint solve: $A_{\rm coupled}^{-T}\phi = \mathbf{q}$ via `A_lu.solve(..., trans='T')`
6. Gradient: $\partial\mathcal{L}/\partial\kappa = -4M\phi + 2\lambda R\kappa$

Total cost per iteration: **two $A_{\rm coupled}$ solves** (forward + adjoint),
reusing the factored SuperLU object.

---

## 13. Regularization Parameter Selection: Morozov's Principle

### 13.1 The Discrepancy Principle

Let $\delta$ denote the noise level. The **Morozov discrepancy principle** selects $\lambda$
such that the reconstruction residual matches the noise level:

$$\|F\kappa_\lambda - \gamma_{\rm obs}\|_{\rm RMS} = c\delta \qquad (c \approx 1)$$

**Theorem (Morozov, 1966; [C\&K \S10.2, Thm 10.4]).** Let $\gamma_{\rm obs} = F\kappa_{\rm true} + \eta$
with $\|\eta\| \leq \delta$. If $\lambda_M$ solves the above, then $\|\kappa_{\lambda_M} - \kappa_{\rm true}\| \to 0$ as $\delta \to 0$.

### 13.2 Implementation

The functional $D(\lambda) = \|F\kappa_\lambda - \gamma_{\rm obs}\|_{\rm RMS} - c\delta$ is monotone decreasing in $\lambda$.
Root-finding uses Brent's method in `regularization.py`, `MorozovSelector.select`.

The discrepancy uses an RMS norm:

$$D(\lambda) = \sqrt{\frac{\|r_1\|^2 + \|r_2\|^2}{n_{\rm data}}} - c\delta, \qquad n_{\rm data} = |\gamma_1| + |\gamma_2|$$

Noise level $\delta$ is estimated from the observed shear using the MAD estimator
in `regularization.py`, `estimate_noise_level`:

$$\delta = 1.4826 \cdot \mathrm{median}\left(|\gamma - \mathrm{median}(\gamma)|\right)$$

`MorozovSelector` also provides `lcurve` for diagnostic plotting.

---

## 14. The Inverse Scattering Connection

### 14.1 Structural Equivalence with the Born Approximation

The weak lensing forward problem is structurally identical to the Born
approximation in acoustic inverse scattering (**[C\&K \S8.1]**):

| Acoustic scattering | Weak lensing |
|---|---|
| Scattered field $u_s$ | Shear $\gamma$ |
| Refractive contrast $n(\mathbf{x})$ | Convergence $\kappa(\mathbf{x})$ |
| Incident field $u_{\rm inc}$ | Uniform (constant) |
| Helmholtz Green's function | Lensing kernel $K(\mathbf{x},\mathbf{y})$ |
| Wavenumber $k > 0$ | $k \to 0$ (Poisson limit) |

The $k \to 0$ limit places the lensing problem in the static scattering regime.

### 14.2 Consequences of Compactness

Since $F$ is compact (**[C\&K \S10.1]**):

1. **Resolution limit.** $\sigma_i \to 0$ imposes a fundamental minimum resolvable feature size.
2. **Range condition.** $F\kappa = \gamma_{\rm obs}$ has a solution only if $\gamma_{\rm obs}$ satisfies the Picard condition (Section 15.3).
3. **Regularization is necessary.** No bounded linear inversion can recover $\kappa$ stably for all right-hand sides.

---

## 15. SVD, Ill-Posedness, and the Picard Condition

### 15.1 The SVD of $F$

Since $F$ is compact, it admits the singular value decomposition:

$$F = \sum_i \sigma_i \mathbf{u}_i \otimes \mathbf{v}_i^*, \qquad \sigma_1 \geq \sigma_2 \geq \cdots \to 0$$

The singular values accumulate only at zero (**[C\&K \S10.1, Thm 10.6]**).
Computed in `svd_analysis.py`, `compute_svd` using randomised Lanczos on
the normal operator $F^*F$.

### 15.2 Noise amplification

With $\gamma_{\rm obs} = \gamma_{\rm true} + \eta$ (noise), the formal inversion
$\sum_i \sigma_i^{-1}\langle\eta, \mathbf{u}_i\rangle\mathbf{v}_i$
diverges since $\sigma_i^{-1} \to \infty$.

### 15.3 The Picard Condition

The equation $F\kappa = \gamma_{\rm true}$ has a solution $\kappa \in L^2(\Omega)$ if and only if:

$$\sum_i \left(\frac{|\langle \gamma_{\rm true}, \mathbf{u}_i\rangle|}{\sigma_i}\right)^2 < \infty$$

(**[C\&K \S10.1, Thm 10.7]**.) For smooth $\kappa$ this holds; for noisy data the
coefficients plateau at the noise floor while $\sigma_i$ continues to decay.

### 15.4 The Picard Plot

Plot $\log\sigma_i$, $\log|\langle\gamma_{\rm obs}, \mathbf{u}_i\rangle|$, and
$\log(|\langle\gamma_{\rm obs}, \mathbf{u}_i\rangle|/\sigma_i)$ versus mode index $i$.
If $\log|\langle\gamma_{\rm obs}, \mathbf{u}_i\rangle|$ decays faster than $\log\sigma_i$, the Picard
condition is satisfied. The crossover gives the effective noise cutoff.

Implemented in `svd_analysis.py`, `picard_plot`.

---

## 16. The Factorization Method for Support Recovery

### 16.1 Motivation

For applications where the goal is to determine only the **support** of $\kappa$,
the factorization method provides a parameter-free alternative.

### 16.2 Range Characterization Theorem

Define the point-source test function at $\mathbf{z} \in \Omega$:

$$\boldsymbol{\Phi}_{\mathbf{z}} = F\delta_{\mathbf{z}} \qquad \text{(shear pattern from a unit point mass at } \mathbf{z}\text{)}$$

**Theorem (Kirsch, 1998; [C\&K \S6.2, Thm 6.15]).**

$$\mathbf{z} \in \mathrm{supp}(\kappa) \iff \boldsymbol{\Phi}_{\mathbf{z}} \in \mathrm{Range}\left(|F|^{1/2}\right)$$

### 16.3 Numerical Implementation

After computing the truncated SVD (modes with $\sigma_i > \delta$), for each test point $\mathbf{z}$:

$$W(\mathbf{z}) = \left(\sum_{\sigma_i > \delta} \frac{|\langle \boldsymbol{\Phi}_{\mathbf{z}}, \mathbf{u}_i\rangle|^2}{\sigma_i}\right)^{-1}$$

$W(\mathbf{z})$ is large where $\mathbf{z} \in \mathrm{supp}(\kappa)$ and small outside.

Probe function computed in `svd_analysis.py`, `_probe_function`.
Indicator evaluated in `FactorizationIndicator.indicator_map`.
The probe function approximates $\Phi_\mathbf{z}$ by concentrating a unit mass at the
nearest mesh node, weighted by the diagonal mass matrix entry $M_{jj}$.

---

## 17. The Linear Sampling Method

### 17.1 The Linear Sampling Equation

For each test point $\mathbf{z}$, seek a density $g_\mathbf{z}$ satisfying $F g_\mathbf{z} = \Phi_\mathbf{z}$.
(**[C\&K \S5.5]**): if $\mathbf{z} \in \mathrm{supp}(\kappa)$, then $\Phi_\mathbf{z} \in \mathrm{Range}(F)$ and the equation has
a bounded solution; if $\mathbf{z} \notin \mathrm{supp}(\kappa)$, then $\|g_\mathbf{z}\| \to \infty$.

### 17.2 The Indicator Functional

Solve via Tikhonov regularization in SVD form:

$$\|g_{\mathbf{z}}^\alpha\|^2 = \sum_i \left(\frac{\sigma_i}{\sigma_i^2 + \alpha}\right)^2 |\langle \boldsymbol{\Phi}_{\mathbf{z}}, \mathbf{u}_i\rangle|^2$$

The support indicator is $\mathcal{I}(\mathbf{z}) = 1/\|g_\mathbf{z}^\alpha\|$, large inside $\mathrm{supp}(\kappa)$.

Implemented in `svd_analysis.py`, `LinearSamplingIndicator.indicator_map`.

---

## 18. Convergence Theory

### 18.1 Cea's Lemma

For the Galerkin approximation $\psi^h$ in $H^1(\Omega)$:

$$\|\psi - \psi^h\|_{H^1} \leq \frac{M}{\alpha}\inf_{v^h \in V^h}\|\psi - v^h\|_{H^1}$$

The bound reduces to best approximation ($M = \alpha = 1$ for the Laplacian).

### 18.2 Approximation theory and shear convergence

For $P_k$ elements and $\psi \in H^{k+1}(\Omega)$:

| Norm | P1 ($k=1$) | P2 ($k=2$) | P3 ($k=3$) |
|------|----------|----------|----------|
| $H^1$ semi-norm | $O(h)$ | $O(h^2)$ | $O(h^3)$ |
| $L^2$ norm | $O(h^2)$ | $O(h^3)$ | $O(h^4)$ |

Shear involves $\nabla^2\psi$; the error in the piecewise Hessian is $O(h^{k-1})$:

| Element | Shear convergence | Why |
|---------|------------------|-----|
| P1 | $\equiv 0$ | $\partial^2/\partial x^2$ of piecewise linear is zero |
| P2 | $O(h^0)$ | Piecewise constant second derivatives |
| P3 | $O(h^2)$ | Piecewise linear second derivatives |

$O(h^4)$ $L^2$ convergence validated in `tests/test_convergence_p3.py`.
$O(h^2)$ shear convergence (deep interior) validated in `tests/test_convergence.py`.

### 18.3 $\psi$ convergence and BEM corner singularities

The $\psi$ (lensing potential) convergence rate is limited by the BEM treatment
of the square domain corners. The logarithmic capacity of the unit square is
approximately 0.59 (less than 1), making $V_h$ negative-definite on this domain ---
mathematically correct and handled correctly by the current implementation.
Square corners introduce reentrant singularities in the exterior solution with
exponent $\pi/(2\pi - \pi/2) = 2/3$, capping the effective $\psi$ convergence at $O(h^{5/3})$
regardless of P3 interior accuracy. Since $\psi$ is never directly observed (only
the shear $\gamma = \partial^2\psi$ is), this is acceptable.

### 18.4 The 64-bit requirement

The condition number of $A_{\rm coupled}$ satisfies $\kappa(A_{\rm coupled}) = O(h^{-2})$. For a
$20 \times 20$ mesh, $\kappa \approx O(1600)$. In 32-bit arithmetic ($\varepsilon_{32} \approx 6 \times 10^{-8}$), solve
errors are $O(\kappa\,\varepsilon_{32}) \approx 2 \times 10^{-5}$, dominating the discretization error $h^4 \approx
6 \times 10^{-6}$ for P3 elements. All FEMMI modules enforce 64-bit via
`jax.config.update("jax_enable_x64", True)` at import in `femmi/__init__.py`.

---

## References

1. Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer.
2. Steinbach, O. (2008). *Numerical Approximation Methods for Elliptic Boundary Value Problems*. Springer.
3. Sauter, S. & Schwab, C. (2011). *Boundary Element Methods*. Springer.
4. Kirsch, A. (1998). Characterization of the shape of a scattering obstacle using the spectral data of the far-field operator. *Inverse Problems*, 14, 1489--1512.
5. Colton, D. & Kirsch, A. (1996). A simple method for solving inverse scattering problems in the resonance region. *Inverse Problems*, 12, 383--393.
6. Tikhonov, A. N. & Arsenin, V. Y. (1977). *Solutions of Ill-Posed Problems*. V. H. Winston & Sons.
7. Morozov, V. A. (1966). On the solution of functional equations by the method of regularization. *Soviet Math. Doklady*, 7, 414--417.
8. Kaiser, N. & Squires, G. (1993). Mapping the dark matter with weak gravitational lensing. *ApJ*, 404, 441--450.
9. Brenner, S. & Scott, R. (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer.
10. Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *IJNME*, 21(6), 1129--1148.
11. Stroud, A. H. (1971). *Approximate Calculation of Multiple Integrals*. Prentice-Hall.
