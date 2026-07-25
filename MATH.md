# FEMMI --- Mathematical Derivation

This document gives a self-contained, rigorous derivation of every mathematical
operation performed by FEMMI, with references to the functions that implement
each formula.

**Citation convention.** Throughout this document, **[C\&K \S X.Y]** and
**[C\&K Thm X.Y]** refer to:

> Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic
> Scattering Theory*, 3rd ed. Springer.


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
derivatives, giving no convergence with refinement. P3 (cubic) elements have
piecewise-linear second derivatives, giving $O(h^2)$ convergence for $\gamma$.

### 1.3 The Green's Function and Exact Solution

The 2D Laplacian fundamental solution satisfying $\nabla^2_y G(x,y) = \delta(x-y)$ is:

$$G(\mathbf{x}, \mathbf{y}) = \frac{1}{2\pi} \ln|\mathbf{x} - \mathbf{y}|$$

The exact solution on $\mathbb{R}^2$ satisfying $\psi \to 0$ at infinity is the volume potential:

$$\psi(\mathbf{x}) = \frac{1}{\pi}\int_{\mathbb{R}^2} \ln|\mathbf{x} - \mathbf{y}|\kappa(\mathbf{y})d^2y$$

The properties of such fundamental solutions are developed in **[C\&K \S2.1]**.
Under the **compact support assumption** ($\kappa = 0$ outside bounded $\Omega$), this is
equivalent to the FEM-BEM formulation derived in Sections 3--6.


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

where $K_{ij} = \int \nabla N_i \cdot \nabla N_jdA$ (stiffness), $M_{ij} = \int N_i N_jdA$ (mass),
$B_{ik} = \oint N_i M_kds$ (boundary coupling). The **Neumann stiffness matrix**
$K$ is assembled **without modifying boundary rows**. Its null space is
$\mathrm{span}\{\mathbf{1}\}$ (constant functions); the BEM coupling and gauge fix remove this.

Assembled in `operators.py`, function `_assemble_operators_from_mesh`.


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


## 6. FEM-BEM Coupling: The Correct System

### 6.1 Assembling the Coupled System

Let $P$ be the restriction operator extracting boundary entries: $P\psi = \psi_b$,
and let $t = \partial\psi/\partial n$ be the boundary flux. The FEM weak form
contributes a boundary term $\oint_{\partial\Omega} v\,t\,ds = (Pv)^\top M_b\,t$, so
the flux enters the interior equation **tested against the trace basis through the
boundary Gram matrix $M_b$** (the Galerkin pairing). The exterior harmonic
extension supplies the discrete Dirichlet-to-Neumann relation
$V_\sigma\,t = (\tfrac12 M_b - K_h)P\psi$, with the $\sigma$-scaled single layer
$V_\sigma$ defined in \S6.5. The full coupled system for unknowns $(\psi, t)$ is:

$$\begin{pmatrix} K & -P^\top M_b \\\ \left(\tfrac{1}{2}M_b - K_h\right)P & -V_\sigma \end{pmatrix} \begin{pmatrix} \psi \\ t \end{pmatrix} = \begin{pmatrix} -2M\kappa \\\ 0 \end{pmatrix}$$

### 6.2 Schur Complement Reduction

From the BEM equation: $t = V_\sigma^{-1} (\tfrac{1}{2}M_b - K_h) P \psi$.
Substituting yields:

$$A_{\rm coupled}\psi = -2M\kappa$$

where:

$$A_{\rm coupled} = K - P^\top M_b\, V_\sigma^{-1}\left(\tfrac{1}{2}M_b - K_h\right)P$$

Implemented in `operators.py`, function `_assemble_operators_from_mesh`
(`coupling='steinbach'`, the default). The dense coupling matrix
$C = -M_b\,V_\sigma^{-1}(\tfrac{1}{2}M_b - K_h)$ is stored in
`FEMOperators.C_dense` (shape $N_b \times N_b$); the boundary-block update
`K[bnd_idx, bnd_idx] += C_dense` produces $A_{\rm coupled}$ as a sparse CSR matrix.
The symmetric Steinbach coupling and its $\sigma$-scaling are derived in \S6.5.

By the Calderon identity, $({\tfrac{1}{2}M_b - K_h})\mathbf{1} \approx 0$, so
$C_{\rm dense}\mathbf{1} \approx 0$ and $A_{\rm coupled}$ retains
$\mathrm{span}\{\mathbf{1}\}$ as its null space. One scalar gauge condition is
therefore required.

### 6.3 Gauge Choice: Single-Node Pin

The null space of $A_{\rm coupled}$ is exactly $\mathrm{span}\{\mathbf{1}\}$: adding any
constant to $\psi$ leaves all shear components unchanged (second derivatives
annihilate constants). This one-dimensional degeneracy is the discrete counterpart
of the **mass-sheet degeneracy** — the physical freedom to add a uniform
$\kappa_0$ to the convergence field.

One might hope that the B-mode constraint $\gamma_B = 0$ could fix this
degeneracy. It cannot. For any $\psi$, the B-mode curl condition
$\partial_x\gamma_2 - \partial_y\gamma_1 = 0$ reduces to an identity via commutativity
of mixed partial derivatives. Adding a constant to $\psi$ changes neither
$\gamma_1$ nor $\gamma_2$, so the B-mode condition carries no information about
the additive constant.

A mean gauge condition $\mathbf{1}^\top\boldsymbol{\psi} = 0$ via a bordered system 

$$\begin{pmatrix}A & \mathbf{1}\\\mathbf{1}^\top & 0\end{pmatrix}$$ 

is the mathematically cleanest fix, but fails in practice because $A_{\rm coupled}$ is
**not symmetric** ($K_h$ is asymmetric). The Fredholm consistency condition
requires the RHS to be orthogonal to the left null vector of $A_{\rm coupled}$,
which is not $\mathbf{1}$ when $A_{\rm coupled} \neq A_{\rm coupled}^\top$. The
Lagrange multiplier $\mu$ then modifies the PDE rather than purely fixing the
gauge, producing a 3.5% interior Poisson residual in tests.

FEMMI instead uses a **single-node pin**: one boundary node $j^{\ast}$ has its
row replaced by an identity row, forcing $\psi_{j^{\ast}} = 0$. The factored system
is stored in `FEMOperators.A_coupled_lu`. To prevent the artificially pinned
value from contaminating shear at adjacent interior nodes (via the columns of
$S_1$, $S_2$), those columns are zeroed after assembly:

```python
S1_lil[:, idx_gauge] = 0;  S2_lil[:, idx_gauge] = 0
```

The gauge node is placed at angle $3\pi/4$ on $\partial\Omega$ (upper-left diagonal),
where $\gamma_1 = 0$ for any centred radially-symmetric lens, minimising the
visible artifact in shear plots.

### 6.3a How far the mass-sheet claim actually goes

The BEM far-field condition makes the uniform sheet **formally observable** in a
way it is not for Kaiser–Squires. This is an operator-level fact and it holds:

$$\|F\mathbf{1}\|/\sqrt{N} > 0 \quad\text{(FEMMI)}, \qquad \|F_{\rm KS}\mathbf{1}\| = 0 \quad\text{exactly},$$

since the KS kernel carries an explicit $1/k^2$ that is singular at $k=0$ and is
set to zero there — the DC mode is in its null space by construction
(`experiments.constant_mode_response`, `ks_constant_mode_response`,
`examples/paper/injectivity.py`).

**It does not follow that FEMMI recovers the absolute normalisation in practice,
and measurement says it does not.** Two findings scope the claim:

1. *The response is a corner effect.* Decomposing $\|F\mathbf{1}\|^2$ by radius on
   a square domain, **99.5–99.9%** of it is carried by the outer collar, and it
   concentrates in the corners. It also **grows** under refinement
   ($\|F\mathbf{1}\|/\sqrt N = 1.66 \to 3.04$ from $n_x=12$ to $24$) rather than
   converging — the signature of the reentrant corner singularity of §18.5, not
   of a resolved physical signal. In the deep interior ($r<1.5$) the uniform
   sheet produces essentially no shear, exactly as the infinite-sheet symmetry
   argument requires.

2. *On neutral truth the advantage disappears.* Against an analytic GalSim NFW
   field (`femmi.truth`, generated by neither method's forward), FEMMI's mean-$\kappa$
   error is $0.047$ versus KS's $0.049$ — it recovers a few percent of the true
   mean. The much-quoted $0.002$ vs $0.085$ comes from generating the test shear
   with FEMMI's *own* forward, which is an inverse crime: there the DC component
   is exactly in the range of $F$ by construction.

The honest statement is therefore: **FEMMI removes the DC mode from the forward
operator's null space; it does not thereby break the mass-sheet degeneracy in
practice.** The multiplicative degeneracy $\kappa \to \lambda\kappa + (1-\lambda)$
is untouched by any pure shear inverter, FEMMI included.

What *does* survive on neutral truth is the boundary claim: FEMMI's exact
far-field condition gives a smaller DC-removed error than KS at every radius, by
$\approx 1.7\times$ in the corner region and growing outward
(`examples/paper/independent_truth.py`).

### 6.4 Guarantees

Solving the gauged system gives $\psi$ satisfying:
1. $\nabla^2\psi = 2\kappa$ in $\Omega$ (interior residual $< 10^{-13}$ of RHS)
2. $\nabla^2\psi = 0$ in $\Omega_{\rm ext}$ (encoded in the BEM Green's representation)
3. $\psi \to 0$ as $|\mathbf{x}| \to \infty$ (the logarithmic representation decays correctly)
4. $\psi$ and $\partial\psi/\partial n$ continuous across $\partial\Omega$
5. $\psi_{j^*} = 0$ (single-node pin at the gauge node)

Uniqueness follows from **[C\&K \S3.3, Thm 3.12]**.

### 6.5 Derivation of the Coupling: Galerkin Pairing and $\sigma$-Scaling

The coupling matrix of \S6.2 is fixed by two requirements, both isolated by direct
spectral testing of the discrete exterior Dirichlet-to-Neumann (DtN) map on a disk.
(A naive *nodal* coupling $V_h^{-1}(\tfrac12 M_b + K_h)$ — used in early versions of
this code — violates both and is neither scale- nor translation-invariant; it has
been removed.)

1. **Galerkin pairing.** The FEM boundary term is $\oint_{\partial\Omega} v\,t\,ds
   = P^\top M_b\,t$, so the flux must be tested against the trace basis through the
   boundary Gram matrix $M_b$. Dropping this outer $M_b$ gives a *nodal* DtN whose
   dimensions do not match $K$: with $V_h\sim L^2$, $M_b,K_h\sim L$, the nodal
   coupling scales as $1/L$ while $K$ is scale-free, so the far-field condition is
   progressively lost as the domain grows and the forward shear error grows with the
   absolute coordinate scale (measured for the nodal form: $\|C\|\propto 1/L$; error
   $0.11\to1.7$ over $\times100$). Retaining the outer $M_b$ — as in \S6.2 — makes
   the coupling scale-free.

2. **$n=0$ log-capacity mode.** In 2D the single layer is elliptic only when the
   logarithmic capacity $\mathrm{cap}(\partial\Omega)<1$. Per-mode testing shows the
   discrete DtN eigenvalues are **exact for every $n\ge1$** ($\lambda_n=-n/R$ to 4–5
   digits), while the $n=0$ (constant / mass-sheet) mode is singular at
   $\mathrm{cap}=R=1$ and uncontrolled otherwise. Only this mode breaks dilation
   invariance, since $V_1 = -R\ln R$ carries a non-homogeneous $\ln R$.

The symmetric Steinbach coupling addresses both, using the physically correct
exterior sign $\tfrac12 M_b - K_h$:

$$A_{\rm coupled} = K \;-\; P^\top\, M_b\, V_\sigma^{-1}\!\left(\tfrac12 M_b - K_h\right)P,
\qquad V_\sigma = V_h - \frac{\ln\sigma}{2\pi}\,\mathbf{w}\mathbf{w}^\top,\quad
\mathbf{w} = M_b\mathbf 1,\quad \sigma = \mathrm{diam}(\partial\Omega).$$

The rank-one update is the Galerkin realization of the **$\sigma$-scaled fundamental
solution** $-\tfrac1{2\pi}\ln(|x-y|/\sigma)$ (**[Steinbach 2008, \S6.6]**); with
$\sigma=\mathrm{diam}$ it non-dimensionalizes the kernel, so it shifts **only** the
$n=0$ eigenvalue (every $n\ge1$ frozen to machine precision — the primary regression
test) and, being $\propto\mathrm{diam}$, restores scale- and translation-invariance.
No hypersingular operator is required: the $n\ge1$ spectrum is already exact.

On the analytic Gaussian lens the coupling is scale-invariant (forward error $0.033$
at every scale) and **more accurate than a Dirichlet truncation** when the boundary
approaches the mass ($3.5\times$ lower error at $\kappa(\partial\Omega)\approx0.14$),
converging to Dirichlet as the boundary recedes — the quantitative statement of the
far-field claim in \S2. Regression tests in `tests/test_steinbach_coupling.py`; the
scale/DtN diagnostics are reproduced by `examples/diagnostics/bem_scaling_diagnostic.py` and
`examples/diagnostics/bem_dtn_diagnostic.py`.


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

The basis satisfies $\sum_i N_i = 1$ (partition of unity) and 

$$N_i(\mathbf{x}_{j}) = \delta_{ij}$$ 

(Kronecker delta). Validated in `tests/test_convergence_p3.py`.


## 8. Element Matrix Assembly

### 8.1 Affine map and Jacobian

FEMMI uses a **subparametric** formulation: the geometry is mapped by only the
3 vertex nodes (affine/linear map). For an element with vertices
$(x_0,y_0)$, $(x_1,y_1)$, $(x_2,y_2)$:

$$\mathbf{x}(\xi,\eta) = \mathbf{x}_0 + J\begin{pmatrix}\xi\\\ \eta\end{pmatrix}, \qquad J = \begin{pmatrix}x_1-x_0 & y_1-y_0\\\ x_2-x_0 & y_2-y_0\end{pmatrix}$$

Because the map is affine, $J$ is **constant over each element**.

### 8.2 Stiffness matrix $K$

The element stiffness matrix:

$$K^e_{ij} = \int_T \nabla N_i \cdot \nabla N_jdA = |T|\sum_q w_q(\nabla_{\mathbf{x}}N_i)_q \cdot (\nabla_{\mathbf{x}}N_j)_q$$

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


## 10. The Complete Forward Operator $F$

### 10.1 The Linear Chain

The complete map from $\kappa$ to $(\gamma_1, \gamma_2)$ is:

$$\kappa \xrightarrow{-2M} \mathbf{f} \xrightarrow{A_{\rm coupled}^{-1}} \psi \xrightarrow{S} (\gamma_1, \gamma_2)$$

Writing this as a single operator: $F = S \cdot A_{\rm coupled}^{-1} \cdot (-2M)$, where
$S = (S_1; S_2)$ stacks the two shear operators.

In `operators.py`: `FEMOperators.psi_from_kappa` solves the gauged system;
`FEMOperators.shear_from_psi` applies $S_1$ and $S_2$.
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

The FEM-BEM system with the single-node gauge has trivial null space. The
boundary condition $\psi \to 0$ at infinity (encoded by the BEM) fixes the
far-field normalization of $\psi$, and the single-node pin removes the remaining
additive constant. Adding a uniform sheet
$\kappa \to \kappa + c$ changes $\mathbf{f} \to \mathbf{f} - 2Mc$, which changes $\psi$, which
changes $\gamma$. The map $F$ is injective in contrast to Kaiser-Squires, where
the Fourier kernel vanishes at $\mathbf{k} = \mathbf{0}$.


## 11. MAP Reconstruction and Tikhonov Regularization

### 11.1 The Tikhonov functional

Tikhonov regularization replaces the ill-posed problem $F\kappa = \gamma_{\rm obs}$ with:

$$\kappa_\lambda = \arg\min_{\kappa} \lbrace\{ \|F\kappa - \gamma_{\rm obs}\|^2 + \lambda\kappa^\top R\kappa \rbrace\}$$

This is exactly the **MAP estimator** with Gaussian likelihood and Gaussian
prior. Existence, uniqueness, and convergence are established in
**[C\&K \S10.2, Thm 10.2]**. Implemented in `inverse.py`, `MAPReconstructor`.

### 11.2 Choosing the regularization operator $R$

- **$H^1$ ($R = K$):** Penalizes $\|\nabla\kappa\|^2$. Smoothness prior.
- **Matern-Wiener ($R = M + \ell^2 K$):** Penalizes $\|\kappa\|^2 + \ell^2\|\nabla\kappa\|^2$. **Recommended.**

The **Matern-Wiener prior** $R = M + \ell^2 K$ has Green's function
$G(r) \approx e^{-r/\ell}$, a Matern-1/2 covariance with correlation length $\ell$.
Setting $\ell = \sigma_{\rm lens}$ matches the prior to the expected spatial scale of $\kappa$.

Assembled in `operators.py`, `build_wiener_regularizer`. Selected by
`wiener_length` parameter in `MAPReconstructor`.

### 11.3 Filtered SVD interpretation

For $R = I$, the Tikhonov filter is $\phi_\lambda(\sigma) = \sigma/(\sigma^2 + \lambda)$:
$\approx 1/\sigma$ for $\sigma \gg \sqrt{\lambda}$
(large modes recovered accurately) and $\approx \sigma/\lambda$ for $\sigma \ll \sqrt{\lambda}$ (small modes
suppressed). This filter interpretation is discussed in **[C\&K \S10.2]**.


## 12. The Adjoint Gradient

### 12.1 The adjoint of $F$

Recall $F = S \cdot A_{\rm coupled}^{-1} \cdot (-2M)$. Using the symmetry of $M$ and noting
that $A_{\rm coupled}$ is not symmetric ($K_h$ is asymmetric), the $L^2$ adjoint is:

$$F^* = (-2M)A_{\rm coupled}^{-T}S^\top$$

### 12.2 The gradient of the MAP loss

Define residuals $r_a = S_a\psi - \gamma_{a,\rm obs}$. The gradient of
$\mathcal{L}(\kappa) = \|F\kappa - \gamma_{\rm obs}\|^2 + \lambda\kappa^\top R\kappa$ is:

$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{\kappa}} = -4MA_{\rm coupled}^{-T}(S_1^\top\mathbf{r}_1 + S_2^\top\mathbf{r}_2) + 2\lambda R\kappa$$

The term $A_{\rm coupled}^{-T}(S_1^\top r_1 + S_2^\top r_2)$ is the **adjoint solve**
using `trans='T'` in the SuperLU factorisation, with the gauge node zeroed in
the RHS.

Per-iteration algorithm in `inverse.py`, `MAPReconstructor._make_obj_and_grad`:

1. Forward: $\mathbf{f} = -2M\kappa$ (gauge node zeroed), solve $A_{\rm coupled}\psi = \mathbf{f}$, compute $\gamma_a = S_a\psi$
2. Residuals: $r_a = \gamma_a - \gamma_{a,\rm obs}$
3. Loss: $\mathcal{L} = \sum_a\|r_a\|^2 + \lambda\kappa^\top R\kappa$
4. Adjoint RHS: $\mathbf{q} = S_1^\top r_1 + S_2^\top r_2$ (gauge node zeroed)
5. Adjoint solve: $A_{\rm coupled}^{-T}\phi = \mathbf{q}$ via `A_coupled_lu.solve(..., trans='T')`
6. Gradient: $\partial\mathcal{L}/\partial\kappa = -4M\phi + 2\lambda R\kappa$

Total cost per iteration: **two $A_{\rm coupled}$ solves** (forward + adjoint),
reusing the factored SuperLU object.


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

Plot the following three quantities versus mode index $i$:

$$\log\sigma_i, \qquad \log\lvert\langle\gamma_{\rm obs}, \mathbf{u}_i\rangle\rvert, \qquad \log\frac{\lvert\langle\gamma_{\rm obs}, \mathbf{u}_i\rangle\rvert}{\sigma_i}$$

When the second decays faster than the first, the Picard condition is satisfied. The crossover index gives the effective noise cutoff.

Implemented in `svd_analysis.py`, `picard_plot`.


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


## 18. Convergence Theory

### 18.1 Cea's Lemma

For the Galerkin approximation $\psi^h$ in $H^1(\Omega)$:

$$\|\psi - \psi^h\|_{H^1} \leq \frac{M}{\alpha}\inf_{v^h \in V^h}\|\psi - v^h\|_{H^1}$$

The bound reduces to best approximation ($M = \alpha = 1$ for the Laplacian).

### 18.2 The potential converges at $O(h^4)$ — the forward operator's validation

For $P_k$ elements and $\psi \in H^{k+1}(\Omega)$:

| Norm | P1 ($k=1$) | P2 ($k=2$) | P3 ($k=3$) |
|------|----------|----------|----------|
| $H^1$ semi-norm | $O(h)$ | $O(h^2)$ | $O(h^3)$ |
| $L^2$ norm | $O(h^2)$ | $O(h^3)$ | $O(h^4)$ |

**$\psi$ is the quantity that validates the forward operator $F$**, and it attains
the full P3 rate $O(h^4)$. This is measured directly, not assumed: a
**fitted order of 4.07** with stable local orders in $[3.8, 4.1]$ over
$h \in [0.125, 0.625]$ (`femmi.experiments.forward_convergence`,
`examples/paper/forward_convergence.py`,
`tests/test_experiments.py::test_forward_potential_converges_at_order_four`).

Two conditions are needed for that rate to be visible, and both are properties of
the *test*, not of $F$:

* **Compact support.** The manufactured potential
  $\psi = c\,(1 - (r/R)^2)^p$ for $r < R$ (else $0$), with $R < $ the half-width,
  vanishes identically near $\partial\Omega$. Comparing instead against an
  *infinite-domain* analytic field imposes a finite-vs-infinite-domain mismatch
  that floors the error near $2\times10^{-2}$ and produces a spurious measured
  order near 1 — the floor, not the operator, is what is being measured.
* **Enough regularity.** $\psi = c\,u^p$ is $C^{2p-1}$. Taking $p = 6$ gives
  $\psi \in C^5 \subset H^6$, comfortably past the $H^4$ that $O(h^4)$ requires.
  A $C^2$ bump ($p = 3$) is regularity-limited and measures $\approx 3.4$.

The additive gauge (FEMMI pins one node) is removed before comparing.

### 18.3 Shear extraction: $O(h^2)$, and how the extraction is done matters

Shear is the traceless Hessian $\gamma_1 = \tfrac12(\psi_{xx} - \psi_{yy})$,
$\gamma_2 = \psi_{xy}$. Differentiating twice costs two orders:

| Element | Shear convergence | Why |
|---------|------------------|-----|
| P1 | $\equiv 0$ | $\partial^2/\partial x^2$ of piecewise linear is zero |
| P2 | $O(h^0)$ | Piecewise constant second derivatives |
| P3 | $O(h^2)$ | Piecewise linear second derivatives |

**The $O(h^2)$ rate is confirmed.** Measured local orders on the compactly
supported manufactured solution (`femmi.experiments.shear_convergence`,
`examples/paper/shear_recovery.py`):

| $h$ | nodal | variational recovery |
|-----|-------|----------------------|
| 0.3125 | 1.38 | 1.60 |
| 0.2083 | 1.64 | 1.66 |
| 0.1562 | 1.81 | 1.80 |
| 0.1250 | 1.88 | 1.87 |
| 0.1042 | 1.92 | 1.91 |
| 0.0893 | **1.94** | **1.93** |

Coarse meshes are strongly pre-asymptotic, so a single fitted slope across the
whole range reads $\approx 1.5$–$1.7$ and *understates* the rate; the local order
is the honest number and it converges cleanly to 2.

Two extraction routes are implemented, and they differ in constant rather than in
rate:

* **Nodal sampling** (the default `S1`/`S2`, `operators._assemble_shear_ops`):
  element Hessians evaluated at the P3 nodes and averaged over adjacent elements.
  Nodes are exactly where a $C^0$ P3 element's second derivative jumps, so this
  pays a large constant.
* **Variational recovery** (`operators.RecoveredShear`): integrate by parts once,
  $\int N_i\,\gamma_1 = \tfrac12\!\left[-\!\int N_{i,x}\psi_{,x} + \int N_{i,y}\psi_{,y}\right] + \text{bdry}$,
  so only *first* derivatives of $\psi_h$ are ever taken, and project the result
  back onto the continuous P3 space with the mass matrix. This gives a
  **$1.8\times$ smaller error at the same $h$** — worth roughly a $1.3\times$
  refinement for free.

  The dropped boundary term $\oint_{\partial\Omega} N_i \psi_{,x} n\,ds$ means
  recovered values on the boundary ring are not meaningful unless $\psi$ and
  $\nabla\psi$ vanish there (as they do for the compactly supported test field).

### 18.3a $C^1$ elements: removing the second-derivative penalty entirely

The $O(h^2)$ ceiling above is a property of the *element*, not of the problem. A
degree-$k$ element gives $O(h^{k-1})$ in the second derivative, so the ceiling
lifts by going to a $C^1$ element of higher degree. `femmi/elements.py`
implements two:

| element | degree | DOF/tri | continuity | shear rate | measured |
|---|---|---|---|---|---|
| P3 Lagrange (current) | 3 | 10 | $C^0$ | $O(h^2)$ | 1.81 |
| HCT macro-element | 3 | 12 | $C^1$ | $O(h^2)$ | 1.95 |
| **Argyris** | **5** | **21** | $C^1$ | $O(h^4)$ | **3.89** |

Measured by interpolating the manufactured potential and differentiating twice
(`experiments.element_shear_convergence`, `examples/paper/element_comparison.py`).
At $h = 0.156$ Argyris is **42× more accurate** than the current P3 nodal path.

**The DOF count does not punish this.** "21 per triangle" is misleading: Argyris
DOFs sit on vertices (6) and edges (1) and are shared, so the global count is
$6(n_x{+}1)^2 + 3n_x^2 \approx 9n_x^2$ — essentially P3's $\approx 9n_x^2$.
Measured ratios are $1.11\times$, $1.06\times$, $1.04\times$ at $n_x = 8, 16, 24$,
tending to $1$. So Argyris buys two orders of convergence at parity cost.

Three precise points, since "$C^1$" is easy to overstate:

1. $C^1$ means a continuous **gradient**, not a continuous Hessian. Across an
   edge interior the tangential-tangential second derivative is continuous
   (differentiate the continuous gradient along the edge) but the normal-normal
   one jumps — measured, and asserted, in `tests/test_elements.py`.
2. What matters for FEMMI is narrower and stronger: Argyris carries
   $\{u_{xx}, u_{xy}, u_{yy}\}$ as **vertex DOFs**, so the Hessian *at a node* is
   single-valued across every adjacent element. Nodal shear extraction becomes
   well posed — no averaging (`_assemble_shear_ops`), no recovery
   (`RecoveredShear`), no reason to zero the boundary ring.
3. HCT gets (1) but not (2): its vertex DOFs stop at the gradient, so its Hessian
   is still multivalued at vertices, and being cubic it stays $O(h^2)$. Its
   appeal is cost — it is *cheaper* than P3 ($0.69\times$ the DOFs) — not shear
   accuracy, where its constant is in fact worse than P3's.

Both elements are constructed in physical coordinates by inverting the
DOF-functional matrix (Argyris is not affine-equivalent, so the usual reference
pullback would mis-transform its derivative DOFs — the classic implementation
trap). Shared-edge normals are oriented from the global vertex indices; get that
wrong and the space is silently non-conforming.

### 18.3b Argyris from a solve, and what still blocks it

The rates in 18.3a are *interpolation* rates. Solving the lensing Poisson problem
$\nabla^2\psi = 2\kappa$ on the Argyris space (`femmi/c1_assembly.py`,
`solved_shear_convergence`) reproduces them: local orders
$3.52 \to 3.17 \to 4.02 \to 3.94$, i.e. $O(h^4)$ from an actual solve.

Two implementation notes that matter for reproducing this:

* **Quadrature.** `femmi.assembly` tops out at a degree-7 rule. Argyris mass
  integrands are degree 10 and stiffness integrands degree 8, so that rule would
  cap the measured order through quadrature error alone. `c1_assembly._quad`
  generates conical-product Gauss rules of arbitrary degree instead, verified
  exact to machine precision through degree 10.
* **Shear extraction becomes a selection.** With Hessian DOFs at the vertices,
  $\gamma_1 = \tfrac12(u_{xx} - u_{yy})$ and $\gamma_2 = u_{xy}$ are read
  directly off the DOF vector (`c1_shear_at_vertices`). There is no assembly, no
  averaging over adjacent elements, and nothing to zero on the boundary — $S_1$
  and $S_2$ collapse to index selection.

**What is not done: the FEM–BEM coupling.** The exterior problem couples through
the boundary trace, and a $C^1$ space has a richer trace than P3 — the
normal-derivative DOFs on boundary edges must be matched against the
Steklov–Poincaré operator. Until that lands, $C^1$ solves use Dirichlet
conditions, which are *exact* for a compactly supported field (hence valid for
the manufactured convergence study above) and *wrong* for an isolated-field
reconstruction. So Argyris is a validated element and solver, not yet a drop-in
replacement for `build_operators`.

### 18.3c Hierarchical compression of the BEM operator

`bem.assemble_single_layer` is dense: $O(N_b^2)$ memory and work, and profiling a
build at $n_x = 20$ puts 2.3s of a 2.6s total in BEM assembly. The single-layer
kernel $G = \tfrac{1}{2\pi}\log|x-y|$ is asymptotically smooth, so blocks between
well-separated boundary pieces are numerically low rank. `femmi/aca.py` builds a
binary cluster tree, applies the admissibility test
$\min(\mathrm{diam}\,s, \mathrm{diam}\,t) \le \eta\,\mathrm{dist}(s,t)$, and
compresses admissible blocks with partially-pivoted ACA.

Measured on a circular boundary, tolerance $10^{-6}$:

| $N_b$ | stored fraction | max block rank | matvec rel. error |
|---|---|---|---|
| 60 | 1.000 | – | $1.4\times10^{-16}$ |
| 120 | 1.000 | – | $1.9\times10^{-16}$ |
| 240 | 0.750 | 5 | $2.3\times10^{-9}$ |
| 480 | 0.383 | 5 | $4.4\times10^{-10}$ |

The block rank stays at 5 while $N_b$ grows, which is the defining H-matrix
property; the stored fraction therefore roughly halves per doubling. At small
$N_b$ it correctly declines to compress and falls back to dense.

This compresses and applies the operator; it does not yet replace the dense
assembly inside `build_operators`, because the coupled solve LU-factorises a
dense $A_{\rm coupled}$. Consuming an H-matrix requires the iterative solver plus
Calderón preconditioning.

### 18.3d Matrix-free coupled solves, and what the BEM block needs

Consuming the H-matrix of 18.3c requires never assembling $A_{\rm coupled}$.
`femmi/iterative.py` applies it as an operator,

$$A x \;=\; K x \;+\; \mathrm{scatter}\!\left(C\,\mathrm{gather}(x)\right), \qquad C = -M_b V_{\rm eff}^{-1} X_m,$$

with the gauge row imposed explicitly, so the BEM is reached only through matvecs
and one $V_{\rm eff}$ solve — which a `v_solve` callable can route through ACA.
Measured against the assembled matrix: matvec and transpose agree to $10^{-16}$,
GMRES with an ILU($K$) preconditioner converges in **20–21 iterations, flat in
mesh size** ($n_x = 10, 14, 18$), and the solution matches the direct LU to
$10^{-12}$. With the BEM supplied by the H-matrix, the coupled solve still
reproduces the dense result to $1.0\times10^{-10}$.

Two corrections this exposed, both worth recording:

* **The transpose is not the obvious thing.** The gauge fix zeroes *row* $g$ but
  leaves *column* $g$ populated, so $(A^{\top}x)_g$ is the whole of column $g$ —
  the gauge term must be *added* to the column contribution, not written over it.
  Overwriting passes a casual check and corrupts the adjoint at the $10^{-4}$
  level, which would silently degrade every MAP gradient.
* **Far-field quadrature must not be used on near blocks.** The ACA entry
  evaluator uses plain Gauss–Legendre, valid across separated clusters but wrong
  where the $\log$ singularity lives. Applying it to inadmissible blocks builds a
  *different* operator — a 69% error in the coupled solve — so `build_hmatrix`
  takes a separate `near_block` evaluator.

**Not Calderón preconditioning.** Calderón preconditioning of $V$ uses the
hypersingular operator $W$ and the identity that $VW$ is a compact perturbation of
$-I/4$; `femmi.bem` assembles $V$, $K$ and $M_b$ but not $W$, so the ingredient
does not exist yet. (`bem.calderon_matrix` is the *coupling* operator
$V^{-1}(\tfrac12 M_b + K_h)$ — an easy name to misread.) Assembling $W$ is what
would make the iteration count provably mesh-independent.

### 18.3e Choosing $\lambda$ when the discrepancy principle does not apply

Morozov selects $\lambda$ from the root of $D(\lambda) = \|F\kappa_\lambda -
\gamma\| - \delta$. Two failures were found by running the benchmark grid:

1. **Selection was skipped for non-quadratic priors.** Nothing about the
   discrepancy requires quadratic structure — it is evaluated by solving the MAP
   problem at each trial $\lambda$ — but the prior was never threaded through, so
   TV/sparsity/max-entropy silently ran at a fixed `lam_reg`. Fixing this moved
   them from shape-$L^2$ $2.7$–$3.8$ to $0.8$–$1.5$ on an NFW field.

2. **No root $\Rightarrow$ the worst possible answer.** If $D(\lambda_{\min}) > 0$
   the model cannot reach the assumed noise level at *any* $\lambda$, and the old
   code returned $\lambda_{\min}$ — the least-regularised solution, i.e. maximal
   noise amplification. On a tapered log-normal field this gave shape $L^2 = 1.55$
   where the best $\lambda$ gave $0.46$ and even $\lambda_{\max}$ gave $0.77$.
   When the residual floor sits above $\delta$, the correct response is *more*
   regularisation, not less. The fallback is now the **L-curve corner**
   (`lcurve_lambda`), which needs no bracket and no reliable $\delta$.

### 18.4 Why $O(h^2)$ is the wrong expectation for catalog-native data

$O(h^2)$ is the correct theory and a poor guide to practice, because the same
second derivative that costs two orders of accuracy also **amplifies noise by
$h^{-2}$**. With a fixed perturbation $\sigma$ in $\psi$ — which is what shape
noise on a galaxy catalog leaves behind after the solve — the total shear error is

$$\mathrm{err}(h) \;\sim\; \underbrace{C\,h^{2}}_{\text{discretisation}} \;+\; \underbrace{\sigma\,h^{-2}}_{\text{amplified noise}},$$

a **U-shaped** curve. Refining helps only down to

$$h_{\rm opt} \sim (\sigma/C)^{1/4},$$

and refining past it actively makes the shear *worse*. This is measured in
`femmi.experiments.shear_noise_amplification` (right panel of
`examples/paper/shear_recovery.py`): at $\sigma = 10^{-4}$ in $\psi$ the error
bottoms out at $h_{\rm opt} \approx 0.42$ and then climbs, and refining from
$h = 0.208$ to $h = 0.125$ (a factor $1.67$) multiplies the error by $2.71
\approx 1.67^2$ — the predicted $h^{-2}$ scaling, to two digits.

The practical consequences for a catalog-native run:

* mesh resolution should be set by the **galaxy density and shape-noise level**,
  not by chasing the asymptotic rate;
* the accuracy gain from variational recovery is worth more than refinement once
  $h \lesssim h_{\rm opt}$, since it lowers $C$ without touching the $\sigma h^{-2}$
  term;
* convergence-rate claims must be demonstrated on noiseless manufactured
  solutions, and must not be extrapolated to noisy data.

### 18.5 $\psi$ convergence and domain geometry

The $\psi$ convergence rate depends critically on the geometry of $\partial\Omega$.

**Square domain.** Square corners introduce reentrant singularities in the
exterior solution with exponent $\pi/(2\pi - \pi/2) = 2/3$, capping the effective
$\psi$ convergence at $O(h^{5/3})$ regardless of P3 interior accuracy. The
logarithmic capacity of the unit square ($\approx 0.59 < 1$) makes $V_h$
negative-definite on this domain — mathematically correct and handled correctly
by the implementation.

**Circular domain.** A circular $\partial\Omega$ has no corners and a smooth exterior
solution, so no singularity exponent caps the convergence. The full P3 rate
$O(h^4)$ in $L^2(\Omega)$ is recovered for $\psi$ as well. The circular domain is the preferred geometry for production runs; a complete
implementation is in progress. Since $\psi$ is never directly observed — only the shear $\gamma = \partial^2\psi$ enters the
data — the $O(h^{5/3})$ cap on the square domain is acceptable for the inverse problem, but the circular domain is preferred when forward model
fidelity matters.

### 18.6 The 64-bit requirement

The condition number of $A_{\rm coupled}$ satisfies $\kappa(A_{\rm coupled}) = O(h^{-2})$. For a
$20 \times 20$ mesh, $\kappa \approx O(1600)$. In 32-bit arithmetic ($\varepsilon_{32} \approx 6 \times 10^{-8}$), solve
errors are $O(\kappa\varepsilon_{32}) \approx 2 \times 10^{-5}$, dominating the discretization error $h^4 \approx
6 \times 10^{-6}$ for P3 elements. All FEMMI modules enforce 64-bit via
`jax.config.update("jax_enable_x64", True)` at import in `femmi/__init__.py`.


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
