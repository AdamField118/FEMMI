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


> **Validation status:** the benchmark numbers below predate the corrected P3
> transpose, missing-data handling, weighted sampler, and HCT observation and
> quadrature rules. They are historical measurements, not validated results for
> the current implementation. Recalibrate all arms before regenerating them.

## 1. Weak Lensing Forward Physics

### 1.1 The lensing potential

A mass distribution with projected surface mass density $\Sigma(\theta)$ produces the
dimensionless convergence:

$$\kappa(\boldsymbol{\theta}) = \frac{\Sigma(\boldsymbol{\theta})}{\Sigma_{\rm cr}}$$

where $\Sigma_{\rm cr}$ is the critical surface density. The lensing potential $\psi$ satisfies
the **2D Poisson equation on all of $\mathbb{R}^2$**:

$$\nabla^2 \psi = 2\kappa \quad \text{in } \mathbb{R}^2$$

### 1.2 Shear from second derivatives of $\psi$

The complex shear $\gamma = \gamma_1 + i\gamma_2$ is related to $\psi$ by:

$$\gamma_1 = \frac{1}{2}\left(\frac{\partial^2\psi}{\partial x^2} - \frac{\partial^2\psi}{\partial y^2}\right), \qquad \gamma_2 = \frac{\partial^2\psi}{\partial x \partial y}$$

Shear requires second derivatives. P1 has zero elementwise Hessians;
P2 has piecewise-constant Hessians, which can approximate a varying field under
refinement. P3 offers higher approximation order, but nodal Hessian recovery
and boundary treatment must be tested separately from potential convergence.

### 1.3 The Green's Function and Exact Solution

The 2D Laplacian fundamental solution satisfying $\nabla^2_y G(x,y) = \delta(x-y)$ is:

$$G(\mathbf{x}, \mathbf{y}) = \frac{1}{2\pi} \ln|\mathbf{x} - \mathbf{y}|$$

A full-plane solution, up to an additive constant, is the volume potential:

$$\psi(\mathbf{x}) = \frac{1}{\pi}\int_{\mathbb{R}^2} \ln|\mathbf{x} - \mathbf{y}|\kappa(\mathbf{y})d^2y$$

For compact support and $m=\int\kappa\,d^2y$, its asymptotic behavior is
$\psi(x)=(m/\pi)\log|x|+O(|x|^{-1})$ in this normalization.
A nonzero total mass is therefore incompatible with $\psi\to0$.
Exterior harmonicity assumes no exterior convergence; it does not model arbitrary
unobserved exterior mass. The implemented scaled BEM and node pin define a
specific discrete boundary model whose consistency is tested numerically.

The properties of such fundamental solutions are developed in **[C\&K \S2.1]**.
Under the **compact support assumption** ($\kappa = 0$ outside bounded $\Omega$), this is
equivalent to the FEM-BEM formulation derived in Sections 3--6.


## 2. Why Naive Dirichlet Boundary Conditions Fail

### 2.1 The systematic error

A standard approach truncates to $\Omega = [-L, L]^2$ and imposes $\psi = 0$ on $\partial\Omega$. For
a Gaussian lens, the true $\psi$ grows logarithmically at large radius and is nonzero at any
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

$$\nabla^2\psi = 2\kappa \quad \text{in } \Omega, \qquad \nabla^2\psi = 0 \quad \text{in } \Omega_{\rm ext}$$

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

In $\Omega_{\rm ext}$, $\psi$ is harmonic with the appropriate logarithmic far-field term. Applying Green's second
identity in $\Omega_{\rm ext}$ yields the **Somigliana identity** for $\mathbf{x} \in \Omega_{\rm ext}$:

$$\psi(\mathbf{x}) = \int_{\partial\Omega} G(\mathbf{x},\mathbf{y})t(\mathbf{y})ds(\mathbf{y}) - \int_{\partial\Omega} \psi(\mathbf{y})\frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})ds(\mathbf{y})$$

This is the direct analogue of **[C\&K \S2.1, Thm 2.5]**.

### 5.2 The Four BEM Operators

All four classical boundary operators map functions on $\partial\Omega$ to functions on
$\partial\Omega$. Their definitions and properties are developed in **[C\&K \S3.1--3.4]**:

$$\text{Single layer: } (Vt)(\mathbf{x}) = \int_{\partial\Omega} G(\mathbf{x},\mathbf{y})t(\mathbf{y})ds(\mathbf{y})$$

$$\text{Double layer: } (K\psi)(\mathbf{x}) = \mathrm{P.V.}\int_{\partial\Omega} \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\psi(\mathbf{y})ds(\mathbf{y})$$

The single layer is symmetric. Signs depend on the chosen fundamental solution
and boundary orientation; the implemented exterior pairing is specified in
Section 6. Compactness statements for smooth boundaries cannot be applied
unqualified to polygonal boundaries. Implemented in `bem.py`.

### 5.3 The Boundary Integral Equation

Taking the limit of the Somigliana identity as $\mathbf{x} \to \partial\Omega$ and applying the jump
relations (**[C\&K \S3.1, Thm 3.1 and Thm 3.3]**):

$$\left(\tfrac{1}{2}I + K\right)\psi\big|_{\partial\Omega} = Vt\big|_{\partial\Omega}, \qquad \mathbf{x} \in \partial\Omega$$

Discretized with $N_b$ boundary nodes (P3 traces on boundary edges):

$$\left(\tfrac{1}{2}M_b + K_h\right)\psi_b = V_ht_b$$

Here $M_b$ is the boundary Gram matrix. This plus-sign relation uses the
interior trace convention; the implemented exterior relation uses the minus
sign shown in Section 6. Do not substitute this equation into the exterior
Schur complement without checking normals and jump conventions.

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

The discrete matrix need not retain an exact constant null vector under this
scaled exterior normalization. Its nonsymmetry also means a symmetric bordered
mean constraint cannot be assumed to be an equivalent gauge treatment.

### 6.3 Gauge Choice: Single-Node Pin

The current implementation replaces one boundary row by an identity row and
zeros the corresponding load, imposing $\psi_{j^*}=0$. This operation is part of
the specified discrete model. Its transpose must include the load projection
(Section 12). Potential offsets leave shear unchanged, but this freedom is
not the convergence mass-sheet ambiguity: a sheet changes the potential
quadratically, not by a constant. No inverse-uniqueness claim follows from
pinning the potential.

The original shear columns are retained. Boundary output rows are zeroed for
P3 because nodal Hessian recovery is unreliable there; these rows should not
be treated as measured shear. Catalogue guard nodes have zero likelihood weight.

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

1. *The response lives entirely at the boundary — intrinsically, not as an
   artifact.* Decomposing $\|F\mathbf{1}\|^2$ by radius, **99.5–99.9%** of it is
   carried by the outer collar, and it **grows** under refinement
   ($\|F\mathbf{1}\|/\sqrt N = 1.66 \to 3.04$ from $n_x=12$ to $24$) rather than
   converging. In the deep interior ($r<1.5$) the uniform sheet produces
   essentially no shear.

   An earlier revision attributed this to the square's reentrant corners and
   predicted a circular domain would fix it. **Measurement says otherwise**: on a
   circular domain the concentration is unchanged (99.9% in the outer 20%) and
   $\|F\mathbf{1}\|/\sqrt N$ still grows, $11.8 \to 23.7$, exactly as on the
   square ($9.97 \to 29.8$). The correct explanation is the simpler one already
   implied by the interior behaviour: an infinite uniform sheet produces **zero**
   shear by symmetry, so a finite sheet's entire observable signature is
   necessarily an edge effect — on *any* domain. This is not a mesh or geometry
   defect that a better discretisation can remove; it is what the physics says.
   The mass-sheet degeneracy therefore cannot be rescued by changing the domain.

2. *On neutral truth the advantage disappears.* Against an analytic GalSim NFW
   field (`femmi.truth`, generated by neither method's forward), FEMMI's mean-$\kappa$
   error is $0.047$ versus KS's $0.049$ — it recovers a few percent of the true
   mean. The much-quoted $0.002$ vs $0.085$ comes from generating the test shear
   with FEMMI's *own* forward, which is an inverse crime: there the DC component
   is exactly in the range of $F$ by construction.

A nonzero response to a constant coefficient vector is a property of this
finite-domain discretization and its exterior assumptions. It is not evidence
that shear observations determine absolute convergence. For linear shear,
adding a global convergence sheet leaves shear unchanged; for reduced shear,
$(\kappa,\gamma)\mapsto(\lambda\kappa+1-\lambda,\lambda\gamma)$ leaves
$g=\gamma/(1-\kappa)$ unchanged (single source plane). These are different
observation models. Benchmark edge-error measurements above must also be rerun
after operator corrections.

### 6.4 Scope of the solve

The gauged finite system enforces its assembled weak equations with a projected
load and a pinned potential coefficient. BEM represents a harmonic exterior under
the chosen isolated-source model, subject to discretization and quadrature error.
A small algebraic residual verifies the solve; it does not prove that the boundary
model is exact for a given catalogue or that the shear inverse is injective.

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

Writing this as a single operator: $F = S \cdot A_{\rm coupled}^{-1} Q \cdot (-2M)$, where
$S = (S_1; S_2)$ stacks the two shear operators.

In `operators.py`: `FEMOperators.psi_from_kappa` solves the gauged system;
`FEMOperators.shear_from_psi` applies $S_1$ and $S_2$.
`FEMOperators.forward` chains both. The JAX-differentiable wrapper lives
in `forward.py`, `DifferentiableForward`.

### 10.2 Continuum order and discrete diagnostics

The ideal full-plane shear map is an order-zero Fourier multiplier. For
$k\ne0$, its components are $(k_1^2-k_2^2)/|k|^2$ and $2k_1k_2/|k|^2$;
their squared magnitudes sum to one. Taking two derivatives of the inverse
Laplacian cancels its two-derivative smoothing. The previous compactness
argument was therefore invalid. The mass matrix is a discrete integration
pairing, not a continuum smoothing operator.

`compute_svd` uses Euclidean coefficient norms and all nodal output rows.
Its leading modes describe that finite matrix, not a physically normalized
catalogue information spectrum. Masks, sampling and noise govern the actual
likelihood; no continuum uniqueness conclusion follows from its finite rank.

### 10.3 Forward solvability versus inverse identifiability

An invertible potential solve does not imply an injective shear map. The shear
operator and observation selection can discard directions. Gauge fixing and
exterior assumptions specify a model; priors can select among compatible maps.
None of these operations supplies missing measurements.

## 11. MAP Reconstruction and Tikhonov Regularization

### 11.1 The Tikhonov functional

The implemented linear-shear objective is

$$J(\kappa)=\sum_{a=1}^2\sum_{i:w_i>0}w_i(F_a\kappa-d_a)_i^2
                 +\lambda\phi(\kappa).$$

Weights are nonnegative relative precisions with
$\mathrm{Var}(n_{a,i})=\sigma_n^2/w_i$. Masked observations have zero weight.
A missing value is not an observation of zero shear. For a quadratic prior,
$\phi=\kappa^T R\kappa$; positive `wiener_length` gives $R=M+\ell^2K$,
while zero retains the historical gradient penalty $R=K$.

The Gaussian likelihood is the data term divided by $2\sigma_n^2$.
Sampling therefore uses $\lambda_{\rm sample}=\lambda_{\rm MAP}/(2\sigma_n^2)$
for the same penalty and posterior mode. A proper posterior additionally
requires positive precision on every retained direction.

### 11.2 Choosing the regularization operator $R$

- **$H^1$ ($R = K$):** Penalizes $\|\nabla\kappa\|^2$. Smoothness prior.
- **Matern-Wiener ($R = M + \ell^2 K$):** Penalizes $\|\kappa\|^2 + \ell^2\|\nabla\kappa\|^2$. **Recommended.**

The **Wiener prior** $R = M + \ell^2 K$ is a discrete mass-plus-gradient
precision. It is not, in two dimensions, automatically a Matérn-1/2 covariance.
Setting $\ell = \sigma_{\rm lens}$ matches the prior to the expected spatial scale of $\kappa$.

Assembled in `operators.py`, `build_wiener_regularizer`. Selected by
`wiener_length` parameter in `MAPReconstructor`.

### 11.3 Filtered SVD interpretation

For $R = I$, the Tikhonov filter is $\phi_\lambda(\sigma) = \sigma/(\sigma^2 + \lambda)$:
$\approx 1/\sigma$ for $\sigma \gg \sqrt{\lambda}$
(large modes recovered accurately) and $\approx \sigma/\lambda$ for $\sigma \ll \sqrt{\lambda}$ (small modes
suppressed). This filter interpretation is discussed in **[C\&K \S10.2]**.


## 12. The Adjoint Gradient

### 12.1 The discrete transpose

Let $Q$ zero the gauge (or prescribed Dirichlet) load entries. Then

$$F=-2SA^{-1}QM,\qquad F^T=-2M^TQA^{-T}S^T.$$

$Q$ acts before the forward solve and after the transpose solve. The transpose
is with respect to Euclidean coefficient coordinates, not an unqualified
finite-element $L^2$ adjoint. JAX's custom VJP and the NumPy gradient implement
the same composition.

### 12.2 The weighted gradient

For residuals $r_a=S_a\psi-d_a$ and $W=\mathrm{diag}(w)$,

$$\nabla J=-4M^TQA^{-T}(S_1^TWr_1+S_2^TWr_2)+\lambda\nabla\phi.$$

The forward and transpose solves reuse one SuperLU factorization. Tests compare
both transpose actions and weighted gradients against an explicitly assembled
small forward matrix, including boundary coefficients and Dirichlet projection.

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

### 14.1 Relation to inverse scattering

FEMMI reconstructs a source under a fixed differential operator. Acoustic
inverse scattering usually infers an operator coefficient from responses to
incident waves. Shared regularization tools do not imply identical inverse
problems or transferable support-recovery theorems.

### 14.2 Practical conditioning

Finite sampling, noise, masks, boundaries and the chosen basis can leave poorly
constrained directions. Regularization controls these directions at the cost of
prior dependence. This does not require a compact continuum shear operator.

## 15. SVD, Ill-Posedness, and the Picard Condition

`svd_analysis.compute_svd` computes leading singular triplets of a finite nodal
matrix using dense SVD or Lanczos on $F^TF$. Returned residuals check both
$Fv=\sigma u$ and $F^Tu=\sigma v$. Small singular values amplify noise in that
matrix's coefficient norm. Leading modes alone do not characterize its kernel.

The Picard plot is an exploratory coefficient diagnostic. A finite-window slope
comparison is not a proof of the continuum Picard condition. For orthonormal
left vectors and independent noise of standard deviation $\delta$, each noise
coefficient has standard deviation $\delta$, not $\delta\sqrt{2n}$.

## 16. The Factorization Method for Support Recovery

The historical `FactorizationIndicator` name is retained for API compatibility.
Its output depends only on the forward operator, probe location, and spectral
cutoff. It never consumes the observed shear, so it cannot identify the support
of an unknown lens. The scattering range characterization formerly quoted here
has not been established for this source-recovery operator.

The implemented diagnostic is the normalized sum
$\sum_i |u_i^T\Phi_z|^2/\sigma_i$. It is not the reciprocal previously documented.
Use it only to inspect operator geometry.

## 17. The Linear Sampling Method

`LinearSamplingIndicator` likewise returns a data-independent geometry score:
the normalized norm of the Tikhonov solution to $Fg=\Phi_z$. Its historical
support-recovery interpretation is unsupported. Neither diagnostic should be
used as a mass-map benchmark or publication claim.

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

### 18.3f Coupling a $C^1$ space to the exterior

Coupling P3 to the BEM is easy because the trace of a P3 field along a boundary
edge is a cubic, so the BEM DOFs can simply *be* the FEM boundary nodes and the
coupling operator $P$ is a selection matrix. Argyris breaks that: its trace along
an edge is fixed by $\{u, u_t, u_{tt}\}$ at each endpoint — six conditions, a
**quintic** — and its DOFs are derivatives, which are not nodal values of
anything.

`femmi/bem_hp.py` therefore generalises the boundary assembly to arbitrary
degree, and `femmi/c1_coupling.py` builds $P$ by **evaluation** rather than
selection: row $i$ holds the Argyris basis functions of the element owning
boundary node $i$, evaluated at that node. The coupled operator is then the same
Steinbach form the P3 path uses,
$A = K + P^\top C P$ with $C = -M_b V_{\rm eff}^{-1}(\tfrac12 M_b - K_h)$.

Validation is anchored on the trusted path: at degree 3 the generalised assembly
reproduces `femmi.bem` to machine precision ($\le 10^{-15}$ relative on $V$, $K$
and $M_b$), so the degree-5 path inherits that confidence.

**The payoff.** A Gaussian convergence has $\psi \sim \log r$ far away, so $\psi$
does *not* vanish on the boundary and a Dirichlet pin is simply the wrong
condition. Shear error against the exact infinite-domain analytic shear:

| $n_x$ | Dirichlet | BEM-coupled | gain |
|---|---|---|---|
| 4 | $7.75\times10^{-1}$ | $1.51\times10^{-1}$ | $5.1\times$ |
| 6 | $7.35\times10^{-1}$ | $9.37\times10^{-2}$ | $7.9\times$ |
| 8 | $5.78\times10^{-1}$ | $7.84\times10^{-2}$ | $7.4\times$ |
| 12 | $3.94\times10^{-1}$ | $4.39\times10^{-2}$ | $9.0\times$ |

**A correction to an earlier claim in this section.** An earlier revision reported
that the coupled solve converges at only $\approx O(h^{1.2})$ and attributed that
to the reentrant corner singularity of §18.5 capping the rate. **That was wrong.**
The measurement had been made against the *infinite-domain* analytic shear of a
Gaussian, which reintroduces exactly the finite-vs-infinite-domain floor
documented in §18.2 — the error stops at $\approx 2.6\times10^{-2}$ and the
apparent "rate" is the floor, not the operator.

Repeating the measurement on the compactly supported manufactured field, where
the exterior is exactly zero and no such floor exists, the coupled solve reaches
the full P3-element... the full **$O(h^4)$** on *both* domains:

| $h$ | square (coupled) | order | circle (coupled) | order |
|---|---|---|---|---|
| — | $2.86\times10^{-1}$ | — | $6.52\times10^{-2}$ | — |
| — | $1.13\times10^{-1}$ | 2.29 | $6.23\times10^{-2}$ | 0.11 |
| — | $4.10\times10^{-2}$ | 3.52 | $2.60\times10^{-2}$ | 3.04 |
| — | $1.13\times10^{-2}$ | 3.17 | $5.58\times10^{-3}$ | 3.79 |
| — | $3.57\times10^{-3}$ | **4.02** | $1.80\times10^{-3}$ | **3.93** |

So the exterior coupling does **not** cost an order, and the square's corners do
not cap the *coupled* rate at these resolutions. The lesson is the one §18.2
already states and which is easy to re-learn the hard way: **any convergence
claim measured against an infinite-domain reference is measuring the floor.**

The circular domain is still the better geometry, but for a plainer reason —
a better constant, not a better rate: $1.80\times10^{-3}$ at 1902 DOFs against
$3.57\times10^{-3}$ at 2534 DOFs, i.e. **2x the accuracy for 25% fewer DOFs**.

Scope: this couples the **Dirichlet** trace, which is what the Steinbach form
needs and is exactly the information the P3 coupling transmits. Argyris also
carries independent normal-derivative DOFs on boundary edges; matching the
Neumann trace (a quartic per edge) as well is a further refinement.

### 18.3h The inverse problem on a C^1 space, and the axis that matters

`femmi/c1_inverse.py` closes the loop: shear observations to kappa, minimising
$\|W(S\psi - \gamma_{\rm obs})\|^2 + \lambda\,\kappa^\top R \kappa$ with
$\psi = A^{-1}(-2M\kappa)$.

**$S$ is a selection matrix.** On P3, extracting shear needs
`_assemble_shear_ops` — element Hessians sampled at nodes, averaged over adjacent
elements, boundary ring zeroed. Argyris carries $\{u_{xx},u_{xy},u_{yy}\}$ as
vertex DOFs, so $\gamma_1 = \tfrac12(u_{xx}-u_{yy})$, $\gamma_2 = u_{xy}$ is a
pure selection: two sparse matrices with two and one entry per row.

*Adjoint note.* $\psi = A^{-1}(Pb)$ with $P$ the gauge projector, so
$dL/d\kappa = -2M^\top P\,(A^{-\top} dL/d\psi)$ — the projector acts **after**
the transpose solve. Applying it to the adjoint right-hand side instead is a
different operator; it leaves most DOFs correct and gets the worst one 34% wrong,
which is why the finite-difference check samples several. $A$ is not symmetric
(the double layer is not), so the transpose solve is required, not optional.

**Measured against independent GalSim NFW truth** (shape = DC-removed relative
$L^2$; the mean-$\kappa$ column is flat at $\approx 0.05$ for everything, as
§6.3a requires):

| method | DOFs | observations | shape $L^2$ | sec |
|---|---|---|---|---|
| Argyris (circle) | 522 | 61 | 0.4380 | 0.4 |
| Argyris (circle) | 1104 | 127 | 0.3392 | 0.7 |
| **Argyris (circle)** | **1902** | **217** | **0.2673** | **1.1** |
| Argyris (circle) | 2916 | 331 | 0.2411 | 1.9 |
| **P3 (square)** | **1849** | **1849** | **0.2676** | **3.9** |
| Kaiser–Squires | 1849 | 1849 | 0.3951 | — |

Two axes, two different conclusions:

* **Per DOF the elements tie.** Argyris spends six DOFs per vertex, so its
  accuracy-per-unknown is no better than P3's. Anyone comparing on this axis
  alone would conclude the C^1 element is not worth it.
* **Per OBSERVATION Argyris needs 8.5x fewer shear measurements** here (217
  against 1849), and runs 3.5x faster.

**But that 8.5x is inflated by the setup, and must not be quoted as the survey
claim.** On a structured mesh P3 is handed a shear observation at *every* node —
edge and interior nodes included — which no real catalog provides. The honest
version of the comparison puts both methods' vertices at galaxy positions
(§18.3i) and measures the axis a survey is actually specified on — effective
source density $n_{\rm eff}$ in gal/arcmin². There the factor against P3 is
**$\ge 3\times$ for $n_{\rm eff}\ge 10$** and unresolved at the sparsest density,
and against Kaiser–Squires it is $3.95\times$ at DES density and $\ge 3\times$
above that.

The reason is structural rather than incidental: a P3 node contributes a shear
estimate only through an average over the elements meeting there, while an
Argyris vertex carries the Hessian outright.

### 18.3g The hypersingular operator, and what Calderón actually buys

`bem_hp.assemble_hypersingular_hp` assembles $W$ through the Nédélec/Maue
integration-by-parts identity, which in 2D reduces the hypersingular form to the
**single-layer form applied to arc-length derivatives**:

$$\langle W u, v\rangle \;=\; -\iint G(x,y)\,\frac{du}{ds}(y)\,\frac{dv}{ds}(x)\,ds(y)\,ds(x).$$

That is why it is cheap here — it reuses the single-layer kernel and its Duffy /
log-Gauss singular treatment with the basis replaced by its derivative. Direct
assembly from the $|x-y|^{-2}$ kernel would need a finite-part regularisation.
Two properties follow and are tested: $W$ is symmetric, and $W\mathbf{1} = 0$
(the derivative of a constant vanishes).

**What Calderón preconditioning delivers here, measured.** Pairing $V$ and $W$ on
the same mesh (with a rank-1 stabilisation of $W$ on constants, mirroring the
log-capacity correction in $V_{\rm eff}$):

| $N_b$ | $\mathrm{cond}(V_{\rm eff})$ | $\mathrm{cond}(V_{\rm eff}M_b^{-1}W)$ | ratio |
|---|---|---|---|
| 48 | $1.31\times10^{2}$ | $5.85\times10^{1}$ | $2.24\times$ |
| 96 | $2.58\times10^{2}$ | $1.14\times10^{2}$ | $2.25\times$ |
| 192 | $5.15\times10^{2}$ | $2.28\times10^{2}$ | $2.26\times$ |
| 384 | $1.03\times10^{3}$ | $4.55\times10^{2}$ | $2.26\times$ |

A flat $2.26\times$ — a genuine constant-factor win, but the conditioning still
grows linearly with $N_b$. This is **not** the mesh-independence Calderón
preconditioning is famous for, and the reason is well known: the discrete pairing
requires **dual (Buffa–Christiansen) basis functions** on a barycentrically
refined mesh. Using the same mesh for both operators cannot give mesh
independence. Implementing the dual bases is what would.

### 18.3i Accuracy versus source density — the catalog-native claim

With the mass-sheet line closed (§6.3a: the DC signature is an edge effect on any
domain, so no mesh rescues it), the candidate claim that replaces it is about
**data efficiency**, and it has to be measured where a survey lives: vertices at
galaxy positions, the same galaxies given to every method, truth from
`femmi.truth`. `femmi/density.py` runs exactly that.

**The axis is source density, not galaxy count.** A weak-lensing survey is
specified by its *effective source density*

$$n_{\rm eff}\ \ [\text{gal}\,\text{arcmin}^{-2}],$$

not by a raw number of galaxies: a count is meaningless without the field area,
while $n_{\rm eff}$ is directly comparable across surveys and is what a proposal,
a forecast or a referee asks for. It is also the one quantity a survey cannot
simply buy more of — it is fixed by depth, seeing and shape-measurement success,
whereas area is bought with survey time. So the honest statement of the claim is
"the same accuracy at *lower source density*", and the experiment is parameterised
that way throughout. Field geometry is in arcmin (as `femmi.truth.galsim_nfw_truth`
already assumes), so on a disk of radius $R$ the conversion
$N = n_{\rm eff}\pi R^2$ is exact rather than nominal.

Sweeping $n_{\rm eff} \in \{5, 10, 20, 30\}$ on a circular field of radius
$3'$ (area $28.3\ \mathrm{arcmin}^2$, so $141/283/565/848$ galaxies) — a range
that brackets **DES Y3** (5.6), **KiDS-1000** (6.2), **CFHTLenS** (11), **HSC Y3**
(19.9), **LSST Y10** (27) and **Euclid** (30), all tabulated in
`density.SURVEY_NEFF`. Shape $L^2$ (DC-removed) against an independent GalSim NFW
field, shape noise 0.05, **averaged over six catalog realisations** with the
standard error of the mean:

Both FEM arms choose their regularisation weight the same way: P3 by its
per-catalog Morozov selection, Argyris by the held-out-calibrated constant
$\lambda_{\rm cal}=1.2111$ (§18.3j — an earlier version of this table pinned
Argyris at $\lambda=0.3$ while P3 tuned itself, which understated the element by
about a fifth).

| $n_{\rm eff}$ [/arcmin²] | galaxies | Argyris (catalog) | P3 (catalog) | Kaiser–Squires |
|---|---|---|---|---|
| 5 | 141 | **0.6288** ± 0.0445 | 0.7082 ± 0.0334 | 0.8781 ± 0.0106 |
| 10 | 283 | **0.4814** ± 0.0118 | 0.6179 ± 0.0071 | 0.7592 ± 0.0062 |
| 20 | 565 | **0.4117** ± 0.0242 | 0.5559 ± 0.0293 | 0.6262 ± 0.0114 |
| 30 | 848 | **0.3798** ± 0.0171 | 0.5182 ± 0.0073 | 0.5047 ± 0.0137 |

**The seed average is not a formality — it is the main methodological finding
here.** On any single realisation the numbers look decisive and are not: across
seeds 0/1/2 the Argyris-vs-P3 equivalence factor comes out anywhere from 0.58x to
2.9x, and P3 beats Argyris outright at $n_{\rm eff}=20$ in two runs of the three.
The three-seed and six-seed averages of that same factor disagree with each other
(4.11/2.04/1.08 against 1.95/—/1.43), which is the clearest possible evidence
that **the equivalence factor is not a stable estimator at this sample size**. The
per-density error comparison above is; the derived factor is not, and it is
reported below only with that attached.

Reading the table against the seed scatter (difference in units of the combined
standard error):

| $n_{\rm eff}$ | Argyris vs P3 | Argyris vs KS |
|---|---|---|
| 5 | $1.4\sigma$ | $5.5\sigma$ |
| 10 | $9.9\sigma$ | $20.9\sigma$ |
| 20 | $3.8\sigma$ | $8.0\sigma$ |
| 30 | $7.4\sigma$ | $5.7\sigma$ |

**What is defensible:**

* **Against Kaiser–Squires, at every density.** $5.5$–$21\sigma$ across the whole
  DES-through-Euclid range. Unlike the earlier pinned-$\lambda$ version of this
  table, the advantage no longer vanishes at Euclid density.
* **Against catalog-native P3, for $n_{\rm eff}\ge 10$** — $9.9\sigma$,
  $3.8\sigma$, $7.4\sigma$. At the sparsest density ($1.4\sigma$) it is not
  resolved, and should not be claimed. This is the statement that the earlier
  table could only call *suggestive*; fixing the regularisation weight (§18.3j)
  is what moved it, not more realisations.

Converted to the survey axis, only the sparsest density yields a finite
equivalence factor at all:

| Argyris at | its error | P3 needs | factor | KS needs | factor |
|---|---|---|---|---|---|
| 5 | 0.6288 | 9.2 | **1.84x** | 19.7 | **3.95x** |
| 10 | 0.4814 | > 30 | **> 3x** | > 30 | **> 3x** |
| 20 | 0.4117 | > 30 | — | > 30 | — |
| 30 | 0.3798 | > 30 | — | > 30 | — |

The dashes are the point rather than a gap. From $n_{\rm eff}=10$ upward
Argyris's error is below anything P3 or KS reach *anywhere in the swept range*,
so the honest report is a bound: **at CFHTLenS-like density (10 gal/arcmin²)
catalog-native Argyris already beats what both baselines achieve at Euclid
density (30)** — a factor of more than three, quoted as an inequality because the
sweep stops there. (`_equivalent_density` returns `nan` outside the measured
range for exactly this reason; `np.interp` would clamp and print a spurious
`1.00x`.)

The $8.5\times$ from the structured-mesh comparison (§18.3h) still does not
survive contact with catalog geometry — that figure came from handing P3 an
observation at every node. But the honest catalog-native factor is $\ge 3\times$
at survey densities rather than the $1.4$–$2.0\times$ the pinned-$\lambda$ table
suggested.

**Two further caveats that belong with the claim.**

1. *Mesh conditioning.* Random galaxy positions produce sliver triangles, and
   Argyris inverts a $21\times21$ Vandermonde per element. Median conditioning is
   a benign $\sim10^5$, but the worst element reaches $10^{11}$–$10^{14}$ — three
   surviving digits or fewer — and the count of ill-conditioned elements grows
   with density (seed-mean $1.2/304$, $1.7/597$, $5.5/1176$, $10.2/1751$ at
   $n_{\rm eff} = 5/10/20/30$), as it must: more points means more chances to draw
   a near-degenerate triple. `density.mesh_quality` reports this. Catalog-native
   C^1 needs mesh conditioning; it is not free, and it remains the most likely
   explanation for the sparsest density still sitting short of its oracle
   (§18.3j).
2. *Cost.* Argyris is also the cheaper of the two FEM methods here in wall-clock
   (5.4/8.9/17.7/24.7 s against P3's 12.0/16.3/25.0/37.3 s), for slightly fewer
   DOFs. But the efficiency being *claimed* is in source density, not time; the
   timings are single-machine and are reported only so nobody assumes C^1 costs
   extra.
3. *One truth field.* Everything above is a single centred analytic NFW halo,
   which is the case a C^1 element should suit best. Task #50 runs the same sweep
   on the lognormal and MassiveNuS truths; until then the scope of the claim is a
   smooth peaked field.

**How this table came to be right**, and what it looked like when it was not:
with Argyris pinned at $\lambda = 0.3$ its error was flat between
$n_{\rm eff}=10$ and $20$ ($0.5159 \to 0.5224$) while both baselines kept
improving, and the P3 comparison was resolved at only one of four densities. The
cause was that the comparison was **tuned against untuned** —
`catalog.reconstruct_catalog` has always run with `use_morozov=True`, so the P3
arm chose its own $\lambda$ per catalog while the Argyris arm did not. §18.3j is
that investigation; it is worth reading before quoting anything above, because
the conclusion it reaches about $\lambda$ selection is itself a result.

### 18.3j The regularisation weight: what the pinned $\lambda$ cost

An **oracle sweep** — scoring against the truth at each $\lambda$ on a 9-point
log grid, three seeds, four densities — settles both questions. Best $\lambda$
and the resulting error, against the pinned $\lambda = 0.3$:

| $n_{\rm eff}$ | best $\lambda$ (3 seeds) | error at best | error at $\lambda=0.3$ | cost of pinning |
|---|---|---|---|---|
| 5 | 0.32, 1.0, 0.32 | 0.5586 | 0.5682 | 1.7% |
| 10 | 1.0, 1.0, 1.0 | 0.4733 | 0.5596 | **15%** |
| 20 | 1.0, 1.0, 1.0 | 0.4471 | 0.5716 | **22%** |
| 30 | 1.0, 1.0, 1.0 | 0.3924 | 0.4845 | **19%** |

Three things follow, and the first one corrects the guess in §18.3i.

**1. $\lambda$ must grow with density, not shrink.** The lead predicted the
opposite — "more sources means less smoothing needed". The data term is an
unnormalised *sum* over observations, so doubling the source count doubles it
while the prior term is unchanged; holding the balance fixed therefore requires
$\lambda \propto n_{\rm obs}$. The optimum moves $0.3 \to 1.0$ over the swept
range for that reason, which is a scaling property of the functional and not a
statement about how much smoothing the physics wants.

**2. The flat spot was the artifact it looked like.** At the oracle $\lambda$ the
Argyris curve is monotone again — $0.5586, 0.4733, 0.4471, 0.3924$ — and the
$10 \to 20$ plateau disappears. Pinning cost nothing at the sparsest density
(where $0.3$ happens to be near-optimal) and 15–22% everywhere else, i.e. it hurt
exactly where the comparison against P3 was weakest.

**3. Neither standard selector finds it.** On the same curves,

* **Morozov ($c=1$)** returns $\lambda = 1.9$–$4.6$, consistently $3$–$4\times$
  too large, and measurably *worse* than the pinned value (at $n_{\rm eff}=5$,
  seed 0: $0.84$ against $0.67$);
* **the L-curve corner** returns $0.03$–$0.32$, consistently too small.

The Morozov failure is *systematic* rather than noisy, which is what suggested it
was fixable. The best $\lambda$ sits where the residual is a fixed fraction of
$\delta$ rather than equal to it, and the mechanism is standard: the discrepancy
principle targets a residual equal to the noise, but here roughly $6\times$ more
unknowns than observations are being fitted (an Argyris vertex carries six DOFs
and supplies two shear components), so the model can drive the residual below
$\delta$ legitimately. The degrees-of-freedom-corrected target is
$\delta\sqrt{1-p/n}$.

Calibrating that single constant against the oracle optimum on **held-out
catalogs — seeds 100–102, disjoint from every seed reported anywhere in this
document** — over the four densities gives

$$c = 0.9119 \pm 0.0157 \quad (n=12,\ \text{spread } 0.791\text{–}0.980),$$

i.e. $p/n \approx 0.17$, and stable across density ($0.87, 0.92, 0.94, 0.92$ at
$n_{\rm eff}=5,10,20,30$). `femmi.c1_lambda.MOROZOV_C` freezes that value;
calibrating on the reported seeds instead would be tuning on the test set.

**And it still does not work.** Run blind on the 24 reported catalogs (seeds
0–5, disjoint from the calibration seeds), the calibrated per-catalog rule is a
coin flip against the pinned value — it wins in **10 of 24** — and it *adds*
variance:

| $n_{\rm eff}$ | pinned $\lambda=0.3$ | Morozov, $c$ calibrated | oracle |
|---|---|---|---|
| 5 | 0.6214 ± 0.0363 | 0.6591 ± 0.0486 | **0.5903** ± 0.0434 |
| 10 | 0.5159 ± 0.0241 | 0.5099 ± 0.0119 | **0.4605** ± 0.0108 |
| 20 | 0.5224 ± 0.0230 | 0.5209 ± 0.0572 | **0.4101** ± 0.0235 |
| 30 | 0.4837 ± 0.0177 | 0.4872 ± 0.0329 | **0.3766** ± 0.0168 |
| all | 0.5358 | 0.5443 | **0.4594** |

**Why it fails, quantitatively.** The residual is nearly flat in $\lambda$ near
the optimum: between $\lambda=1$ and $\lambda=3.16$ it moves only $0.0421 \to
0.0493$. The sensitivity is therefore

$$\frac{d\log\lambda}{d\log \mathrm{resid}} \;\approx\; 7,$$

so the $\pm 10\%$ catalog-to-catalog spread in the *correct* $c$ maps to a factor
$\sim 2$ spread in the selected $\lambda$ — which is exactly the range observed
($0.91$ to $1.79$). **The discrepancy signal does not contain enough information
to locate $\lambda$ on this problem**, and no amount of recalibrating $c$ fixes
that; the map being inverted is ill-conditioned. That is a property of the
selection problem, not of the constant.

**The fix is the constant, not the adaptation.** If the per-catalog *signal* is
useless but the per-catalog *optimum* is stable, then the transferable quantity
is the typical optimum. Its geometric mean over the same held-out catalogs is

$$\lambda_{\rm cal} = 1.2111,$$

and freezing that single number and applying it blind to the 24 reported
catalogs gives:

| $n_{\rm eff}$ | pinned $\lambda=0.3$ | $\lambda_{\rm cal}=1.21$ | oracle (ceiling) |
|---|---|---|---|
| 5 | 0.6214 ± 0.0363 | 0.6325 ± 0.0426 | 0.5903 ± 0.0434 |
| 10 | 0.5159 ± 0.0241 | **0.4804** ± 0.0115 | 0.4605 ± 0.0108 |
| 20 | 0.5224 ± 0.0230 | **0.4102** ± 0.0240 | 0.4101 ± 0.0235 |
| 30 | 0.4837 ± 0.0177 | **0.3825** ± 0.0189 | 0.3766 ± 0.0168 |
| all | 0.5358 | **0.4764** | 0.4594 |

It beats the pinned value in **19 of 24** catalogs, by 11% overall and 21% at
$n_{\rm eff} \ge 20$ — and for $n_{\rm eff}\ge 20$ it **matches the per-catalog
oracle to within noise** ($0.4102$ against $0.4101$; $0.3825$ against $0.3766$).
The curve is monotone again ($0.6325, 0.4804, 0.4102, 0.3825$), so the flat spot
is gone.

The conclusion is sharper than the lead expected. Per-catalog adaptation is worth
essentially **nothing** on this problem; getting the *constant* right is worth a
fifth of the error at survey densities. `femmi.c1_lambda.CALIBRATED_LAM` is that
constant and is now the default in `density.argyris_catalog_run`; `lam="auto"`
remains available and is documented as the negative result it is.

The remaining gap is at the sparsest density ($0.6325$ against the oracle's
$0.5903$), where the per-catalog optimum genuinely does vary. Closing it needs a
different *signal*, not a better constant: **cross-validation on held-out
galaxies** measures predictive error directly rather than inferring it from the
flat residual curve (task #52).

### 18.3ja Robustness of the density claim — where it holds and where it stops

§18.3i is measured in one setting: a single centred analytic NFW halo, uniform
galaxies, a clean disk. Each block below changes exactly one of those, three
seeds, otherwise identical. One of them limits the claim materially.

**A non-Gaussian truth field removes the advantage over P3.** Shape $L^2$ on the
shifted-lognormal field (`truth.lognormal_truth`, peaked and non-Gaussian):

| $n_{\rm eff}$ | Argyris | P3 | KS | A vs P3 | A vs KS |
|---|---|---|---|---|---|
| 5 | 0.6223 ± 0.0772 | **0.5854** ± 0.0731 | 0.8688 ± 0.0279 | $0.35\sigma$ (P3 ahead) | $3.0\sigma$ |
| 10 | 0.5977 ± 0.0232 | **0.5695** ± 0.0390 | 0.7905 ± 0.0162 | $0.62\sigma$ (P3 ahead) | $6.8\sigma$ |
| 20 | 0.4908 ± 0.0404 | **0.4798** ± 0.0473 | 0.6524 ± 0.0292 | $0.18\sigma$ (P3 ahead) | $3.2\sigma$ |

P3 is nominally ahead at all three densities, every gap well inside $1\sigma$ —
so against P3 this is a **tie**, not a reversal, but it is emphatically not the
$3.8$–$9.9\sigma$ Argyris advantage the NFW field shows. **The
Argyris-over-P3 result in §18.3i is specific to a smooth, peaked, analytic
field**, which is the case a C¹ element is most suited to and the sympathetic
case a referee would ask about first. That scope limit is measured, not
suspected.

The advantage over **Kaiser–Squires survives the change of field** at
$3.0$–$6.8\sigma$, which makes it the more robust of the two comparisons — the
same conclusion §18.3i reached, now on independent grounds.

**Masking does not hurt, and slightly helps.** Three circular holes (bright
stars / bad CCDs), NFW truth, galaxy count held fixed so masking is not also a
density change:

| $n_{\rm eff}$ | Argyris | P3 | KS | A vs P3 |
|---|---|---|---|---|
| 20 | **0.4127** ± 0.0380 | 0.5942 ± 0.0157 | 0.6239 ± 0.0125 | $4.4\sigma$ |

Against the unmasked $3.8\sigma$ at the same density. Argyris is essentially
unchanged by the holes ($0.4127$ against $0.4117$) while P3 degrades
($0.5559 \to 0.5942$). This is the predicted mechanism: a hole is an **interior
boundary**, and exact BEM far-field handling is worth more there than a
truncation. It is the one perturbation that made the advantage larger.

**Clustered positions hurt everyone, Argyris least.** 40% of galaxies drawn in
tight groups — the pessimistic mesh case, since clusters make slivers:

| $n_{\rm eff}$ | Argyris | P3 | KS |
|---|---|---|---|
| 5 | **0.6748** ± 0.0724 | 0.7717 ± 0.0651 | 0.9347 ± 0.0259 |
| 10 | **0.5587** ± 0.0352 | 0.6995 ± 0.0305 | 0.8426 ± 0.0065 |
| 20 | **0.5159** ± 0.0418 | 0.6335 ± 0.0485 | 0.7490 ± 0.0190 |

Every method degrades against its uniform-catalog baseline; Argyris stays ahead
of both, though the P3 margin thins to $1.8\sigma$ at $n_{\rm eff}=20$ from
$3.8\sigma$.

**Net.** The KS comparison is robust to all three perturbations. The P3
comparison is robust to masking, weakened by clustering, and **absent on a
non-Gaussian field** — so it should be stated as a result about smooth peaked
fields, not about catalog-native reconstruction in general.

*(All $\sigma$ in this section are marginal, not paired; see §18.3o — they
understate the significances, and the qualitative conclusions are unchanged.)*

### 18.3jb HCT: the cheap C¹ element is not the cheap route to the C¹ result

HCT is the obvious economy: a C¹ macro-element at **12 DOF per triangle against
Argyris's 21**, inverting a 12×12 system instead of a 21×21, which is the direct
answer to the sliver caveat. If the accuracy survived, it would be the cheaper
claim to defend.

Taking it through the inverse path needed one new piece. Argyris gets shear for
free — the Hessian *is* three of its DOFs, so $S$ is a selection with one or two
entries per row. HCT's vertex block is only $\{u, u_x, u_y\}$, so the Hessian has
to be **recovered**. C¹ does not imply a unique Hessian. The current operator
uses an area-weighted average of all incident subtriangle traces. The old
first-element convention depended on triangle order. HCT mass, stiffness and
load quadrature is now split over its three polynomial pieces. The following
table predates both corrections and must be regenerated after recalibration.

Three seeds, NFW truth, everything else as §18.3i:

| $n_{\rm eff}$ | Argyris | HCT | P3 | KS | DOFs (A / H / P3) |
|---|---|---|---|---|---|
| 5 | **0.6131** ± 0.0548 | 0.6930 ± 0.0415 | 0.7266 ± 0.0505 | 0.8763 ± 0.0168 | 1458 / 963 / 2668 |
| 10 | **0.4789** ± 0.0182 | 0.6332 ± 0.0219 | 0.6245 ± 0.0054 | 0.7531 ± 0.0109 | 2816 / 1865 / 3946 |
| 20 | **0.4437** ± 0.0317 | 0.6659 ± 0.0109 | 0.5713 ± 0.0373 | 0.6422 ± 0.0048 | 5466 / 3627 / 6412 |

Argyris beats HCT by $1.2\sigma$, $5.4\sigma$, $6.6\sigma$ — and the gap **grows
with density**. HCT lands roughly level with P3 at low density and falls behind
it by $2.4\sigma$ at $n_{\rm eff}=20$. It is not cheaper per DOF either: Argyris
at 2816 DOFs (0.4789) beats HCT at 3627 (0.6659), so it loses on both axes.

**The honest qualifier, and it is the same bug as before.** HCT ran with
`CALIBRATED_LAM = 1.2111`, which was calibrated *for Argyris* (§18.3j). HCT has a
different DOF count and a differently scaled prior matrix $R$, so its optimal
weight is not the same number — **HCT is the untuned arm here**, exactly the
asymmetry that made the Argyris-vs-P3 comparison misleading before it was fixed.

The supporting evidence is the shape of its curve: HCT goes
$0.6930 \to 0.6332 \to 0.6659$, flat and non-monotone, which is precisely the
fixed-$\lambda$ signature that a pinned weight produced for Argyris. So the
result to quote is *"HCT does not beat Argyris at Argyris's $\lambda$"*, and a
per-element recalibration is what would settle whether the element or the tuning
is responsible. What can already be said is that HCT is not a free lunch: the DOF
saving is real and does not buy accuracy on its own.

### 18.3k Vandermonde equilibration — a null result worth keeping

The sliver caveat in §18.3i is real as stated: the worst catalog element's raw
Vandermonde reaches $10^{11}$–$10^{14}$. The obvious repair is to stop inverting
it in bad units. `elements.equilibrated_inverse` applies Ruiz-style two-sided
scaling and inverts $D_r V D_c$ instead, using

$$V^{-1} \;=\; D_c\,(D_r V D_c)^{-1} D_r,$$

an algebraic identity — the returned inverse is the inverse of the original $V$,
so nothing is approximated. What changes is that the rounding committed during
the inversion is governed by the *scaled* condition number.

It works, on its own terms:

| $n_{\rm eff}$ | median raw | median scaled | worst raw | worst scaled | ill-conditioned |
|---|---|---|---|---|---|
| 5 | $1.3\times10^{4}$ | $1.8\times10^{3}$ | $8.3\times10^{11}$ | $4.6\times10^{10}$ | 2/304 → 2/304 |
| 30 | $1.1\times10^{5}$ | $5.4\times10^{3}$ | $8.7\times10^{11}$ | $2.7\times10^{10}$ | 6/1752 → 1/1752 |

**And it buys no accuracy at all.** A/B over six catalogs, everything else held
fixed:

$$\text{mean change in shape } L^2 = -0.15\% \pm 0.34\%.$$

Zero, to the precision the seeds allow. The explanation is arithmetic rather than
subtle: double precision carries ~16 digits, so losing three to a $10^{13}$
condition number still leaves ten — orders of magnitude below the shape-noise
floor that actually limits the reconstruction. **Conditioning was never the
binding constraint**, and the caveat in §18.3i, while true, was not costing
anything.

The scaling is kept because it is free, strictly safer, and makes `mesh_quality`
report both numbers so the distinction stays visible. It is *not* an accuracy
improvement and must not be quoted as one.

### 18.3l The fast solvers: measured, and both stay off

`femmi.aca` (H-matrix BEM) and `femmi.iterative` (matrix-free coupled solve) were
both built, tested, and then used by nothing. Switching them on and reporting a
speedup would have been wrong.

**Making ACA usable at all required a fix first.** `single_layer_entry_fn` was
hardcoded to P3 (`_p3_boundary_basis`, four nodes per element) while the C¹
coupling runs its boundary at degree 5 — so ACA could not serve the one path in
the project that would have used it. It now takes `degree` and `clustering` like
the rest of `bem_hp`, and reproduces the dense single layer to $10^{-9}$.

**And then it loses.** Against dense assembly:

| geometry | $N_b$ | dense / ACA |
|---|---|---|
| uniform circle (deg 3) | 144 | $0.52$–$0.78\times$ |
| uniform circle (deg 5) | 240 | $0.56$–$0.68\times$ |
| catalog guard ring | 120 | $0.65\times$ |
| catalog guard ring | 240 | $0.65\times$ |

ranging over ACA tolerance $10^{-6}$–$10^{-9}$ and admissibility $\eta \in [1,2]$.
Slower everywhere, and **flat in $N_b$** — so this is a per-entry cost
difference, not an overhead that amortises, and no crossover is approaching below
the sizes this project runs ($N_b = 120$–$580$).

The reasoning that predicted a win is worth recording because it is the tempting
one. Dense assembly's cost is the $O(N^2)$ Galerkin **quadrature** in Python, not
linear algebra, so ACA's skipping of most entries ought to pay even at small $N$.
It does not: the cluster tree, the per-block pivoting, and the near-field blocks —
which still need the tuned Duffy/log-Gauss treatment, and are most of a boundary
this size — cost more than the quadrature saved. `bem_hp.ACA_MIN_NB` is therefore
set beyond any reachable size rather than at a crossover, because none was found.
The compression and the accuracy are real; only the speed is not.

**The iterative solver is a separate question and the answer is also no, here.**
The coupled operator's cost is not the boundary at all. Measured build times:

| $n_{\rm eff}$ | $n_{\rm dofs}$ | $N_b$ | FEM asm | BEM asm | trace | total |
|---|---|---|---|---|---|---|
| 5 | 1458 | 120 | 0.60 s | 2.58 s | 0.05 s | 3.72 s |
| 30 | 8093 | 290 | 3.50 s | 6.83 s | 0.10 s | 11.86 s |

At 8k DOFs a sparse `splu` plus its triangular solves beats a matrix-free
iteration; that crossover is in $n_{\rm dofs}$, not $N_b$, and lies above
anything this project runs. `femmi.iterative` stays available and unused by
default, which is the honest state of it.

### 18.3m Buffa–Christiansen dual bases: the mesh independence, delivered

§18.3g left this open explicitly: pairing $V$ and $W$ on the same mesh gives a
flat $2.26\times$ while $\mathrm{cond}$ still grows linearly in $N_b$, and the
textbook fix is a genuinely **dual** basis on a barycentrically refined mesh.
`femmi.calderon` builds it.

The construction is the standard one — split every element at its midpoint, and
associate with each coarse **node** a piecewise constant supported on the two
fine half-elements touching it, normalised to unit integral.

**The pairing must cross spaces, and this is the whole trap.** The first
implementation paired those node-indexed duals against the element-indexed coarse
*constants*. That gives a bidiagonal circulant whose rows sum to 1, with
eigenvalues $(1+\omega^k)/2$ over the $n$-th roots of unity — which **vanishes at
$\omega=-1$**, so the matrix is exactly singular for every even $n$. Measured:
$\mathrm{cond}\sim10^{16}$, which reads as "dual bases do not help".

The correct partner is the space on the other side of the Calderón identity:
$V$ acts on densities in $H^{-1/2}$ (element-indexed constants) while $W$ acts on
traces in $H^{+1/2}$ (node-indexed continuous linears). The duals are
node-indexed, so they pair with the **hats**, and that pairing is available in
closed form. On a fine half-element running from a node to a midpoint the hat at
that node goes $1 \to \tfrac12$ (mean $\tfrac34$) and its neighbour $0 \to
\tfrac12$ (mean $\tfrac14$), so

$$G_{ii} = \tfrac34 \ \text{ exactly, for any mesh}, \qquad
G_{i,i\pm1} = \tfrac14\,\frac{L_{\rm half}}{L_{\rm total}}.$$

The diagonal carries **no mesh dependence whatsoever**. On a uniform mesh the
eigenvalues are $\tfrac34 + \tfrac14\cos\theta \in [\tfrac12, 1]$, so

| $n$ | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| $\mathrm{cond}(G_{\rm BC})$ | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 |

Exactly 2, independent of $N$ — the growth §18.3g reported is gone. And on
*irregular* meshes, where a same-mesh Gram matrix of piecewise constants is
$\mathrm{diag}(L_e)$ and its conditioning is just the element-length ratio:

| $n$ | $\mathrm{cond}(G_{\rm BC})$ | $\mathrm{cond}(\mathrm{diag}\,L)$ |
|---|---|---|
| 32 | 2.06 | $7.3\times10^{2}$ |
| 128 | 2.11 | $7.5\times10^{2}$ |
| 512 | 2.14 | $1.2\times10^{5}$ |

**Scope.** What is delivered and verified is the load-bearing ingredient — a dual
pairing whose conditioning does not grow with the mesh. Assembling $V$ and $W$
*against* the dual basis to obtain the fully preconditioned operator is the
remaining step; the pairing was the part §18.3g identified as missing.

### 18.3n Cross-validated $\lambda$: the signal Morozov did not have

§18.3j ended with a diagnosis rather than a fix. The discrepancy principle cannot
locate $\lambda$ on this problem because the fitting residual is nearly flat near
the optimum — $d\log\lambda / d\log\mathrm{resid} \approx 7$, so the $\pm10\%$
catalog-to-catalog spread in the correct constant $c$ becomes a factor $\sim2$ in
the selected $\lambda$. That is an ill-conditioned inversion, and no
recalibration of $c$ repairs it. The way out had to be a **different signal**.

**K-fold cross-validation does not go through that curve at all.** It measures
how well the reconstruction predicts shear at galaxies it never saw, which is
directly the quantity being optimised and is steep in $\lambda$ on both sides of
the optimum. The fold structure is the natural one: a fold is a subset of the
*data-carrying vertices*, dropped from the data weight rather than from the mesh
— removing them from the mesh would change the discretisation between folds and
compare different function spaces, which is not a cross-validation of $\lambda$.

Five folds, seven-point grid, warm-started along $\lambda$ within each fold. Run
blind on the same 24 catalogs (seeds 0–5):

| $n_{\rm eff}$ | $\lambda_{\rm cal}=1.21$ | 5-fold CV | oracle |
|---|---|---|---|
| 5 | 0.6326 ± 0.0437 | **0.5991** ± 0.0460 | 0.5893 ± 0.0419 |
| 10 | 0.4818 ± 0.0119 | **0.4736** ± 0.0084 | 0.4588 ± 0.0101 |
| 20 | 0.4099 ± 0.0243 | 0.4068 ± 0.0233 | 0.4100 ± 0.0223 |
| 30 | 0.3807 ± 0.0176 | 0.3801 ± 0.0178 | 0.3760 ± 0.0167 |
| all | 0.4763 | **0.4649** | 0.4585 |

**It works, and exactly where it was predicted to.** CV wins in **20 of 24**
catalogs — the comparison is paired, same catalogs and same noise, so the win
count is the meaningful statistic rather than the marginal error bars. Set
against Morozov's 10 of 24, the contrast is the point: same problem, same grid,
same solver, different signal.

Where it pays is the sparsest density, which is precisely where §18.3j left the
gap. At $n_{\rm eff}=5$ CV recovers **77% of the remaining oracle gap**
($0.6326 \to 0.5991$ against the oracle's $0.5893$). At $n_{\rm eff}\ge20$ it
adds nothing measurable, for the good reason that the fixed constant is *already
at the oracle* there.

**Why it is not the default.** Cost. CV is $5\times7 = 35$ MAP solves per catalog
against one, and buys 2.4% overall. The recommendation the numbers support: keep
`CALIBRATED_LAM` as the default, and use `lam="cv"` on sparse catalogs, where it
buys 5% and the extra solves are cheapest anyway.

**What this settles about $\lambda$ selection generally.** Four rules were
measured on identical data: Morozov with the textbook $c=1$ (worse than a pinned
value), Morozov with a held-out-calibrated $c$ (a coin flip), a single held-out
constant (near-oracle for $n_{\rm eff}\ge20$), and cross-validation (best
everywhere, 20/24). The ordering is not about sophistication — it is about
whether the estimator's signal is *sensitive to what is being estimated*. The
residual is not; predictive error is.

### 18.3o Kaiser–Squires was the untuned arm, and the significances were unpaired

Two defects in how §18.3i was measured, both found after the fact, both fixed in
`femmi.density`.

**KS was running at an uncalibrated grid.** After §18.3j the Argyris arm chose a
calibrated $\lambda$ and the P3 arm ran its own per-catalog Morozov, while
Kaiser–Squires stayed pinned at `grid_size=32, smoothing_px=1.0` — numbers that
were never measured. Grid resolution **is** the KS regularisation knob, since a
coarser pixel averages more galaxies, so the comparison had become
tuned-against-untuned in the direction that flatters the claim. This is the same
class of bug as §18.3j, one arm later.

Calibrated on the same held-out catalogs as everything else (seeds 100–102),
sweeping `grid_size` over 6–48 and `smoothing_px` over 0–2:

| $n_{\rm eff}$ | best (grid, smooth) | error | at (32, 1.0) | gain |
|---|---|---|---|---|
| 5 | (12, 1.0) | 0.5943 | 0.8671 | **+31.5%** |
| 10 | (12, 1.0) | 0.4932 | 0.7768 | **+36.5%** |
| 20 | (16, 1.0) | 0.3617 | 0.5875 | **+38.4%** |
| 30 | (24, 1.0) | 0.4032 | 0.4730 | **+14.8%** |

**15–38% of KS's error was the untuned grid.** The optima are interior only after
extending the search down to `grid_size=6`; the first sweep bottomed out at its
own lower edge and would have reported a boundary value as the optimum — the same
mistake the equivalence-factor interpolation makes when it clamps. Resolution
rises with density, as it should, and `smoothing_px = 1.0` was right all along.
`ks_params_for_density` interpolates between the anchors.

This **will move the headline**: the KS comparison is the one §18.3ja identified
as robust, and it was measured against a handicapped baseline. The tables in
§18.3i are not yet regenerated against calibrated KS.

**The significances were computed unpaired.** Every $\sigma$ in §18.3i, §18.3ja
and §18.3jb was $\Delta / \sqrt{\mathrm{se}_A^2 + \mathrm{se}_B^2}$, treating the
two arms as independent samples. They are not: every arm sees the same catalogs,
the same galaxy positions, the same noise realisation and the same truth, so the
catalog-to-catalog scatter is **shared and cancels** under pairing.

The size of the error is easy to show. With a true gap of $0.02$ under shared
scatter of $0.10$, the paired $t$ is $9$–$11$ while the marginal $\sigma$ is
$0.3$ — the same data, and one test sees nothing. `density.paired_comparison`
reports the mean paired difference, its standard error, the paired $t$, and a
distribution-free win count (which matters at $n=6$); rows without a `seed` are
refused rather than silently aggregated. The win-count statistics already used
for $\lambda$ selection (20/24 against 10/24, §18.3n) were paired all along, and
were decisive precisely where marginal bars overlapped.

Both fixes point the same way: **regenerate §18.3i once, with calibrated KS and
paired statistics**, rather than quoting the current table.

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
