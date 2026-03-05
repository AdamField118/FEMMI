---
title: "Finite Element Methods for Weak Gravitational Lensing: From Poisson's Equation to Mass Reconstruction"
date: "2025-12-05"
tags: "Computational Astrophysics, FEM, Gravitational Lensing"
snippet: "A complete derivation of the finite element method for solving the 2D lensing Poisson equation, from weak formulation through numerical implementation. Building the bridge between differential geometry and computational reconstruction."
---

## Abstract

Weak gravitational lensing requires solving a 2D Poisson equation $\nabla^2\psi=2\kappa$ to compute the lensing potential from observed convergence maps. We derive the finite element method (FEM) for this problem from first principles: starting with the strong form PDE, deriving the weak (variational) formulation, discretizing with piecewise linear basis functions, assembling the global stiffness matrix, and solving for nodal potentials. Interactive demonstrations show mesh generation, element assembly, and the complete solve pipeline. This framework provides the foundation for GPU-accelerated lensing reconstruction.

## 1. The Lensing Poisson Equation

### 1.1 Recap: From Einstein to Poisson

In [this blog post](https://www.adamfield.org/pages/blog.html?blog=Weak+Gravitational+Lensing%3A+From+the+Geodesic+Equation+to+Mass+Reconstruction), we derived how light deflection in the weak-field limit leads to the lensing potential $\psi(\theta)$ satisfying:

$$\nabla^2 \psi(\boldsymbol{\theta}) = 2\kappa(\boldsymbol{\theta}) \qquad\boldsymbol{(1)}$$

where:
- $\boldsymbol{\psi}$: Dimensionless lensing potential (related to projected Newtonian potential)
- $\boldsymbol{\kappa}$: Convergence (dimensionless surface density) = $\Sigma / \Sigma_{\text{crit}}$
- $\boldsymbol{\theta}$: Angular position on the sky (2D coordinates)

The deflection angle is the gradient:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \nabla\psi(\boldsymbol{\theta}) \qquad\boldsymbol{(2)}$$

And the shear (shape distortion) comes from second derivatives:

$$\gamma_1 = \frac{1}{2}(\psi_{,11} - \psi_{,22}), \quad \gamma_2 = \psi_{,12} \qquad\boldsymbol{(3)}$$

### 1.2 The Forward Problem

**Given**: Convergence field $\kappa(\theta)$ (mass distribution)  
**Find**: Lensing potential $\psi(\theta)$ satisfying equation (1)  
**Then**: Compute deflection $\alpha$ and shear $\gamma$ from derivatives of $\psi$

This is a **boundary value problem (BVP)** for an elliptic PDE. We need:
1. The domain $\Omega\subset\mathbb{R}^2$ (region of sky)
2. Boundary conditions on $\partial\Omega$ (edges of domain)

### 1.3 Boundary Conditions

For an isolated lens (e.g., galaxy cluster), physically $\psi\to 0$ as $|\theta|\to\infty$. We approximate this by:

**Dirichlet BC**: $\psi = 0$ on $\partial\Omega$ (boundary far from lens)

Or sometimes:

**Neumann BC**: $\partial\psi/\partial n = 0$ on $\partial\Omega$ (zero-flux, appropriate if $\partial\Omega$ is at large radius)

For computational domains, we typically use Dirichlet (easier to implement).

### 1.4 Why Not Just Use FFT?

The Fast Fourier Transform (FFT) can solve the Poisson equation on a **regular grid** with **periodic boundary conditions**. Indeed, Kaiser-Squires inversion uses this! However:

**FFT limitations**:
- Requires rectangular domain with uniform spacing
- Periodic BCs (not always appropriate)
- Hard to handle irregular boundaries (e.g., survey footprints)
- Cannot adapt resolution locally

**FEM advantages**:
- Arbitrary domain shapes (match survey geometry)
- Local mesh refinement (resolve mass peaks)
- Flexible boundary conditions
- Natural extension to curved surfaces (spherical sky)

For GPU acceleration and adaptive meshing, FEM is ideal.

## 2. The Weak Formulation

### 2.1 From Strong to Weak Form

The **strong form** (equation 1) requires $\psi$ to be twice differentiable. The **weak form** relaxes this: we only need $\psi$ and its first derivatives to be square-integrable.

**Key idea**: Multiply equation (1) by a **test function** $v(\theta)$ and integrate:

$$\int_\Omega \nabla^2\psi \, v \, d^2\theta = \int_\Omega 2\kappa \, v \, d^2\theta \qquad\boldsymbol{(4)}$$

### 2.2 Integration by Parts

Use the **vector identity**:

$$\nabla \cdot (\nabla\psi \, v) = \nabla^2\psi \, v + \nabla\psi \cdot \nabla v \qquad\boldsymbol{(5)}$$

Rearranging:

$$\nabla^2\psi \, v = \nabla \cdot (\nabla\psi \, v) - \nabla\psi \cdot \nabla v$$

Substitute into (4):

$$\int_\Omega [\nabla \cdot (\nabla\psi \, v) - \nabla\psi \cdot \nabla v] d^2\theta = \int_\Omega 2\kappa \, v \, d^2\theta$$

### 2.3 Apply Divergence Theorem

The divergence theorem states:

$$\int_\Omega \nabla \cdot \mathbf{F} \, d^2\theta = \oint_{\partial\Omega} \mathbf{F} \cdot \mathbf{n} \, ds \qquad\boldsymbol{(6)}$$

where $\mathbf{n}$ is the outward normal. Applying to $\mathbf{F} = \nabla\psi \, v$:

$$\int_\Omega \nabla \cdot (\nabla\psi \, v) d^2\theta = \oint_{\partial\Omega} v \frac{\partial\psi}{\partial n} ds$$

For **Dirichlet BC** ($\psi = 0$ on $\partial\Omega$), we choose test functions $\boldsymbol{v} \boldsymbol{=} \boldsymbol{0}$ **on** $\boldsymbol{\partial\Omega}$, so the boundary term vanishes.

### 2.4 The Weak Formulation

Putting it all together:

$$\boxed{\int_\Omega \nabla\psi \cdot \nabla v \, d^2\theta = \int_\Omega 2\kappa \, v \, d^2\theta} \qquad\boldsymbol{(7)}$$

This is the **weak form** of the Poisson equation.

**Mathematical statement**: Find $\psi \in H^1_0(\Omega)$ such that equation (7) holds for all test functions $v \in H^1_0(\Omega)$.

Where $H^1_0(\Omega)$ is the **Sobolev space** of functions with square-integrable first derivatives that vanish on the boundary.

### 2.5 Why This Is Better

The weak form has several advantages:
1. **Only first derivatives** appear (less smoothness required)
2. **Symmetric bilinear form** $a(\psi,v) = \int\nabla\psi\cdot\nabla v$ (nice mathematical properties)
3. **Natural for FEM** (piecewise polynomials fit naturally)
4. **Boundary conditions** automatically incorporated

## 3. Finite Element Discretization

### 3.1 Mesh and Elements

We **discretize** the domain $\Omega$ into **elements**:

$$\Omega \approx \Omega_h = \bigcup_{e=1}^{N_e} T_e \qquad\boldsymbol{(8)}$$

where each $T_e$ is a **triangle** (2D) or **quadrilateral**.

**Triangular elements** are most common:
- Handle arbitrary geometries
- Easy refinement (split triangles)
- Simple basis functions

The mesh has:
- $N_v$ vertices (nodes): $\{\theta_i\}_{i=1}^{N_v}$
- $N_e$ elements (triangles): $\{T_e\}_{e=1}^{N_e}$
- Connectivity: which nodes belong to which elements

### 3.2 Basis Functions

We approximate $\psi$ as a **linear combination** of basis functions:

$$\psi_h(\boldsymbol{\theta}) = \sum_{i=1}^{N_v} \psi_i \phi_i(\boldsymbol{\theta}) \qquad\boldsymbol{(9)}$$

where:
- $\psi_i$: Nodal values (unknowns)
- $\phi_i(\theta)$: Basis functions ("hat functions")

**Piecewise linear basis** (P1 FEM): On each triangle $T_e$, $\phi_i$ is linear. Globally:

$$\phi_i(\boldsymbol{\theta}_j) = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases} \qquad\boldsymbol{(10)}$$

So $\psi_i = \psi_h(\theta_i)$ — the values at nodes.

### 3.3 Shape Functions on the Reference Triangle

For a triangle with vertices $\boldsymbol{\theta}^{(1)}$, $\boldsymbol{\theta}^{(2)}$, $\boldsymbol{\theta}^{(3)}$, the **barycentric coordinates** $(\lambda_1, \lambda_2, \lambda_3)$ satisfy:

$$\boldsymbol{\theta} = \lambda_1 \boldsymbol{\theta}^{(1)} + \lambda_2 \boldsymbol{\theta}^{(2)} + \lambda_3 \boldsymbol{\theta}^{(3)} \qquad\boldsymbol{(11)}$$

with $\lambda_1 + \lambda_2 + \lambda_3 = 1$. The shape functions are simply:

$$N_1 = \lambda_1, \quad N_2 = \lambda_2, \quad N_3 = \lambda_3 \qquad\boldsymbol{(12)}$$

On element $T_e$, we have:

$$\psi|_{T_e} = \psi_1^e N_1 + \psi_2^e N_2 + \psi_3^e N_3 \qquad\boldsymbol{(13)}$$

### 3.4 Interactive Demo: Mesh and Basis Functions

Let's visualize how a 2D mesh is constructed and how basis functions work:

[codeContainer](/weak_lensing_poisson/web/js/fem-mesh-demo.js)

## 4. Element-Level Computation

### 4.1 Element Stiffness Matrix

For element $T_e$, define the **element stiffness matrix**:

$$K_{ij}^e = \int_{T_e} \nabla N_i \cdot \nabla N_j \, d^2\theta \qquad\boldsymbol{(14)}$$

This is a $3\times 3$ matrix (3 nodes per triangle).

### 4.2 Computing Gradients

For a linear function on a triangle:

$$\nabla N_i = \text{const on } T_e$$

We can compute these explicitly. Let:

$$\boldsymbol{\theta}^{(i)} = (x_i, y_i)$$

The area of the triangle is:

$$A_e = \frac{1}{2}|(x_2 - x_1)(y_3 - y_1) - (x_3 - x_1)(y_2 - y_1)| \qquad\boldsymbol{(15)}$$

The gradients are:

$$\nabla N_1 = \frac{1}{2A_e}\begin{pmatrix} y_2 - y_3 \\ x_3 - x_2 \end{pmatrix} \qquad\boldsymbol{(16)}$$

$$\nabla N_2 = \frac{1}{2A_e}\begin{pmatrix} y_3 - y_1 \\ x_1 - x_3 \end{pmatrix} \qquad\boldsymbol{(17)}$$

$$\nabla N_3 = \frac{1}{2A_e}\begin{pmatrix} y_1 - y_2 \\ x_2 - x_1 \end{pmatrix} \qquad\boldsymbol{(18)}$$

### 4.3 Explicit Element Stiffness

Since gradients are constant:

$$K_{ij}^e = \nabla N_i \cdot \nabla N_j \cdot A_e \qquad\boldsymbol{(19)}$$

For example:

$$K_{11}^e = \frac{1}{4A_e}[(y_2 - y_3)^2 + (x_3 - x_2)^2] \qquad\boldsymbol{(20)}$$

And similarly for all 9 entries. This can be vectorized easily on GPU!

### 4.4 Element Load Vector

Similarly, define:

$$f_i^e = \int_{T_e} 2\kappa \, N_i \, d^2\theta \qquad\boldsymbol{(21)}$$

If $\kappa$ is **constant** on $T_e$ (piecewise constant):

$$f_i^e = 2\kappa_e \int_{T_e} N_i \, d^2\theta = 2\kappa_e \cdot \frac{A_e}{3} \qquad\boldsymbol{(22)}$$

The factor $1/3$ comes from $\int N_i = A_e/3$ for linear shape functions.

If $\kappa$ is **nodal** (given at vertices):

$$\kappa|_{T_e} = \kappa_1^e N_1 + \kappa_2^e N_2 + \kappa_3^e N_3$$

Then:

$$f_i^e = 2\sum_{j=1}^3 \kappa_j^e \int_{T_e} N_i N_j \, d^2\theta \qquad\boldsymbol{(23)}$$

Using the formula:

$$\int_{T_e} N_i N_j \, d^2\theta = \frac{A_e}{12}(1 + \delta_{ij}) \qquad\boldsymbol{(24)}$$

We get:

$$f_i^e = 2 \cdot \frac{A_e}{12}(2\kappa_i^e + \kappa_j^e + \kappa_k^e) \qquad (j,k \neq i) \qquad\boldsymbol{(25)}$$

## 5. Global Assembly

### 5.1 Assembly Process

The global stiffness matrix $K$ and load vector $\mathbf{f}$ are built by **summing element contributions**:

$$K_{IJ} = \sum_{e=1}^{N_e} K_{ij}^e \quad \text{where node } i \text{ in element } e \text{ maps to global index } I \qquad\boldsymbol{(26)}$$

This is called **assembly**.

**Algorithm**:
```
Initialize K = 0 (N_v × N_v sparse matrix)
Initialize f = 0 (N_v vector)

For each element e = 1 to N_e:
    Compute K^e (3×3 matrix)
    Compute f^e (3 vector)
    
    For i = 1 to 3:  # local node index
        I = global_index(e, i)
        
        For j = 1 to 3:
            J = global_index(e, j)
            K[I,J] += K^e[i,j]
        
        f[I] += f^e[i]
```

### 5.2 Sparsity Structure

The matrix $K$ is **sparse**: node $I$ is only connected to nodes that share an element with $I$.

For a typical 2D mesh:
- Each node has $\sim 6$ neighbors
- $K$ has $\sim 7N_v$ nonzeros (vs. $N_v^2$ for dense)

**Storage**: Use **Compressed Sparse Row (CSR)** format or **COO** (Coordinate format) for GPU.

### 5.3 Interactive Demo: Element Assembly

Let's visualize how individual element matrices combine into the global system:

[codeContainer](/weak_lensing_poisson/web/js/fem-assembly-demo.js)

## 6. Boundary Conditions

### 6.1 Dirichlet Boundary Conditions

For $\boldsymbol{\psi} \boldsymbol{=} \boldsymbol{0}$ **on** $\boldsymbol{\partial\Omega}$, we need to enforce $\psi_i = 0$ for boundary nodes.

**Method 1: Elimination**
- Remove rows/columns for boundary nodes from $K$
- Solve smaller system for interior nodes only

**Method 2: Penalty Method**
- Add large number to diagonal: $K[i,i] \mathrel{+}= 10^{15}$
- Set $f[i] = 0$
- Effectively enforces $\psi_i \approx 0$

**Method 3: Direct Substitution**
- Set $K[i,:] = 0$, $K[i,i] = 1$, $f[i] = 0$
- Explicitly enforces $\psi_i = 0$

We'll use **Method 3** (cleanest for coding).

### 6.2 Neumann Boundary Conditions

For $\partial\psi/\partial n = 0$, we add a boundary integral term. But the weak form (7) with homogeneous Neumann BC requires **no modification** — the boundary term automatically vanishes!

This is called a **natural boundary condition**.

### 6.3 Mixed Boundary Conditions

In practice, we might have:
- Dirichlet on some parts of $\partial\Omega$
- Neumann on others

Handle by applying Dirichlet modifications only to those nodes.

## 7. Solving the Linear System

### 7.1 The Discrete Problem

After assembly and BC application, we have:

$$\boxed{K\boldsymbol{\psi} = \mathbf{f}} \qquad\boldsymbol{(27)}$$

where:
- $K$: $N_v \times N_v$ sparse symmetric positive-definite matrix
- $\boldsymbol{\psi}$: $N_v$ vector of unknowns (nodal potentials)
- $\mathbf{f}$: $N_v$ vector (source term)

This is a **large sparse linear system**.

### 7.2 Direct Solvers

**LU decomposition** or **Cholesky** (for symmetric positive-definite):

$$K = LL^T \quad \Rightarrow \quad L(L^T\boldsymbol{\psi}) = \mathbf{f}$$

Solve in two steps:
1. Forward: $Ly = f$
2. Backward: $L^T \psi = y$

**Cost**: $O(N_v^{3/2})$ for sparse Cholesky with good ordering  
**Memory**: Stores $L$ (can be large)

**Good for**: Moderate-sized problems ($\sim 10^4$ nodes)

### 7.3 Iterative Solvers

**Conjugate Gradient (CG)**: Exploits symmetry and positive-definiteness

**Algorithm**:
```
Initialize ψ₀ = 0
r₀ = f - Kψ₀ = f
p₀ = r₀

For k = 0, 1, 2, ...
    α_k = (r_k · r_k) / (p_k · Kp_k)
    ψ_{k+1} = ψ_k + α_k p_k
    r_{k+1} = r_k - α_k Kp_k
    β_k = (r_{k+1} · r_{k+1}) / (r_k · r_k)
    p_{k+1} = r_{k+1} + β_k p_k
    
    If ||r_{k+1}|| < tol: break
```

**Cost**: $O(\text{iterations} \times N_v \times \text{nnz})$ where nnz is nonzeros in $K$  
**Memory**: $O(N_v)$ — only store vectors

**Convergence**: Typically $O(\sqrt{\kappa(K)})$ iterations where $\kappa$ is condition number

### 7.4 Preconditioning

To accelerate CG, use a **preconditioner** $M \approx K^{-1}$:

**Preconditioned CG (PCG)**: Solve $M^{-1}K\psi = M^{-1}f$

Common choices:
- **Jacobi**: $M = \text{diag}(K)$ (simplest, modest improvement)
- **Incomplete Cholesky (IC)**: $M = \tilde{L}\tilde{L}^T$ where $\tilde{L}$ is approximate Cholesky
- **Algebraic Multigrid (AMG)**: Hierarchical solver (best for large problems)

For GPU, **Jacobi** is easiest to implement.

### 7.5 GPU Implementation

**JAX/Flax** provides:
```python
from jax.scipy.sparse.linalg import cg

psi, info = cg(K, f, tol=1e-6, maxiter=1000)
```

For custom kernels, use **jax.lax.scan** to implement CG loop.

**Key optimization**: Matrix-vector product $Kp$ should be:
1. Sparse (don't store zeros)
2. Parallelized (each row independently)
3. Fused with other operations (reduce memory traffic)

## 8. Computing Derivatives: Deflection and Shear

### 8.1 Deflection Angle

Once we have nodal potentials $\{\psi_i\}$, we need:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \nabla\psi(\boldsymbol{\theta}) \qquad\boldsymbol{(28)}$$

On element $T_e$:

$$\nabla\psi|_{T_e} = \sum_{i=1}^3 \psi_i^e \nabla N_i = \text{const} \qquad\boldsymbol{(29)}$$

So deflection is **piecewise constant** (discontinuous across elements).

**At nodes**: Average from neighboring elements:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}_I) = \frac{1}{|\mathcal{E}_I|}\sum_{e \in \mathcal{E}_I} \nabla\psi|_{T_e} \qquad\boldsymbol{(30)}$$

where $\mathcal{E}_I$ is the set of elements containing node $I$.

### 8.2 Shear Components

Shear requires **second derivatives**:

$$\gamma_1 = \frac{1}{2}(\psi_{,11} - \psi_{,22}), \quad \gamma_2 = \psi_{,12}$$

For **piecewise linear** $\psi$, second derivatives are **zero within elements** and **singular at edges**.

**Solution**: Use **recovery techniques**:

**Option 1: L² Projection**  
Solve for smooth approximation:

$$\int_\Omega \gamma_1^h \, v \, d^2\theta = \int_\Omega \psi_{,11} \, v \, d^2\theta$$

This gives a mass matrix problem.

**Option 2: Superconvergent Patch Recovery (SPR)**  
Fit polynomials locally using nodal values from patch of elements.

**Option 3: Higher-Order Elements**  
Use P2 (quadratic) elements $\to$ linear gradients $\to$ constant second derivatives

For our purposes, **Option 3** is cleanest: use quadratic basis functions.

### 8.3 Quadratic (P2) Elements

On a triangle with 6 nodes (3 vertices + 3 midpoints), we have quadratic shape functions.

**Advantage**: Smooth first derivatives, piecewise linear second derivatives

**Cost**: Larger system (more nodes), but better accuracy

For lensing, P2 elements are worth it for shear computation.

## 9. Complete FEM Pipeline

### 9.1 Algorithm Summary

**Input**: Convergence field $\kappa(\theta)$, domain $\Omega$, boundary conditions  
**Output**: Lensing potential $\psi$, deflection $\alpha$, shear $\gamma$

**Steps**:
1. **Mesh generation**: Triangulate $\Omega$ (adaptive refinement near mass peaks)
2. **Assembly**: Compute $K$ and $\mathbf{f}$ from all elements
3. **Apply BC**: Modify $K$, $\mathbf{f}$ for boundary conditions
4. **Solve**: Use CG to solve $K\psi = \mathbf{f}$
5. **Post-process**: Compute $\alpha = \nabla\psi$, $\gamma$ from second derivatives
6. **Visualize**: Plot $\psi$, $\alpha$, $\gamma$; validate against analytic solutions

### 9.2 JAX Implementation Outline

```python
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

def assemble_stiffness(nodes, elements):
    """Assemble global stiffness matrix K"""
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    
    # Preallocate (COO format)
    rows = []
    cols = []
    vals = []
    
    for e in range(n_elements):
        # Get element nodes
        idx = elements[e]  # shape (3,)
        x = nodes[idx]     # shape (3, 2)
        
        # Compute element stiffness
        K_e = element_stiffness(x)
        
        # Add to global arrays
        for i in range(3):
            for j in range(3):
                rows.append(idx[i])
                cols.append(idx[j])
                vals.append(K_e[i, j])
    
    # Convert to sparse matrix
    K = sparse_matrix(rows, cols, vals, shape=(n_nodes, n_nodes))
    return K

def element_stiffness(x):
    """Compute 3x3 stiffness for one triangle"""
    # x: (3, 2) array of vertex coordinates
    
    # Compute area
    x1, x2, x3 = x[0], x[1], x[2]
    A = 0.5 * jnp.abs((x2[0] - x1[0]) * (x3[1] - x1[1]) - 
                       (x3[0] - x1[0]) * (x2[1] - x1[1]))
    
    # Gradients of shape functions
    grad_N = jnp.array([
        [x2[1] - x3[1], x3[0] - x2[0]],
        [x3[1] - x1[1], x1[0] - x3[0]],
        [x1[1] - x2[1], x2[0] - x1[0]]
    ]) / (2 * A)
    
    # K_ij = grad_N_i · grad_N_j * A
    K_e = jnp.einsum('ia,ja->ij', grad_N, grad_N) * A
    return K_e

def solve_lensing(kappa, nodes, elements, boundary_nodes):
    """Main solver"""
    # Assemble
    K = assemble_stiffness(nodes, elements)
    f = assemble_load(kappa, nodes, elements)
    
    # Apply Dirichlet BC
    K, f = apply_bc(K, f, boundary_nodes)
    
    # Solve
    psi, info = cg(K, f, tol=1e-6)
    
    # Compute derivatives
    alpha = compute_deflection(psi, nodes, elements)
    gamma = compute_shear(psi, nodes, elements)
    
    return psi, alpha, gamma
```

### 9.3 Interactive Demo: Complete Solver

Let's run a full FEM solve for a simple lens configuration:

[codeContainer](/weak_lensing_poisson/web/js/fem-solver-demo.js)

## 10. Validation: Analytic Solutions

### 10.1 Point Mass Lens

For a point mass $M$ at origin, the analytic solution is:

$$\psi(\theta) = \theta_E^2 \ln|\boldsymbol{\theta}|/\theta_0 \qquad\boldsymbol{(31)}$$

where $\theta_E$ is the Einstein radius and $\theta_0$ is a reference scale.

The deflection is:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \frac{\theta_E^2}{|\boldsymbol{\theta}|^2}\boldsymbol{\theta} \qquad\boldsymbol{(32)}$$

**Test**: Set $\kappa = M \delta(\theta)$, solve FEM, compare $\alpha$ to equation (32).

**Expected error**: $O(h^2)$ where $h$ is mesh size (for P1 elements)

### 10.2 Singular Isothermal Sphere (SIS)

The SIS has:

$$\kappa(\theta) = \frac{\theta_E}{2|\boldsymbol{\theta}|} \qquad\boldsymbol{(33)}$$

$$\alpha(\theta) = \theta_E \qquad\boldsymbol{(34)}$$

Constant deflection! Easy to verify.

### 10.3 Convergence Study

**Mesh refinement test**:
1. Solve on mesh with spacing $h$
2. Solve on mesh with spacing $h/2$
3. Compute error: $\|\psi_h - \psi_{h/2}\|$

**Expected**: Error $\propto h^p$ where $p$ is convergence rate
- P1 elements: $p = 2$ (quadratic convergence)
- P2 elements: $p = 3$ (cubic convergence)

Plot $\log(\text{error})$ vs $\log(h)$ $\to$ slope gives $p$.

## 11. Adaptive Mesh Refinement

### 11.1 Why Adapt?

Lensing features span multiple scales:
- Large-scale cluster (Mpc)
- Individual galaxies (kpc)
- Strong lensing arcs (arcsec)

Uniform mesh is wasteful. **Adaptive refinement** concentrates resolution where needed.

### 11.2 Error Estimators

**Residual-based**: Compute $\|\nabla^2\psi - 2\kappa\|$ on each element. Refine where large.

**Gradient recovery**: Compare FEM gradient to smoothed gradient. Refine where they differ.

**A posteriori**: After solving, compute local error indicators, then refine mesh and re-solve.

### 11.3 Refinement Strategy

**Red-Green refinement** for triangles:
- **Red**: Split each edge $\to$ 4 child triangles
- **Green**: Add one edge to maintain conformity

**Hanging nodes**: Allowed in some codes (simplifies data structure)

**JAX-AMR approach**: Pre-define multiple fixed meshes (coarse, medium, fine). Switch between them without dynamic allocation (JIT-friendly).

### 11.4 Multi-Level Solvers

For large problems, use **multigrid**:
1. Solve on coarse mesh
2. Interpolate to fine mesh
3. Solve correction on fine mesh
4. Iterate

**V-cycle** or **W-cycle** strategies.

**Advantage**: $O(N)$ complexity (vs. $O(N^{3/2})$ for sparse direct)

## 12. Extensions and Applications

### 12.1 3D Lensing (Multi-Plane)

For sources at different redshifts, we have **multiple lenses** at different distances.

**Approach**:
1. Solve 2D Poisson at each lens plane
2. Ray-trace through each plane sequentially
3. Cumulative deflection

Still uses 2D FEM at each plane!

### 12.2 Non-Linear Lensing

Strong lensing ($\kappa \sim 1$) requires solving the **full lens equation** non-linearly.

**Newton-Raphson**:
1. Guess $\psi_0$
2. Linearize: $K\delta\psi = \mathbf{f} - K\psi_k$
3. Update: $\psi_{k+1} = \psi_k + \delta\psi$
4. Iterate until convergence

FEM handles this naturally ($K$ reassembled each iteration).

### 12.3 Bayesian Reconstruction

**Inverse problem**: Given observed shear $\gamma_{\text{obs}}$, infer $\kappa$ (and thus $\psi$).

**Maximum likelihood**:

$$\text{minimize} \quad \frac{1}{2}\|L(\gamma[\psi]) - \gamma_{\text{obs}}\|^2 + \lambda R(\psi)$$

where:
- $L$: Likelihood operator (compares shear)
- $R$: Regularization (smoothness prior)

**Gradient-based optimization**: JAX autodiff computes $\partial/\partial\psi_i$ automatically!

### 12.4 GPU Acceleration

**Speedups** from JAX GPU:
- Element assembly: $100\times$ (fully parallel)
- CG iterations: $50\times$ (sparse matvec + vector ops)
- Overall: $\sim 30$-$50\times$ for end-to-end pipeline

For $10^6$ nodes, solve in seconds on V100 (vs. minutes on CPU).

## 13. Connection to Neural Networks

### 13.1 FEM as a Neural Network Layer

The FEM solve $K\psi = \mathbf{f}$ can be viewed as:

$$\boldsymbol{\psi} = K^{-1}\mathbf{f} = \mathcal{F}(\kappa; \text{mesh}) \qquad\boldsymbol{(35)}$$

This is a **differentiable function** $\kappa \to \psi$!

**JAX autodiff**: Can compute $\partial\psi/\partial\kappa$ for optimization.

### 13.2 Hybrid NN-FEM Approaches

**Idea**: Use neural network to:
1. Predict initial guess $\psi_0$ from $\kappa$ (fast, approximate)
2. Refine with FEM solver (accurate)

**Training**: Minimize $\|\text{FEM}(\kappa) - \text{NN}(\kappa)\|$ on training set.

**Inference**: NN gives good starting point, CG converges in fewer iterations.

### 13.3 Learning-Based Mesh Adaptation

**Neural mesh refinement**:
- Train NN to predict where to refine given $\kappa$
- Use as error indicator instead of residual
- Faster than traditional estimators

Active research area!

## 14. Practical Considerations

### 14.1 Mesh Generation Tools

**Python libraries**:
- **meshio**: Read/write mesh formats
- **pygmsh**: Wrapper for Gmsh (powerful mesher)
- **Triangle**: Constrained Delaunay triangulation
- **CGAL**: Computational geometry (via Python bindings)

**For JAX**:
- Generate mesh externally, load as arrays
- Or use **jax-fem** library (provides meshing + solve)

### 14.2 Boundary Condition Subtleties

For **real surveys**:
- Boundaries are irregular (survey footprint)
- May have holes (masked regions)
- Require careful mesh construction

**Approach**:
1. Define polygon boundary
2. Constrained triangulation (Triangle/Gmsh)
3. Apply BC only on true boundary (not internal holes)

### 14.3 Numerical Stability

**Condition number**: $\kappa(K)$ grows with mesh refinement.

**Mitigation**:
- Use preconditioner (IC or AMG)
- Iterative refinement: Solve $K(\psi + \delta\psi) \approx \mathbf{f}$ iteratively
- Higher-order elements (better condition number)

**Monitor**: Check CG residual $\|K\psi - \mathbf{f}\| / \|\mathbf{f}\|$

## 15. Summary and Next Steps

### 15.1 What We've Accomplished

We derived the complete FEM pipeline for weak lensing:

1. **Weak formulation**: $\int\nabla\psi\cdot\nabla v = \int 2\kappa v$ (integration by parts)
2. **Discretization**: Piecewise linear (or quadratic) basis functions
3. **Element computation**: $K_{ij}^e = \int\nabla N_i\cdot\nabla N_j$ (explicitly computed)
4. **Assembly**: Sum element contributions $\to$ global $K$, $\mathbf{f}$
5. **Boundary conditions**: Modify $K$, $\mathbf{f}$ for Dirichlet BC
6. **Solve**: Conjugate Gradient (GPU-accelerated in JAX)
7. **Post-processing**: $\alpha = \nabla\psi$, $\gamma$ from second derivatives
8. **Validation**: Compare to analytic lens models

**Key equations**:

$$\nabla^2\psi = 2\kappa \quad \xrightarrow{\text{weak form}} \quad \int_\Omega \nabla\psi \cdot \nabla v = \int_\Omega 2\kappa v$$

$$\boldsymbol{\psi}_h = \sum_{i=1}^{N_v} \psi_i \phi_i \quad \Rightarrow \quad K_{IJ} = \sum_e K_{ij}^e$$

$$K\boldsymbol{\psi} = \mathbf{f} \quad \xrightarrow{\text{CG}} \quad \boldsymbol{\psi}$$

### 15.2 Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **FFT (Kaiser-Squires)** | Fast $O(N \log N)$, simple | Uniform grid, periodic BC, no adaptivity |
| **FEM** | Flexible geometry, adaptive, accurate BCs | More complex implementation |
| **Finite Difference** | Simple stencils | Regular grids, harder for complex domains |
| **Neural Networks** | Ultra-fast inference | Requires training data, less interpretable |

FEM shines for:
- Complex survey geometries
- Adaptive resolution
- Coupling with inverse problems

### 15.3 Implementation Roadmap

**Phase 1**: Basic solver
- Uniform triangular mesh
- P1 elements
- Direct solver (Cholesky)
- Validate on point mass

**Phase 2**: GPU acceleration
- JAX implementation
- Sparse matrix COO format
- CG solver
- Benchmark vs. CPU

**Phase 3**: Advanced features
- Adaptive mesh refinement
- P2 elements (for shear)
- Preconditioned CG
- Multi-lens planes

**Phase 4**: Integration
- Couple with shear measurement (ShearNet output $\to$ FEM input)
- Bayesian inversion
- Real data pipeline

### 15.4 Open Challenges

**Theoretical**:
- Optimal mesh adaptation strategies for lensing
- Error bounds for realistic mass distributions
- Convergence guarantees for non-linear lensing

**Computational**:
- Scaling to $10^9$ elements (full-sky surveys)
- Real-time reconstruction (for transient events)
- Uncertainty quantification (Bayesian posterior sampling)

**Astrophysical**:
- Baryonic effects on small scales (need high resolution)
- Multi-wavelength data fusion (X-ray, SZ, optical)
- Time-domain lensing (for moving lenses)

### 15.5 Closing Thoughts

The finite element method provides a mathematically rigorous and computationally efficient framework for solving the lensing Poisson equation. By leveraging GPU acceleration (JAX), adaptive meshing, and modern linear algebra, we can reconstruct mass distributions at unprecedented resolution and speed.

This methodology forms the computational backbone of my weak lensing research, connecting differential geometry (geodesics), functional analysis (weak formulation), numerical analysis (FEM discretization), and machine learning (ShearNet shear estimation) into a unified pipeline.

## Appendix A: Barycentric Coordinates

For a triangle with vertices $\boldsymbol{\theta}^{(1)}$, $\boldsymbol{\theta}^{(2)}$, $\boldsymbol{\theta}^{(3)}$, any point $\boldsymbol{\theta}$ inside can be written:

$$\boldsymbol{\theta} = \lambda_1\boldsymbol{\theta}^{(1)} + \lambda_2\boldsymbol{\theta}^{(2)} + \lambda_3\boldsymbol{\theta}^{(3)}$$

with $\lambda_1 + \lambda_2 + \lambda_3 = 1$ and $\lambda_i \geq 0$.

**Geometric interpretation**: $\lambda_i$ is the ratio of the area of the sub-triangle opposite vertex $i$ to the total area.

**Computation**: Given $\boldsymbol{\theta} = (x, y)$, solve:

$$\begin{pmatrix} x \\ y \\ 1 \end{pmatrix} = \begin{pmatrix} x_1 & x_2 & x_3 \\ y_1 & y_2 & y_3 \\ 1 & 1 & 1 \end{pmatrix}\begin{pmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \end{pmatrix}$$

Invert ($3\times 3$ matrix) to get $\lambda$ from $\theta$.

## Appendix B: Integration Formulas

For a triangle $T$ with area $A$:

$$\int_T 1 \, d^2\theta = A$$

$$\int_T N_i \, d^2\theta = \frac{A}{3}$$

$$\int_T N_i N_j \, d^2\theta = \frac{A}{12}(1 + \delta_{ij})$$

These follow from barycentric coordinate integration.

## Appendix C: Sparse Matrix Formats

**COO (Coordinate)**: Store (row, col, value) triplets
- Simple to construct
- Not efficient for operations

**CSR (Compressed Sparse Row)**: Store row_ptr, col_ind, values
- Efficient for matrix-vector product
- Harder to modify

**JAX**: Use `jax.experimental.sparse` or convert to dense for small problems.

For GPU, **COO is fine** for assembly; convert to CSR for CG solve.

## References

1. **Brenner, S. C., & Scott, L. R. (2008).** *The Mathematical Theory of Finite Element Methods* (3rd ed.). Springer. — The definitive FEM textbook.

2. **Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2013).** *The Finite Element Method: Its Basis and Fundamentals* (7th ed.). Butterworth-Heinemann.

3. **Ern, A., & Guermond, J.-L. (2004).** *Theory and Practice of Finite Elements*. Springer.

4. **Logg, A., Mardal, K.-A., & Wells, G. (2012).** *Automated Solution of Differential Equations by the Finite Element Method*. Springer. — FEniCS book.

5. **Bartelmann, M., & Schneider, P. (2001).** "Weak gravitational lensing." *Physics Reports*, 340(4-5), 291-472. [arXiv:astro-ph/9912508](https://arxiv.org/abs/astro-ph/9912508)

6. **Seitz, S., Schneider, P., & Ehlers, J. (1994).** "Light propagation in arbitrary spacetimes and the gravitational lens approximation." *Classical and Quantum Gravity*, 11(9), 2345.

7. **JAX-FEM Documentation.** [https://github.com/deepmodeling/jax-fem](https://github.com/deepmodeling/jax-fem)

8. **Bradbury, J., et al. (2018).** *JAX: Composable transformations of Python+NumPy programs*. [http://github.com/google/jax](http://github.com/google/jax)

9. **Shewchuk, J. R. (1996).** "Triangle: Engineering a 2D Quality Mesh Generator and Delaunay Triangulator." [https://www.cs.cmu.edu/~quake/triangle.html](https://www.cs.cmu.edu/~quake/triangle.html)

10. **Saad, Y. (2003).** *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. — CG and preconditioning.

---

*Questions or corrections? Email me at adfield@wpi.edu*