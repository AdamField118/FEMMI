# FEMMI — Finite Element Mass Map Inversion

**GPU-accelerated P3 finite element pipeline for weak gravitational lensing mass reconstruction.**

FEMMI solves the inverse problem of recovering a projected mass distribution κ (convergence) from observed weak gravitational shear γ. It replaces the classical Kaiser-Squires FFT estimator with a physics-informed MAP (maximum a posteriori) estimator built on cubic (P3) Lagrange finite elements, achieving a **63% reduction in L2 reconstruction error** on masked survey data.

---

## Table of Contents

- [Background](#background)
- [Method Overview](#method-overview)
- [Key Results](#key-results)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Running Examples](#running-examples)
- [Architecture](#architecture)
- [Why P3 Elements?](#why-p3-elements)
- [The Wiener Prior](#the-wiener-prior)
- [Implementation Notes](#implementation-notes)
- [References](#references)

---

## Background

In weak gravitational lensing, the projected mass distribution of a galaxy cluster (the convergence κ) distorts the apparent shapes of background galaxies. This distortion is measured as the complex shear field γ = γ₁ + iγ₂. Reconstructing κ from observed γ is a central problem in observational cosmology.

The standard approach — **Kaiser-Squires (KS)** — applies an FFT-based Fourier kernel to invert the lensing operator. It is fast and unbiased on full-sky data, but degrades significantly in the presence of:

- **Irregular survey boundaries** — the FFT assumes periodic boundary conditions, causing aliasing artifacts at edges
- **Masked regions** — bright stars, detector gaps, and satellite trails leave regions with no shear data
- **Noise** — shape measurement noise amplified near the survey boundary

FEMMI addresses all three by casting the reconstruction as a variational problem on a triangulated mesh, using the FEM Poisson solver to enforce the exact lensing PDE with natural Dirichlet boundary conditions, and a physics-informed Matérn prior to regularize against noise.

---

## Method Overview

The forward model is the weak lensing Poisson equation:

```
∇²ψ = 2κ    in Ω
ψ  = 0      on ∂Ω
```

where ψ is the lensing potential. The shear components follow from second derivatives:

```
γ₁ = ½(∂²ψ/∂x² − ∂²ψ/∂y²)
γ₂ = ∂²ψ/∂x∂y
```

**FEM discretization** turns this into a sparse linear system using P3 (cubic Lagrange) triangular elements. Three precomputed sparse operators drive the entire pipeline:

| Operator | Mathematical definition | Role |
|----------|------------------------|------|
| **K** | K[i,j] = ∫ ∇Nᵢ · ∇Nⱼ dA | Stiffness — discretizes Laplacian |
| **M** | M[i,j] = ∫ Nᵢ Nⱼ dA | Mass — L² inner product |
| **S1, S2** | Nodal-averaged element Hessians | Map ψ to (γ₁, γ₂) at nodes |

The complete forward chain is purely linear once operators are assembled:

```
κ  →  f = −2Mκ  →  ψ = K⁻¹f  →  γ₁ = S1ψ,  γ₂ = S2ψ
```

**MAP reconstruction** minimizes:

```
L(κ) = ‖γ_pred(κ) − γ_obs‖² + λ κᵀRκ
```

where R = M + ℓ²K is a Matérn-like Wiener prior with correlation length ℓ set to the lens scale.

For the full mathematical derivation — weak form, basis functions, Hessian transformation, adjoint gradient — see [`MATH.md`](MATH.md).

---

## Key Results

Benchmarked on a synthetic Gaussian lens (A=1, σ=0.5 arclengths) on a 20×20 P3 mesh over [−2.5, 2.5]² with 10% shear noise and a circular mask (r=0.6) near the centre:

| Method | L2 Error | Improvement over KS |
|--------|----------|---------------------|
| Kaiser-Squires | 0.098 | — |
| FEM-MAP (H1 prior, R=K) | 0.047 | +52% |
| **FEM-MAP (Wiener prior, ℓ=0.5)** | **0.036** | **+63%** |
| FEM-MAP (adaptive mesh only) | 0.056 | +43% |
| FEM-MAP (Wiener + adaptive) | 0.059 | +40% |

**The headline finding:** The Wiener prior alone is the dominant improvement. The combined Wiener + adaptive method underperforms Wiener-only because the adaptive mesh increases the effective masked node fraction from 4.3% to 17.7%, requiring re-tuned regularization. Physics-informed regularization outperforms geometric mesh refinement for this problem.

---

## Installation

### Prerequisites

- Python 3.11+
- conda (recommended)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AdamField118/FEMMI.git
cd FEMMI

# 2. Create and activate a conda environment
conda create --name femmi python=3.11 -y
conda activate femmi

# 3. Install FEMMI and all dependencies in editable mode
pip install -e .
```

**GPU support (optional):** After the standard install, replace the CPU JAX with a CUDA build:

```bash
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `jax` / `jaxlib` | ≥0.4.20 | Autodiff, JIT, GPU; 64-bit enabled throughout |
| `numpy` | ≥1.24 | Array operations |
| `scipy` | ≥1.10 | SuperLU sparse solve, L-BFGS-B, FFT |
| `matplotlib` | ≥3.7 | Visualization |

---

## Repository Structure

```
FEMMI/
├── README.md               ← You are here
├── MATH.md                 ← Full mathematical derivation with code citations
├── pyproject.toml          ← PEP 517/518 package metadata and dependencies
├── run.sh                  ← Helper: sets PYTHONPATH and runs a script
│
├── femmi/                  ← Main package
│   ├── __init__.py         ← Public API exports
│   │
│   │   ── Active pipeline (use these) ──────────────────────────
│   ├── operators.py        ← Assembles K, M, S1, S2; FEMOperators dataclass
│   ├── forward.py          ← Differentiable κ→γ via JAX custom_vjp
│   ├── inverse.py          ← MAP reconstructor, Wiener prior, KS reference
│   │
│   │   ── Low-level FEM (called by operators.py) ───────────────
│   ├── p3_shape_functions.py   ← 10-node cubic Lagrange basis functions (JAX)
│   ├── p3_assembly.py          ← Element Kₑ, Mₑ, Fₑ; Dunavant quadrature
│   ├── p3_mesh_generator.py    ← Structured + adaptive P3 mesh generation
│   │
│   │   ── Legacy P1 modules (kept for reference/comparison) ────
│   ├── fem_solver.py           ← P1 solver, analytic lens classes, Mesh dataclass
│   ├── mesh_generator.py       ← P1 mesh generation
│   └── autodiff_integration.py ← P1-era autodiff (superseded by forward.py)
│
├── tests/
│   ├── test_pipeline.py        ← Integration test: 5-step κ→γ→κ round-trip
│   ├── test_convergence_p3.py  ← P3 convergence study (expects O(h⁴))
│   └── test_convergence_p1.py  ← P1 convergence study (expects O(h²))
│
└── examples/
    ├── demo_p3_pipeline.py     ← Full forward model demo (κ, ψ, α, γ₁, γ₂)
    └── cluster_example.py      ← Single/multi-cluster P1 legacy example
```

### Module dependency graph

```
p3_shape_functions.py ←──┐
p3_assembly.py        ←──┤
p3_mesh_generator.py  ←──┴── operators.py ←── forward.py ←── inverse.py
                                  │
                             fem_solver.py  (Mesh dataclass only)
```

---

## Quick Start

### Forward model: κ → (γ₁, γ₂)

```python
import numpy as np
from femmi import build_operators

# Build P3 FEM operators on a 20×20 mesh over [-2.5, 2.5]²
ops = build_operators(20, 20)

# Gaussian lens centred at the origin
nodes = np.array(ops.mesh.nodes)
kappa = np.exp(-(nodes[:,0]**2 + nodes[:,1]**2) / (2 * 0.5**2))

# One sparse LU solve + two sparse matrix-vector products
gamma1, gamma2 = ops.forward(kappa)
print(f"max|γ₁| = {np.abs(gamma1).max():.4f}")
```

### MAP reconstruction: (γ₁, γ₂) → κ

```python
import numpy as np
from femmi import build_operators, DifferentiableForward, MAPReconstructor

ops  = build_operators(20, 20)
fwd  = DifferentiableForward(ops, lam_reg=2e-2)
rec  = MAPReconstructor(fwd, wiener_length=0.5)   # ℓ = σ_lens

kappa_map, result = rec.reconstruct(gamma1_obs, gamma2_obs)
print(f"Converged: {result.converged}  |  iterations: {result.n_iter}")
```

### Full benchmark against Kaiser-Squires

```python
from femmi import run_comparison

kappa_map, kappa_ks, kappa_true, result = run_comparison(
    nx            = 20,
    noise_level   = 0.10,
    lam_reg       = 2e-2,
    wiener_length = 0.5,
    apply_mask    = True,
    mask_radius   = 0.6,
)
# Prints L2 errors and saves map_reconstruction.png
```

### Adaptive mesh near a mask boundary

```python
from femmi import build_operators_adaptive

ops = build_operators_adaptive(
    20, 20,
    mask_center   = (0.0, 0.0),
    mask_radius   = 0.5,
    refine_factor = 3,        # 3× finer resolution in an annular band
)
```

---

## Running Tests

```bash
# Run all tests with pytest
pytest tests/ -v

# Or use run.sh (sets PYTHONPATH automatically, no install needed)
bash run.sh tests/test_pipeline.py
bash run.sh tests/test_convergence_p3.py
bash run.sh tests/test_convergence_p1.py
```

### Test descriptions

**`test_pipeline.py`** — Integration test with 5 subtests:

1. **Operator assembly** — K, M, S1, S2 have correct shapes and nonzero entries
2. **Forward pass** — ψ and γ are finite and within an order of magnitude of the analytic reference
3. **Gradient validation** — autodiff ∂L/∂κ matches central-difference finite differences to < 1e-4 relative error at 8 randomly chosen interior nodes
4. **Noiseless MAP** — reconstructed κ from exact FEM-generated γ achieves L2 error < 0.15
5. **Noisy MAP vs KS** — FEM-MAP L2 error stays within 1.5× of KS at 10% noise

**`test_convergence_p3.py`** — P3 convergence study on three manufactured solutions:

| Solution | Domain | κ (source) |
|----------|--------|-----------|
| sin(πx)sin(πy) | [0,1]² | −π²sin(πx)sin(πy) |
| (x²−1)(y²−1) | [−1,1]² | x²+y²−2 |
| x(1−x)y(1−y) | [0,1]² | −y(1−y)−x(1−x) |

All three should achieve average L2 convergence rate ≥ 3.5. Expected output:

```
Average L² convergence rate: 3.93  (expected ~4.0 for P3)
✅ O(h⁴) confirmed
```

---

## Running Examples

```bash
# Full P3 forward model demo — generates 3 PNG figures
bash run.sh examples/demo_p3_pipeline.py

# P1 cluster example (legacy)
bash run.sh examples/cluster_example.py
```

`demo_p3_pipeline.py` produces:

- `p3_pipeline_gaussian.png` — 6-panel: κ, ψ, |α|, γ₁, γ₂, |γ| for a Gaussian cluster
- `p3_shear_validation.png` — FEM γ₁, γ₂ vs. analytic Gaussian reference with residuals
- `p3_two_cluster.png` — Same 6 panels for a two-cluster system

---

## Architecture

### Operator assembly (`femmi/operators.py`)

`build_operators` assembles four sparse matrices once for a given mesh, then caches a SuperLU factorization of K:

```
Mesh generation
    │
    ├── K assembly  (element stiffness, Dunavant order-5 quadrature)
    │       └── Dirichlet BC enforced: identity rows at boundary nodes (L237)
    │
    ├── M assembly  (element mass, same quadrature)
    │       └── Boundary rows zeroed (L247–L248)
    │
    ├── H_ref  (10×10×2×2 reference Hessians, JAX jacfwd∘jacrev, L70–L78)
    │
    ├── S1, S2 assembly  (nodal-averaged physical Hessians, L108–L132)
    │
    └── SuperLU factorization of K  →  K_lu  (L269, reused every solve)
```

All four matrices are stored in a `FEMOperators` dataclass (L139–L186). Every forward evaluation of κ → (γ₁, γ₂) costs only:

- 1 sparse triangular solve using the cached LU (dominant cost, O(n) for structured mesh)
- 3 sparse matrix-vector products (M, S1, S2)

### Differentiable forward model (`femmi/forward.py`)

`DifferentiableForward` wraps the scipy sparse operations in `jax.pure_callback` closures with manually defined `custom_vjp` rules (L20–L71). This makes the full κ → γ pipeline differentiable through JAX's autodiff system while keeping the actual computation in scipy (which supports 64-bit sparse LU). The key primitives are:

- `fem_solve(b)` at L30–L42: implements K⁻¹b with VJP rule K⁻ᵀg = K⁻¹g (K symmetric)
- `matvec(x)` at L60–L70: implements Ax with VJP rule Aᵀg

### MAP solver (`femmi/inverse.py`)

`MAPReconstructor` uses scipy's L-BFGS-B with an explicit **numpy adjoint gradient** computed without JAX JIT, for maximum compatibility with the SuperLU solver. The adjoint gradient (L144–L149) is:

```
dL/dκ = −4 Mᵀ K⁻¹(S1ᵀr₁ + S2ᵀr₂) + 2λ Rκ
```

where r₁, r₂ are the shear residuals and R is the regularization matrix. See `MATH.md` §7.3 for the full chain-rule derivation.

---

## Why P3 Elements?

Shear requires second derivatives of ψ. The polynomial degree of those derivatives determines the convergence rate:

| Element | ψ convergence (L2) | γ convergence (L2) |
|---------|-------------------|-------------------|
| P1 (linear) | O(h²) | ≡ 0 — second derivatives identically zero |
| P2 (quadratic) | O(h³) | O(h⁰) — piecewise constant, no convergence |
| **P3 (cubic)** | **O(h⁴)** | **O(h²)** |

P1 and P2 are fundamentally insufficient for shear computation. P3 is the minimum viable element order — not a performance choice but a mathematical necessity.

The P3 shear operators use a precomputed table of 10×10×2×2 reference Hessians (`_build_ref_hessians`, `operators.py` L70–L78) differentiated via JAX forward-over-reverse autodiff, then transformed to physical coordinates via the affine Jacobian. The Hessian coordinate transformation uses the einsum `'ja,kb,njk->nab'` at `operators.py` L121. See `MATH.md` §5.3 for why this index ordering is critical.

---

## The Wiener Prior

The standard H1 regularizer R = K penalizes ‖∇κ‖², suppressing high-frequency noise but treating all spatial scales equally above the mesh scale.

The **Matérn-like Wiener prior** replaces R with:

```
R = M + ℓ²K
```

The MAP cost term κᵀ(M + ℓ²K)κ = ∫[κ² + ℓ²|∇κ|²] dA corresponds to a Gaussian process prior with covariance proportional to the Green's function of the operator (I − ℓ²∇²) — a Matérn-½ covariance with correlation length ℓ.

Setting ℓ = σ_lens encodes the prior belief that κ should vary smoothly at the lens scale while strongly penalizing sub-resolution noise. This is the primary source of the 63% improvement over KS.

Implemented in `femmi/operators.py` L337–L358:

```python
def build_wiener_regularizer(ops, wiener_length):
    return (ops.M + wiener_length**2 * ops.K).tocsr()
```

Selected at construction time in `MAPReconstructor.__init__` (`inverse.py` L88–L91):

```python
if wiener_length > 0.0:
    self._R = build_wiener_regularizer(fwd.ops, wiener_length)
else:
    self._R = fwd.ops.K    # plain H1 prior
```

---

## Implementation Notes

**64-bit float is mandatory.** JAX defaults to 32-bit. For P3 elements, the condition number of K grows as O(h⁻²), and 32-bit arithmetic introduces cancellation errors that destroy the theoretical O(h⁴) convergence on fine meshes. All FEMMI modules call `jax.config.update("jax_enable_x64", True)` at import time (`operators.py` L29, `forward.py` L11, `inverse.py` L25). Missing this single line was one of the two root causes of the original P3 convergence failure.

**Dunavant order-5 rule.** The mass integrand NᵢNⱼ has degree 6 (product of two cubic polynomials), requiring a quadrature rule exact for degree-6 polynomials. The Dunavant 13-point degree-7 rule is used (`p3_assembly.py`, `get_gauss_quadrature_triangle(order=5)`). The original implementation had scrambled S111 orbit parameters that collapsed 6 nominally distinct quadrature points to only 3, silently degrading the rule to degree-5 accuracy and breaking all mass matrix integrals. See `MATH.md` §4.3 for the correct parameter values and the orbital geometry explanation.

**Hessian transformation einsum.** The physical Hessian uses `np.einsum('ja,kb,njk->nab', A, A, H_ref[li])` where A = (J⁻¹)ᵀ (`operators.py` L121). Swapping to `'aj,bk'` transposes A in both slots, giving wrong results for upper triangles where the Jacobian is non-symmetric. The bug is invisible on lower triangles because their Jacobians are diagonal. See `MATH.md` §5.3.

**P1 legacy code.** `fem_solver.py`, `mesh_generator.py`, and `autodiff_integration.py` are retained for historical comparison and because `fem_solver.py` defines the `Mesh` dataclass used by `operators.py`. They are not part of the active P3 pipeline.

---

## References

1. Kaiser, N. & Squires, G. (1993). *Mapping the dark matter with weak gravitational lensing.* ApJ, 404, 441.

2. Brenner, S. & Scott, R. (2008). *The Mathematical Theory of Finite Element Methods* (3rd ed.). Springer. Primary FEM reference; see §3.2 for P3 element theory and §4.4 for Aubin-Nitsche duality.

3. Dunavant, D.A. (1985). *High degree efficient symmetrical Gaussian quadrature rules for the triangle.* IJNME, 21(6), 1129–1148. Source of the 13-point degree-7 rule used in `p3_assembly.py`.

4. Bartelmann, M. & Schneider, P. (2001). *Weak gravitational lensing.* Phys. Rep., 340, 291–472. Comprehensive lensing review; shear/convergence relations in §3.

---

*Independent Study Project — Computational Physics, Worcester Polytechnic Institute. Spring 2026.*