# FEMMI Implementation Plan: FEM-BEM Coupling + Morozov + Inverse Scattering

**Purpose**: Systematic plan for upgrading FEMMI from its current Dirichlet-BC FEM
implementation to the full FEM-BEM coupled pipeline described in MATH.md.

Each change is tagged with a **phase** (1–4), a **priority** (blocking / high / medium),
and its coupling dependencies.

---

## Phase Overview

| Phase | Scope | Blocks |
|-------|-------|--------|
| 1 | BEM matrix assembly (`bem.py`) | Everything else |
| 2 | Coupled system in `operators.py` | Forward, inverse |
| 3 | Morozov λ selection (`regularization.py`) | Inverse |
| 4 | Inverse scattering (`svd_analysis.py`) | Standalone |

---

## Phase 1 — Create `femmi/bem.py`  *(blocking)*

**Goal**: Assemble the three BEM matrices V_h, K_h, M_b on ∂Ω.

### 1.1 — Boundary mesh extraction

```python
# bem.py
def extract_boundary_edges(mesh: Mesh) -> BoundaryMesh:
    """
    Returns nodes on ∂Ω in counter-clockwise order, edge lengths, and
    outward unit normals.  Used by all BEM assembly routines.
    """
```

**Inputs**: `femmi.mesh.Mesh`
**Outputs**: `BoundaryMesh(nodes, edge_lengths, normals, n_boundary_dofs)`
**Tests**: `test_fem_bem_coupling.py::test_boundary_extraction`

---

### 1.2 — Logarithmic Gauss-Jacobi quadrature

```python
def log_gauss_jacobi_points(n: int) -> tuple[Array, Array]:
    """
    Returns n-point quadrature (nodes, weights) for integrals of the form
    ∫₀¹ f(t) (-ln t) dt.  Used for diagonal blocks of V_h.
    Relative error < 1e-12 for n=10; use n=25 for production.
    """
```

**Reference**: MATH.md §5.2; Sauter & Schwab Table 5.3.
**Tests**: verify ∫₀¹ tᵏ(-ln t) dt = 1/(k+1)² to 1e-12 for k=0..4.

---

### 1.3 — Single-layer matrix V_h

```python
def assemble_single_layer(bnd: BoundaryMesh, n_quad: int = 25) -> Array:
    """
    V_h[i,j] = ∫_{Γᵢ} ∫_{Γⱼ} G(x,y) φᵢ(x) φⱼ(y) ds(x) ds(y)
    G(x,y) = (1/2π) ln|x−y|   (2-D fundamental solution)

    Diagonal (i==j): logarithmic-singular, use log_gauss_jacobi_points.
    Off-diagonal:    non-singular, use standard Gauss-Legendre (n=8).

    Returns symmetric positive definite (N_b × N_b) matrix.
    """
```

**Critical**: diagonal blocks require `log_gauss_jacobi_points`, not standard
Gauss-Legendre (which would give O(1) error on the singularity).

**Reference**: C&K §3.1, Steinbach §4.1.
**Tests**:
- Symmetry: `‖V_h − V_hᵀ‖ / ‖V_h‖ < 1e-12`
- Coercivity: `min(eigs(V_h)) > 0`
- Patch test: V_h applied to constant density reproduces analytic result

---

### 1.4 — Double-layer matrix K_h

```python
def assemble_double_layer(bnd: BoundaryMesh, n_quad: int = 8) -> Array:
    """
    K_h[i,j] = ∫_{Γᵢ} ∫_{Γⱼ} (∂G/∂n(y))(x,y) φᵢ(x) φⱼ(y) ds(x) ds(y)
    ∂G/∂n = (1/2π) (x−y)·n(y) / |x−y|²

    Off-diagonal: weakly singular (integrable); standard Gauss-Legendre.
    Diagonal:     finite-part integral, use Cauchy principal value formula.
    """
```

**Reference**: C&K §3.2, jump relation [C&K (3.14)].
**Tests**:
- Row sums: `K_h @ ones = 0` (double-layer preserves constants up to ½)
- Calderon identity: `½M_b + K_h` has known eigenvalue distribution

---

### 1.5 — Boundary mass matrix M_b

```python
def assemble_boundary_mass(bnd: BoundaryMesh) -> Array:
    """
    M_b[i,j] = ∫_{∂Ω} φᵢ(s) φⱼ(s) ds
    Standard 1-D Gauss-Legendre (n=4 sufficient for P1 boundary elements).
    """
```

**Tests**: `M_b @ ones ≈ |∂Ω|` (total boundary length).

---

### 1.6 — Schur complement: Calderon preconditioner

```python
def calderon_matrix(V_h: Array, K_h: Array, M_b: Array) -> Array:
    """
    C = V_h^{-1} (½M_b + K_h)
    Returned as a LinearOperator (sparse factored form of V_h applied via
    solve, not explicit dense inverse).
    Complexity: O(N_b³) LU of V_h (one-time), O(N_b²) matvec.
    """
```

---

## Phase 2 — Modify `femmi/operators.py`  *(blocking)*

### 2.1 — Remove Dirichlet boundary condition enforcement

**Current code** (`operators.py`, approximately lines 236–237):
```python
# REMOVE THESE LINES:
K = K.at[boundary_dofs, :].set(0)
K = K.at[boundary_dofs, boundary_dofs].set(1)
```

**Reason**: This enforces ψ = 0 on ∂Ω.  We need the Neumann stiffness matrix
(no row modification) so that the FEM weak form retains the boundary flux term
∫_{∂Ω} (∂ψ/∂n) v ds, which the BEM exterior couples into.

**Risk**: Without a corresponding BEM coupling, the Neumann system is singular
(pure Neumann problem has a one-dimensional null space = constants).  Phase 2.2
**must** be applied atomically with 2.1.

---

### 2.2 — Assemble A_coupled

```python
# operators.py
def build_operators(mesh: Mesh) -> Operators:
    K  = _assemble_neumann_stiffness(mesh)     # no Dirichlet rows
    M  = _assemble_mass(mesh)
    S1, S2 = _assemble_shear_operators(mesh)

    # BEM coupling
    bnd    = extract_boundary_edges(mesh)
    V_h    = assemble_single_layer(bnd)
    K_h    = assemble_double_layer(bnd)
    M_b    = assemble_boundary_mass(bnd)
    C      = calderon_matrix(V_h, K_h, M_b)   # V_h^{-1}(½M_b + K_h)
    P      = _restriction_matrix(mesh, bnd)   # sparse, shape (N_b, N_dof)
    A_coupled = K + P.T @ C @ P               # (N_dof × N_dof) sparse+dense

    return Operators(A_coupled=A_coupled, M=M, S1=S1, S2=S2, mesh=mesh)
```

**Note on A_coupled structure**: K is large sparse (N_dof × N_dof); Pᵀ C P is a
rank-N_b perturbation (N_b ≪ N_dof for fine meshes).  Store as
`SparsePlusDense(K, P, C)` for efficient LU factorisation via the
matrix determinant lemma / Woodbury identity if N_b is small, or as
a sparse direct sum otherwise.

---

### 2.3 — Shear operator index fix (already applied — verify)

Confirm the einsum index order is:
```python
gamma = jnp.einsum('ka,lb,jkl->jab', A, A, H_ref)   # CORRECT
# NOT: jnp.einsum('aj,bk,jkl->lab', A, A, H_ref)     # WRONG
```

Run `test_shear_consistency.py` to confirm.  No code change needed if already
correct; document it here so it is not accidentally reverted.

---

## Phase 3 — Create `femmi/regularization.py`  *(high priority)*

### 3.1 — Noise level estimation

```python
def estimate_noise_level(gamma_obs: Array, method: str = "mad") -> float:
    """
    Estimate per-component noise standard deviation δ from observed shear.
    method='mad': δ = 1.4826 * median(|γ - median(γ)|)  (robust)
    method='high_freq': fit power spectrum tail.
    """
```

---

### 3.2 — Morozov discrepancy function

```python
def discrepancy(lam: float, ops: Operators, gamma_obs: Array,
                delta: float, c: float = 1.0) -> float:
    """
    D(λ) = ‖F κ_λ − γ_obs‖ − c δ

    κ_λ = (FᵀF + λR)^{-1} Fᵀ γ_obs  (solved via one forward + one adjoint)
    Monotone decreasing in λ (guaranteed by compact F; see MATH.md §13).
    """
```

**Implementation notes**:
- Do NOT form FᵀF explicitly; use the adjoint gradient structure
- Reuse the factored A_coupled from `ops` (don't re-assemble)
- Regularisation matrix R = M + ℓ²K (Matérn prior); ℓ defaults to mesh.h * 5

---

### 3.3 — MorozovSelector

```python
class MorozovSelector:
    """
    Select λ by Morozov's discrepancy principle using Brent's method.

    Usage:
        selector = MorozovSelector(ops, noise_std=0.02)
        lam = selector.select(gamma_obs)
        # Typical: 15–25 forward solves; converges to 6 significant figures.
    """
    def __init__(self, ops: Operators, noise_std: float | None = None,
                 c: float = 1.0, lam_min: float = 1e-8, lam_max: float = 1.0):
        ...

    def select(self, gamma_obs: Array) -> float:
        # Brent root finding on discrepancy(lam, ...)
        ...

    def lcurve(self, gamma_obs: Array, n_points: int = 50) -> dict:
        # Compute L-curve for diagnostic plotting
        ...
```

**Tests** (`test_morozov.py`):
- Monotonicity: `D(λ₁) > D(λ₂)` whenever `λ₁ < λ₂`
- Recovery: with known κ and δ = noise level, check `‖κ_{λ*} − κ_true‖ < C δ^{2/3}`
  (optimal convergence rate for compact operators; C&K Thm 10.2)

---

### 3.4 — Modify `femmi/inverse.py`: MAPReconstructor

**Current**: manual `lam_reg = 2e-2` hardcoded in `MAPReconstructor.__init__`.

**Change**:
```python
class MAPReconstructor:
    def __init__(self, ops: Operators, noise_std: float | None = None,
                 lam: float | None = None, ...):
        """
        lam=None (default): select by Morozov's principle before each solve.
        lam=float: fix λ manually (legacy behaviour).
        noise_std: required if lam is None.
        """

    def reconstruct(self, gamma_obs: Array) -> Array:
        if self.lam is None:
            self.lam = MorozovSelector(self.ops, self.noise_std).select(gamma_obs)
        # ... existing L-BFGS loop, replacing K_LU with A_coupled solves
```

**Forward solve change** (inside the loop):
```python
# OLD:
psi = jax.scipy.linalg.solve_triangular(...)   # K_LU^{-1}
# NEW:
psi = solve_coupled(self.ops.A_coupled, f)      # A_coupled^{-1}
```

---

## Phase 4 — Create `femmi/svd_analysis.py`  *(medium priority, standalone)*

### 4.1 — Truncated SVD of F

```python
def compute_svd(ops: Operators, n_singular: int = 40,
                method: str = "lanczos") -> SVDResult:
    """
    Compute leading n_singular singular triplets (σᵢ, uᵢ, vᵢ) of F.
    F: L²(Ω) → L²(Ω)²,  F κ = (γ₁, γ₂).

    method='lanczos': randomised SVD via FᵀF matvec (one forward + one adjoint
                      per Lanczos step); O(n_singular × n_dof) cost.
    method='dense':   form F explicitly (n_dof columns); only for small meshes.

    Returns SVDResult(sigma, U, V, residuals).
    """
```

---

### 4.2 — Picard plot diagnostic

```python
def picard_plot(ops: Operators, gamma_obs: Array, noise_std: float,
                n_singular: int = 40, save: str | None = "picard.pdf"):
    """
    Three-panel Picard plot:
      Panel 1: log σᵢ vs i   (singular values — shows decay rate)
      Panel 2: log|⟨γ_obs, uᵢ⟩| vs i   (Fourier coefficients of data)
      Panel 3: log(|⟨γ_obs, uᵢ⟩| / σᵢ) vs i   (amplified noise)

    The Picard condition holds if panel 2 decays faster than panel 1.
    The crossing point in panel 3 gives a visual noise cutoff.

    Reference: MATH.md §15, C&K §10.1.
    """
```

---

### 4.3 — Factorization method indicator

```python
class FactorizationIndicator:
    """
    Support recovery via the Kirsch factorization method (C&K Thm 6.15).

    W(z)^{-1} = Σ_{σᵢ > δ}  |⟨Φ_z, uᵢ⟩|² / σᵢ

    W(z) is small iff z ∈ support(κ).

    Usage:
        fi = FactorizationIndicator(ops, n_singular=40)
        W  = fi.indicator_map(test_points)   # shape (n_test,)
        fi.plot(mesh)                         # 2-D indicator map
    """
    def __init__(self, ops: Operators, n_singular: int = 40,
                 noise_floor: float | None = None): ...

    def probe_function(self, z: Array) -> Array:
        """Φ_z = shear of Green's function centred at z."""
        ...

    def indicator_map(self, test_points: Array) -> Array:
        """Evaluate W at all test points (vectorised)."""
        ...
```

**Tests** (`test_factorization.py`):
- For a disc-shaped κ: indicator correctly identifies interior/exterior
- Noise robustness: adding 5% Gaussian noise doesn't move boundary by more than 2h

---

### 4.4 — Linear sampling method indicator

```python
class LinearSamplingIndicator:
    """
    Support recovery via the linear sampling method (C&K §5.5).

    I(z) = 1 / ‖g_z^α‖

    where g_z^α = (FᵀF + α I)^{-1} Fᵀ Φ_z  is the Tikhonov solution to F g = Φ_z.
    I(z) is large iff z ∉ support(κ).

    More numerically stable than factorization method near corners/edges.
    """
    def __init__(self, ops: Operators, n_singular: int = 40,
                 alpha: float | None = None): ...

    def indicator_map(self, test_points: Array) -> Array: ...
```

---

## Cross-Cutting Concerns

### 64-bit enforcement

Every module must begin with:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

Add a module-level guard in `femmi/__init__.py`:
```python
import jax
jax.config.update("jax_enable_x64", True)

def _check_64bit():
    import jax.numpy as jnp
    x = jnp.array(1.0)
    assert x.dtype == jnp.float64, (
        "FEMMI requires 64-bit JAX.  Set JAX_ENABLE_X64=1 or call "
        "jax.config.update('jax_enable_x64', True) before importing femmi."
    )
_check_64bit()
```

### Invariants that must not change

The following are **correct in the current code** and must not be modified:

| Component | Current state | Note |
|-----------|--------------|-------|
| P3 basis functions | Correct | Do not downgrade to P2/P1 |
| Dunavant quadrature | Bug fixed | Do not revert |
| Shear einsum index | `'ka,lb,jkl->jab'` | Do not revert |
| L-BFGS optimizer | Unchanged | Phase 2 only changes the linear solve |
| 64-bit enforcement | Present | Extend, do not remove |

---

## File Change Summary

| File | Action | Phase |
|------|--------|-------|
| `femmi/bem.py` | **CREATE** | 1 |
| `femmi/operators.py` | **MODIFY** (lines 236–237, `build_operators`) | 2 |
| `femmi/forward.py` | **MODIFY** (`A_coupled` solve replaces `K_LU`) | 2 |
| `femmi/inverse.py` | **MODIFY** (Morozov λ, `A_coupled` solve) | 2+3 |
| `femmi/regularization.py` | **CREATE** | 3 |
| `femmi/svd_analysis.py` | **CREATE** | 4 |
| `femmi/__init__.py` | **MODIFY** (64-bit guard) | 1 |
| `tests/test_fem_bem_coupling.py` | **CREATE** | 1 |
| `tests/test_morozov.py` | **CREATE** | 3 |
| `tests/test_factorization.py` | **CREATE** | 4 |

---

## Testing Checklist

### Phase 1 (BEM)
- [ ] `extract_boundary_edges`: CCW ordering, normals point outward
- [ ] `log_gauss_jacobi_points`: ∫₀¹ tᵏ (−ln t) dt exact for k=0..4
- [ ] `assemble_single_layer`: symmetric, positive definite
- [ ] `assemble_double_layer`: row sums zero, Calderon identity
- [ ] `assemble_boundary_mass`: total = |∂Ω|
- [ ] Transmission conditions: [ψ] = 0 and [∂ψ/∂n] = 0 at ∂Ω

### Phase 2 (Coupling)
- [ ] `build_operators`: A_coupled positive definite
- [ ] Forward solve: known κ (NFW) → γ matches Kaiser-Squires on large domain
- [ ] Adjoint: `⟨F κ, γ⟩ = ⟨κ, Fᵀ γ⟩` to 1e-10
- [ ] Convergence: O(h⁴) in L², O(h²) in shear (mesh sequence 4→32)
- [ ] No spurious mass at boundary (compare with Dirichlet code on same problem)

### Phase 3 (Morozov)
- [ ] `discrepancy`: strictly decreasing as function of λ
- [ ] `MorozovSelector.select`: |D(λ*)| < 1e-8 after convergence
- [ ] Recovery rate: ‖κ_{λ*} − κ_true‖ scales as O(δ^{2/3})

### Phase 4 (Inverse Scattering)
- [ ] Picard plot: visual separation of signal and noise components
- [ ] Factorization indicator: disc support recovered within 2h for SNR ≥ 10
- [ ] LSM indicator: consistent with factorization method on smooth shapes

---

## Complexity Budget (32×32 mesh, N_dof ≈ 10⁴, N_b ≈ 500)

| Operation | Cost | Frequency |
|-----------|------|-----------|
| BEM assembly (V_h, K_h, M_b) | O(N_b² × n_quad) ≈ 6×10⁶ flops | Once |
| V_h LU factorisation | O(N_b³) ≈ 1.25×10⁸ flops | Once |
| A_coupled factorisation | O(N_dof log N_dof) | Once per λ |
| Forward solve (A_coupled^{-1}) | O(N_dof log N_dof) | Per iteration |
| Morozov λ selection | 15–25 forward solves | Per reconstruction |
| SVD of F (n_singular=40) | 80 forward/adjoint pairs | Once |
| Factorization indicator (64² grid) | 4096 probe evaluations | Once per field |

For a 32×32 mesh the BEM cost is O(1.25×10⁸) — sub-second on a workstation.
BEM assembly scales as O(N_b²); for N_b > 2000 consider H-matrix compression.
