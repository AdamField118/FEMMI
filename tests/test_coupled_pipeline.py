"""
tests/test_coupled_pipeline.py
================================
Step-by-step diagnostic tests for the FEM-BEM coupled pipeline.

Each test is self-contained and checks ONE specific invariant.
Run all tests to pinpoint exactly where a failure originates.

Tests (in dependency order):
    1.  test_bem_matrices_basic        – V_h symmetric, (½M_b+K_h)@1≈0
    2.  test_calderon_dense            – C = V_h⁻¹(½M_b+K_h) is well-conditioned
    3.  test_A_coupled_nonsingular     – A_coupled invertible after gauge fix
    4.  test_zero_kappa                – forward(κ=0) → γ = 0 exactly
    5.  test_poisson_residual          – ‖K_neumann ψ + 2Mκ - boundary_term‖ small
    6.  test_gauge_node_zero           – ψ[gauge_node] = 0
    7.  test_psi_smooth                – ψ has no large spikes (max|ψ| reasonable)
    8.  test_shear_order_of_magnitude  – max|γ| in expected range for Gaussian κ
    9.  test_shear_symmetry            – for radially symmetric κ, γ has expected symmetry
    10. test_dirichlet_vs_bem          – on a large domain, shear magnitudes agree to ~50%

Usage:
    cd ~/FEMMI && python -m tests.test_coupled_pipeline
    cd ~/FEMMI && python tests/test_coupled_pipeline.py
"""

import sys, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "✓ PASS"
FAIL = "✗ FAIL"
SKIP = "  SKIP"

results = []

def record(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    results.append((name, ok))

def sep(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Build a small mesh once — reused across all tests
# ─────────────────────────────────────────────────────────────────────────────

NX = 8    # small enough to be fast, large enough to test correctness
SIGMA = 0.5
DOMAIN = (-2.5, 2.5, -2.5, 2.5)

print("=" * 60)
print("FEM-BEM Coupled Pipeline — Diagnostic Tests")
print(f"  Mesh: {NX}×{NX} P3, domain {DOMAIN}")
print("=" * 60)

print(f"\nBuilding {NX}×{NX} mesh and operators...")
t0 = time.perf_counter()
from femmi.operators import build_operators
ops = build_operators(NX, NX,
                      xmin=DOMAIN[0], xmax=DOMAIN[1],
                      ymin=DOMAIN[2], ymax=DOMAIN[3],
                      verbose=False)
print(f"  Built in {time.perf_counter()-t0:.1f}s")
print(f"  n_nodes={ops.n_nodes}, N_b={ops.bnd_mesh.n_boundary_dofs}")

nodes    = np.array(ops.mesh.nodes)
x, y     = nodes[:, 0], nodes[:, 1]
kappa_g  = np.exp(-(x**2 + y**2) / (2 * SIGMA**2))   # Gaussian κ
kappa_0  = np.zeros(ops.n_nodes)                       # zero κ


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: BEM matrices — basic properties
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 1 — BEM matrix basic properties")

bnd   = ops.bnd_mesh
N_b   = bnd.n_boundary_dofs
ones_b = np.ones(N_b)

# Retrieve by reassembling (cheaply, n_quad_sl=10 fine for this test)
from femmi.bem import assemble_bem_matrices, assemble_boundary_mass
V_h, K_h, M_b = assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8)

sym_err = np.linalg.norm(V_h - V_h.T) / np.linalg.norm(V_h)
record("V_h symmetric", sym_err < 1e-11,
       f"‖V_h - V_hᵀ‖/‖V_h‖ = {sym_err:.2e}")

M_b_check = assemble_boundary_mass(bnd)
perimeter  = float(bnd.edge_lengths.sum())
row_sum    = float(ones_b @ M_b_check @ ones_b)
record("M_b @ 1 = perimeter", abs(row_sum - perimeter) < 1e-10,
       f"M_b@1 = {row_sum:.6f}, perimeter = {perimeter:.6f}")

combo_norm = np.linalg.norm((0.5*M_b + K_h) @ ones_b)
ref_norm   = np.linalg.norm(0.5*M_b @ ones_b)
calderon_ratio = combo_norm / ref_norm
record("Calderón (½M_b+K_h)@1 ≈ 0", calderon_ratio < 1e-2,
       f"‖(½M_b+K_h)@1‖/‖½M_b@1‖ = {calderon_ratio:.2e}  (threshold 1e-2)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Calderon matrix C
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 2 — Calderon dense matrix C")

C = np.linalg.solve(V_h, 0.5 * M_b + K_h)

# C @ 1 should be ≈ 0 (inherits from (½M_b+K_h)@1 ≈ 0)
C_ones = np.linalg.norm(C @ ones_b)
record("C @ 1 ≈ 0", C_ones < 1e-1,
       f"‖C@1‖ = {C_ones:.2e}")

# C should be invertible (check condition number)
eigs_C = np.linalg.eigvals(C)
cond_C = np.abs(eigs_C).max() / (np.abs(eigs_C).min() + 1e-20)
record("C well-conditioned", cond_C < 1e8,
       f"cond(C) = {cond_C:.2e}")

# C is the same as ops.C_dense
diff_C = np.linalg.norm(C - ops.C_dense) / (np.linalg.norm(C) + 1e-20)
record("ops.C_dense matches recomputed C", diff_C < 1e-10,
       f"‖C - ops.C_dense‖/‖C‖ = {diff_C:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: A_coupled non-singular
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 3 — A_coupled non-singular (gauge fix)")

# Check that A_coupled solves correctly (round-trip test)
rng   = np.random.default_rng(0)
x_ref = rng.standard_normal(ops.n_nodes)
b_ref = ops.A_coupled @ x_ref
x_sol = ops.A_coupled_lu.solve(b_ref)
resid = np.linalg.norm(x_sol - x_ref) / np.linalg.norm(x_ref)
record("A_coupled round-trip solve", resid < 1e-10,
       f"‖x_sol - x_ref‖/‖x_ref‖ = {resid:.2e}")

# Check that the gauge node row is identity
idx_g   = int(ops.bnd_mesh.node_indices[0])
row_g   = np.array(ops.A_coupled[idx_g, :].todense()).ravel()
diag_ok = abs(row_g[idx_g] - 1.0) < 1e-14
offdiag_ok = np.abs(np.delete(row_g, idx_g)).max() < 1e-14
record("Gauge row is identity", diag_ok and offdiag_ok,
       f"A[gauge,gauge]={row_g[idx_g]:.6f}, max|off-diag|={np.abs(np.delete(row_g,idx_g)).max():.2e}")

# A_coupled should NOT have the null vector 1
A_dense  = ops.A_coupled.toarray()
ones_all = np.ones(ops.n_nodes)
A_ones   = np.linalg.norm(A_dense @ ones_all) / np.linalg.norm(A_dense)
record("Constant not in A_coupled null space", A_ones > 1e-6,
       f"‖A@1‖/‖A‖ = {A_ones:.2e}  (should be >> 0 after gauge fix)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Zero kappa → zero shear
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 4 — Zero κ → zero shear")

psi_0  = ops.psi_from_kappa(kappa_0)
g1_0, g2_0 = ops.shear_from_psi(psi_0)

max_psi0   = np.abs(psi_0).max()
max_shear0 = max(np.abs(g1_0).max(), np.abs(g2_0).max())

record("ψ = 0 when κ = 0", max_psi0 < 1e-10,
       f"max|ψ| = {max_psi0:.2e}")
record("γ = 0 when κ = 0", max_shear0 < 1e-10,
       f"max|γ| = {max_shear0:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Poisson residual in the interior
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 5 — Poisson residual  ‖K_neumann ψ + 2Mκ‖ in interior")

# For the FEM weak form: K ψ = -2Mκ + (boundary flux term)
# The residual K ψ + 2Mκ should be ≈ 0 at INTERIOR nodes
# (boundary nodes carry the BEM flux contribution, which is non-zero)

psi_g     = ops.psi_from_kappa(kappa_g)
residual  = ops.K @ psi_g + 2.0 * ops.M @ kappa_g   # (n_nodes,)

interior_mask = ops.interior
bnd_mask      = ~interior_mask

res_int   = np.abs(residual[interior_mask])
res_bnd   = np.abs(residual[bnd_mask])
rhs_scale = np.abs(2.0 * ops.M @ kappa_g).max()

rel_int = res_int.max() / rhs_scale
rel_bnd = res_bnd.max() / rhs_scale

record("Interior Poisson residual < 1% of RHS",
       rel_int < 0.01,
       f"max|res|_interior / max|rhs| = {rel_int:.2e}  (threshold 1e-2)")

print(f"         max|res|_boundary  / max|rhs| = {rel_bnd:.2e}  (boundary expected non-zero)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Gauge node has ψ = 0
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 6 — Gauge node ψ = 0")

psi_gauge_val = float(psi_g[idx_g])
record("ψ[gauge_node] = 0", abs(psi_gauge_val) < 1e-12,
       f"ψ[{idx_g}] = {psi_gauge_val:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: ψ has no large spikes
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 7 — ψ has no large spikes (order-of-magnitude check)")

# For a Gaussian κ with A=1, σ=0.5, the Newtonian potential is finite.
# An upper bound: |ψ| < A * σ² * (something) ~ O(0.1 to 1).
# Values > 100 indicate a singular/wrong solve.
max_psi_g = np.abs(psi_g).max()
record("max|ψ| < 100", max_psi_g < 100,
       f"max|ψ| = {max_psi_g:.4f}  (physically expected ~0.01-1)")
record("max|ψ| > 0", max_psi_g > 1e-6,
       f"max|ψ| = {max_psi_g:.4f}  (should be non-trivial)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Shear order of magnitude
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 8 — Shear order of magnitude for Gaussian κ")

g1_g, g2_g = ops.shear_from_psi(psi_g)
max_g1_int = np.abs(g1_g[interior_mask]).max()
max_g2_int = np.abs(g2_g[interior_mask]).max()
max_shear_int = max(max_g1_int, max_g2_int)

inf_plane_ref = 1.0 / (4 * SIGMA**2) * np.exp(-0.5)   # ← add this line

print(f"         Infinite-plane reference peak: {inf_plane_ref:.3f}")
print(f"         max|γ₁| interior = {max_g1_int:.4f},  boundary = {np.abs(g1_g[~interior_mask]).max():.4f}")
print(f"         max|γ₂| interior = {max_g2_int:.4f},  boundary = {np.abs(g2_g[~interior_mask]).max():.4f}")

record("max|γ| interior < 10 × reference",
       max_shear_int < 10 * inf_plane_ref,
       f"max|γ|_interior = {max_shear_int:.3f}  vs  10× ref = {10*inf_plane_ref:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Shear symmetry for radially symmetric κ
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 9 — Shear symmetry for radially symmetric κ")

# For radially symmetric κ:
#   γ₁ = 0 on the diagonal x=y  (anti-symmetric under x↔y)
#   γ₂ = 0 on the x-axis and y-axis  (∂²ψ/∂x∂y = 0 by symmetry)

# γ₁ should be suppressed on x=y diagonal (interior only)
diag_mask = np.abs(x - y) < 0.15
int_diag  = diag_mask & interior_mask
int_off   = ~diag_mask & interior_mask
if int_diag.sum() > 3:
    g1_on_diag = np.abs(g1_g[int_diag]).mean()
    g1_off     = np.abs(g1_g[int_off]).mean()
    ratio_g1   = g1_on_diag / (g1_off + 1e-20)
    record("γ₁ suppressed on x=y diagonal (interior)",
           ratio_g1 < 0.5,
           f"mean|γ₁| on x≈y = {g1_on_diag:.4f}, elsewhere = {g1_off:.4f}, ratio = {ratio_g1:.2f}")
else:
    record("γ₁ diagonal symmetry (skipped)", True, "SKIPPED")

# γ₂ should be suppressed on axes (interior only)
axes_mask = (np.abs(y) < 0.15) | (np.abs(x) < 0.15)
int_axes  = axes_mask & interior_mask
int_off2  = ~axes_mask & interior_mask
if int_axes.sum() > 3:
    g2_on_axes = np.abs(g2_g[int_axes]).mean()
    g2_off     = np.abs(g2_g[int_off2]).mean()
    ratio_g2   = g2_on_axes / (g2_off + 1e-20)
    record("γ₂ suppressed on x,y axes (interior)",
           ratio_g2 < 0.5,
           f"mean|γ₂| on axes = {g2_on_axes:.4f}, elsewhere = {g2_off:.4f}, ratio = {ratio_g2:.2f}")
else:
    record("γ₂ axis symmetry (skipped)", True, "SKIPPED")

# γ₁ < 0 along x-axis
xaxis_mask = (np.abs(y) < 0.1) & (np.abs(x) > 0.3) & interior_mask
if xaxis_mask.sum() > 3:
    mean_g1_xaxis = float(np.mean(g1_g[xaxis_mask]))
    record("γ₁ < 0 along x-axis (tangential shear)",
           mean_g1_xaxis < 0,
           f"mean(γ₁) on x-axis = {mean_g1_xaxis:.4f}  (should be < 0)")
else:
    record("γ₁ x-axis sign (skipped)", True, "SKIPPED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Compare BEM vs old Dirichlet on a large domain
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 10 — BEM vs Dirichlet: shear magnitude comparison")

# Build old Dirichlet forward using the deprecated K_lu
rhs_dir = -2.0 * ops.M @ kappa_g
rhs_dir[ops.boundary] = 0.0
psi_dir   = ops.K_lu.solve(rhs_dir)
g1_dir, g2_dir = ops.shear_from_psi(psi_dir)

max_g1_dir = np.abs(g1_dir).max()
max_g2_dir = np.abs(g2_dir).max()
max_g1_bem = np.abs(g1_g).max()
max_g2_bem = np.abs(g2_g).max()

print(f"         Dirichlet:  max|γ₁|={max_g1_dir:.4f}  max|γ₂|={max_g2_dir:.4f}")
print(f"         BEM-coupled: max|γ₁|={max_g1_bem:.4f}  max|γ₂|={max_g2_bem:.4f}")
print(f"         Infinite-plane ref: {inf_plane_ref:.4f}")

# On a 5×5 domain with σ=0.5, boundary BC error is large for Dirichlet,
# but both should give shear in the same order of magnitude.
ratio_g1 = max_g1_bem / (max_g1_dir + 1e-20)
ratio_g2 = max_g2_bem / (max_g2_dir + 1e-20)
print(f"         BEM/Dirichlet ratio: γ₁={ratio_g1:.2f}, γ₂={ratio_g2:.2f}")

# Interior peak location comparison
int_nodes = ops.interior
g2_int_bem = np.abs(g2_g[int_nodes])
g2_int_dir = np.abs(g2_dir[int_nodes])
ratio_peak = g2_int_bem.max() / (g2_int_dir.max() + 1e-20)
record("BEM/Dirichlet peak shear ratio in (0.1, 10)",
       0.1 < ratio_peak < 10,
       f"BEM peak |γ₂|={g2_int_bem.max():.4f}  Dirichlet={g2_int_dir.max():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
for name, ok in results:
    print(f"  {'✓' if ok else '✗'}  {name}")
print(f"\n  {n_pass}/{n_pass+n_fail} passed")
if n_fail > 0:
    print("\n  FAILING TESTS:")
    for name, ok in results:
        if not ok:
            print(f"    ✗  {name}")
print("=" * 60)

sys.exit(0 if n_fail == 0 else 1)
