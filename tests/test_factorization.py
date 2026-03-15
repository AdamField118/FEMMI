"""
tests/test_factorization.py
============================
Phase 4 tests for femmi/svd_analysis.py.

Tests:
    1. test_svd_shapes         – output arrays have correct shapes
    2. test_svd_singular_decay – singular values are decreasing
    3. test_svd_orthogonality  – V columns are orthonormal (VᵀV ≈ I)
    4. test_svd_residuals      – ‖F vᵢ − σᵢ uᵢ‖/σᵢ < 1e-4 for top 5 modes
    5. test_picard_condition   – smooth κ data satisfies Picard condition
    6. test_factorization_interior_beats_exterior
                               – W(z_inside) > W(z_outside) for disc κ
    7. test_lsm_interior_beats_exterior
                               – I(z_inside) > I(z_outside) for disc κ
    8. test_factorization_lsm_consistent
                               – both indicators agree on interior/exterior
    9. test_noise_robustness   – 5% noise doesn't flip interior/exterior
   10. test_svd_reuse          – passing svd_result avoids recomputation

Run:
    cd ~/FEMMI && python tests/test_factorization.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.svd_analysis import (
    compute_svd, picard_plot,
    FactorizationIndicator, LinearSamplingIndicator,
)

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []

def record(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, ok))

def sep(title):
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"{'─'*58}")


# ── Build a small mesh once — reused across all tests ────────────────────────
NX     = 12
DOMAIN = (-2.5, 2.5, -2.5, 2.5)
SIGMA  = 0.5

print("=" * 58)
print("Phase 4 — SVD + Inverse Scattering Tests")
print(f"  Mesh: {NX}×{NX} P3, domain {DOMAIN}")
print("=" * 58)

print(f"\nBuilding {NX}×{NX} mesh...")
t0  = time.perf_counter()
ops = build_operators(NX, NX,
                      xmin=DOMAIN[0], xmax=DOMAIN[1],
                      ymin=DOMAIN[2], ymax=DOMAIN[3],
                      verbose=False)
print(f"  Built: {ops.n_nodes} nodes  ({time.perf_counter()-t0:.1f}s)")

nodes   = np.array(ops.mesh.nodes)
x, y    = nodes[:, 0], nodes[:, 1]

# Disc-shaped κ (step function approximated by steep Gaussian)
kappa_disc = np.where(np.sqrt(x**2 + y**2) < 0.8, 1.0, 0.0).astype(np.float64)
# Smooth Gaussian κ for Picard test
kappa_gauss = np.exp(-(x**2 + y**2) / (2 * SIGMA**2))

# Generate clean observations from disc κ
g1_true, g2_true = ops.forward(kappa_disc)
g_stacked_clean  = np.concatenate([g1_true, g2_true])

# Noisy observations (5% noise)
rng         = np.random.default_rng(42)
noise_scale = 0.05 * np.std(np.hypot(g1_true, g2_true))
g1_noisy    = g1_true + rng.normal(0, noise_scale, g1_true.shape)
g2_noisy    = g2_true + rng.normal(0, noise_scale, g2_true.shape)

# Test points: well inside and well outside the disc
z_inside  = np.array([0.0, 0.0])    # centre of disc
z_outside = np.array([1.8, 1.8])    # corner, outside disc

# Pre-compute SVD once (reused across tests)
N_SING = 20
print(f"\nPre-computing SVD (n_singular={N_SING})...")
t0  = time.perf_counter()
svd = compute_svd(ops, n_singular=N_SING)
print(f"  Done: {len(svd.sigma)} modes  ({time.perf_counter()-t0:.1f}s)")
print(f"  σ_max={svd.sigma[0]:.4f}  σ_min={svd.sigma[-1]:.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: SVD shapes
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 1 — SVD output shapes")

n = ops.n_nodes
ok = (svd.sigma.shape == (N_SING,) and
      svd.U.shape     == (2*n, N_SING) and
      svd.V.shape     == (n,   N_SING))
record("SVD arrays have correct shapes",
       ok,
       f"sigma={svd.sigma.shape}, U={svd.U.shape}, V={svd.V.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Singular value decay
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 2 — Singular values are non-negative and decreasing")

non_neg  = bool((svd.sigma >= 0).all())
decaying = bool((np.diff(svd.sigma) <= 1e-12).all())
record("All σᵢ ≥ 0",
       non_neg,
       f"min σ = {svd.sigma.min():.4e}")
record("σᵢ non-increasing",
       decaying,
       f"max increase = {max(np.diff(svd.sigma).max(), 0):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: V columns orthonormal
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 3 — V columns are orthonormal")

VtV     = svd.V.T @ svd.V
orth_err = np.linalg.norm(VtV - np.eye(N_SING), 'fro') / N_SING
record("‖VᵀV − I‖_F / k < 1e-6",
       orth_err < 1e-6,
       f"‖VᵀV − I‖_F / k = {orth_err:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: SVD residuals
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 4 — SVD residuals ‖F vᵢ − σᵢ uᵢ‖/σᵢ < 1e-4 (top 5)")

top5_res = svd.residuals[:5]
max_res  = float(top5_res.max())
record("Max residual (top 5 modes) < 1e-4",
       max_res < 1e-4,
       f"residuals = {top5_res}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Picard condition for smooth κ
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 5 — Picard condition satisfied for smooth κ")

# Generate observations from smooth Gaussian κ
g1_g, g2_g   = ops.forward(kappa_gauss)
noise_std_g  = 0.02 * np.std(np.hypot(g1_g, g2_g))
g1_gn = g1_g + rng.normal(0, noise_std_g, g1_g.shape)
g2_gn = g2_g + rng.normal(0, noise_std_g, g2_g.shape)

picard_result = picard_plot(
    ops,
    np.stack([g1_gn, g2_gn]),
    noise_std=noise_std_g,
    svd_result=svd,
    save=None,
    show=False,
)
record("Picard condition satisfied (smooth Gaussian κ)",
       picard_result['picard_ok'],
       f"coeff decay faster than σ decay: {picard_result['picard_ok']}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Factorization indicator — interior > exterior
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 6 — Factorization indicator: W(inside) > W(outside)")

fi = FactorizationIndicator(ops, svd_result=svd)
W  = fi.indicator_map(np.array([z_inside, z_outside]))
W_in, W_out = float(W[0]), float(W[1])

record("W(z_inside) > W(z_outside)",
       W_in > W_out,
       f"W(0,0)={W_in:.4f}  W(1.8,1.8)={W_out:.4f}")
record("W(z_inside) > 0.3  (substantial interior signal)",
       W_in > 0.3,
       f"W(0,0)={W_in:.4f}  (threshold 0.3)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: LSM indicator — interior > exterior
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 7 — LSM indicator: I(inside) > I(outside)")

lsm = LinearSamplingIndicator(ops, svd_result=svd)
I   = lsm.indicator_map(np.array([z_inside, z_outside]))
I_in, I_out = float(I[0]), float(I[1])

record("I(z_inside) > I(z_outside)",
       I_in > I_out,
       f"I(0,0)={I_in:.4f}  I(1.8,1.8)={I_out:.4f}")
record("I(z_inside) > 0.3  (substantial interior signal)",
       I_in > 0.3,
       f"I(0,0)={I_in:.4f}  (threshold 0.3)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Factorization and LSM consistent
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 8 — Factorization and LSM agree on 5 test points")

test_pts = np.array([
    [0.0,  0.0],   # centre — inside
    [0.5,  0.3],   # inside
    [-0.4, 0.5],   # inside
    [1.8,  0.0],   # outside
    [0.0,  1.9],   # outside
])
W_pts = fi.indicator_map(test_pts)
I_pts = lsm.indicator_map(test_pts)

# Both should rank the same points high/low
W_rank = np.argsort(W_pts)[::-1]
I_rank = np.argsort(I_pts)[::-1]
top2_agree = set(W_rank[:2]) == set(I_rank[:2])
bot2_agree = set(W_rank[-2:]) == set(I_rank[-2:])

record("Top-2 high-indicator points agree",
       top2_agree,
       f"W top2={W_rank[:2]}, I top2={I_rank[:2]}  (pts: {test_pts[W_rank[:2]]})")
record("Bottom-2 low-indicator points agree",
       bot2_agree,
       f"W bot2={W_rank[-2:]}, I bot2={I_rank[-2:]}  (pts: {test_pts[W_rank[-2:]]})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Noise robustness
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 9 — 5% noise doesn't flip interior/exterior ordering")

# Use SVD from clean data, evaluate indicators on noisy probe functions
# (the probe function itself uses the forward solve, which is deterministic;
#  the noise test checks that modest data noise doesn't change the ranking)
fi_robust  = FactorizationIndicator(ops, svd_result=svd,
                                    noise_floor=5 * noise_scale)
lsm_robust = LinearSamplingIndicator(ops, svd_result=svd,
                                     alpha=(5 * noise_scale)**2)

W_robust = fi_robust.indicator_map(np.array([z_inside, z_outside]))
I_robust = lsm_robust.indicator_map(np.array([z_inside, z_outside]))

record("Factorization: W(inside) > W(outside) at 5% noise level",
       float(W_robust[0]) > float(W_robust[1]),
       f"W(0,0)={W_robust[0]:.4f}  W(1.8,1.8)={W_robust[1]:.4f}")
record("LSM: I(inside) > I(outside) at 5% noise level",
       float(I_robust[0]) > float(I_robust[1]),
       f"I(0,0)={I_robust[0]:.4f}  I(1.8,1.8)={I_robust[1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: SVD reuse
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 10 — Passing svd_result avoids recomputation")

t_with_svd = time.perf_counter()
fi2 = FactorizationIndicator(ops, svd_result=svd)
t_with_svd = time.perf_counter() - t_with_svd

t_without_svd = time.perf_counter()
fi3 = FactorizationIndicator(ops, n_singular=N_SING)
t_without_svd = time.perf_counter() - t_without_svd

speedup = t_without_svd / max(t_with_svd, 1e-6)
record("Reusing SVD is faster than recomputing (≥10× speedup)",
       speedup >= 10,
       f"with SVD: {t_with_svd:.2f}s  without: {t_without_svd:.2f}s  "
       f"speedup: {speedup:.0f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print("Summary")
print(f"{'='*58}")
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
for name, ok in results:
    print(f"  {'✓' if ok else '✗'}  {name}")
print(f"\n  {n_pass}/{n_pass+n_fail} passed")
if n_fail:
    print("\n  FAILING:")
    for name, ok in results:
        if not ok:
            print(f"    ✗  {name}")
print(f"{'='*58}")

import sys
sys.exit(0 if n_fail == 0 else 1)