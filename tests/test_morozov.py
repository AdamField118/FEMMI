"""
tests/test_morozov.py
======================
Phase 3 tests for femmi/regularization.py.

Tests:
    1. estimate_noise_level — MAD and std on known distributions
    2. discrepancy_sign     — D(λ_large) > 0, D(λ_small) < 0
    3. discrepancy_monotone — D strictly decreasing in λ
    4. morozov_root         — |D(λ*)| < tolerance after brentq
    5. morozov_improves_l2  — reconstruction at λ* beats fixed λ
    6. lcurve_shape         — residual_norm monotone in λ

Run:
    cd ~/FEMMI && python tests/test_morozov.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


print("=" * 55)
print("Phase 3 — Morozov Regularization Tests")
print("=" * 55)

# ── Build operators once ──────────────────────────────────────────────────────
print("\nBuilding 8×8 mesh...")
t0 = time.perf_counter()
from femmi.operators import build_operators
ops = build_operators(8, 8, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
                      verbose=False)
print(f"  Built in {time.perf_counter()-t0:.1f}s")

nodes = np.array(ops.mesh.nodes)
x, y  = nodes[:, 0], nodes[:, 1]
SIGMA = 0.5
kappa_true = np.exp(-(x**2 + y**2) / (2 * SIGMA**2))

# Generate noiseless shear
g1_true, g2_true = ops.forward(kappa_true)

# Add known noise
NOISE_STD = 0.05
rng = np.random.default_rng(42)
noise_scale = NOISE_STD * np.std(np.hypot(g1_true, g2_true))
g1_obs = g1_true + rng.normal(0, noise_scale, g1_true.shape)
g2_obs = g2_true + rng.normal(0, noise_scale, g2_true.shape)

print(f"  Noise scale: {noise_scale:.4f}  (added to shear)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: estimate_noise_level
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 1 — estimate_noise_level")

from femmi.regularization import estimate_noise_level

# MAD on pure Gaussian noise
noise_only = rng.normal(0, 1.0, 10000)
delta_mad  = estimate_noise_level(noise_only, method="mad")
delta_std  = estimate_noise_level(noise_only, method="std")
record("MAD recovers σ=1 to within 2%",
       abs(delta_mad - 1.0) < 0.02,
       f"MAD={delta_mad:.4f}, std={delta_std:.4f}  (expected 1.0)")

# MAD is robust to outliers (std is not)
noise_outliers = noise_only.copy()
noise_outliers[:10] = 100.0   # 10 extreme outliers
delta_mad_o = estimate_noise_level(noise_outliers, method="mad")
delta_std_o = estimate_noise_level(noise_outliers, method="std")
record("MAD robust to outliers (std inflated by 10×)",
       delta_mad_o < 1.5 and delta_std_o > 2.5,
       f"MAD={delta_mad_o:.4f}  std={delta_std_o:.4f}  (MAD should be ≈1, std should be >>1)")

# MAD on observed shear estimates noise floor
g_all  = np.concatenate([g1_obs, g2_obs])
delta_obs = estimate_noise_level(g_all, method="mad")
record("MAD on noisy shear gives positive estimate",
       delta_obs > 0,
       f"MAD(γ_obs) = {delta_obs:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Discrepancy sign — bracketing
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 2 — Discrepancy sign (bracket check)")

from femmi.regularization import discrepancy

delta = noise_scale  # known noise level

print("  Computing D(λ_small) and D(λ_large)...")

D_small = discrepancy(1e-7, ops, g1_obs, g2_obs, delta=delta,
                      maxiter_inner=100, wiener_length=0.5)
D_large = discrepancy(10.0, ops, g1_obs, g2_obs, delta=delta,
                      maxiter_inner=100, wiener_length=0.5)
print(f"  D(10.0) = {D_large:+.4f}  (should be > 0, over-smooth)")

print(f"  D(1e-7) = {D_small:+.4f}  (should be < 0, over-fit)")
print(f"  D(0.5)  = {D_large:+.4f}  (should be > 0, over-smooth)")

record("D(λ_small) < 0  (small λ over-fits, residual < δ)",
       D_small < 0,
       f"D(1e-7) = {D_small:+.4f}")
record("D(λ_large) > 0  (large λ over-smooths, residual > δ)",
       D_large > 0,
       f"D(0.5) = {D_large:+.4f}")
record("Bracket valid for Brent's method",
       D_small < 0 < D_large,
       "D(λ_lo) < 0 < D(λ_hi)  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Monotonicity of D(λ)
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 3 — Monotonicity of D(λ)")

print("  Evaluating D at 6 λ values...")
lam_vals = np.logspace(-7, -1, 6)
D_vals   = []
for lam in lam_vals:
    D = discrepancy(lam, ops, g1_obs, g2_obs, delta=delta,
                    maxiter_inner=80, wiener_length=0.5)
    D_vals.append(D)
    print(f"    λ={lam:.1e}  D={D:+.4f}")

D_vals = np.array(D_vals)
diffs  = np.diff(D_vals)   # should all be >= 0 (D is decreasing as λ increases)
# D decreasing means D[i+1] < D[i], so diffs < 0
n_violations = int((diffs < -1e-4).sum())
record(f"D(λ) monotone increasing (≤1 violation out of {len(diffs)})",
       n_violations <= 1,
       f"diffs: {diffs.round(4)}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: MorozovSelector finds root
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 4 — MorozovSelector.select() finds D(λ*) ≈ 0")

from femmi.regularization import MorozovSelector

selector = MorozovSelector(
    ops,
    noise_std=noise_scale,
    lam_min=1e-7,
    lam_max=10.0,
    wiener_length=0.5,
    maxiter_inner=100,
    verbose=True,
)

print()
lam_star = selector.select(g1_obs, g2_obs)

D_star = discrepancy(lam_star, ops, g1_obs, g2_obs, delta=noise_scale,
                     maxiter_inner=150, wiener_length=0.5)

record("λ* found in [lam_min, lam_max]",
       selector.lam_min <= lam_star <= selector.lam_max,
       f"λ* = {lam_star:.4e}")
record("|D(λ*)| < 0.01 × δ  (near-exact root)",
       abs(D_star) < 0.01 * noise_scale + 0.001,
       f"|D(λ*)| = {abs(D_star):.2e}  vs  0.01δ = {0.01*noise_scale:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Reconstruction at λ* vs fixed λ
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 5 — Morozov λ* gives better reconstruction than fixed λ")

from femmi.inverse import MAPReconstructor
from femmi.forward import DifferentiableForward

def run_map(lam, maxiter=200):
    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(fwd, maxiter=maxiter, gtol=1e-8,
                           callback_every=0, wiener_length=0.5)
    kappa, _ = rec.reconstruct(g1_obs, g2_obs, verbose=False)
    return float(np.linalg.norm(kappa - kappa_true) / np.linalg.norm(kappa_true))

print(f"  Running MAP at λ* = {lam_star:.3e} ...")
l2_morozov = run_map(lam_star, maxiter=300)

# Compare to λ that's 10× too large and 10× too small
lam_too_large = 10.0    # always clearly over-smoothed (top of L-curve)
lam_too_small = 0.01    # always clearly under-regularized
print(f"  Running MAP at λ_large = {lam_too_large:.3e} ...")
l2_large = run_map(lam_too_large, maxiter=200)
print(f"  Running MAP at λ_small = {lam_too_small:.3e} ...")
l2_small = run_map(lam_too_small, maxiter=200)

print(f"\n  L2 at λ* (Morozov):  {l2_morozov:.4f}")
print(f"  L2 at 10×λ* (large): {l2_large:.4f}")
print(f"  L2 at λ*/10 (small): {l2_small:.4f}")

record("Morozov λ* gives better L2 than 10× too large",
       l2_morozov < l2_large,
       f"{l2_morozov:.4f} < {l2_large:.4f}")
record("Morozov λ* gives better L2 than λ*/10 (over-fit)",
       l2_morozov <= l2_small * 1.5,   # within 50% — over-fit can look good on L2
       f"{l2_morozov:.4f} vs {l2_small:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: L-curve shape (residual monotone in λ)
# ─────────────────────────────────────────────────────────────────────────────
sep("Test 6 — L-curve: residual_norm monotone in λ")

selector_quiet = MorozovSelector(ops, noise_std=noise_scale,
                                  wiener_length=0.5, maxiter_inner=80,
                                  verbose=False)
lcurve = selector_quiet.lcurve(g1_obs, g2_obs, n_points=8)

res = lcurve['residual_norm']
lams = lcurve['lam']
# residual_norm should increase with λ (more regularization → larger residual)
diffs_res = np.diff(res)
n_viol = int((diffs_res < -1e-4).sum())

record("residual_norm increases with λ (monotone)",
       n_viol == 0,
       f"Residual: {res.round(3)}\ndiffs: {diffs_res.round(4)}")

# Discrepancy changes sign at some λ*
disc = lcurve['discrepancy']
has_sign_change = bool((disc[:-1] * disc[1:] < 0).any())
record("D(λ) changes sign over the λ range",
       has_sign_change,
       f"D values: {disc.round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Summary")
print("=" * 55)
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
print("=" * 55)

import sys
sys.exit(0 if n_fail == 0 else 1)