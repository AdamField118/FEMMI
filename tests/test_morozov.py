"""
tests/test_morozov.py
Phase 3 tests for femmi/regularization.py.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

results = []

def record(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}")
    if detail:
        print(f"       {detail}")
    results.append((name, ok))


print("Building 8x8 mesh...")
t0 = time.perf_counter()
from femmi.operators import build_operators
ops = build_operators(8, 8, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5, verbose=False)
print(f"  built in {time.perf_counter()-t0:.1f}s")

nodes      = np.array(ops.mesh.nodes)
x, y       = nodes[:, 0], nodes[:, 1]
kappa_true = np.exp(-(x**2 + y**2) / (2 * 0.5**2))

g1_true, g2_true = ops.forward(kappa_true)
NOISE_STD   = 0.05
rng         = np.random.default_rng(42)
noise_scale = NOISE_STD * np.std(np.hypot(g1_true, g2_true))
g1_obs      = g1_true + rng.normal(0, noise_scale, g1_true.shape)
g2_obs      = g2_true + rng.normal(0, noise_scale, g2_true.shape)


# estimate_noise_level
from femmi.regularization import estimate_noise_level

noise_only = rng.normal(0, 1.0, 10000)
delta_mad  = estimate_noise_level(noise_only, method='mad')
record("MAD recovers sigma=1 within 2%", abs(delta_mad - 1.0) < 0.02,
       f"MAD={delta_mad:.4f}")

noise_outliers = noise_only.copy(); noise_outliers[:10] = 100.0
delta_mad_o = estimate_noise_level(noise_outliers, method='mad')
delta_std_o = estimate_noise_level(noise_outliers, method='std')
record("MAD robust to outliers", delta_mad_o < 1.5 and delta_std_o > 2.5,
       f"MAD={delta_mad_o:.4f}  std={delta_std_o:.4f}")


# discrepancy sign
from femmi.regularization import discrepancy

delta = noise_scale
D_small = discrepancy(1e-7, ops, g1_obs, g2_obs, delta=delta,
                      maxiter_inner=100, wiener_length=0.5)
D_large = discrepancy(10.0, ops, g1_obs, g2_obs, delta=delta,
                      maxiter_inner=100, wiener_length=0.5)

record("D(small lambda) < 0", D_small < 0, f"D(1e-7)={D_small:+.4f}")
record("D(large lambda) > 0", D_large > 0, f"D(10.0)={D_large:+.4f}")
record("Bracket valid for Brent", D_small < 0 < D_large)


# Monotonicity
lam_vals = np.logspace(-7, -1, 6)
D_vals   = []
for lam in lam_vals:
    D = discrepancy(lam, ops, g1_obs, g2_obs, delta=delta,
                    maxiter_inner=80, wiener_length=0.5)
    D_vals.append(D)
    print(f"    lambda={lam:.1e}  D={D:+.4f}")

D_vals = np.array(D_vals)
n_violations = int((np.diff(D_vals) < -1e-4).sum())
record(f"D(lambda) monotone (<=1 violation out of {len(D_vals)-1})",
       n_violations <= 1, f"diffs={np.diff(D_vals).round(4)}")


# MorozovSelector
from femmi.regularization import MorozovSelector

selector = MorozovSelector(ops, noise_std=noise_scale, lam_min=1e-7, lam_max=10.0,
                           wiener_length=0.5, maxiter_inner=100, verbose=True)
lam_star = selector.select(g1_obs, g2_obs)

D_star = discrepancy(lam_star, ops, g1_obs, g2_obs, delta=noise_scale,
                     maxiter_inner=150, wiener_length=0.5)
record("lambda* in [lam_min, lam_max]",
       selector.lam_min <= lam_star <= selector.lam_max,
       f"lambda*={lam_star:.4e}")
record("|D(lambda*)| near zero", abs(D_star) < 0.01 * noise_scale + 0.001,
       f"|D(lambda*)|={abs(D_star):.2e}")


# Reconstruction at lambda* vs fixed lambda
from femmi.inverse import MAPReconstructor
from femmi.forward import DifferentiableForward

def run_map(lam, maxiter=200):
    fwd = DifferentiableForward(ops, lam_reg=lam)
    rec = MAPReconstructor(fwd, maxiter=maxiter, gtol=1e-8,
                           callback_every=0, wiener_length=0.5)
    kappa, _ = rec.reconstruct(g1_obs, g2_obs, verbose=False)
    return float(np.linalg.norm(kappa - kappa_true) / np.linalg.norm(kappa_true))

l2_morozov = run_map(lam_star, maxiter=300)
l2_large   = run_map(10.0)
l2_small   = run_map(0.01)
print(f"  L2: Morozov={l2_morozov:.4f}  large={l2_large:.4f}  small={l2_small:.4f}")

record("Morozov L2 < too-large lambda L2", l2_morozov < l2_large,
       f"{l2_morozov:.4f} < {l2_large:.4f}")


# L-curve shape
selector_q = MorozovSelector(ops, noise_std=noise_scale, wiener_length=0.5,
                              maxiter_inner=80, verbose=False)
lcurve     = selector_q.lcurve(g1_obs, g2_obs, n_points=8)

res   = lcurve['residual_norm']
disc  = lcurve['discrepancy']
n_viol = int((np.diff(res) < -1e-4).sum())
record("residual_norm increases with lambda", n_viol == 0,
       f"diffs={np.diff(res).round(4)}")
record("D(lambda) changes sign", bool((disc[:-1] * disc[1:] < 0).any()),
       f"D={disc.round(3)}")


if __name__ == "__main__":
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{n_pass}/{n_pass+n_fail} passed")
    if n_fail:
        for name, ok in results:
            if not ok:
                print(f"  FAIL  {name}")
    sys.exit(0 if n_fail == 0 else 1)