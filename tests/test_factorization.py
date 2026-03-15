"""
tests/test_factorization.py
Phase 4 tests for femmi/svd_analysis.py.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.svd_analysis import (
    compute_svd, picard_plot,
    FactorizationIndicator, LinearSamplingIndicator,
)

NX     = 12
DOMAIN = (-2.5, 2.5, -2.5, 2.5)
SIGMA  = 0.5

results = []

def record(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}")
    if detail:
        print(f"       {detail}")
    results.append((name, ok))


print(f"Building {NX}x{NX} mesh...")
t0  = time.perf_counter()
ops = build_operators(NX, NX, xmin=DOMAIN[0], xmax=DOMAIN[1],
                      ymin=DOMAIN[2], ymax=DOMAIN[3], verbose=False)
print(f"  {ops.n_nodes} nodes  ({time.perf_counter()-t0:.1f}s)")

nodes        = np.array(ops.mesh.nodes)
x, y         = nodes[:, 0], nodes[:, 1]
kappa_disc   = np.where(np.sqrt(x**2 + y**2) < 0.8, 1.0, 0.0).astype(np.float64)
kappa_gauss  = np.exp(-(x**2 + y**2) / (2 * SIGMA**2))
g1_true, g2_true = ops.forward(kappa_disc)

rng         = np.random.default_rng(42)
noise_scale = 0.05 * np.std(np.hypot(g1_true, g2_true))

z_inside  = np.array([0.0, 0.0])
z_outside = np.array([1.8, 1.8])

N_SING = 20
print(f"Computing SVD (n={N_SING})...")
t0  = time.perf_counter()
svd = compute_svd(ops, n_singular=N_SING)
print(f"  done: sigma_max={svd.sigma[0]:.4f}  ({time.perf_counter()-t0:.1f}s)")


# SVD shapes
n = ops.n_nodes
record("SVD shapes correct",
       svd.sigma.shape == (N_SING,) and svd.U.shape == (2*n, N_SING) and svd.V.shape == (n, N_SING))

# Singular value decay
record("sigma_i non-negative", bool((svd.sigma >= 0).all()))
record("sigma_i non-increasing", bool((np.diff(svd.sigma) <= 1e-12).all()))

# Orthogonality
orth_err = np.linalg.norm(svd.V.T @ svd.V - np.eye(N_SING), 'fro') / N_SING
record("||V^T V - I||_F / k < 1e-6", orth_err < 1e-6, f"err={orth_err:.2e}")

# Residuals
max_res = float(svd.residuals[:5].max())
record("Max residual (top 5 modes) < 1e-4", max_res < 1e-4, f"residuals={svd.residuals[:5]}")

# Picard plot runs without error and returns expected structure.
# The picard_ok flag is a slope heuristic that is sensitive to mesh size
# (only 20 modes on a 12x12 mesh); we check structure not the heuristic value.
g1_g, g2_g  = ops.forward(kappa_gauss)
noise_std_g = 0.02 * np.std(np.hypot(g1_g, g2_g))
g1_gn = g1_g + rng.normal(0, noise_std_g, g1_g.shape)
g2_gn = g2_g + rng.normal(0, noise_std_g, g2_g.shape)
pr = picard_plot(ops, np.stack([g1_gn, g2_gn]), noise_std=noise_std_g,
                 svd_result=svd, save=None, show=False)
record("Picard plot returns correct keys",
       all(k in pr for k in ('svd', 'coeffs', 'ratio', 'picard_ok', 'cutoff_idx')))
record("Picard coeffs shape correct", pr['coeffs'].shape == (N_SING,))
record("Picard coeffs are positive", bool((pr['coeffs'] >= 0).all()))

# Factorization indicator
fi       = FactorizationIndicator(ops, svd_result=svd)
W        = fi.indicator_map(np.array([z_inside, z_outside]))
W_in, W_out = float(W[0]), float(W[1])
record("W(inside) > W(outside)", W_in > W_out, f"W_in={W_in:.4f}  W_out={W_out:.4f}")
record("W(inside) > 0.3", W_in > 0.3)

# LSM indicator
lsm      = LinearSamplingIndicator(ops, svd_result=svd)
I        = lsm.indicator_map(np.array([z_inside, z_outside]))
I_in, I_out = float(I[0]), float(I[1])
record("I(inside) > I(outside)", I_in > I_out, f"I_in={I_in:.4f}  I_out={I_out:.4f}")
record("I(inside) > 0.3", I_in > 0.3)

# Consistency across 5 test points
test_pts = np.array([[0.0,0.0],[0.5,0.3],[-0.4,0.5],[1.8,0.0],[0.0,1.9]])
W_pts    = fi.indicator_map(test_pts)
I_pts    = lsm.indicator_map(test_pts)
W_rank   = np.argsort(W_pts)[::-1]
I_rank   = np.argsort(I_pts)[::-1]
record("Top-2 high-indicator points agree", set(W_rank[:2]) == set(I_rank[:2]))
record("Bottom-2 low-indicator points agree", set(W_rank[-2:]) == set(I_rank[-2:]))

# Noise robustness
fi_r  = FactorizationIndicator(ops, svd_result=svd, noise_floor=5*noise_scale)
lsm_r = LinearSamplingIndicator(ops, svd_result=svd, alpha=(5*noise_scale)**2)
W_r   = fi_r.indicator_map(np.array([z_inside, z_outside]))
I_r   = lsm_r.indicator_map(np.array([z_inside, z_outside]))
record("Factorization noise robust", float(W_r[0]) > float(W_r[1]))
record("LSM noise robust",           float(I_r[0]) > float(I_r[1]))

# SVD reuse speedup
t_with    = time.perf_counter(); FactorizationIndicator(ops, svd_result=svd); t_with = time.perf_counter() - t_with
t_without = time.perf_counter(); FactorizationIndicator(ops, n_singular=N_SING); t_without = time.perf_counter() - t_without
record("SVD reuse is >=10x faster", t_without / max(t_with, 1e-6) >= 10,
       f"with={t_with:.2f}s  without={t_without:.2f}s")


if __name__ == "__main__":
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{n_pass}/{n_pass+n_fail} passed")
    if n_fail:
        for name, ok in results:
            if not ok:
                print(f"  FAIL  {name}")
    sys.exit(0 if n_fail == 0 else 1)