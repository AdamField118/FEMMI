"""
tests/test_regression.py
End-to-end regression test for the full FEM-BEM MAP reconstruction pipeline.

Tests the chain: NFW kappa_true -> (gamma1, gamma2) + noise -> MAP -> kappa_MAP.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor

NX     = 16
DOMAIN = (-2.5, 2.5, -2.5, 2.5)
NOISE  = 0.05

results = []

def record(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}")
    if detail:
        print(f"       {detail}")
    results.append((name, ok))


def nfw_kappa(nodes, kappa_s=0.5, r_s=0.8, r_core=0.05):
    r      = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    r_soft = np.maximum(r, r_core)
    u      = r_soft / r_s
    return kappa_s / (u * (1.0 + u)**2)


print(f"Building {NX}x{NX} mesh...")
t0  = time.perf_counter()
ops = build_operators(NX, NX, xmin=DOMAIN[0], xmax=DOMAIN[1],
                      ymin=DOMAIN[2], ymax=DOMAIN[3], verbose=False)
print(f"  {ops.n_nodes} nodes  ({time.perf_counter()-t0:.1f}s)")

nodes      = np.array(ops.mesh.nodes)
kappa_true = nfw_kappa(nodes)
x, y       = nodes[:, 0], nodes[:, 1]
int_mask   = ops.interior


# Forward: finite and plausible shear
g1_true, g2_true = ops.forward(kappa_true)
record("gamma finite", np.all(np.isfinite(g1_true)) and np.all(np.isfinite(g2_true)))

max_g = max(np.abs(g1_true[int_mask]).max(), np.abs(g2_true[int_mask]).max())
record("max|gamma| interior in [0.01, 20.0]", 0.01 < max_g < 20.0,
       f"max|gamma|={max_g:.4f}")


# Adjoint: <F kappa, gamma> = <kappa, F* gamma>
rng       = np.random.default_rng(7)
kappa_rnd = rng.standard_normal(ops.n_nodes)
g_rnd     = rng.standard_normal(ops.n_nodes)
idx_g     = int(ops.bnd_mesh.node_indices[0])
M, S1, S2 = ops.M, ops.S1, ops.S2
A_lu      = ops.A_coupled_lu

def _fwd(kappa):
    rhs = -2.0 * M @ kappa; rhs[idx_g] = 0.0
    psi = A_lu.solve(rhs)
    return S1 @ psi, S2 @ psi

def _adj(g1, g2):
    rhs = S1.T @ g1 + S2.T @ g2; rhs[idx_g] = 0.0
    phi = A_lu.solve(rhs, trans='T')
    return -2.0 * (M.T @ phi)

g1_k, g2_k = _fwd(kappa_rnd)
kappa_adj  = _adj(g_rnd, g_rnd)
lhs        = float(np.dot(g1_k, g_rnd) + np.dot(g2_k, g_rnd))
rhs_val    = float(np.dot(kappa_rnd, kappa_adj))
adj_err    = abs(lhs - rhs_val) / (abs(lhs) + 1e-14)
record("<Fk, g> ~ <k, F*g>  (rel err < 5e-3)", adj_err < 5e-3,
       f"lhs={lhs:.6e}  rhs={rhs_val:.6e}  rel={adj_err:.2e}")


# MAP reconstruction
noise_scale = NOISE * np.std(np.hypot(g1_true, g2_true))
g1_obs = g1_true + rng.normal(0, noise_scale, g1_true.shape)
g2_obs = g2_true + rng.normal(0, noise_scale, g2_true.shape)

fwd = DifferentiableForward(ops, lam_reg=1e-2)
rec = MAPReconstructor(fwd, maxiter=300, gtol=1e-8, wiener_length=0.5, callback_every=0)

t0 = time.perf_counter()
kappa_map, result = rec.reconstruct(g1_obs, g2_obs, verbose=False)
print(f"  MAP: {result.n_iter} iters, {time.perf_counter()-t0:.1f}s, converged={result.converged}")

g1_init, g2_init = ops.forward(np.zeros(ops.n_nodes))
res_init = np.sqrt(np.mean((g1_init - g1_obs)**2 + (g2_init - g2_obs)**2))
res_map  = np.sqrt(np.mean((result.gamma1_pred - g1_obs)**2 + (result.gamma2_pred - g2_obs)**2))

record("MAP residual < initial residual", res_map < res_init,
       f"res(0)={res_init:.4f}  res(MAP)={res_map:.4f}")
record("MAP residual <= 5x noise", res_map <= 5.0 * noise_scale,
       f"res_map={res_map:.4f}  noise={noise_scale:.4f}")


# Spatial structure
r_nodes  = np.sqrt(x**2 + y**2)
peak_r   = float(r_nodes[int_mask][np.argmax(kappa_map[int_mask])])
peak_val = float(kappa_map[int_mask].max())
mean_out = float(kappa_map[int_mask][r_nodes[int_mask] > 1.5].mean())

record("kappa_MAP peaks near centre (r < 1.0)", peak_r < 1.0,
       f"peak at r={peak_r:.3f}")
record("kappa_MAP peak > outer mean", peak_val > mean_out,
       f"peak={peak_val:.4f}  outer_mean={mean_out:.4f}")


# Lower noise -> better L2
def run_map_l2(noise_frac, lam):
    ns   = noise_frac * np.std(np.hypot(g1_true, g2_true))
    g1n  = g1_true + rng.normal(0, ns, g1_true.shape)
    g2n  = g2_true + rng.normal(0, ns, g2_true.shape)
    fwd_ = DifferentiableForward(ops, lam_reg=lam)
    rec_ = MAPReconstructor(fwd_, maxiter=200, gtol=1e-7, wiener_length=0.5, callback_every=0)
    km, _ = rec_.reconstruct(g1n, g2n, verbose=False)
    return float(np.sqrt(np.mean((km[int_mask] - kappa_true[int_mask])**2)))

l2_high = run_map_l2(0.20, 3e-2)
l2_low  = run_map_l2(0.05, 1e-2)
record("L2 decreases from 20% -> 5% noise", l2_low < l2_high,
       f"L2(20%)={l2_high:.4f}  L2(5%)={l2_low:.4f}")


if __name__ == "__main__":
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{n_pass}/{n_pass+n_fail} passed")
    if n_fail:
        for name, ok in results:
            if not ok:
                print(f"  FAIL  {name}")
    sys.exit(0 if n_fail == 0 else 1)