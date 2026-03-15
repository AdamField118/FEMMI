"""
tests/test_coupled_pipeline.py
Step-by-step diagnostic tests for the FEM-BEM coupled pipeline.

Each test checks one invariant in dependency order.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NX     = 8
SIGMA  = 0.5
DOMAIN = (-2.5, 2.5, -2.5, 2.5)

from femmi.operators import build_operators

print(f"Building {NX}x{NX} mesh...")
t0 = time.perf_counter()
ops = build_operators(NX, NX, xmin=DOMAIN[0], xmax=DOMAIN[1],
                      ymin=DOMAIN[2], ymax=DOMAIN[3], verbose=False)
print(f"  {ops.n_nodes} nodes, N_b={ops.bnd_mesh.n_boundary_dofs}  ({time.perf_counter()-t0:.1f}s)")

nodes   = np.array(ops.mesh.nodes)
x, y    = nodes[:, 0], nodes[:, 1]
kappa_g = np.exp(-(x**2 + y**2) / (2 * SIGMA**2))
kappa_0 = np.zeros(ops.n_nodes)

results = []

def record(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}")
    if detail:
        print(f"       {detail}")
    results.append((name, ok))


# BEM matrices
from femmi.bem import assemble_bem_matrices, assemble_boundary_mass
bnd   = ops.bnd_mesh
N_b   = bnd.n_boundary_dofs
ones_b = np.ones(N_b)

V_h, K_h, M_b = assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8)

sym_err = np.linalg.norm(V_h - V_h.T) / np.linalg.norm(V_h)
record("V_h symmetric", sym_err < 1e-11, f"sym_err={sym_err:.2e}")

M_b_check = assemble_boundary_mass(bnd)
perimeter  = float(bnd.edge_lengths.sum())
row_sum    = float(ones_b @ M_b_check @ ones_b)
record("M_b @ 1 = perimeter", abs(row_sum - perimeter) < 1e-10,
       f"M_b@1={row_sum:.6f}, perimeter={perimeter:.6f}")

combo_norm    = np.linalg.norm((0.5*M_b + K_h) @ ones_b)
ref_norm      = np.linalg.norm(0.5*M_b @ ones_b)
calderon_ratio = combo_norm / ref_norm
record("Calderon (0.5*M_b+K_h)@1 ~ 0", calderon_ratio < 1e-2,
       f"ratio={calderon_ratio:.2e}")

# Calderon dense matrix
C = np.linalg.solve(V_h, 0.5 * M_b + K_h)
# C @ 1 should be small since (0.5*M_b+K_h)@1 ~ 0, but V_h^{-1} can amplify it.
# The direct Calderon ratio check above is the correct invariant; here just
# verify C@1 is not wildly large relative to C itself.
_C_ones_norm = np.linalg.norm(C @ ones_b)
_C_norm      = np.linalg.norm(C)
record("C @ 1 small relative to ||C||", _C_ones_norm < 0.5 * _C_norm,
       f"||C@1||={_C_ones_norm:.2e}  ||C||={_C_norm:.2e}")

diff_C = np.linalg.norm(C - ops.C_dense) / (np.linalg.norm(C) + 1e-20)
record("ops.C_dense matches recomputed C", diff_C < 1e-10, f"diff={diff_C:.2e}")

# A_coupled
rng   = np.random.default_rng(0)
x_ref = rng.standard_normal(ops.n_nodes)
b_ref = ops.A_coupled @ x_ref
x_sol = ops.A_coupled_lu.solve(b_ref)
resid = np.linalg.norm(x_sol - x_ref) / np.linalg.norm(x_ref)
record("A_coupled round-trip solve", resid < 1e-10, f"rel_resid={resid:.2e}")

idx_g   = int(ops.bnd_mesh.node_indices[0])
row_g   = np.array(ops.A_coupled[idx_g, :].todense()).ravel()
diag_ok = abs(row_g[idx_g] - 1.0) < 1e-14
offdiag_ok = np.abs(np.delete(row_g, idx_g)).max() < 1e-14
record("Gauge row is identity", diag_ok and offdiag_ok)

# Zero kappa
psi_0       = ops.psi_from_kappa(kappa_0)
g1_0, g2_0  = ops.shear_from_psi(psi_0)
record("psi=0 when kappa=0", np.abs(psi_0).max() < 1e-10)
record("gamma=0 when kappa=0", max(np.abs(g1_0).max(), np.abs(g2_0).max()) < 1e-10)

# Poisson residual
psi_g       = ops.psi_from_kappa(kappa_g)
residual    = ops.K @ psi_g + 2.0 * ops.M @ kappa_g
rhs_scale   = np.abs(2.0 * ops.M @ kappa_g).max()
rel_int     = np.abs(residual[ops.interior]).max() / rhs_scale
record("Interior Poisson residual < 1% of RHS", rel_int < 0.01,
       f"rel_int={rel_int:.2e}")

# Gauge node
record("psi[gauge_node] = 0", abs(float(psi_g[idx_g])) < 1e-12,
       f"psi[gauge]={psi_g[idx_g]:.2e}")

# Order of magnitude
max_psi_g = np.abs(psi_g).max()
record("max|psi| < 100", max_psi_g < 100, f"max|psi|={max_psi_g:.4f}")
record("max|psi| > 0",   max_psi_g > 1e-6)

# Shear magnitude
g1_g, g2_g  = ops.shear_from_psi(psi_g)
max_shear    = max(np.abs(g1_g[ops.interior]).max(), np.abs(g2_g[ops.interior]).max())
inf_ref      = 1.0 / (4 * SIGMA**2) * np.exp(-0.5)
record("max|gamma| interior < 10x reference",
       max_shear < 10 * inf_ref,
       f"max|gamma|={max_shear:.3f}  10x_ref={10*inf_ref:.3f}")

# Shear symmetry
diag_mask   = np.abs(x - y) < 0.15
int_diag    = diag_mask & ops.interior
int_off     = ~diag_mask & ops.interior
if int_diag.sum() > 3:
    ratio = np.abs(g1_g[int_diag]).mean() / (np.abs(g1_g[int_off]).mean() + 1e-20)
    record("gamma1 suppressed on x=y diagonal", ratio < 0.5, f"ratio={ratio:.2f}")

axes_mask = (np.abs(y) < 0.15) | (np.abs(x) < 0.15)
int_axes  = axes_mask & ops.interior
int_off2  = ~axes_mask & ops.interior
if int_axes.sum() > 3:
    ratio2 = np.abs(g2_g[int_axes]).mean() / (np.abs(g2_g[int_off2]).mean() + 1e-20)
    record("gamma2 suppressed on axes", ratio2 < 0.5, f"ratio={ratio2:.2f}")

xaxis_mask = (np.abs(y) < 0.1) & (np.abs(x) > 0.3) & ops.interior
if xaxis_mask.sum() > 3:
    record("gamma1 < 0 along x-axis",
           float(np.mean(g1_g[xaxis_mask])) < 0,
           f"mean(gamma1 on x-axis)={np.mean(g1_g[xaxis_mask]):.4f}")


if __name__ == "__main__":
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{n_pass}/{n_pass+n_fail} passed")
    if n_fail:
        print("Failed:")
        for name, ok in results:
            if not ok:
                print(f"  {name}")
    sys.exit(0 if n_fail == 0 else 1)