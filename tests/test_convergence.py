"""
tests/test_convergence.py
Mesh convergence test for the FEM-BEM coupled forward operator.

Tests O(h^p) convergence of kappa -> psi -> gamma as the mesh is refined,
using a fine reference mesh (nx=64) as the ground truth.

Theoretical rates:
  psi: O(h^{5/3}) - BEM on a square has re-entrant corner singularities
       with exponent pi/(2pi - pi/2) = 2/3. The L2 boundary approximation
       error is capped at O(h^{5/3}) regardless of BEM polynomial order.
       Acceptable since psi is never directly observed.

  gamma: O(h^2) - P3 nodal Hessian averaging gives piecewise-linear second
         derivatives. Measured on deep interior nodes (|x|,|y| < 1.5).

Run:
    python tests/test_convergence.py
"""

import sys, os, time
import numpy as np
from scipy.interpolate import griddata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators

SIGMA    = 1.5
DOMAIN   = (-2.5, 2.5, -2.5, 2.5)
NX_REF   = 64
DEEP_LIM = 1.5


def kappa_fn(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.exp(-(x**2 + y**2) / (2 * SIGMA**2))


def ml2(v):
    return float(np.sqrt(np.mean(v**2)))


print(f"Building reference mesh (nx={NX_REF})...")
t0 = time.perf_counter()
ops_ref        = build_operators(NX_REF, NX_REF,
                                 xmin=DOMAIN[0], xmax=DOMAIN[1],
                                 ymin=DOMAIN[2], ymax=DOMAIN[3],
                                 verbose=False)
nodes_ref      = np.array(ops_ref.mesh.nodes)
kappa_ref      = kappa_fn(nodes_ref)
psi_ref        = ops_ref.psi_from_kappa(kappa_ref)
g1_ref, g2_ref = ops_ref.shear_from_psi(psi_ref)

int_ref  = np.array(ops_ref.interior, dtype=bool)
x_r, y_r = nodes_ref[:, 0], nodes_ref[:, 1]
deep_ref = int_ref & (np.abs(x_r) < DEEP_LIM) & (np.abs(y_r) < DEEP_LIM)
print(f"  {ops_ref.n_nodes} nodes  ({time.perf_counter()-t0:.1f}s)")


res_psi = []
res_gam = []

for nx in [8, 12, 16, 24, 32]:
    h = 5.0 / nx
    t0  = time.perf_counter()
    ops = build_operators(nx, nx,
                          xmin=DOMAIN[0], xmax=DOMAIN[1],
                          ymin=DOMAIN[2], ymax=DOMAIN[3],
                          verbose=False)

    nodes      = np.array(ops.mesh.nodes)
    psi_h      = ops.psi_from_kappa(kappa_fn(nodes))
    g1_h, g2_h = ops.shear_from_psi(psi_h)

    int_mask  = np.array(ops.interior, dtype=bool)
    x_c, y_c  = nodes[:, 0], nodes[:, 1]
    deep_mask = int_mask & (np.abs(x_c) < DEEP_LIM) & (np.abs(y_c) < DEEP_LIM)

    psi_interp = griddata(nodes_ref, psi_ref, nodes, method='linear')
    psi_nn     = griddata(nodes_ref, psi_ref, nodes, method='nearest')
    psi_interp = np.where(np.isfinite(psi_interp), psi_interp, psi_nn)

    g1_interp = griddata(nodes_ref[deep_ref], g1_ref[deep_ref], nodes, method='linear')
    g1_nn     = griddata(nodes_ref[deep_ref], g1_ref[deep_ref], nodes, method='nearest')
    g1_interp = np.where(np.isfinite(g1_interp), g1_interp, g1_nn)

    g2_interp = griddata(nodes_ref[deep_ref], g2_ref[deep_ref], nodes, method='linear')
    g2_nn     = griddata(nodes_ref[deep_ref], g2_ref[deep_ref], nodes, method='nearest')
    g2_interp = np.where(np.isfinite(g2_interp), g2_interp, g2_nn)

    psi_offset     = float(np.mean((psi_h - psi_interp)[int_mask]))
    err_psi        = ml2((psi_h - psi_offset - psi_interp)[int_mask])
    norm_psi       = ml2(psi_interp[int_mask])
    err_gam        = ml2((g1_h - g1_interp)[deep_mask]) + ml2((g2_h - g2_interp)[deep_mask])
    norm_gam       = ml2(g1_interp[deep_mask]) + ml2(g2_interp[deep_mask])

    print(f"  nx={nx:2d}  h={h:.3f}  "
          f"||psi_err||={err_psi:.3e} (rel {err_psi/norm_psi:.3e})  "
          f"||gam_err||={err_gam:.3e} (rel {err_gam/norm_gam:.3e})  "
          f"({time.perf_counter()-t0:.1f}s)")

    res_psi.append((h, err_psi))
    res_gam.append((h, err_gam))


def rate_table(label, data, exp_rate, pass_threshold):
    print(f"\n{label}")
    print(f"  {'h':>6}  {'error':>12}  {'rate':>6}")
    rates = []
    for i, (h, err) in enumerate(data):
        if i == 0:
            print(f"  {h:6.3f}  {err:12.3e}    -")
        else:
            h0, e0 = data[i-1]
            r = np.log(err / e0) / np.log(h / h0)
            rates.append(r)
            print(f"  {h:6.3f}  {err:12.3e}  {r:+.2f}")
    if rates:
        mean_r = float(np.mean(rates[-2:]))
        ok     = mean_r >= pass_threshold
        print(f"  asymptotic rate (last 2 steps): {mean_r:.2f}  "
              f"(theory >= {exp_rate})  {'PASS' if ok else 'FAIL'}")
    return rates


print("\nConvergence Summary")
rate_table("psi (lensing potential, interior nodes)", res_psi,
           exp_rate=1.67, pass_threshold=1.4)
rate_table("gamma (shear, deep interior |x|,|y| < 1.5)", res_gam,
           exp_rate=2.0, pass_threshold=1.5)