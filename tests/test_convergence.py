"""
tests/test_convergence.py
==========================
Mesh convergence test for the FEM-BEM coupled FORWARD operator.

Tests O(h^p) convergence of the forward map (κ → ψ → γ) as the mesh
is refined, using a fine reference mesh (nx=64) as the "exact" solution.

Why forward and not inverse?
-----------------------------
The ill-posed inverse problem converges at rate O(δ^{2/3}) regardless
of h (Morozov theory).  With noiseless data + tiny λ, the optimizer
amplifies discretization errors in the near-null-space of F.
The FORWARD problem has a clean theoretical convergence rate.

Theoretical rates
-----------------
  ψ: O(h^{5/3}) — BEM on a polygon with 90° corners has re-entrant
     corner singularities with exponent π/(2π − π/2) = 2/3.  The L²
     boundary approximation error is capped at O(h^{5/3}) regardless
     of BEM polynomial order.  Upgrading from P1→P3 BEM does not help
     here; only corner-graded meshes or a circular domain would recover
     the full polynomial rate.  For the weak lensing application this
     is acceptable: ψ is never directly observed, only γ is.

  γ: O(h^2) — P3 nodal Hessian averaging gives piecewise-linear second
     derivatives.  The corner singularity in ψ decays to O(h^{4/3}) in
     H², so it does not contaminate the interior shear rate.  Measured
     on deep interior nodes (|x|,|y| < 1.5) to exclude the ring of
     near-boundary nodes where P3 nodal averaging is unreliable.

Strategy
--------
- σ = 1.5: κ(corner) ≈ 0.004 (compact), σ/h ≥ 2.4 on all meshes.
- ψ error: interior nodes, mean-offset subtracted (gauge ambiguity).
- γ error: deep interior nodes |x|,|y| < 1.5 only.
- Reference mesh nx=64 (2× finest test mesh).

Run
---
    cd ~/FEMMI && python tests/test_convergence.py
"""

import sys, os, time
import numpy as np
from scipy.interpolate import griddata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators

# ── Parameters ────────────────────────────────────────────────────────────────
SIGMA    = 1.5
DOMAIN   = (-2.5, 2.5, -2.5, 2.5)
NX_REF   = 64
DEEP_LIM = 1.5   # |x|,|y| < DEEP_LIM for γ error measurement

def kappa_fn(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.exp(-(x**2 + y**2) / (2 * SIGMA**2))

def ml2(v):
    return float(np.sqrt(np.mean(v**2)))

# ── Build reference solution ───────────────────────────────────────────────────
print("=" * 62)
print("FEM-BEM Forward Operator Convergence Test")
print(f"  σ = {SIGMA},  reference mesh nx={NX_REF}")
print(f"  Coarse meshes: nx = 8, 12, 16, 24, 32")
print("=" * 62)

print(f"\nBuilding reference mesh (nx={NX_REF})...")
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

print(f"  {ops_ref.n_nodes} nodes, {ops_ref.bnd_mesh.n_boundary_dofs} boundary DOFs  "
      f"({time.perf_counter()-t0:.1f}s)")
print(f"  max|ψ_ref|              = {np.abs(psi_ref).max():.4f}")
print(f"  max|γ_ref| all nodes    = {max(np.abs(g1_ref).max(), np.abs(g2_ref).max()):.4f}")
print(f"  max|γ_ref| interior     = {max(np.abs(g1_ref[int_ref]).max(), np.abs(g2_ref[int_ref]).max()):.4f}")
print(f"  max|γ_ref| deep int.    = {max(np.abs(g1_ref[deep_ref]).max(), np.abs(g2_ref[deep_ref]).max()):.4f}")

# ── Coarse mesh sweep ──────────────────────────────────────────────────────────
mesh_sizes = [8, 12, 16, 24, 32]
res_psi    = []
res_gam    = []

for nx in mesh_sizes:
    h = 5.0 / nx
    print(f"\n{'─'*55}")
    print(f"  nx={nx}  h={h:.3f}  σ/h={SIGMA/h:.1f}")
    print(f"{'─'*55}")

    t0  = time.perf_counter()
    ops = build_operators(nx, nx,
                          xmin=DOMAIN[0], xmax=DOMAIN[1],
                          ymin=DOMAIN[2], ymax=DOMAIN[3],
                          verbose=False)
    print(f"  Built: {ops.n_nodes} nodes  ({time.perf_counter()-t0:.1f}s)")

    nodes      = np.array(ops.mesh.nodes)
    kappa      = kappa_fn(nodes)
    psi_h      = ops.psi_from_kappa(kappa)
    g1_h, g2_h = ops.shear_from_psi(psi_h)

    int_mask  = np.array(ops.interior, dtype=bool)
    x_c, y_c  = nodes[:, 0], nodes[:, 1]
    deep_mask = int_mask & (np.abs(x_c) < DEEP_LIM) & (np.abs(y_c) < DEEP_LIM)

    # ψ: interpolate from all reference nodes (boundary included — ψ is smooth,
    #    and both meshes pin ψ=0 at the same corner so bdry values are correct)
    psi_interp = griddata(nodes_ref, psi_ref, nodes, method='linear')
    psi_nn     = griddata(nodes_ref, psi_ref, nodes, method='nearest')
    psi_interp = np.where(np.isfinite(psi_interp), psi_interp, psi_nn)

    # γ: interpolate from deep interior reference nodes only (excludes
    #    near-boundary nodes where P3 nodal averaging is unreliable)
    g1_interp = griddata(nodes_ref[deep_ref], g1_ref[deep_ref], nodes, method='linear')
    g1_nn     = griddata(nodes_ref[deep_ref], g1_ref[deep_ref], nodes, method='nearest')
    g1_interp = np.where(np.isfinite(g1_interp), g1_interp, g1_nn)

    g2_interp = griddata(nodes_ref[deep_ref], g2_ref[deep_ref], nodes, method='linear')
    g2_nn     = griddata(nodes_ref[deep_ref], g2_ref[deep_ref], nodes, method='nearest')
    g2_interp = np.where(np.isfinite(g2_interp), g2_interp, g2_nn)

    # ψ gauge: subtract mean interior offset (residual ambiguity from
    #    mesh-dependent gauge-fix corner location)
    psi_offset     = float(np.mean((psi_h - psi_interp)[int_mask]))
    psi_h_centered = psi_h - psi_offset

    err_psi  = ml2((psi_h_centered - psi_interp)[int_mask])
    norm_psi = ml2(psi_interp[int_mask])

    err_gam  = ml2((g1_h - g1_interp)[deep_mask]) + ml2((g2_h - g2_interp)[deep_mask])
    norm_gam = ml2(g1_interp[deep_mask]) + ml2(g2_interp[deep_mask])

    print(f"  psi gauge offset = {psi_offset:.2e}")
    print(f"  ‖ψ_h - ψ_ref‖  = {err_psi:.3e}  (rel {err_psi/norm_psi:.3e})")
    print(f"  ‖γ_h - γ_ref‖  = {err_gam:.3e}  (rel {err_gam/norm_gam:.3e})")

    res_psi.append((h, err_psi))
    res_gam.append((h, err_gam))

# ── Convergence rate tables ────────────────────────────────────────────────────
def rate_table(label, data, exp_rate, pass_threshold):
    print(f"\n  {label}")
    print(f"  {'h':>6}  {'error':>12}  {'rate':>6}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*6}")
    rates = []
    for i, (h, err) in enumerate(data):
        if i == 0:
            print(f"  {h:6.3f}  {err:12.3e}    —")
        else:
            h0, e0 = data[i-1]
            r = np.log(err / e0) / np.log(h / h0)
            rates.append(r)
            print(f"  {h:6.3f}  {err:12.3e}  {r:+.2f}")
    if rates:
        mean_r = float(np.mean(rates[-2:]))
        ok = mean_r >= pass_threshold
        print(f"  Mean rate (asymptotic, last 2 steps): {mean_r:.2f}  "
              f"(theory ≥ {exp_rate})")
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  rate ≥ {pass_threshold}")
    return rates

print(f"\n{'='*62}")
print("Convergence Summary")
print(f"{'='*62}")
rates_psi = rate_table(
    "ψ (lensing potential, interior nodes)",
    res_psi,
    exp_rate=1.67,    # O(h^{5/3}) — polygon corner singularity limit
    pass_threshold=1.4,
)
rates_gam = rate_table(
    "γ (shear, deep interior |x|,|y| < 1.5)",
    res_gam,
    exp_rate=2.0,     # O(h^2) — P3 nodal Hessian averaging
    pass_threshold=1.5,
)
print(f"{'='*62}")
print("""
Notes:
  ψ rate ~5/3: expected for BEM on square domain (90° corner singularity).
               Not fixable by higher BEM order; requires corner-graded mesh
               or circular domain.  Acceptable since γ, not ψ, is observed.
  γ rate ~2:   confirms P3 Hessian averaging is working correctly.
""")