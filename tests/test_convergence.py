"""
tests/test_convergence.py
==========================
Mesh convergence test for the FEM-BEM coupled FORWARD operator.

Tests O(h^p) convergence of the forward map (κ → ψ → γ) as the mesh
is refined, using a fine reference mesh (nx=20) as the "exact" solution.

Why forward and not inverse?
-----------------------------
The ill-posed inverse problem converges at rate O(δ^{2/3}) regardless
of h (Morozov theory).  With noiseless data + tiny λ, the optimizer
amplifies discretization errors in the near-null-space of F.
The FORWARD problem has a clean theoretical convergence rate:
  ψ: O(h^4)   (P3 FEM optimal for smooth κ)
  γ: O(h^2)   (nodal Hessian averaging reduces by two orders)

Strategy
--------
- Build a fine REFERENCE mesh at nx=20.
- Build coarse meshes at nx = 4, 6, 8, 12, 16.
- For each, compute ψ_h and γ_h from the SAME κ_true (wide Gaussian, σ=2.0
  so σ/h >= 1.6 even on the coarsest mesh → asymptotic regime).
- Interpolate ψ_ref and γ_ref onto coarse mesh nodes via griddata.
- Measure ‖ψ_h - ψ_ref‖ and ‖γ_h - γ_ref‖.

Expected rates
--------------
  ψ convergence: ≥ 1.5  (limited by P1 BEM boundary approximation, O(h^2))
  γ convergence: ≥ 0.8  (interior only, excludes boundary Hessian artifacts)

Run
---
    cd ~/FEMMI && python tests/test_convergence.py
"""

import sys, os, time
import numpy as np
from scipy.interpolate import griddata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators

SIGMA   = 2.0          # wide Gaussian: resolved on all test meshes
DOMAIN  = (-2.5, 2.5, -2.5, 2.5)
NX_REF  = 32           # reference mesh — treated as "exact"

def kappa_fn(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.exp(-(x**2 + y**2) / (2 * SIGMA**2))

print("=" * 62)
print("FEM-BEM Forward Operator Convergence Test")
print(f"  σ = {SIGMA},  reference mesh nx={NX_REF}")
print(f"  Coarse meshes: nx = 4, 6, 8, 12, 16")
print("=" * 62)

# ── Build reference solution ──────────────────────────────────────────────
print(f"\nBuilding reference mesh (nx={NX_REF})...")
t0 = time.perf_counter()
ops_ref  = build_operators(NX_REF, NX_REF,
                           xmin=DOMAIN[0], xmax=DOMAIN[1],
                           ymin=DOMAIN[2], ymax=DOMAIN[3],
                           verbose=False)
nodes_ref      = np.array(ops_ref.mesh.nodes)
kappa_ref      = kappa_fn(nodes_ref)
psi_ref        = ops_ref.psi_from_kappa(kappa_ref)
g1_ref, g2_ref = ops_ref.shear_from_psi(psi_ref)
print(f"  {ops_ref.n_nodes} nodes, {ops_ref.bnd_mesh.n_boundary_dofs} boundary DOFs  "
      f"({time.perf_counter()-t0:.1f}s)")
print(f"  max|ψ_ref| = {np.abs(psi_ref).max():.4f}")
print(f"  max|γ_ref| = {max(np.abs(g1_ref).max(), np.abs(g2_ref).max()):.4f}")

# ── Coarse mesh sweep ─────────────────────────────────────────────────────
mesh_sizes = [4, 6, 8, 12, 16]
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

    # Interpolate reference solution onto coarse mesh nodes
    psi_interp = griddata(nodes_ref, psi_ref, nodes,
                          method='cubic', fill_value=0.0)
    g1_interp  = griddata(nodes_ref, g1_ref,  nodes,
                          method='cubic', fill_value=0.0)
    g2_interp  = griddata(nodes_ref, g2_ref,  nodes,
                          method='cubic', fill_value=0.0)

    # L2 errors (mass-weighted via M)
    M = ops.M
    def ml2(v):
        return float(np.sqrt(np.abs(v @ (M @ v))))

    # ψ is gauge-fixed (up to a constant) — subtract mean offset before comparing
    psi_offset     = float(np.mean(psi_h - psi_interp))
    psi_h_centered = psi_h - psi_offset

    # Use interior nodes only — boundary spikes are a known numerical artifact
    int_mask = ops.interior   # boolean (n_nodes,)

    err_psi  = ml2((psi_h_centered - psi_interp)[int_mask])
    err_gam  = (ml2((g1_h - g1_interp)[int_mask]) +
                ml2((g2_h - g2_interp)[int_mask]))
    norm_psi = ml2(psi_interp[int_mask])
    norm_gam = (ml2(g1_interp[int_mask]) + ml2(g2_interp[int_mask]))

    print(f"  ‖ψ_h - ψ_ref‖  = {err_psi:.3e}  (rel {err_psi/norm_psi:.3e})")
    print(f"  ‖γ_h - γ_ref‖  = {err_gam:.3e}  (rel {err_gam/norm_gam:.3e})")

    res_psi.append((h, err_psi))
    res_gam.append((h, err_gam))

# ── Convergence rate tables ───────────────────────────────────────────────
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
            r = np.log(err/e0) / np.log(h/h0)
            rates.append(r)
            print(f"  {h:6.3f}  {err:12.3e}  {r:+.2f}")
    if rates:
        mean_r = np.mean(rates[-2:])   # asymptotic: last 3 refinements
        ok = mean_r >= pass_threshold
        print(f"  Mean rate (asymptotic): {mean_r:.2f}  "
              f"(expected ≥ {exp_rate})")
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  rate ≥ {pass_threshold}")
    return rates

print(f"\n{'='*62}")
print("Convergence Summary")
print(f"{'='*62}")
rates_psi = rate_table("ψ (lensing potential, interior)", res_psi,
                        exp_rate=2.0, pass_threshold=1.5)
rates_gam = rate_table("γ (shear = ∂²ψ, interior)",      res_gam,
                        exp_rate=1.5, pass_threshold=0.8)
print(f"{'='*62}")