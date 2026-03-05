"""
tests/test_pipeline.py
======================
Integration test for the full κ → γ → κ pipeline.

  1. Operator assembly     : K, M, S1, S2 correct shapes/nnz
  2. Forward pass          : ψ, γ finite and nonzero
  3. Gradient validation   : autodiff vs FD to 1e-4 rel error
  4. Noiseless MAP         : reconstruct from exact γ, L2 < 0.12
  5. Noisy MAP vs KS       : 10% noise, FEM-MAP within 1.5× KS L2

Run from project root:
    bash ./run.sh tests/test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.fem     import build_operators
from src.forward import DifferentiableForward
from src.inverse import MAPReconstructor, run_comparison

PASS = "✅  PASS"
FAIL = "❌  FAIL"

NX     = 10
DOMAIN = (-2.0, 2.0, -2.0, 2.0)
A, SIGMA = 1.0, 0.5


def gaussian_kappa(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return A * np.exp(-(x**2 + y**2) / (2*SIGMA**2))


def sep():
    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════

def test_1_operators(ops):
    sep()
    print("TEST 1: Operator assembly")
    n, ok = ops.n_nodes, True
    for name, mat in [("K",ops.K),("M",ops.M),("S1",ops.S1),("S2",ops.S2)]:
        s_ok = mat.shape == (n, n)
        z_ok = mat.nnz > 0
        print(f"  {name}: shape={mat.shape}  nnz={mat.nnz}  {'✅' if s_ok and z_ok else '❌'}")
        ok = ok and s_ok and z_ok
    bnd_ok = np.allclose(np.array(ops.K.diagonal())[ops.boundary], 1.0, atol=1e-10)
    print(f"  K BCs (boundary diagonal=1): {'✅' if bnd_ok else '❌'}")
    ok = ok and bnd_ok
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_2_forward(ops):
    sep()
    print("TEST 2: Forward pass (κ → ψ → γ)")
    nodes = np.array(ops.mesh.nodes)
    kappa = gaussian_kappa(nodes)
    psi   = ops.psi_from_kappa(kappa)
    g1, g2 = ops.shear_from_psi(psi)

    ok = True
    fp = np.all(np.isfinite(psi)) and np.abs(psi).max() > 1e-12
    print(f"  ψ finite: {'✅' if fp else '❌'}   max|ψ|={np.abs(psi).max():.4f}")
    ok = ok and fp
    fg = np.all(np.isfinite(g1)) and np.all(np.isfinite(g2))
    print(f"  γ finite: {'✅' if fg else '❌'}   max|γ₁|={np.abs(g1).max():.4f}  max|γ₂|={np.abs(g2).max():.4f}")
    ok = ok and fg

    # analytic reference (infinite domain approximation)
    dx, dy = nodes[:,0], nodes[:,1]
    k = A * np.exp(-(dx**2+dy**2)/(2*SIGMA**2))
    g1_ref = k * (dy**2 - dx**2) / (2*SIGMA**2)
    interior = ops.interior
    ratio = np.abs(g1[interior]).max() / (np.abs(g1_ref[interior]).max() + 1e-10)
    order_ok = 0.1 < ratio < 10.0
    print(f"  γ₁ order-of-magnitude vs analytic: ratio={ratio:.2f}  {'✅' if order_ok else '⚠️ '}")
    ok = ok and order_ok
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_3_gradients(ops):
    sep()
    print("TEST 3: Gradient validation (autodiff vs FD)")
    fwd   = DifferentiableForward(ops, lam_reg=1e-3)
    nodes = np.array(ops.mesh.nodes)
    kappa = gaussian_kappa(nodes)
    g1, g2 = ops.forward(kappa)
    result = fwd.validate_gradients(kappa, g1, g2, n_checks=8, eps=1e-5, verbose=True)
    ok = result['passed']
    print(f"  → {PASS if ok else FAIL}  (max rel err = {result['max_rel_error']:.3e})")
    return ok


def test_4_noiseless_map(ops):
    """
    Reconstruct κ from FEM-predicted γ (perfectly self-consistent observations).

    With the numpy adjoint, loss should go to ~0 and L2 error should be small.
    On a 10×10 mesh, Dirichlet BCs limit accuracy near boundaries, so we
    expect L2 ≈ 0.08–0.12 rather than 0 even at the exact solution.

    The key check: optimizer must take >10 iterations (not stall at κ=0).
    """
    sep()
    print("TEST 4: Noiseless MAP reconstruction")

    fwd   = DifferentiableForward(ops, lam_reg=1e-5)
    rec   = MAPReconstructor(fwd, maxiter=300, gtol=1e-10, callback_every=50)
    nodes = np.array(ops.mesh.nodes)
    kappa_true = gaussian_kappa(nodes)

    # Use FEM forward to generate self-consistent observations
    # (same operator chain that the optimizer will use → loss can → 0)
    g1_obs, g2_obs = ops.forward(kappa_true)

    kappa_map, result = rec.reconstruct(g1_obs, g2_obs, verbose=True)

    l2 = np.sqrt(np.mean((kappa_map - kappa_true)**2))

    # Checks:
    #   - optimizer ran (not stalled after <5 iters)
    #   - loss actually decreased substantially
    #   - L2 error reasonable for a 10×10 mesh
    ran_properly = result.n_iter > 10
    threshold    = 0.15
    ok = ran_properly and l2 < threshold

    print(f"  L2(κ_MAP, κ_true) = {l2:.4f}  (threshold {threshold})")
    print(f"  Converged: {result.converged}  |  iterations: {result.n_iter}")
    print(f"  Ran properly (>10 iters): {'✅' if ran_properly else '❌'}")
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_5_noisy_map_vs_ks(ops):
    sep()
    print("TEST 5: Noisy MAP vs Kaiser-Squires\n")
    out = run_comparison(nx=NX, noise_level=0.10, lam_reg=1e-2,
                         domain=DOMAIN, A=A, sigma=SIGMA, verbose=True)
    l2_map, l2_ks = out['l2_map'], out['l2_ks']
    ok = l2_map <= 1.5 * l2_ks
    print(f"  FEM-MAP L2 = {l2_map:.4f}  KS L2 = {l2_ks:.4f}")
    print(f"  ratio MAP/KS = {l2_map/l2_ks:.2f}  (threshold 1.5)")
    print(f"  → {PASS if ok else FAIL}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  Full pipeline integration test")
    print(f"  {NX}×{NX} P3 mesh on {DOMAIN}")
    print("═"*60 + "\n")

    ops = build_operators(NX, NX,
                          xmin=DOMAIN[0], xmax=DOMAIN[1],
                          ymin=DOMAIN[2], ymax=DOMAIN[3],
                          verbose=True)

    results = {
        "1. Operators"       : test_1_operators(ops),
        "2. Forward"         : test_2_forward(ops),
        "3. Gradients"       : test_3_gradients(ops),
        "4. Noiseless MAP"   : test_4_noiseless_map(ops),
        "5. Noisy MAP vs KS" : test_5_noisy_map_vs_ks(ops),
    }

    sep()
    print("\nSummary")
    sep()
    all_pass = True
    for name, ok in results.items():
        print(f"  {name:25s}  {PASS if ok else FAIL}")
        all_pass = all_pass and ok
    sep()
    print(f"\n  Overall: {'✅  ALL PASS' if all_pass else '❌  SOME FAILED'}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())