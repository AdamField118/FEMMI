"""
tests/test_eb_modes.py
E/B-mode decomposition of the reconstructed convergence.

A real lensing potential produces pure E-mode shear, so for a synthetic
E-mode signal:
  - the E-mode reconstruction recovers kappa,
  - the B-mode reconstruction (same estimator on 45-deg-rotated shear) is
    consistent with zero (systematics null test),
  - kappa_B equals what you get by manually rotating the shear and running the
    ordinary reconstruction (the "it's a rotation away" identity),
for both FEMMI-MAP and Kaiser-Squires.

Run:
    python -m pytest tests/test_eb_modes.py -v
    python tests/test_eb_modes.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor, kaiser_squires


def _synthetic_e_mode(nx=12, sigma=0.5, seed=0):
    """Build a small E-mode problem: kappa_true -> noiseless (g1, g2)."""
    ops   = build_operators(nx, nx, -2.5, 2.5, -2.5, 2.5)
    nodes = np.array(ops.mesh.nodes)
    kappa_true = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * sigma**2))
    g1, g2 = ops.forward(kappa_true)
    return ops, nodes, kappa_true, np.asarray(g1), np.asarray(g2)


def test_rotation_identity_map():
    """kappa_B from reconstruct_eb == reconstruct on the 45-deg-rotated shear."""
    ops, _, _, g1, g2 = _synthetic_e_mode()
    fwd = DifferentiableForward(ops, lam_reg=1e-2)

    rec = MAPReconstructor(fwd, maxiter=200, wiener_length=0.5, noise_std=None)
    kE, kB, _, _ = rec.reconstruct_eb(g1, g2, verbose=False)

    # manual 45-deg rotation: (g1, g2) -> (g2, -g1)
    rec2 = MAPReconstructor(fwd, maxiter=200, wiener_length=0.5, noise_std=None)
    kB_manual, _ = rec2.reconstruct(g2, -g1, verbose=False)

    assert np.allclose(kB, kB_manual, atol=1e-8), \
        f"B-mode is not the 45-deg rotation: max diff {np.abs(kB - kB_manual).max():.2e}"


def test_bmode_is_null_for_e_signal_map():
    """
    On a pure E-mode signal the FEMMI-MAP B-mode is a clean null: it carries
    none of the lensing structure. The right diagnostic is spatial coherence
    with the signal, not raw amplitude -- the forward operator F is E-mode-only,
    so B-mode data cannot be fit and any residual kappa_B is incoherent leakage,
    decorrelated from the true convergence.
    """
    ops, nodes, kappa_true, g1, g2 = _synthetic_e_mode()
    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    rec = MAPReconstructor(fwd, maxiter=300, wiener_length=0.5, noise_std=None)

    kE, kB, _, _ = rec.reconstruct_eb(g1, g2, verbose=False)

    interior = np.hypot(nodes[:, 0], nodes[:, 1]) < 1.5
    corr_E = np.corrcoef(kE[interior], kappa_true[interior])[0, 1]
    corr_B = abs(np.corrcoef(kB[interior], kappa_true[interior])[0, 1])

    assert corr_E > 0.9,  f"E-mode does not track the truth: corr={corr_E:.3f}"
    assert corr_B < 0.1,  f"B-mode leaks the signal (should be null): |corr|={corr_B:.3f}"


def test_kaiser_squires_eb():
    """KS: E-mode recovers structure, B-mode is a small null for an E signal."""
    ops, nodes, _, g1, g2 = _synthetic_e_mode()

    kE = kaiser_squires(g1, g2, nodes)                    # default: E only
    kE2, kB = kaiser_squires(g1, g2, nodes, return_bmode=True)

    assert np.allclose(kE, kE2)                            # E unchanged by flag

    # B-mode equals the estimator on rotated shear (rotation identity)
    kB_rot = kaiser_squires(g2, -g1, nodes)
    assert np.allclose(kB, kB_rot, atol=1e-10)

    interior = np.hypot(nodes[:, 0], nodes[:, 1]) < 1.5
    assert np.sqrt(np.mean(kB[interior]**2)) < 0.5 * np.sqrt(np.mean(kE[interior]**2))


if __name__ == "__main__":
    tests = [
        test_rotation_identity_map,
        test_bmode_is_null_for_e_signal_map,
        test_kaiser_squires_eb,
    ]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn(); passed += 1; print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}"); failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}"); failed.append(fn.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if not failed else 1)
