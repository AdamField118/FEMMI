"""
tests/test_bmode_diagnostics.py
B-mode quality diagnostics (MAPReconstructor.bmode_diagnostics):

  - a clean E-mode signal flags 'clean' and the B-channel noise floor
    (delta_noise) undercuts the signal-biased MAD estimate,
  - injecting coherent B-mode shear raises the coherent B/E power and the
    B-mode SNR, tripping the flag toward 'contaminated',
  - delta_noise (both coherent modes removed) tracks the true per-component
    noise far better than MAD on the raw shear.

Run:
    python -m pytest tests/test_bmode_diagnostics.py -v
    python tests/test_bmode_diagnostics.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators      import build_operators
from femmi.forward        import DifferentiableForward
from femmi.inverse        import MAPReconstructor, BModeDiagnostics
from femmi.regularization import estimate_noise_level


def _setup(nx=14, noise_level=0.10, seed=42):
    ops   = build_operators(nx, nx, -2.5, 2.5, -2.5, 2.5, verbose=False)
    nodes = np.array(ops.mesh.nodes)
    kt    = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * 0.5**2))
    g1, g2 = (np.asarray(a) for a in ops.forward(kt))
    rng   = np.random.default_rng(seed)
    noise = noise_level * np.std(np.hypot(g1, g2))
    g1o   = g1 + rng.normal(0, noise, g1.shape)
    g2o   = g2 + rng.normal(0, noise, g2.shape)
    return ops, nodes, g1o, g2o, noise


def _make_rec(ops, g1, g2):
    ns  = estimate_noise_level(np.concatenate([g1, g2]), method='mad')
    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    return MAPReconstructor(fwd, maxiter=300, wiener_length=0.5, noise_std=ns)


def _inject_bmode(ops, nodes, g1, g2, amp, center=(0.6, -0.4), sigma=0.4):
    """Add a coherent B-mode: E-shear of a blob, rotated 45 deg -> (s2, -s1)."""
    ksys   = amp * np.exp(-((nodes[:, 0] - center[0])**2 +
                            (nodes[:, 1] - center[1])**2) / (2 * sigma**2))
    s1, s2 = (np.asarray(a) for a in ops.forward(ksys))
    return g1 + s2, g2 - s1


def test_clean_signal_flags_clean():
    ops, nodes, g1, g2, noise = _setup()
    diag, _, _ = _make_rec(ops, g1, g2).bmode_diagnostics(g1, g2, verbose=False)

    assert isinstance(diag, BModeDiagnostics)
    assert diag.flag == 'clean', diag.summary()
    assert diag.bmode_snr < 1.0
    # coherent B is a small fraction of the E signal (only numerical leakage)
    assert diag.bmode_to_emode < 0.25


def test_noise_floor_beats_mad():
    """delta_noise (both modes removed) is closer to the true noise than MAD."""
    ops, nodes, g1, g2, noise = _setup()
    diag, _, _ = _make_rec(ops, g1, g2).bmode_diagnostics(g1, g2, verbose=False)

    # MAD on raw shear is biased high by the E signal; the noise floor is not.
    assert diag.delta_noise < diag.delta_mad
    assert diag.delta_consistency < 1.0
    # and delta_noise is within a factor ~2 of the injected per-component noise
    assert 0.5 * noise < diag.delta_noise < 3.0 * noise


def test_injected_bmode_trips_flag():
    ops, nodes, g1, g2, noise = _setup()
    clean, _, _ = _make_rec(ops, g1, g2).bmode_diagnostics(g1, g2, verbose=False)

    g1c, g2c = _inject_bmode(ops, nodes, g1, g2, amp=0.7)
    contam, _, _ = _make_rec(ops, g1c, g2c).bmode_diagnostics(g1c, g2c, verbose=False)

    # coherent B power and SNR both rise under injection
    assert contam.bmode_to_emode > 2.0 * clean.bmode_to_emode
    assert contam.bmode_snr      > clean.bmode_snr
    assert contam.flag in ('marginal', 'contaminated')
    assert contam.flag != 'clean'


def test_estimate_noise_bmode_recovers_true_noise():
    """
    estimate_noise_bmode (delta_noise at a self-consistent Morozov lambda) is a
    far less biased noise estimate than MAD on the raw shear, which the E-mode
    signal inflates. This is what makes it worth feeding back to Morozov.
    """
    ops, nodes, g1, g2, noise = _setup(nx=14, noise_level=0.10, seed=7)
    from femmi.operators import build_operators  # noqa: F401 (kept for parity)
    rec = _make_rec(ops, g1, g2)
    rec.noise_std = None

    mad = estimate_noise_level(np.concatenate([g1, g2]), method='mad')
    dnb = rec.estimate_noise_bmode(g1, g2, maxiter=150)

    assert dnb < mad, f"delta_noise={dnb:.3e} not below MAD={mad:.3e}"
    assert abs(dnb - noise) < abs(mad - noise), \
        f"delta_noise={dnb:.3e} not closer to true={noise:.3e} than MAD={mad:.3e}"


if __name__ == "__main__":
    tests = [
        test_clean_signal_flags_clean,
        test_noise_floor_beats_mad,
        test_injected_bmode_trips_flag,
        test_estimate_noise_bmode_recovers_true_noise,
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
