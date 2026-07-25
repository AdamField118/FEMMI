"""
tests/test_experiments.py
The paper's core experiments (femmi.experiments) -- the structural FEMMI-vs-KS
claims, on small meshes.

  - DC / mass-sheet mode: FEMMI's forward responds to a uniform sheet; the KS/FFT
    forward annihilates it exactly (the injectivity-at-DC claim, P1.6);
  - absolute-normalisation recovery: FEMMI recovers the mean kappa far better than
    KS on self-consistent shear (P0.1);
  - boundary bias: FEMMI's error at the domain edge beats KS (P0.2);
  - forward shear converges with resolution (P1.5).

Run:
    python -m pytest tests/test_experiments.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.experiments import (square_ops, constant_mode_response,
                               ks_constant_mode_response, mass_sheet_recovery,
                               boundary_error_profile, forward_convergence)


def test_dc_mode_femmi_observes_ks_annihilates():
    """FEMMI's forward gives an O(1) response to a uniform mass sheet; the KS/FFT
    forward gives exactly zero (the DC mode is in its null space)."""
    assert constant_mode_response(square_ops(12)) > 0.1
    assert ks_constant_mode_response() < 1e-10


def test_femmi_recovers_absolute_normalisation():
    """On self-consistent shear, FEMMI recovers the mean kappa much better than KS,
    which floats by an unconstrained additive constant."""
    d = mass_sheet_recovery(nx=12, noise_std=0.02, seed=0)
    assert d["err_femmi"] < 0.3 * d["err_ks"]          # FEMMI clearly closer to truth mean
    assert d["err_ks"] > 0.5 * abs(d["mean_truth"])    # KS misses ~the whole mean


def test_femmi_beats_ks_at_the_boundary():
    d = boundary_error_profile(nx=16, seed=0)
    assert d["err_femmi"][-1] < d["err_ks"][-1]         # smaller error at the outer edge


def test_forward_potential_converges_at_order_four():
    """FEMMI's recovered potential psi converges at the P3 theory rate O(h^4),
    validating the forward operator F (no floor: the manufactured psi is compact)."""
    h, err, order = forward_convergence(nxs=(8, 12, 16, 24, 32))
    assert err[-1] < err[0]                             # error drops with resolution
    assert order > 3.5                                  # clean O(h^4), not regularity-limited


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
