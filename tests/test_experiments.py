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
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.experiments import (square_ops, constant_mode_response,
                               ks_constant_mode_response, mass_sheet_recovery,
                               boundary_error_profile, forward_convergence,
                               femmi_forward_shear, independent_truth_recovery,
                               shear_convergence, shear_noise_amplification)


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


# --------------------------------------------------------------------------- #
# the same claims on INDEPENDENT truth (no inverse crime)
# --------------------------------------------------------------------------- #
def test_independent_truth_is_not_femmis_own_forward():
    """The point of femmi.truth: its shear must differ materially from what FEMMI's
    forward would produce for the same kappa. If these ever agreed closely, the
    'independent' test would have quietly become an inverse crime again."""
    pytest.importorskip("galsim")
    from femmi.truth import galsim_nfw_truth
    ops = square_ops(10)
    nodes = np.array(ops.mesh.nodes)
    kt, g1t, g2t = galsim_nfw_truth(nodes)
    g1f, g2f = femmi_forward_shear(ops, kt)
    mismatch = np.sqrt(np.mean((g1f - g1t)**2 + (g2f - g2t)**2))
    signal = np.sqrt(np.mean(g1t**2 + g2t**2))
    assert mismatch > 0.2 * signal


def test_femmi_beats_ks_at_the_boundary_on_independent_truth():
    """P1.4 on neutral truth. For a compact, isolated halo -- the far-field-zero
    regime FEMMI's BEM actually assumes -- FEMMI's shape error near the domain edge
    beats KS. This is the claim that SURVIVES removing the inverse crime."""
    pytest.importorskip("galsim")
    d = independent_truth_recovery(nx=12, noise_std=0.02, seed=0,
                                   truth_kw={"halos": ((2.0e14, 4.0, (0.0, 0.0)),)})
    edge = d["r_err"] >= d["half_width"]
    assert np.nanmean(d["err_femmi_r"][edge]) < np.nanmean(d["err_ks_r"][edge])


def test_dc_mode_is_not_recovered_on_independent_truth():
    """The HONEST SCOPE of the mass-sheet claim, pinned as a regression guard.

    On self-consistent shear FEMMI recovers the mean kappa ~40x better than KS
    (test_femmi_recovers_absolute_normalisation). On INDEPENDENT truth it does not:
    both land near zero mean. FEMMI's forward does respond to a uniform sheet where
    KS's annihilates it (test_dc_mode_femmi_observes_ks_annihilates), but that
    response is concentrated in the domain corners, so it does not translate into
    practical DC recovery from realistic data.

    If a future change genuinely fixes this, THIS TEST SHOULD FAIL -- and the
    mass-sheet claim can then be strengthened."""
    pytest.importorskip("galsim")
    d = independent_truth_recovery(nx=12, noise_std=0.02, seed=0,
                                   truth_kw={"halos": ((2.0e14, 4.0, (0.0, 0.0)),)})
    assert d["err_femmi"] > 0.5 * d["err_ks"]        # NOT the ~0.02x of the crime case
    assert abs(d["mean_femmi"]) < 0.5 * abs(d["mean_truth"])


# --------------------------------------------------------------------------- #
# P0.3 -- shear extraction
# --------------------------------------------------------------------------- #
def test_shear_extraction_approaches_second_order():
    """Both extractions converge toward the P3 theory rate O(h^2); the local order
    on the finest pair is the asymptotic number (a single fitted slope across
    coarse meshes understates it)."""
    d = shear_convergence(nxs=(16, 24, 32))
    assert d["local_nodal"][-1] > 1.7
    assert d["local_recovered"][-1] > 1.7


def test_variational_recovery_beats_nodal_sampling():
    """Same rate, better constant: recovering the shear variationally instead of
    sampling element Hessians at the nodes cuts the error substantially."""
    d = shear_convergence(nxs=(16, 24, 32))
    assert np.all(d["err_recovered"] < d["err_nodal"])


def test_shear_noise_is_amplified_by_h_squared():
    """A second derivative amplifies noise in psi by h^-2, so with fixed noise the
    error is U-shaped in h: the finest mesh must be WORSE than the optimum. This is
    why the O(h^2) rate is not reachable catalog-native."""
    d = shear_noise_amplification(nxs=(8, 12, 16, 24, 32), noise_std=1e-4)
    assert d["err_noisy"][-1] > d["err_noisy"].min()      # refining past h_opt hurts
    assert d["h_opt"] > d["h"].min()                      # optimum is not the finest mesh
    assert d["err_clean"][-1] < d["err_clean"][0]         # noiseless still converges


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
