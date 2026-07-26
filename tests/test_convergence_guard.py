"""
tests/test_convergence_guard.py
The floor guard (femmi.convergence).

This exists because the project reported a floored curve as a convergence order
TWICE -- once for the forward shear against infinite-domain analytic Gaussian
shear, and once for the coupled C^1 solve, which was written up as the square's
corner singularity capping the rate before being retracted (MATH.md 18.3f).

The regression tests below use the ACTUAL historical numbers from both mistakes.
If a future change makes the guard permissive enough to let those through again,
these fail.

Run:
    python -m pytest tests/test_convergence_guard.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.convergence import fit_order, floor_diagnosis, local_orders, FloorError


# the two curves that fooled us, verbatim
FLOORED_SQUARE = ([1.25, 0.8333, 0.625, 0.4167, 0.3125],
                  [1.3256e-1, 7.3675e-2, 4.4053e-2, 2.9428e-2, 2.6186e-2])
FLOORED_CIRCLE = ([1.3090, 0.8727, 0.6545, 0.4363, 0.3272],
                  [2.6119e-1, 1.5988e-1, 1.1387e-1, 9.7882e-2, 8.9910e-2])
# and three genuine ones that must still be accepted
GOOD_SQUARE = ([1.25, 0.8333, 0.625, 0.4167, 0.3125],
               [2.8627e-1, 1.1291e-1, 4.0992e-2, 1.1332e-2, 3.5650e-3])
GOOD_CIRCLE = ([1.3090, 0.8727, 0.6545, 0.4363, 0.3272],
               [6.5206e-2, 6.2268e-2, 2.5959e-2, 5.5832e-3, 1.8025e-3])
GOOD_PSI = ([0.625, 0.4167, 0.3125, 0.2083, 0.15625],
            [1.1052e-1, 5.3163e-2, 3.3546e-2, 1.7086e-2, 1.0189e-2])


@pytest.mark.parametrize("h,err", [FLOORED_SQUARE, FLOORED_CIRCLE])
def test_the_two_historical_floors_are_refused(h, err):
    """Both of these returned an innocent-looking slope (~1.2, ~0.9) that was
    reported as a convergence order. They must now raise."""
    with pytest.raises(FloorError, match="floored reference"):
        fit_order(h, err)
    d = floor_diagnosis(h, err)
    assert d["floored"] and d["reasons"]


@pytest.mark.parametrize("h,err", [GOOD_SQUARE, GOOD_CIRCLE, GOOD_PSI])
def test_genuine_convergence_is_accepted(h, err):
    """The guard must not be so strict that real results stop working -- these
    are the measured O(h^4) and O(h^2) curves currently quoted in MATH.md."""
    assert not floor_diagnosis(h, err)["floored"]
    assert fit_order(h, err) > 1.0


def test_tail_flatness_is_caught_even_when_the_coarse_end_looks_fine():
    """The failure mode that actually bit: a curve that converges nicely at first
    and then plateaus. Averaging over the whole range hides it; the local order at
    the fine end does not."""
    h = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    err = np.array([1.0, 0.25, 0.0625, 0.05, 0.049])      # O(h^2) then flat
    d = floor_diagnosis(h, err)
    assert d["total_decay"] > 4.0                          # decay test alone passes
    assert d["floored"]                                    # tail test still catches it
    assert "gone flat" in " ".join(d["reasons"])


def test_on_floor_modes():
    h, err = FLOORED_SQUARE
    with pytest.warns(RuntimeWarning, match="floored reference"):
        v = fit_order(h, err, on_floor="warn")
    assert np.isfinite(v)
    assert np.isfinite(fit_order(h, err, on_floor="ignore"))
    with pytest.raises(ValueError, match="unknown on_floor"):
        fit_order(h, err, on_floor="nonsense")


def test_short_sweep_is_not_mistaken_for_a_floor():
    """Regression for a false positive the guard itself caused. A 3-point O(h^2)
    sweep spanning only 2x in h can decay at most 4x, so an ABSOLUTE "must fall
    4x" rule rejected a perfectly healthy curve (this is the real
    shear_convergence(nxs=(16,24,32)) data). The decay threshold is range-aware
    for exactly this reason."""
    h = [0.3125, 0.2083, 0.15625]
    err = [5.9728e-2, 3.0660e-2, 1.8202e-2]
    d = floor_diagnosis(h, err)
    assert d["total_decay"] < 4.0          # would have failed an absolute rule
    assert d["tail_order"] > 1.5           # but it is plainly converging
    assert not d["floored"]
    assert fit_order(h, err) > 1.0


def test_two_point_fit_is_flagged_unverified():
    """A 2-point sweep gives a slope but no way to distinguish a rate from a
    plateau, so it must warn rather than either failing or being trusted."""
    with pytest.warns(RuntimeWarning, match="unverified"):
        v = fit_order([1.0, 0.5], [1.0, 0.25])
    assert abs(v - 2.0) < 1e-12


def test_input_validation():
    with pytest.raises(ValueError, match="strictly decreasing"):
        floor_diagnosis([0.1, 0.2, 0.4], [1.0, 0.5, 0.25])      # h increasing
    with pytest.raises(ValueError, match="at least 3"):
        floor_diagnosis([1.0, 0.5], [1.0, 0.25])
    with pytest.raises(ValueError, match="positive"):
        floor_diagnosis([1.0, 0.5, 0.25], [1.0, 0.0, 0.25])


def test_local_orders_match_a_known_power_law():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    assert np.allclose(local_orders(h, h**3), 3.0)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
