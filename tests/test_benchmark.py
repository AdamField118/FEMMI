"""
tests/test_benchmark.py
The configuration sweep (femmi.benchmark).

What matters here is the comparison PROTOCOL, not the numbers: every cell must
see the same truth and noise, the KS baseline must always be present, a failing
cell must not abort the sweep, and unimplemented combinations must be reported
rather than silently skipped.

Run:
    python -m pytest tests/test_benchmark.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.benchmark import sweep, to_table, _metrics

pytest.importorskip("galsim", reason="benchmark scores against the GalSim NFW truth")


def test_metrics_are_zero_for_a_perfect_reconstruction():
    t = np.array([0.1, 0.4, 0.2, 0.9])
    m = _metrics(t, t)
    assert m["rel_l2"] < 1e-12
    assert m["rel_l2_dc_removed"] < 1e-12
    assert m["mean_err"] < 1e-12


def test_metrics_separate_dc_error_from_shape_error():
    """A reconstruction that is right in shape but offset by a constant must show
    a large mean error and ~zero shape error -- the exact distinction the
    mass-sheet result turns on."""
    t = np.array([0.1, 0.4, 0.2, 0.9])
    m = _metrics(t + 0.25, t)
    assert m["rel_l2_dc_removed"] < 1e-12
    assert abs(m["mean_err"] - 0.25) < 1e-12


def test_sweep_includes_the_ks_baseline_and_ranks_results():
    rows = sweep(dict(element=["p3"], prior=["wiener"], method=["map"]),
                 nx=10, verbose=False)
    methods = {r["method"] for r in rows}
    assert "ks" in methods and "map" in methods
    for r in rows:
        if "error" not in r:
            assert r["dofs"] > 0 and r["seconds"] >= 0
            assert np.isfinite(r["rel_l2"])
    assert "element" in to_table(rows)


def test_unimplemented_element_is_reported_not_skipped():
    """C^1 elements are validated but have no FEM-BEM coupling yet; the sweep must
    say so instead of dropping the row."""
    rows = sweep(dict(element=["argyris"], prior=["wiener"], method=["map"]),
                 nx=10, include_ks=False, verbose=False)
    assert len(rows) == 1
    assert "error" in rows[0] and "coupling" in rows[0]["error"]


def test_a_failing_cell_does_not_abort_the_sweep():
    rows = sweep(dict(element=["p3"], prior=["wiener"], method=["nonsense"]),
                 nx=10, include_ks=True, verbose=False)
    assert any("error" in r for r in rows)
    assert any(r.get("method") == "ks" and "error" not in r for r in rows)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
