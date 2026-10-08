import numpy as np
import pytest
from femmi.protocol import validate_config, fits_roundtrip, iteration_stability
from femmi.calibration import make_catalogue
from femmi.aperture import convergence_aperture, quadrature_control
from femmi.catalog import analytic_gaussian_shear


def config():
    return dict(
        n_eff=3,
        methods=["p3", "smpy_ks_plus"],
        calibration_seeds=list(range(100, 105)),
        evaluation_seeds=list(range(20)),
        evaluation_grid=40,
        radius=3,
        ks_plus_iterations=100,
        ks_plus_iteration_check=[50, 100, 200],
    )


def test_protocol_guards():
    validate_config(config())
    for kw in (
        {"evaluation_seeds": [100]},
        {"calibration_seeds": [100]},
        {"allow_unresolved": True},
        {"allow_unstable_iterations": True},
        {"ks_plus_iteration_check": [50, 100]},
        {"selection_metric": "training_error"},
        {"aperture_radius_arcmin": 0.001},
    ):
        with pytest.raises(ValueError):
            validate_config(config() | kw)


def test_fits_transport(tmp_path):
    pytest.importorskip("astropy")
    pytest.importorskip("galsim")
    c = make_catalogue(3, 21, radius=2.0, truth="nfw")
    d = fits_roundtrip(c, tmp_path / "cat.fits")
    for key in ("x", "y", "g1", "g2", "weight", "truth"):
        np.testing.assert_allclose(getattr(c, key), getattr(d, key), rtol=0, atol=1e-10)


def test_iteration_gate(monkeypatch):
    from femmi import smpy

    pytest.importorskip("galsim")
    c = make_catalogue(3, 21, radius=2.0, truth="nfw")

    def reconstruct(c, m, a, b, iterations):
        grid = np.arange(64).reshape(8, 8) * (1 + 1 / iterations)
        return None, None, None, grid, None

    monkeypatch.setattr(smpy, "reconstruct", reconstruct)
    assert iteration_stability([c], (8, 0), [50, 100, 200], 100)["accepted"]
    assert not iteration_stability([c], (8, 0), [50, 100, 200], 100, tolerance=0.001)[
        "accepted"
    ]


def test_aperture_quadrature_refines():
    pytest.importorskip("smpy")
    errors = []
    for n in (40, 80, 160):
        axis = (np.arange(n) + 0.5) * 6 / n - 3
        x, y = np.meshgrid(axis, axis)
        xy = np.c_[x.ravel(), y.ravel()]
        k, a, b = analytic_gaussian_shear(xy, sigma=0.7, amp=0.1, center=(0.3, -0.2))
        regions = {"field": np.linalg.norm(xy, axis=1) < 3}
        report = quadrature_control(k, a, b, regions, dict(evaluation_grid=n), 3.0)
        errors.append(report["relative_l2"])
        const, _ = convergence_aperture(np.ones((n, n)), n / 8)
        assert np.max(np.abs(const[n // 3 : 2 * n // 3, n // 3 : 2 * n // 3])) < 1e-14
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert errors[-1] < 0.005


def test_aperture_gate_precedes_calibration(tmp_path):
    pytest.importorskip("galsim")
    pytest.importorskip("smpy")
    import json
    from femmi.calibration import calibrate_and_evaluate, CalibrationFailure

    c = config() | dict(
        methods=["p3"],
        evaluation_grid=24,
        aperture_comparison=True,
        aperture_quadrature_tolerance=1e-8,
    )
    with pytest.raises(CalibrationFailure, match="refine evaluation_grid"):
        calibrate_and_evaluate(c, tmp_path)
    report = json.loads((tmp_path / "calibration.json").read_text())
    assert report["calibrations"] == {}
    assert not report["aperture_quadrature_check"]["accepted"]
    assert not (tmp_path / "evaluation.json").exists()
