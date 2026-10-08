"""Algebraic acceptance criteria for the fixed catalogue estimator."""

import numpy as np
import pytest
from femmi import FlatCatalog, MapperConfig, FEMMapper
from femmi.diagnostics import mass_norm, operator_checks


@pytest.mark.parametrize("kind", ["p3", "argyris", "hct"])
def test_superposition_and_conditional_b(kind):
    rng = np.random.default_rng(731)
    xy = rng.uniform(-1, 1, (28, 2))
    c = FlatCatalog(
        *xy.T, rng.normal(size=28), rng.normal(size=28), rng.uniform(0.2, 2, 28)
    )
    m = FEMMapper(c, MapperConfig(kind, 0.3, 0.6, 1.5))
    a, b = m.solver.forward(rng.normal(size=m.dofs))
    fit = m.reconstruct(a, b)
    r = m.diagnose_b(a, b, e_fit=fit)
    checks = operator_checks(m)
    assert checks["adjoint_relative"] < 1e-8
    assert checks["cross_skew_relative"] < 1e-8
    assert r["diagnostics"]["closure_relative"] < 2e-5
    # Discrete E inputs need not have zero rotated-shear response.
    assert r["diagnostics"]["raw_b_l2"] > 1e-6 * mass_norm(m, fit.coefficients)
    other = m.reconstruct()
    combined = m.reconstruct(a + c.g1, b + c.g2)
    assert mass_norm(
        m, combined.coefficients - fit.coefficients - other.coefficients
    ) < 2e-5 * mass_norm(m, combined.coefficients)
    with pytest.raises(ValueError, match="observations"):
        m.diagnose_b(e_fit=fit)
    with pytest.raises(ValueError, match="same prior"):
        m.diagnose_b(a, b, e_fit=fit, b_fit=m.reconstruct(b, -a, lam=0.1))


def test_boundary_configuration():
    for kw in (
        {"boundary_padding": 1},
        {"boundary_nodes": 11},
        {"boundary_nodes": True},
    ):
        with pytest.raises(ValueError):
            MapperConfig("p3", 0.3, 0.6, 2.0, **kw)
