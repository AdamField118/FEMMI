"""Discrete SVD checks; geometry indicators are not support estimators."""
import numpy as np
import pytest
from femmi.operators import build_operators
from femmi.svd_analysis import compute_svd, FactorizationIndicator, LinearSamplingIndicator


@pytest.fixture(scope="module")
def ops():
    return build_operators(3,3,-2,2,-2,2,verbose=False)


def test_lanczos_matches_dense_singular_values_and_both_equations(ops):
    dense=compute_svd(ops,n_singular=8,method="dense")
    iterative=compute_svd(ops,n_singular=8,tol=1e-11)
    np.testing.assert_allclose(iterative.sigma,dense.sigma,rtol=1e-8)
    assert max(iterative.residuals) < 1e-8
    assert max(dense.residuals) < 1e-10
    np.testing.assert_allclose(iterative.U.T@iterative.U,np.eye(8),atol=1e-8)


def test_legacy_indicators_match_documented_geometry_scores(ops):
    svd=compute_svd(ops,n_singular=8,method="dense")
    points=np.array([[0.,0.],[.7,.2],[1.,1.]])
    fi=FactorizationIndicator(ops,svd_result=svd)
    li=LinearSamplingIndicator(ops,svd_result=svd)
    coeff=np.array([svd.U.T@fi.probe_function(p) for p in points])
    active=svd.sigma>fi.noise_floor
    expected=np.sum(coeff[:,active]**2/svd.sigma[active],axis=1)
    np.testing.assert_allclose(fi.indicator_map(points),expected/expected.max())
    expected=np.sqrt(np.sum((coeff*svd.sigma/(svd.sigma**2+li.alpha))**2,axis=1))
    np.testing.assert_allclose(li.indicator_map(points),expected/expected.max())
