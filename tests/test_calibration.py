"""Checks for the actual mathematical/study-design failure modes."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu
from femmi.quadratic import QuadraticMAP
from femmi.calibration import adaptive_grid,make_catalogue,catalogue_mapper,fem_fit,calibrate_and_evaluate
from femmi.density import paired_comparison,sample_catalog


def test_quadratic_map_matches_dense_optimum_with_gauge_and_weights():
    rng=np.random.default_rng(3);n=9;m=5
    a=rng.normal(size=(n,n));A=a.T@a+np.eye(n)
    M=sparse.diags(np.arange(1,n+1,dtype=float))
    K=sparse.diags(np.arange(n,dtype=float))
    s1=sparse.csr_matrix(rng.normal(size=(m,n)))
    s2=sparse.csr_matrix(rng.normal(size=(m,n)))
    w=np.array([1,0,2,3,4.]);d=rng.normal(size=2*m)
    model=QuadraticMAP(M,K,s1,s2,splu(sparse.csc_matrix(A)),[0,2],w)
    F=np.vstack([np.column_stack([model.forward(v)[j] for v in np.eye(n)]) for j in (0,1)])
    R=(M+.7**2*K).toarray();ww=np.r_[w,w]
    expected=np.linalg.solve(F.T@(ww[:,None]*F)+.3*R,F.T@(ww*d))
    got,info=model.solve(d[:m],d[m:],.3,.7,rtol=1e-11)
    np.testing.assert_allclose(got,expected,rtol=1e-9,atol=1e-10)
    x=rng.normal(size=n);y=rng.normal(size=2*m)
    assert np.dot(F@x,y)==pytest.approx(np.dot(x,model.transpose(y[:m],y[m:])),rel=1e-12)
    with pytest.raises(RuntimeError,match='did not converge'):
        model.solve(d[:m],d[m:],.3,.7,maxiter=1)


def test_search_expands_and_refines_without_hiding_edges():
    ev=lambda a,b:dict(score=(np.log(a/27))**2+(np.log(b/.03))**2)
    r=adaptive_grid(ev,[[1,3],[.3,1]],max_expansions=6)
    assert not r['boundary_unresolved']
    assert r['parameters'][0]>3 and r['parameters'][1]<.3
    r=adaptive_grid(lambda a,b:dict(score=-a-b),[[1,3],[1,3]],max_expansions=1)
    assert r['boundary_unresolved']


def test_seed_leakage_rejected_before_computation(tmp_path):
    with pytest.raises(ValueError,match='disjoint'):
        calibrate_and_evaluate(dict(calibration_seeds=[0],evaluation_seeds=[0]),tmp_path)


def test_pairing_rejects_mismatches_and_duplicates_and_separates_scenarios():
    rows=[dict(method=m,scenario=sc,n_eff_nominal=5,seed=i,shape_l2=i+v,catalogue_hash=str(i))
          for sc in ('one','two') for i in range(3) for m,v in [('a',0),('b',.2)]]
    assert len(paired_comparison(rows,'a','b'))==2
    with pytest.raises(ValueError,match='duplicate'):
        paired_comparison(rows+[rows[0]],'a','b')
    rows[1]['catalogue_hash']='different'
    with pytest.raises(ValueError,match='different catalogues'):
        paired_comparison(rows,'a','b')


def test_mask_remains_empty_after_clustering():
    x,y=sample_catalog(500,seed=13,masks=[(0,0,1.5)],clustering=.8)
    assert np.min(np.hypot(x,y))>1.5
    with pytest.raises(ValueError,match='entire'):
        sample_catalog(10,masks=[(0,0,4)])


@pytest.mark.parametrize('kind',['p3','argyris','hct'])
def test_catalogue_operator_adjoint_and_converged_map(kind):
    pytest.importorskip('galsim')
    c=make_catalogue(1,71,radius=2,catalog_kw=dict(weight_scatter=.5))
    model=catalogue_mapper(c,kind);rng=np.random.default_rng(2)
    x=rng.normal(size=model.dofs);a=rng.normal(size=len(c.x));b=rng.normal(size=len(c.x))
    u,v=model.solver.forward(x)
    assert np.dot(a,u)+np.dot(b,v)==pytest.approx(np.dot(x,model.solver.transpose(a,b)),rel=1e-9,abs=1e-9)
    values,info,_=fem_fit(model,1,1)
    assert info['converged'] and len(values)==len(c.x)
    assert c.fingerprint==make_catalogue(1,71,radius=2,catalog_kw=dict(weight_scatter=.5)).fingerprint


def test_massivenus_split_rejects_duplicate_contents(tmp_path):
    a=tmp_path/'train';b=tmp_path/'eval';a.mkdir();b.mkdir()
    np.save(a/'a.npy',np.ones((16,16)))
    np.save(b/'renamed.npy',np.ones((16,16)))
    config=dict(truth='massivenus',n_eff=1,calibration_seeds=[100],evaluation_seeds=[0],
        calibration_truth_kw=dict(data_dir=str(a)),evaluation_truth_kw=dict(data_dir=str(b)))
    with pytest.raises(ValueError,match='contents overlap'):
        calibrate_and_evaluate(config,tmp_path/'out')


def test_massivenus_truth_retains_actual_patch_mean(tmp_path):
    from femmi.neural_prior.massivenus import MassiveNuSMaps
    from femmi.truth import massivenus_truth
    y,x=np.mgrid[:16,:16];original=.2+.03*np.cos(x)+.02*np.sin(y)
    np.save(tmp_path/'one.npy',original)
    pool=MassiveNuSMaps(str(tmp_path),16,kappa_std=None)
    np.testing.assert_allclose(pool.sample(1,0,subtract_mean=False)[0],original,atol=2e-8)
    nodes=np.array([[0.,0.],[.2,.3]])
    full=massivenus_truth(nodes,str(tmp_path),1.,n_pix=16,subtract_mean=False)
    zero=massivenus_truth(nodes,str(tmp_path),1.,n_pix=16,subtract_mean=True)
    np.testing.assert_allclose(full[0]-zero[0],original.mean(),atol=2e-8)


def test_equilibrated_lu_forward_transpose_and_multiple_rhs():
    from femmi.linear_solve import EquilibratedLU
    rng=np.random.default_rng(9);a=rng.normal(size=(12,12));a+=10*np.eye(12)
    scale=np.geomspace(1e-5,1e5,12);A=scale[:,None]*a*scale[None,:]
    lu=EquilibratedLU(sparse.csc_matrix(A));b=rng.normal(size=(12,3))
    for trans in ('N','T'):
        mat=A if trans=='N' else A.T
        expected=np.linalg.solve(mat,b)
        np.testing.assert_allclose(lu.solve(b,trans=trans),expected,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(lu.solve(b[:,0],trans=trans),expected[:,0],rtol=1e-10,atol=1e-10)


def test_argyris_dense_catalogue_fresh_residual_regression():
    pytest.importorskip('galsim')
    # This catalogue produced a misleading recursive CG success while the
    # fresh normal-equation residual remained ~7e-6 before LU equilibration.
    c=make_catalogue(20,2);model=catalogue_mapper(c,'argyris')
    _,info,_=fem_fit(model,.09486832980505137,5.4)
    assert info['relative_residual']<1e-7


def test_failed_search_keeps_raw_candidates():
    from femmi.calibration import CalibrationFailure
    with pytest.raises(CalibrationFailure) as exc:
        adaptive_grid(lambda a,b:dict(score=float('inf'),error='solve failed'),[[1,2],[1,2]])
    assert len(exc.value.candidates)==4
    assert all(r['error']=='solve failed' for r in exc.value.candidates)
