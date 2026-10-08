import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu
from femmi import FEMMapper, MapperConfig, FlatCatalog
from femmi.quadratic import QuadraticMAP
from femmi.eb import joint_map, sampled_eb_overlap


def toy():
    rng = np.random.default_rng(175)
    n, q = 7, 12
    f1, f2 = rng.normal(size=(2, q, n))
    weights = np.linspace(.2, 2, q); weights[0] = 0
    m = eye(n, format='csc'); k = csc_matrix(np.diag(np.arange(1, n+1)))
    s = QuadraticMAP(m,k,csc_matrix(f1),csc_matrix(f2),splu(-2*m),np.array([],dtype=int),weights)
    a,b=rng.normal(size=(2,q))
    c=FlatCatalog(np.arange(q),np.zeros(q),a,b,weights)
    model=SimpleNamespace(solver=s,dofs=n,catalogue=c,value_indices=np.arange(n),
        config=MapperConfig('p3',.3,.6,20,rtol=1e-11,residual_tolerance=1e-8))
    return model,f1,f2


def test_joint_matches_independent_dense_normal_solve_and_gradient():
    model,f1,f2=toy();c=model.catalogue
    result=joint_map(model,lam_b=.7,length_b=.2)
    a=np.block([[f1,-f2],[f2,f1]])
    w=np.r_[c.weight,c.weight];d=np.r_[c.g1,c.g2]
    r1=.3*(np.eye(model.dofs)+.6**2*model.solver.K.toarray())
    r2=.7*(np.eye(model.dofs)+.2**2*model.solver.K.toarray())
    from scipy.linalg import block_diag
    h=a.T@(w[:,None]*a)+block_diag(r1,r2)
    rhs=a.T@(w*d)
    truth=np.linalg.solve(h,rhs)
    z=np.r_[result.e_coefficients,result.b_coefficients]
    np.testing.assert_allclose(z,truth,rtol=1e-8,atol=1e-10)
    direction=np.linspace(-1,1,len(z));eps=1e-5
    def loss(x):return np.dot(w,(a@x-d)**2)+x@block_diag(r1,r2)@x
    assert (loss(z+eps*direction)-loss(z-eps*direction))/(2*eps)==pytest.approx(0,abs=1e-7)
    assert result.diagnostics['objective']==pytest.approx(loss(z))
    e,_=model.solver.solve(c.g1,c.g2,.3,.6)
    assert loss(z)<=loss(np.r_[e,np.zeros(model.dofs)])+1e-10


@pytest.mark.parametrize('kind',['p3','argyris','hct'])
def test_real_joint_rotation_and_persistence(kind,tmp_path):
    rng=np.random.default_rng(74);xy=rng.uniform(-.7,.7,(24,2))
    c=FlatCatalog(*xy.T,*rng.normal(size=(2,24)),np.linspace(.2,2,24))
    m=FEMMapper(c,MapperConfig(kind,.3,.6,1.))
    fit=m.reconstruct_eb()
    rotated=m.reconstruct_eb(-c.g2,c.g1)
    np.testing.assert_allclose(rotated.e_coefficients,-fit.b_coefficients,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(rotated.b_coefficients,fit.e_coefficients,atol=1e-6,rtol=1e-5)
    e1,e2=m.solver.forward(fit.e_coefficients);b1,b2=m.solver.forward(fit.b_coefficients)
    np.testing.assert_allclose(fit.predicted_g1,e1-b2)
    np.testing.assert_allclose(fit.predicted_g2,e2+b1)
    e,b=fit.evaluate(xy)
    np.testing.assert_allclose(e,fit.kappa_e,atol=1e-8)
    np.testing.assert_allclose(b,fit.kappa_b,atol=1e-8)
    fit.save(tmp_path/'joint.npz')
    with np.load(tmp_path/'joint.npz',allow_pickle=False) as saved:
        assert json.loads(str(saved['metadata']))['diagnostics']['converged']
    report=sampled_eb_overlap(m)
    assert report['e_rank']==report['observation_dimension']
    assert min(report['principal_cosines'])>1-1e-9
    with pytest.raises(ValueError,match='max_entries'):
        sampled_eb_overlap(m,max_entries=1)


def test_orthogonal_modes_recover_real_b_without_suppression():
    model,_,_=toy();n=model.dofs;q=len(model.catalogue.x)
    f=np.zeros((q,n));f[:n]=np.eye(n)
    model.solver.S1=csc_matrix(f);model.solver.S2=csc_matrix(np.zeros_like(f))
    e=np.linspace(-.1,.2,n);b=np.linspace(.2,-.05,n)
    result=joint_map(model,f@e,f@b,lam_e=1e-7,lam_b=1e-7)
    np.testing.assert_allclose(result.e_coefficients[1:],e[1:],atol=1e-7)
    np.testing.assert_allclose(result.b_coefficients[1:],b[1:],atol=1e-7)


def test_joint_missing_values_invalid_priors_and_failure():
    model,_,_=toy();c=model.catalogue
    a=c.g1.copy();b=c.g2.copy();a[0]=b[0]=np.nan
    x=joint_map(model,a,b);y=joint_map(model)
    np.testing.assert_allclose(x.kappa_b,y.kappa_b)
    for kw in (dict(lam_b=0),dict(length_e=-1),dict(lam_e=np.inf)):
        with pytest.raises(ValueError):joint_map(model,**kw)
    with pytest.raises(ValueError):joint_map(model,a)
    from dataclasses import replace
    model.config=replace(model.config,maxiter=1)
    with pytest.raises(RuntimeError,match='did not converge'):joint_map(model)
