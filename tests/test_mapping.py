import json
from dataclasses import replace
import numpy as np
import pytest
from femmi import FlatCatalog, MapperConfig, FEMMapper, map_mass
from femmi.catalog import analytic_gaussian_catalog


def catalogue():
    rng=np.random.default_rng(4)
    xy=rng.uniform(-.7,.7,(24,2))
    return FlatCatalog(*xy.T,np.full(24,.01),np.full(24,-.02),np.linspace(.5,2,24))


@pytest.mark.parametrize('kind',['p3','argyris','hct'])
def test_mapper_fe_evaluation_reuse_and_save(kind,tmp_path):
    c=catalogue(); mapper=FEMMapper(c,MapperConfig(kind,.3,.6,1.))
    result=mapper.reconstruct()
    np.testing.assert_allclose(result.evaluate(np.column_stack([c.x,c.y])),result.kappa,atol=1e-9)
    assert np.isnan(result.evaluate([[10,10]])[0])
    lu=mapper.solver.lu
    other=mapper.reconstruct(-c.g1,-c.g2)
    assert mapper.solver.lu is lu
    np.testing.assert_allclose(other.kappa,-result.kappa,atol=1e-8)
    result.save(tmp_path/'map.npz')
    with np.load(tmp_path/'map.npz',allow_pickle=False) as saved:
        assert json.loads(str(saved['metadata']))['diagnostics']['converged']
        np.testing.assert_array_equal(saved['coefficients'],result.coefficients)
    # Interpolate a polynomial independently of the inverse problem.
    p=np.array([[.11,.12],[-.21,.13]])
    if kind=='p3':
        v=mapper.vertices; k=1+v[:,0]+2*v[:,1]+v[:,0]*v[:,1]
    else:
        def polynomial(p,dx,dy):
            x,y=p
            return {(0,0):1+x+2*y+x*y,(1,0):1+y,(0,1):2+x,(1,1):1}.get((dx,dy),0.)
        k=mapper.space.interpolate(polynomial)
    np.testing.assert_allclose(mapper.evaluate(k,p),1+p[:,0]+2*p[:,1]+p[:,0]*p[:,1],atol=1e-10)


def test_weight_scale_and_lambda_scale_leave_solution_unchanged():
    c=catalogue(); cfg=MapperConfig('p3',.3,.6,1.)
    a=map_mass(c,cfg)
    b=map_mass(replace(c,weight=17*c.weight),replace(cfg,lam=17*cfg.lam))
    np.testing.assert_allclose(a.kappa,b.kappa,atol=1e-8)


def test_fail_fast_config_and_catalogue(tmp_path):
    cfg=MapperConfig('hct',.3,.6,1.)
    cfg.save(tmp_path/'config.yaml')
    assert MapperConfig.from_file(tmp_path/'config.yaml').method=='hct'
    for kw in [dict(observable='reduced_shear'),dict(lam=0),dict(length=-1),dict(rtol=.1)]:
        with pytest.raises(ValueError):replace(cfg,**kw)
    c=catalogue()
    for bad in [replace(c,units='degrees'),replace(c,z=np.ones(24)),
                replace(c,weight=-c.weight),replace(c,g1=np.full(24,np.nan)),
                replace(c,x=np.ones(24),y=np.ones(24))]:
        with pytest.raises(ValueError):FEMMapper(bad,cfg)


def test_cross_validation_has_no_truth_dependency():
    mapper=FEMMapper(catalogue(),MapperConfig('p3',.3,.6,1.))
    report=mapper.select_regularization([.1,1.],[0.,1.],folds=2,seed=7)
    assert len(report['candidates'])==4
    assert all(np.isfinite(r['score']) for r in report['candidates'])
    assert report==mapper.select_regularization([.1,1.],[0.,1.],folds=2,seed=7)


def test_gaussian_reconstruction():
    c=analytic_gaussian_catalog(n_gal=250,sigma=.5,shape_noise=.01,seed=1)
    flat=FlatCatalog(c['x'],c['y'],c['g1'],c['g2'],np.ones(len(c['x'])))
    radius=float(np.max(np.hypot(flat.x,flat.y)))
    result=map_mass(flat,MapperConfig('p3',.1,.5,radius))
    inner=np.hypot(flat.x,flat.y)<1.5
    assert np.corrcoef(result.kappa[inner],c['kappa_true'][inner])[0,1]>.85


def test_cli_round_trip_and_repeated_input_persistence(tmp_path):
    from femmi.cli import main
    c=catalogue(); cfg=MapperConfig('p3',.3,.6,1.)
    cfg.save(tmp_path/'mapper.yaml')
    np.savez(tmp_path/'input.npz',**{k:getattr(c,k) for k in ('x','y','g1','g2','weight')})
    main(['map','--catalogue',str(tmp_path/'input.npz'),'--config',str(tmp_path/'mapper.yaml'),
          '--output',str(tmp_path/'output.npz')])
    with np.load(tmp_path/'output.npz') as data:
        assert len(data['kappa'])==c.n
        assert json.loads(str(data['metadata']))['diagnostics']['relative_residual']<1e-6
    result=FEMMapper(c,cfg).reconstruct(-c.g1,-c.g2)
    result.save(tmp_path/'repeated.npz')
    with np.load(tmp_path/'repeated.npz') as data:
        np.testing.assert_array_equal(data['g1'],-c.g1)


def test_research_fits_pipeline_keeps_weights(monkeypatch):
    from types import SimpleNamespace
    from femmi.config import load_config
    from femmi.pipeline import build_forward_and_data
    import femmi.io
    c=catalogue()
    monkeypatch.setattr(femmi.io,'read_fits_catalog',lambda *a,**kw:
        SimpleNamespace(to_tangent_plane=lambda **kw:c))
    cfg=load_config(None)
    for key,value in {'forward.geometry':'catalog','data.source':'catalog_fits',
                      'data.fits':'unused.fits','forward.radius':1.,'forward.n_boundary':18}.items():
        cfg.set(key,value)
    result=build_forward_and_data(cfg);mesh=result['catalog_mesh']
    np.testing.assert_array_equal(result['weight'][mesh.galaxy_nodes],c.weight[mesh.source_index])
