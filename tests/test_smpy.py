import numpy as np
import pytest
pytest.importorskip('smpy')
from femmi.smpy import create_maps,sample_grid


def test_upstream_ks_fourier_sign_orientation_and_bmode():
    y,x=np.mgrid[:24,:32]
    phase=2*np.pi*(3*x/32+2*y/24)
    k=np.cos(phase); kx,ky=3/32,2/24
    a=(kx*kx-ky*ky)/(kx*kx+ky*ky); b=2*kx*ky/(kx*kx+ky*ky)
    e,cross,_=create_maps(a*k,b*k,np.ones_like(k),'smpy_ks')
    np.testing.assert_allclose(e,k,atol=2e-14)
    np.testing.assert_allclose(cross,0,atol=2e-14)


def test_ksplus_availability_uses_weights_and_shear_mode(monkeypatch):
    from smpy.mapping_methods.ks_plus.ks_plus import KSPlusMapper
    original=KSPlusMapper._create_mask
    seen=[]
    def mask(self,a,b):
        out=original(self,a,b);seen.append(out.copy());return out
    monkeypatch.setattr(KSPlusMapper,'_create_mask',mask)
    a=np.zeros((12,12)); w=np.ones_like(a);w[3:7,4:8]=0
    e,b,cfg=create_maps(a,a,w,'smpy_ks_plus',iterations=4)
    np.testing.assert_array_equal(seen[0],w)
    assert cfg['methods']['ks_plus']['reduced_shear_iterations']==1
    np.testing.assert_allclose(e,0,atol=1e-12)


def test_aperture_output_is_separate_and_finite():
    a=np.ones((12,12))*.01
    e,b,_=create_maps(a,a,np.ones_like(a),'smpy_aperture',aperture_scale=2.)
    assert e.shape==a.shape and np.all(np.isfinite(e))


def test_grid_centres_and_spatial_regions():
    from femmi.comparison import spatial_metrics
    grid=np.arange(16.).reshape(4,4)
    points=np.array([[-.75,-.75],[.75,.75],[3,3]])
    np.testing.assert_allclose(sample_grid(grid,points,1),[0,15,np.nan])
    p=np.array([[.1,0],[.6,0],[.9,0]])
    truth=np.array([3.,2.,1.]);regions=dict(field=np.ones(3,bool),mask=np.array([1,0,0],bool))
    m=spatial_metrics(truth+7,truth,p,regions,1.)
    assert m['field_shape_l2']==pytest.approx(0)
    assert m['mask_rmse']==pytest.approx(0)
    assert m['aperture_contrast_error']==pytest.approx(0)
