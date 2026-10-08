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


@pytest.mark.parametrize('shape',[(15,17),(16,18)])
def test_corrected_ksplus_forward_retains_b_and_roundtrips_resolved_modes(shape):
    from smpy.config import Config
    from smpy.mapping_methods.ks_plus.ks_plus import KSPlusMapper
    from femmi.smpy import ks_plus_forward_shear
    m=KSPlusMapper(Config.from_defaults('ks_plus').to_dict())
    y,x=np.indices(shape)
    e=.2*np.cos(2*np.pi*(x/shape[1]+2*y/shape[0]))
    b=.3*np.sin(2*np.pi*(2*x/shape[1]-y/shape[0]))
    g1,g2=ks_plus_forward_shear(e,b)
    ee,bb=m._gamma_to_kappa(g1,g2)
    np.testing.assert_allclose(ee,e,atol=1e-14)
    np.testing.assert_allclose(bb,b,atol=1e-14)
    with np.errstate(invalid='ignore',divide='ignore'):
        old1,old2=m._kappa_to_gamma(np.zeros(shape),b)
    assert np.linalg.norm(old1)+np.linalg.norm(old2)<1e-13
    # Preserve upstream E-only convention, including its even-grid Nyquist rule.
    with np.errstate(invalid='ignore',divide='ignore'):
        a,c=m._kappa_to_gamma(e,np.zeros(shape))
    aa,cc=ks_plus_forward_shear(e,np.zeros(shape))
    np.testing.assert_allclose(a,aa,atol=1e-14)
    np.testing.assert_allclose(c,cc,atol=1e-14)


def test_ksplus_schedule_and_correction_are_explicit():
    a=np.zeros((8,8));w=np.ones_like(a)
    for forward in ('upstream','corrected'):
        _,_,cfg=create_maps(a,a,w,'smpy_ks_plus',iterations=3,
            threshold_tau=25.,ks_plus_forward=forward)
        assert cfg['methods']['ks_plus']['threshold_tau']==25.
        assert cfg['femmi_adapter']['forward_transform']==forward
    with pytest.raises(ValueError,match='threshold_tau'):
        create_maps(a,a,w,'smpy_ks_plus',threshold_tau=0.)
