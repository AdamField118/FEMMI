import numpy as np
import pytest
from femmi.coherence import map_coherence,coherence_length,coherence_samples


def test_registered_coherence_sign_window_and_censoring():
    a=np.random.default_rng(2).normal(size=(32,32))
    for sign in (1,-1):
        s=map_coherence(a,sign*a+3.,.2)
        np.testing.assert_allclose(s['coherence'][s['count']>0],sign,atol=1e-12)
        assert coherence_length(s)['status']==('below-resolution' if sign==1 else 'above-field-scale')
    a[0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):map_coherence(a,a,1.)
    w=np.ones_like(a);w[0]=0
    assert np.isfinite(map_coherence(a,a,1.,window=w)['coherence']).any()


def test_crossing_and_samples_do_not_discard_censoring():
    s=dict(q=np.array([1,2,3.]),coherence=np.array([1,.95,.7]),count=np.ones(3))
    r=coherence_length(s)
    assert r['status']=='crossing' and r['bracket']==[1/3,1/2]
    a=np.random.default_rng(0).normal(size=(16,16))
    r=coherence_samples([a,-a],a,1.)
    assert r['n_samples']==2 and r['n_crossings']==0 and r['conditional_quantiles'] is None
