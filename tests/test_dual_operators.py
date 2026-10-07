import numpy as np
from femmi.calderon import dual_boundary_operators,dual_conditioning


def circle(n):
    t=np.arange(n)*2*np.pi/n
    return np.column_stack([np.cos(t),np.sin(t)])


def test_dual_operators_singular_quadrature_null_and_mesh_growth():
    cond=[];raw=[]
    for n in (12,24,48):
        a=dual_conditioning(circle(n));cond.append(a['cond_preconditioned']);raw.append(a['cond_V'])
        assert a['constant_residual']<1e-12
    assert max(cond)/min(cond)<1.2
    assert raw[-1]>3*raw[0]
    x=circle(16);x[:,0]*=1.7
    a=dual_boundary_operators(x,12);b=dual_boundary_operators(x,24)
    np.testing.assert_allclose(a['V'],b['V'],rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(a['W'],a['W'].T,atol=1e-13)
    assert np.linalg.eigvalsh(a['W'])[0]>-1e-12
