"""HCT vertex recovery, piecewise quadrature, and weighted C1 gradients."""
import numpy as np
import pytest
from femmi.elements import C1Space, structured_triangulation
from femmi.c1_assembly import assemble_c1, assemble_c1_load, c1_shear_at_vertices
from femmi.c1_inverse import C1MAPReconstructor, shear_operators


def polynomial(p,dx=0,dy=0):
    x,y=p
    return {(0,0): x*x+3*x*y-2*y*y, (1,0):2*x+3*y,
            (0,1):3*x-4*y,(2,0):2.,(1,1):3.,(0,2):-4.}.get((dx,dy),0.)


def test_hct_reproduces_quadratic_and_ignores_triangle_order():
    v,t=structured_triangulation(2,1.)
    s=C1Space(v,t,kind='hct'); rev=C1Space(v,t[::-1],kind='hct')
    a,b=shear_operators(s)
    k=s.interpolate(polynomial)
    np.testing.assert_allclose(a@k,3.,atol=1e-10)
    np.testing.assert_allclose(b@k,3.,atol=1e-10)
    # Edge DOF numbering changes with triangle order: map by edge identity.
    rng=np.random.default_rng(5); k=rng.normal(size=s.n_dofs)
    kr=k.copy(); base=3*s.n_vertices
    for edge,i in s.edges.items(): kr[base+rev.edges[edge]]=k[base+i]
    ar,br=shear_operators(rev)
    np.testing.assert_allclose(a@k,ar@kr,atol=1e-10)
    np.testing.assert_allclose(b@k,br@kr,atol=1e-10)
    c,d=c1_shear_at_vertices(s,k)
    np.testing.assert_allclose(c,a@k); np.testing.assert_allclose(d,b@k)
    el=s.element(0); point=el.verts[:1]
    assert np.linalg.norm(el.basis_on_subtriangle(point,1,2,0)
                          -el.basis_on_subtriangle(point,2,2,0)) > 1e-3


def test_hct_piecewise_quadrature_exact_for_mass_and_stiffness():
    v,t=structured_triangulation(1,1.)
    s=C1Space(v,t,kind='hct')
    k5,m5=assemble_c1(s,5); k9,m9=assemble_c1(s,9)
    np.testing.assert_allclose(k5.toarray(),k9.toarray(),atol=2e-12)
    np.testing.assert_allclose(m5.toarray(),m9.toarray(),atol=2e-12)
    constant=np.zeros(s.n_dofs); constant[:3*s.n_vertices:3]=1
    np.testing.assert_allclose(k5@constant,0,atol=2e-12)
    assert constant @ (m5@constant) == pytest.approx(4.)
    np.testing.assert_allclose(assemble_c1_load(s,lambda p: np.ones(len(p)),5),
                               m5@constant,atol=2e-12)


@pytest.mark.parametrize('kind',['hct','argyris'])
def test_c1_nonbinary_weighted_gradient_and_missing_values(kind):
    v,t=structured_triangulation(1,1.)
    s=C1Space(v,t,kind=kind)
    w=np.array([0.,.2,2.,4.])
    rec=C1MAPReconstructor(s,data_weight=w,lam=.3)
    rng=np.random.default_rng(12)
    k=rng.normal(scale=.01,size=s.n_dofs); direction=rng.normal(size=s.n_dofs)
    y=rng.normal(scale=.1,size=(2,s.n_vertices)); y[:,0]=np.nan
    loss,grad=rec._obj_grad(k,*y)
    eps=1e-6
    fd=(rec._obj_grad(k+eps*direction,*y)[0]-rec._obj_grad(k-eps*direction,*y)[0])/(2*eps)
    assert grad@direction == pytest.approx(fd,rel=2e-6,abs=1e-8)
    assert np.isfinite(loss)
