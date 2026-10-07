"""Parity against main fa079a9, including singular and shared-endpoint blocks."""
from pathlib import Path
import numpy as np
import pytest
from femmi import bem, bem_hp
from femmi import _bem_assembly as assembly
from femmi.elements import C1Space, structured_triangulation, HCTElement


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('degree',[3,5])
@pytest.mark.parametrize('quad',[7,25])
@pytest.mark.parametrize('scale',[.3,2.])
def test_bem_matches_pre_acceleration_reference(backend,degree,quad,scale):
    if backend=='numba':
        pytest.importorskip('numba')
    with np.load(Path(__file__).parent/'fixtures_cpu/bem_fa079a9.npz') as ref:
        b=bem_hp.build_boundary_mesh(ref['points']*scale,degree)
        prefix=f'd{degree}_q{quad}_s{scale}'
        values={'V':bem_hp.assemble_single_layer_hp(b,degree,quad,backend=backend),
                'K':bem_hp.assemble_double_layer_hp(b,degree,quad,backend=backend)}
        if degree==3:
            values['P3V']=bem.assemble_single_layer(b,quad,backend=backend)
            values['P3K']=bem.assemble_double_layer(b,quad,backend=backend)
        for key,value in values.items():
            assert value.dtype==np.float64
            np.testing.assert_allclose(value,ref[prefix+'_'+key],rtol=2e-12,atol=3e-15)


def test_backend_selection_and_missing_numba(monkeypatch):
    b=bem_hp.build_boundary_mesh([[0.,0.],[1.,0.],[0.,1.]],3)
    monkeypatch.setattr(assembly,'_numba_kernel',lambda:None)
    reference=bem.assemble_single_layer(b,backend='numpy')
    monkeypatch.setenv('FEMMI_BEM_BACKEND','auto')
    np.testing.assert_array_equal(bem.assemble_single_layer(b),reference)
    with pytest.raises(ImportError,match='speed'):
        bem.assemble_single_layer(b,backend='numba')
    with pytest.raises(ValueError,match='backend'):
        bem.assemble_single_layer(b,backend='typo')
    b.elements[0,0]=b.n_boundary_dofs
    with pytest.raises(ValueError,match='boundary mesh'):
        bem.assemble_single_layer(b,backend='numpy')


@pytest.mark.parametrize('kind',['argyris','hct'])
def test_element_cache_owns_fixed_geometry(kind):
    v,t=structured_triangulation(2)
    space=C1Space(v,t,kind)
    el=space.element(0)
    before=el.basis(el.verts)
    assert space.element(0) is el
    v[:]=999; t[:]=0
    np.testing.assert_array_equal(space.element(0).basis(el.verts),before)
    with pytest.raises(ValueError):
        space.vertices[0,0]=3.
    with pytest.raises(ValueError):
        space.triangles[0,0]=1
    with pytest.raises(ValueError):
        el._C[0,0]=0.
    space.clear_element_cache()
    assert space.element(0) is not el
    np.testing.assert_array_equal(space.element(0).basis(el.verts),before)


@pytest.mark.parametrize('derivative',[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)])
def test_batched_hct_matches_scalar_piece_selection(derivative):
    rng=np.random.default_rng(2)
    v=np.array([[.1,-.4],[1.7,.1],[.2,1.3]])
    el=HCTElement(v)
    # Interior, all vertices, centroid, shared subtriangle edges and exterior.
    bary=rng.dirichlet([1,1,1],size=40)
    pts=np.vstack([bary@v,v,el.centroid[None,:],.5*(v+el.centroid),[[5.,3.]]])
    dx,dy=derivative
    expected=np.vstack([el.basis_on_subtriangle(p,el._which_sub(p),dx,dy) for p in pts])
    np.testing.assert_allclose(el.basis(pts,dx,dy),expected,atol=3e-14,rtol=2e-13)
