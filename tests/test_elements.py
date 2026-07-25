"""
tests/test_elements.py
The C^1 elements (femmi.elements): Argyris P5 and Hsieh-Clough-Tocher.

These elements exist to fix FEMMI's second-derivative problem, so the tests are
about the three properties that actually deliver that:

  * exact reproduction of the element's polynomial space, INCLUDING the Hessian
    (a basis that gets values right but Hessians wrong would pass a naive test
    and be useless here);
  * genuine C^1 conformity across a shared edge, checked with ARBITRARY global
    DOFs -- interpolating a smooth function would be continuous no matter how
    broken the element was, so it proves nothing;
  * the convergence rate of the extracted shear, which is the payoff:
    O(h^2) for HCT (cubic) and O(h^4) for Argyris (quintic).

Run:
    python -m pytest tests/test_elements.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.elements import (ArgyrisElement, HCTElement, C1Space,
                            structured_triangulation, _edge_normal, make_element)
from femmi.experiments import element_shear_convergence


TRI = np.array([[0.0, 0.0], [1.0, 0.2], [0.3, 0.9]])


def _poly(coefs, deg):
    """A polynomial of the given degree and its partials, from monomial coefs."""
    import math
    P = [(i, j) for d in range(deg + 1) for i in range(d + 1) for j in [d - i]]

    def f(p, dx=0, dy=0):
        x, y = p[0], p[1]
        s = 0.0
        for c, (i, j) in zip(coefs, P):
            if i < dx or j < dy:
                continue
            ci = math.factorial(i) // math.factorial(i - dx)
            cj = math.factorial(j) // math.factorial(j - dy)
            s += c * ci * cj * x ** (i - dx) * y ** (j - dy)
        return s
    return f, len(P)


def _dofs(f, el, nv):
    """DOF vector of f for an element with nv value/derivative DOFs per vertex."""
    d = []
    for v in range(3):
        p = el.verts[v]
        d += [f(p, 0, 0), f(p, 1, 0), f(p, 0, 1)]
        if nv == 6:
            d += [f(p, 2, 0), f(p, 1, 1), f(p, 0, 2)]
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        n = _edge_normal(el.verts[a], el.verts[b], el.gverts[a], el.gverts[b])
        m = 0.5 * (el.verts[a] + el.verts[b])
        d.append(n[0] * f(m, 1, 0) + n[1] * f(m, 0, 1))
    return np.array(d)


@pytest.mark.parametrize("cls,deg,nv", [(ArgyrisElement, 5, 6), (HCTElement, 3, 3)])
def test_reproduces_its_polynomial_space_including_hessian(cls, deg, nv):
    """Argyris must reproduce P5 and HCT must reproduce P3 exactly -- values,
    gradients AND second derivatives."""
    rng = np.random.default_rng(0)
    n_monomials = (deg + 1) * (deg + 2) // 2
    f, _ = _poly(rng.normal(size=n_monomials), deg)

    el = cls(TRI, gverts=(0, 1, 2))
    c = _dofs(f, el, nv)
    pts = np.array([[0.2, 0.2], [0.5, 0.15], [0.35, 0.5], [0.12, 0.06]])

    assert np.allclose(el.value(c, pts), [f(p) for p in pts], atol=1e-11)
    g = el.grad(c, pts)
    assert np.allclose(g[:, 0], [f(p, 1, 0) for p in pts], atol=1e-10)
    assert np.allclose(g[:, 1], [f(p, 0, 1) for p in pts], atol=1e-10)
    H = el.hessian(c, pts)
    assert np.allclose(H[:, 0, 0], [f(p, 2, 0) for p in pts], atol=1e-9)
    assert np.allclose(H[:, 0, 1], [f(p, 1, 1) for p in pts], atol=1e-9)
    assert np.allclose(H[:, 1, 1], [f(p, 0, 2) for p in pts], atol=1e-9)


@pytest.mark.parametrize("cls,nv", [(ArgyrisElement, 6), (HCTElement, 3)])
def test_is_C1_across_a_shared_edge_for_arbitrary_dofs(cls, nv):
    """The defining property. Two elements share an edge and agree only on the
    DOFs attached to that edge's vertices plus its normal-derivative DOF; the
    field and its GRADIENT must still match along the edge for ARBITRARY values.

    This is what fails silently if the shared edge normal is oriented
    inconsistently between the two sides."""
    p0, p1 = np.array([0.0, 0.0]), np.array([1.0, 0.2])
    p2, p3 = np.array([0.3, 0.9]), np.array([0.75, -0.8])
    rng = np.random.default_rng(1)

    A = cls(np.array([p0, p1, p2]), gverts=(0, 1, 2))
    B = cls(np.array([p0, p1, p3]), gverts=(0, 1, 3))
    ca = rng.normal(size=cls.n_dofs)
    cb = rng.normal(size=cls.n_dofs)
    cb[:2 * nv] = ca[:2 * nv]              # shared vertices p0, p1
    cb[3 * nv + 2] = ca[3 * nv + 2]        # shared edge (local edge 2)

    s = np.linspace(0.08, 0.92, 9)[:, None]
    pts = p0 + s * (p1 - p0)
    assert np.allclose(A.value(ca, pts), B.value(cb, pts), atol=1e-10)
    assert np.allclose(A.grad(ca, pts), B.grad(cb, pts), atol=1e-9)

    # C^1 also forces the tangential-tangential second derivative to agree
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    Ha, Hb = A.hessian(ca, pts), B.hessian(cb, pts)
    tt = lambda H: np.einsum('i,pij,j->p', t, H, t)
    assert np.allclose(tt(Ha), tt(Hb), atol=1e-8)


def test_argyris_hessian_is_single_valued_at_shared_vertices():
    """Argyris carries the second derivatives as vertex DOFs, so the Hessian at a
    vertex is shared by every adjacent element. That is precisely what makes
    NODAL shear extraction well posed -- the thing P3 cannot do, and the reason
    operators._assemble_shear_ops has to average over adjacent elements."""
    p0, p1 = np.array([0.0, 0.0]), np.array([1.0, 0.2])
    p2, p3 = np.array([0.3, 0.9]), np.array([0.75, -0.8])
    rng = np.random.default_rng(2)
    A = ArgyrisElement(np.array([p0, p1, p2]), gverts=(0, 1, 2))
    B = ArgyrisElement(np.array([p0, p1, p3]), gverts=(0, 1, 3))
    ca = rng.normal(size=21); cb = rng.normal(size=21)
    cb[:12] = ca[:12]; cb[20] = ca[20]
    pv = np.array([p0, p1])
    assert np.allclose(A.hessian(ca, pv), B.hessian(cb, pv), atol=1e-9)


def test_global_space_dof_count_and_sharing():
    """C1Space numbering: n_vert_dofs per vertex plus one per edge, and adjacent
    elements must resolve a shared edge to the SAME global DOF."""
    verts, tris = structured_triangulation(3, 2.5)
    for kind, nv in (("argyris", 6), ("hct", 3)):
        S = C1Space(verts, tris, kind=kind)
        assert S.n_dofs == len(verts) * nv + len(S.edges)
        # every element's DOFs are in range and unique
        for t in range(len(tris)):
            idx = S.local_dofs(t)
            assert len(set(idx.tolist())) == len(idx)
            assert idx.min() >= 0 and idx.max() < S.n_dofs
        # the two triangles of the first quad share exactly one edge DOF
        e0 = set(S.local_dofs(0)[3 * nv:]); e1 = set(S.local_dofs(1)[3 * nv:])
        assert len(e0 & e1) == 1


def test_hct_shear_is_second_order():
    """HCT is cubic, so its second derivative converges at O(h^2) -- the same rate
    as P3, with C^1 continuity and 12 DOF."""
    d = element_shear_convergence(kind="hct", nxs=(8, 12, 16, 24))
    assert d["local"][-1] > 1.7
    assert d["err"][-1] < d["err"][0]


def test_argyris_shear_is_fourth_order():
    """The payoff: Argyris is quintic, so the shear converges at O(h^4) -- two
    orders better than P3, which is what removes the accuracy penalty FEMMI
    currently pays for differentiating twice."""
    d = element_shear_convergence(kind="argyris", nxs=(8, 12, 16, 24))
    assert d["local"][-1] > 3.3
    assert d["err"][-1] < 0.05 * d["err"][0]


def test_argyris_beats_hct_at_equal_resolution():
    """Same mesh, far smaller shear error -- the accuracy-per-mesh comparison the
    benchmark will need."""
    a = element_shear_convergence(kind="argyris", nxs=(8, 16))
    h = element_shear_convergence(kind="hct", nxs=(8, 16))
    assert a["err"][-1] < 0.1 * h["err"][-1]


def test_make_element_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown C1 element"):
        make_element("p3", TRI)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
