"""
femmi/elements.py
C^1-conforming triangular elements: Argyris (P5) and Hsieh-Clough-Tocher (HCT).

WHY THIS EXISTS
---------------
FEMMI's shear is the traceless Hessian of the lensing potential, gamma_1 =
1/2(psi_xx - psi_yy), gamma_2 = psi_xy -- a SECOND derivative. The P3 Lagrange
element is only C^0, so its second derivative is discontinuous across element
boundaries, and every awkward thing in the shear path descends from that one
fact:

  * element Hessians are sampled exactly at the nodes where they jump
    (operators._assemble_shear_ops);
  * boundary rows of S1/S2 had to be zeroed as "unreliable"
    (operators.py), which then imposes a spurious zero-shear constraint;
  * operators.RecoveredShear exists purely to work around the jump, and it drops
    a boundary term it cannot currently evaluate;
  * noise in psi is amplified by h^-2 (MATH.md 18.4).

A C^1 element fixes this -- but it is worth being precise about HOW, because C^1
does NOT mean the Hessian is globally continuous. C^1 gives a continuous
GRADIENT; across an edge interior the tangential-tangential second derivative is
then continuous too, while the normal-normal one is free to jump (and does --
tests/test_elements.py measures it). What actually changes for FEMMI:

  1. psi_h lies in H^2, so grad^2 psi_h is a genuine L^2 field. For P3, psi_h is
     not in H^2 at all and the Hessian only exists elementwise.
  2. For ARGYRIS the second derivatives at each vertex are themselves DOFs, so
     the Hessian AT THE NODES is single-valued and shared between all adjacent
     elements. That is exactly where FEMMI samples shear -- so nodal extraction
     becomes well posed: no averaging over adjacent elements, no jump at the
     sampling point, no reason to special-case the boundary ring.
  3. Approximation order improves: the Hessian of a degree-k element converges at
     O(h^(k-1)), so Argyris gives O(h^4) shear against P3's O(h^2).

HCT shares (1) but not (2): its vertex DOFs stop at the gradient, so its Hessian
is still multivalued at vertices, and being cubic it is O(h^2) in shear like P3.
Its appeal is cost -- 12 DOF against 21 -- and C^1 continuity for the solve.

  Argyris  P5, 21 DOF/triangle: at each vertex {u, u_x, u_y, u_xx, u_xy, u_yy}
           (18) plus the normal derivative at each edge midpoint (3). C^1, and
           contains P5 exactly.
  HCT      macro-element, 12 DOF/triangle: the triangle is split at its centroid
           into three sub-triangles carrying separate cubics glued C^1; DOFs are
           {u, u_x, u_y} at each vertex (9) plus the edge-midpoint normal
           derivatives (3). C^1, contains P3, and much cheaper than Argyris.

CONSTRUCTION
------------
Both are built numerically, directly in PHYSICAL coordinates, by imposing the DOF
functionals (and, for HCT, the C^1 matching conditions) on a monomial basis and
solving for the nodal basis. Two consequences worth knowing:

  * Derivative DOFs are expressed in global Cartesian components, so vertex DOFs
    are shared between adjacent elements with no transformation -- which is
    exactly what makes global C^1 assembly straightforward. (This is why the
    usual reference-element pullback is avoided: Argyris is NOT affine
    equivalent, and mapping its derivative DOFs correctly is the classic
    implementation trap.)
  * Monomials up to degree 5 in raw physical coordinates would be badly
    conditioned, so each element is built in a local frame translated to its
    centroid and scaled by its diameter; the h-powers are folded into the DOF
    functionals so the resulting basis still corresponds to PHYSICAL derivative
    DOFs.

Edge normal orientation must agree between the two elements sharing an edge or
C^1 matching silently fails; `_edge_normal` fixes it from the GLOBAL vertex
indices (always lower index -> higher index), which is why element constructors
take `gverts`.
"""

from __future__ import annotations
import math
import numpy as np


# --------------------------------------------------------------------------- #
# monomial helpers
# --------------------------------------------------------------------------- #
def _powers(degree):
    """[(i, j), ...] for all monomials xi^i eta^j with i + j <= degree."""
    return [(i, j) for d in range(degree + 1) for i in range(d + 1)
            for j in [d - i]]


def _mono(powers, xi, eta, dx=0, dy=0):
    """Evaluate d^(dx+dy)/dxi^dx deta^dy of each monomial at the given points.

    Returns (n_pts, n_powers). Used both to build the DOF-functional matrix and
    to evaluate the finished basis."""
    xi = np.atleast_1d(np.asarray(xi, float))
    eta = np.atleast_1d(np.asarray(eta, float))
    out = np.zeros((len(xi), len(powers)))
    for m, (i, j) in enumerate(powers):
        if i < dx or j < dy:
            continue                                  # derivative kills it
        ci = math.factorial(i) // math.factorial(i - dx)
        cj = math.factorial(j) // math.factorial(j - dy)
        out[:, m] = ci * cj * xi ** (i - dx) * eta ** (j - dy)
    return out


def _edge_normal(pa, pb, ga, gb):
    """Unit normal of the edge (pa, pb), oriented canonically by the GLOBAL vertex
    indices so both elements sharing this edge agree on its sign.

    Without this the two sides impose normal-derivative DOFs of opposite sign and
    the assembled space is not C^1 -- a failure that is invisible elementwise and
    only shows up as a jump across the shared edge."""
    if ga > gb:                                        # canonical: low -> high
        pa, pb = pb, pa
    t = np.asarray(pb, float) - np.asarray(pa, float)
    t /= np.linalg.norm(t)
    return np.array([t[1], -t[0]])


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class C1Element:
    """Shared machinery: a local frame, and evaluation of a coefficient array.

    Subclasses supply `degree`, `n_dofs`, and `_build(verts, gverts)` returning
    the basis coefficients.
    """
    degree = None
    n_dofs = None
    name = None

    def __init__(self, verts, gverts=None):
        self.verts = np.asarray(verts, float)          # (3, 2) physical vertices
        if self.verts.shape != (3, 2):
            raise ValueError("verts must be (3, 2)")
        self.gverts = tuple(range(3)) if gverts is None else tuple(gverts)
        self.centroid = self.verts.mean(axis=0)
        # local scaling: diameter of the element
        d = max(np.linalg.norm(self.verts[i] - self.verts[j])
                for i, j in ((0, 1), (1, 2), (2, 0)))
        self.h = float(d)
        self.powers = _powers(self.degree)
        self._build()

    # -- local frame ------------------------------------------------------- #
    def _to_local(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return (p - self.centroid) / self.h

    def _build(self):
        raise NotImplementedError

    # -- evaluation -------------------------------------------------------- #
    def basis(self, pts, dx=0, dy=0):
        """Physical derivative d^(dx+dy) of every basis function at `pts`.

        Returns (n_pts, n_dofs). The 1/h**(dx+dy) converts the local-frame
        derivative back to a physical one."""
        raise NotImplementedError

    def value(self, coeffs, pts):
        return self.basis(pts) @ np.asarray(coeffs, float)

    def grad(self, coeffs, pts):
        c = np.asarray(coeffs, float)
        return np.stack([self.basis(pts, 1, 0) @ c,
                         self.basis(pts, 0, 1) @ c], axis=-1)

    def hessian(self, coeffs, pts):
        """(n_pts, 2, 2) Hessian -- the quantity that is CONTINUOUS for these
        elements and discontinuous for P3."""
        c = np.asarray(coeffs, float)
        uxx = self.basis(pts, 2, 0) @ c
        uxy = self.basis(pts, 1, 1) @ c
        uyy = self.basis(pts, 0, 2) @ c
        H = np.empty((len(uxx), 2, 2))
        H[:, 0, 0] = uxx; H[:, 0, 1] = uxy
        H[:, 1, 0] = uxy; H[:, 1, 1] = uyy
        return H

    def shear(self, coeffs, pts):
        """(gamma1, gamma2) = (1/2(u_xx - u_yy), u_xy) -- FEMMI's convention.
        Continuous across elements, which is the entire point of this module."""
        H = self.hessian(coeffs, pts)
        return 0.5 * (H[:, 0, 0] - H[:, 1, 1]), H[:, 0, 1]


# --------------------------------------------------------------------------- #
# Argyris P5
# --------------------------------------------------------------------------- #
class ArgyrisElement(C1Element):
    """The classical C^1 quintic. 21 DOF, contains P5 exactly.

    DOF order: vertex 0 {u, u_x, u_y, u_xx, u_xy, u_yy}, vertex 1 {...},
    vertex 2 {...}, then edge normal derivatives for edges 0, 1, 2 where edge k
    joins vertices (k+1) % 3 and (k+2) % 3.
    """
    degree = 5
    n_dofs = 21
    name = "argyris"

    def _build(self):
        P = self.powers
        h = self.h
        rows = []

        for v in range(3):                             # 6 DOFs per vertex
            L = self._to_local(self.verts[v])
            xi, eta = L[0, 0], L[0, 1]
            # physical derivative = local derivative / h**order, so the row that
            # represents a physical DOF carries the matching 1/h**order
            rows.append(_mono(P, xi, eta, 0, 0)[0])
            rows.append(_mono(P, xi, eta, 1, 0)[0] / h)
            rows.append(_mono(P, xi, eta, 0, 1)[0] / h)
            rows.append(_mono(P, xi, eta, 2, 0)[0] / h**2)
            rows.append(_mono(P, xi, eta, 1, 1)[0] / h**2)
            rows.append(_mono(P, xi, eta, 0, 2)[0] / h**2)

        for k in range(3):                             # 3 edge normal derivatives
            a, b = (k + 1) % 3, (k + 2) % 3
            n = _edge_normal(self.verts[a], self.verts[b],
                             self.gverts[a], self.gverts[b])
            mid = 0.5 * (self.verts[a] + self.verts[b])
            L = self._to_local(mid)
            xi, eta = L[0, 0], L[0, 1]
            rows.append((n[0] * _mono(P, xi, eta, 1, 0)[0]
                         + n[1] * _mono(P, xi, eta, 0, 1)[0]) / h)

        V = np.array(rows)                             # (21, 21) DOF x monomial
        self._V = V
        self._C = np.linalg.inv(V)                     # monomial x DOF

    def basis(self, pts, dx=0, dy=0):
        L = self._to_local(pts)
        M = _mono(self.powers, L[:, 0], L[:, 1], dx, dy)
        return (M @ self._C) / self.h ** (dx + dy)


# --------------------------------------------------------------------------- #
# Hsieh-Clough-Tocher macro element
# --------------------------------------------------------------------------- #
class HCTElement(C1Element):
    """C^1 cubic macro-element. 12 DOF, contains P3.

    The triangle is split at its centroid G into three sub-triangles
    T_k = (G, v_(k+1)%3, v_(k+2)%3), each carrying an independent cubic (10
    coefficients, 30 total). Those 30 unknowns are pinned by

      * 12 DOF conditions -- {u, u_x, u_y} at each vertex, and the normal
        derivative at each edge midpoint; and
      * C^1 matching across the three INTERNAL edges G-v_k: the value and the
        normal derivative must agree between the two sub-triangles meeting there.

    The matching conditions are imposed by sampling along each internal edge
    (a cubic restricted to a line is a 1D cubic, its normal derivative a 1D
    quadratic, so a handful of sample points captures them exactly) and the
    combined system is solved in least-squares sense. That keeps the construction
    robust to redundant constraints -- and `tests/test_elements.py` verifies the
    resulting basis really is C^1 rather than trusting the count.
    """
    degree = 3
    n_dofs = 12
    name = "hct"
    _n_sub = 3

    def _sub_verts(self, k):
        g = self.centroid
        return np.array([g, self.verts[(k + 1) % 3], self.verts[(k + 2) % 3]])

    def _build(self):
        P = self.powers                                # 10 cubic monomials
        npow = len(P)
        h = self.h
        ndof = self.n_dofs
        N = self._n_sub * npow                         # 30 unknowns

        def block(k, row_local):
            """place a per-sub-triangle monomial row into the global 30-vector"""
            r = np.zeros(N)
            r[k * npow:(k + 1) * npow] = row_local
            return r

        def which_sub(pt):
            """index of the sub-triangle containing pt (by barycentric sign)"""
            for k in range(self._n_sub):
                sv = self._sub_verts(k)
                T = np.array([sv[1] - sv[0], sv[2] - sv[0]]).T
                lam = np.linalg.solve(T, np.asarray(pt, float) - sv[0])
                if lam[0] >= -1e-9 and lam[1] >= -1e-9 and lam.sum() <= 1 + 1e-9:
                    return k
            return 0

        # --- DOF rows (12) ------------------------------------------------- #
        dof_rows = []
        for v in range(3):
            # vertex v belongs to the two sub-triangles that list it; either
            # gives the same value once matching is imposed -- pick one.
            k = (v + 1) % 3                            # T_k has v as a corner
            L = self._to_local(self.verts[v]); xi, eta = L[0, 0], L[0, 1]
            dof_rows.append(block(k, _mono(P, xi, eta, 0, 0)[0]))
            dof_rows.append(block(k, _mono(P, xi, eta, 1, 0)[0] / h))
            dof_rows.append(block(k, _mono(P, xi, eta, 0, 1)[0] / h))
        for k in range(3):
            a, b = (k + 1) % 3, (k + 2) % 3
            n = _edge_normal(self.verts[a], self.verts[b],
                             self.gverts[a], self.gverts[b])
            mid = 0.5 * (self.verts[a] + self.verts[b])
            ks = which_sub(mid)
            L = self._to_local(mid); xi, eta = L[0, 0], L[0, 1]
            dof_rows.append(block(ks, (n[0] * _mono(P, xi, eta, 1, 0)[0]
                                       + n[1] * _mono(P, xi, eta, 0, 1)[0]) / h))

        # --- C^1 matching across the internal edges G-v_k ------------------ #
        con_rows = []
        for v in range(3):
            # internal edge from centroid to vertex v is shared by the two
            # sub-triangles that both have v as a corner
            ks = [k for k in range(3) if v in ((k + 1) % 3, (k + 2) % 3)]
            k1, k2 = ks
            pa, pb = self.centroid, self.verts[v]
            t = pb - pa; t = t / np.linalg.norm(t)
            nrm = np.array([t[1], -t[0]])
            for s in np.linspace(0.0, 1.0, 5):         # value: 1D cubic -> >=4
                p = pa + s * (pb - pa)
                L = self._to_local(p); xi, eta = L[0, 0], L[0, 1]
                m = _mono(P, xi, eta, 0, 0)[0]
                con_rows.append(block(k1, m) - block(k2, m))
            for s in np.linspace(0.0, 1.0, 4):         # normal deriv: quadratic
                p = pa + s * (pb - pa)
                L = self._to_local(p); xi, eta = L[0, 0], L[0, 1]
                m = (nrm[0] * _mono(P, xi, eta, 1, 0)[0]
                     + nrm[1] * _mono(P, xi, eta, 0, 1)[0])
                con_rows.append(block(k1, m) - block(k2, m))

        A = np.array(dof_rows + con_rows)              # (12 + 27, 30)
        rhs = np.zeros((A.shape[0], ndof))
        rhs[:ndof, :] = np.eye(ndof)                   # DOF k -> unit, others 0
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # (30, 12)
        self._C = sol
        self._which_sub = which_sub

    def basis(self, pts, dx=0, dy=0):
        p = np.atleast_2d(np.asarray(pts, float))
        L = self._to_local(p)
        npow = len(self.powers)
        out = np.zeros((len(p), self.n_dofs))
        for i in range(len(p)):
            k = self._which_sub(p[i])
            M = _mono(self.powers, L[i, 0], L[i, 1], dx, dy)[0]
            out[i] = M @ self._C[k * npow:(k + 1) * npow, :]
        return out / self.h ** (dx + dy)


# --------------------------------------------------------------------------- #
# global C^1 space
# --------------------------------------------------------------------------- #
class C1Space:
    """Global C^1 finite-element space over a triangulation.

    DOF layout: `n_vert_dofs` per vertex (6 for Argyris, 3 for HCT) followed by
    one normal-derivative DOF per edge. Because the element bases are built in
    physical coordinates with global Cartesian derivative DOFs, assembling the
    global space is just this numbering -- no per-element transformation -- and
    the space is C^1 provided every shared edge uses the same normal, which
    `_edge_normal` guarantees from the global vertex indices.

    This is deliberately an interpolation/evaluation layer, not yet a solver:
    it is what convergence-rate measurements need, and it is the foundation the
    FEM-BEM operator assembly will sit on.
    """

    def __init__(self, vertices, triangles, kind="argyris"):
        self.vertices = np.asarray(vertices, float)
        self.triangles = np.asarray(triangles, int)
        self.kind = kind
        self.cls = ELEMENTS[kind]
        self.n_vert_dofs = 6 if kind == "argyris" else 3

        edges = {}
        for tri in self.triangles:
            for k in range(3):
                a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
                edges.setdefault((min(a, b), max(a, b)), len(edges))
        self.edges = edges
        self.n_vertices = len(self.vertices)
        self.n_dofs = self.n_vertices * self.n_vert_dofs + len(edges)

    def element(self, t):
        tri = self.triangles[t]
        return self.cls(self.vertices[tri], gverts=tuple(int(i) for i in tri))

    def local_dofs(self, t):
        """Global DOF indices for element t, in the element's local DOF order."""
        tri = self.triangles[t]
        nv = self.n_vert_dofs
        idx = []
        for v in tri:
            idx.extend(range(int(v) * nv, int(v) * nv + nv))
        base = self.n_vertices * nv
        for k in range(3):
            a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            idx.append(base + self.edges[(min(a, b), max(a, b))])
        return np.array(idx, int)

    def interpolate(self, f):
        """Global DOF vector interpolating a callable f(p, dx, dy) that returns
        the requested partial derivative at a point."""
        u = np.zeros(self.n_dofs)
        nv = self.n_vert_dofs
        for v, p in enumerate(self.vertices):
            u[v * nv + 0] = f(p, 0, 0)
            u[v * nv + 1] = f(p, 1, 0)
            u[v * nv + 2] = f(p, 0, 1)
            if nv == 6:
                u[v * nv + 3] = f(p, 2, 0)
                u[v * nv + 4] = f(p, 1, 1)
                u[v * nv + 5] = f(p, 0, 2)
        base = self.n_vertices * nv
        for (a, b), e in self.edges.items():
            n = _edge_normal(self.vertices[a], self.vertices[b], a, b)
            m = 0.5 * (self.vertices[a] + self.vertices[b])
            u[base + e] = n[0] * f(m, 1, 0) + n[1] * f(m, 0, 1)
        return u

    def eval_shear(self, u, t, pts):
        """(gamma1, gamma2) of the global field u, evaluated inside element t."""
        el = self.element(t)
        return el.shear(np.asarray(u)[self.local_dofs(t)], pts)

    def eval_value(self, u, t, pts):
        el = self.element(t)
        return el.value(np.asarray(u)[self.local_dofs(t)], pts)


def structured_triangulation(nx, half_width=2.5):
    """Vertices + triangles of a uniform right-triangle mesh on the square
    [-half_width, half_width]^2 -- the same geometry femmi.mesh uses, but at the
    P1 vertex level, which is all a C^1 element needs (its extra DOFs live on
    those same vertices and edges)."""
    c = np.linspace(-half_width, half_width, nx + 1)
    X, Y = np.meshgrid(c, c)
    verts = np.stack([X.ravel(), Y.ravel()], axis=1)
    tris = []
    for j in range(nx):
        for i in range(nx):
            v00 = j * (nx + 1) + i
            v10, v01, v11 = v00 + 1, v00 + nx + 1, v00 + nx + 2
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    return verts, np.array(tris, int)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
ELEMENTS = {"argyris": ArgyrisElement, "hct": HCTElement}


def make_element(kind, verts, gverts=None):
    """Build a C^1 element by name ('argyris' or 'hct')."""
    try:
        cls = ELEMENTS[kind]
    except KeyError:
        raise ValueError(f"unknown C1 element {kind!r}; expected one of "
                         f"{sorted(ELEMENTS)}") from None
    return cls(verts, gverts=gverts)
