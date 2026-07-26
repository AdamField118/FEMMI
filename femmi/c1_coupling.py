"""
femmi/c1_coupling.py
FEM-BEM coupling for the C^1 spaces -- the piece that turns Argyris from a
validated element into an actual isolated-field inverter.

THE PROBLEM
-----------
`femmi.c1_assembly` can solve grad^2 psi = 2 kappa on an Argyris space and gets
O(h^4) shear, but only with DIRICHLET boundary conditions. Those are exact for a
compactly supported manufactured field and wrong for a real reconstruction, where
the whole point of FEMMI is the exact exterior (far-field-zero) condition supplied
by the BEM.

Coupling P3 to the BEM is easy because the boundary trace of a P3 field is a cubic
per edge, so the BEM DOFs can simply BE the FEM boundary nodes and the coupling
operator P is a selection matrix. That is not available here: the trace of an
Argyris field along an edge is fixed by {u, u_t, u_tt} at each endpoint -- six
conditions, a QUINTIC -- and its DOFs are derivatives, which are not boundary
node values of anything.

THE CONSTRUCTION
----------------
Use degree-5 boundary elements (femmi.bem_hp), whose trace space matches the
Argyris trace exactly, and build P by EVALUATION rather than selection:

    P[i, :]  =  the Argyris basis functions of the element owning boundary node i,
                evaluated at that node's position.

P is then a sparse (N_b x n_dofs) matrix and the coupled operator is the same
Steinbach form the P3 path uses,

    A_coupled = K + P^T C P,     C = -M_b V_eff^{-1} (1/2 M_b - K_h),

with V_eff carrying the log-capacity correction and one node pinned for gauge.

SCOPE / HONEST NOTE
-------------------
This couples the DIRICHLET trace, which is what the Steinbach form above needs.
An Argyris space also carries independent NORMAL-derivative DOFs on boundary
edges; those are not separately constrained here, so the exterior sees the field
through its Dirichlet trace only. That is the same information the P3 coupling
transmits, so this is not a regression -- but a fully C^1 coupling that also
matches the Neumann trace (a quartic per edge) would use both, and is the natural
next refinement.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial.distance import pdist

from .elements import C1Space
from .bem_hp import build_boundary_mesh, assemble_bem_hp
from .c1_assembly import assemble_c1, assemble_c1_load


def boundary_loop(space: C1Space):
    """Ordered CCW list of boundary vertex indices of the triangulation.

    A boundary edge belongs to exactly one triangle; chaining those edges gives
    the loop. Orientation is fixed by the signed area so the outward normals in
    bem_hp point out of the domain."""
    count = {}
    for tri in space.triangles:
        for k in range(3):
            a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    bedges = [e for e, c in count.items() if c == 1]

    nxt = {}
    for a, b in bedges:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)

    start = bedges[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        cands = [v for v in nxt[cur] if v != prev]
        if not cands:
            break
        nxt_v = cands[0]
        if nxt_v == start:
            break
        loop.append(nxt_v)
        prev, cur = cur, nxt_v

    pts = space.vertices[loop]
    area2 = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                   - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area2 < 0:                                   # make it counter-clockwise
        loop = loop[::-1]
    return np.array(loop, int)


def trace_operator(space: C1Space, bnd, loop, degree):
    """Sparse (N_b x n_dofs) matrix P with (P u)_i = u(x_i) for each boundary node.

    Each boundary node lies on one triangulation edge, so the Argyris basis of the
    single triangle owning that edge is evaluated there. Endpoint nodes are shared
    between two boundary segments; either owning element gives the same value
    because the space is C^0 across the edge, so the first is used.
    """
    # map each boundary edge (vertex pair) to its owning triangle
    owner = {}
    for t, tri in enumerate(space.triangles):
        for k in range(3):
            a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            owner.setdefault((min(a, b), max(a, b)), t)

    rows, cols, vals = [], [], []
    seen = set()
    n_seg = len(loop)
    for e in range(n_seg):
        va, vb = int(loop[e]), int(loop[(e + 1) % n_seg])
        t = owner[(min(va, vb), max(va, vb))]
        el = space.element(t)
        gdofs = space.local_dofs(t)
        for a_loc, gnode in enumerate(bnd.elements[e]):
            if gnode in seen:
                continue
            seen.add(int(gnode))
            phi = el.basis(bnd.nodes[gnode][None, :])[0]      # (n_local_dofs,)
            nz = np.abs(phi) > 1e-14
            rows.extend([int(gnode)] * int(nz.sum()))
            cols.extend(gdofs[nz].tolist())
            vals.extend(phi[nz].tolist())

    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(bnd.n_boundary_dofs, space.n_dofs)).tocsr()


class C1CoupledOperators:
    """Assembled C^1 FEM-BEM system: A_coupled, its factorisation, and the trace
    operator, for a given C1Space."""

    def __init__(self, space: C1Space, degree=5, quad_order=7, sigma_scale=1.0,
                 n_quad_sl=25, n_quad_dl=8, verbose=False):
        self.space = space
        self.degree = degree

        self.K, self.M = assemble_c1(space, quad_order=quad_order)
        self.loop = boundary_loop(space)
        self.bnd = build_boundary_mesh(space.vertices[self.loop], degree)
        V_h, K_h, M_b = assemble_bem_hp(self.bnd, degree,
                                        n_quad_sl=n_quad_sl, n_quad_dl=n_quad_dl)
        self.P = trace_operator(space, self.bnd, self.loop, degree)

        N_b = self.bnd.n_boundary_dofs
        w = M_b @ np.ones(N_b)
        Xm = 0.5 * M_b - K_h
        sigma = float(sigma_scale) * float(pdist(self.bnd.nodes).max())
        V_eff = V_h - (np.log(sigma) / (2.0 * np.pi)) * np.outer(w, w)
        C = -(M_b @ np.linalg.solve(V_eff, Xm))

        A = (self.K + (self.P.T @ sp.csr_matrix(C) @ self.P)).tolil()
        # gauge: pin the value DOF of one boundary vertex
        self.idx_gauge = int(self.loop[0]) * space.n_vert_dofs
        A[self.idx_gauge, :] = 0.0
        A[self.idx_gauge, self.idx_gauge] = 1.0
        self.A = A.tocsr()
        self.A_lu = spla.splu(self.A.tocsc())
        if verbose:
            print(f"  C1 coupling: n_dofs={space.n_dofs} N_b={N_b} degree={degree}")

    def solve_psi(self, rhs):
        rhs = np.asarray(rhs, float).copy()
        rhs[self.idx_gauge] = 0.0
        return self.A_lu.solve(rhs)

    def psi_from_kappa(self, kappa_fn, quad_order=7):
        """Solve grad^2 psi = 2 kappa with the exterior condition from the BEM."""
        b = -2.0 * assemble_c1_load(self.space, kappa_fn, quad_order)
        return self.solve_psi(b)


def c1_coupled_shear_convergence(kind="argyris", nxs=(4, 6, 8, 12), half_width=2.5,
                                 R=1.5, pw=6, degree=5, quad_order=7):
    """Convergence of the shear from a COUPLED (BEM far-field) C^1 solve.

    Same manufactured field as c1_assembly.solved_shear_convergence, but the
    boundary condition now comes from the exterior problem instead of a Dirichlet
    pin -- so this measures the coupled operator, not just the element.
    """
    from .elements import structured_triangulation
    from .experiments import manufactured_potential_derivs
    from .c1_assembly import _quad

    qp, qw = _quad(quad_order)
    f = lambda p, dx=0, dy=0: manufactured_potential_derivs(p, dx, dy, R=R, pw=pw)
    kappa = lambda P: np.array([0.5 * (f(p, 2, 0) + f(p, 0, 2)) for p in P])

    hs, errs = [], []
    for nx in nxs:
        verts, tris = structured_triangulation(nx, half_width)
        S = C1Space(verts, tris, kind=kind)
        ops = C1CoupledOperators(S, degree=degree, quad_order=quad_order)
        psi = ops.psi_from_kappa(kappa, quad_order=quad_order)

        num = den = 0.0
        for t in range(len(tris)):
            v = verts[tris[t]]
            area = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0]]))) / 2.0
            pts = v[0] + qp[:, 0:1] * (v[1] - v[0]) + qp[:, 1:2] * (v[2] - v[0])
            a1, a2 = S.eval_shear(psi, t, pts)
            e1 = np.array([0.5 * (f(q, 2, 0) - f(q, 0, 2)) for q in pts])
            e2 = np.array([f(q, 1, 1) for q in pts])
            num += area * np.sum(qw * ((a1 - e1)**2 + (a2 - e2)**2))
            den += area * np.sum(qw * (e1**2 + e2**2))
        hs.append(2.0 * half_width / nx)
        errs.append(np.sqrt(num / (den + 1e-300)))

    hs = np.array(hs); errs = np.array(errs)
    from .convergence import fit_order, local_orders
    return dict(h=hs, err=errs, kind=kind, degree=degree,
                order=fit_order(hs, errs, what="C^1 coupled shear"),
                local=local_orders(hs, errs))
