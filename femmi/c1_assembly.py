"""
femmi/c1_assembly.py
FEM assembly and solves on the C^1 spaces of femmi.elements (Argyris, HCT).

femmi.elements gives the bases and the global DOF numbering; this turns them into
operators -- stiffness, mass, load -- and solves the lensing Poisson problem
grad^2 psi = 2 kappa on them, so the O(h^4) shear measured for Argyris by
interpolation is also available from an actual SOLVE.

What is here, and what is not
-----------------------------
Here: K, M, load, Dirichlet constraints, a Poisson solve, and shear extraction
that reads the Hessian straight off the Argyris vertex DOFs (no averaging, no
recovery, no boundary special-casing -- see MATH.md 18.3a).

Not here yet: the FEM-BEM coupling. The exterior problem couples through the
boundary trace, and a C^1 space has a richer trace than P3 -- the normal-
derivative DOFs on boundary edges have to be matched against the Steklov-Poincare
operator. Until that lands, C^1 solves use Dirichlet conditions, which is exact
for a compactly supported field (the manufactured solutions FEMMI validates
against) and wrong for a real isolated-field reconstruction. So this is a
validated element+solver, not yet a drop-in replacement for build_operators.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elements import C1Space, structured_triangulation


def _quad(n=7):
    """Conical-product Gauss rule on the reference triangle, exact to degree
    2n-1 in each variable.

    femmi.assembly tops out at a degree-7 rule, which is fine for P3 but NOT for
    Argyris: its mass integrand phi*phi is degree 10 and the stiffness integrand
    is degree 8, so a degree-7 rule would leave a quadrature error that pollutes
    the measured convergence rate. Generating the rule here keeps the P3 path's
    tuned rules untouched.

    Weights are normalised to sum to 1, matching this codebase's convention that
    the physical triangle area is applied separately.
    """
    x, wx = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (x + 1.0); wu = 0.5 * wx           # Gauss-Legendre on [0, 1]
    U, V = np.meshgrid(u, u, indexing="ij")
    WU, WV = np.meshgrid(wu, wu, indexing="ij")
    # (u, v) -> (xi, eta) = (u, v(1-u)), Jacobian (1-u); x2 normalises sum -> 1
    pts = np.stack([U.ravel(), (V * (1.0 - U)).ravel()], axis=1)
    wts = (2.0 * WU * WV * (1.0 - U)).ravel()
    return pts, wts


def assemble_c1(space: C1Space, quad_order=7):
    """Stiffness K[i,j] = int grad phi_i . grad phi_j and mass M[i,j] = int phi_i phi_j.

    Assembled elementwise from the physical-coordinate bases, so no reference
    pullback of the derivative DOFs is involved (see femmi.elements)."""
    qp, qw = _quad(quad_order)
    n = space.n_dofs
    rows, cols, kv, mv = [], [], [], []

    for t in range(len(space.triangles)):
        el = space.element(t)
        v = el.verts
        area = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0]]))) / 2.0
        pts = v[0] + qp[:, 0:1] * (v[1] - v[0]) + qp[:, 1:2] * (v[2] - v[0])

        N = el.basis(pts)                       # (nq, ndof)
        Gx = el.basis(pts, 1, 0)
        Gy = el.basis(pts, 0, 1)

        Ke = area * (np.einsum('q,qi,qj->ij', qw, Gx, Gx)
                     + np.einsum('q,qi,qj->ij', qw, Gy, Gy))
        Me = area * np.einsum('q,qi,qj->ij', qw, N, N)

        idx = space.local_dofs(t)
        rows.append(np.repeat(idx, len(idx)))
        cols.append(np.tile(idx, len(idx)))
        kv.append(Ke.ravel()); mv.append(Me.ravel())

    r = np.concatenate(rows); c = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kv), (r, c)), shape=(n, n)).tocsr()
    M = sp.coo_matrix((np.concatenate(mv), (r, c)), shape=(n, n)).tocsr()
    return K, M


def assemble_c1_load(space: C1Space, f, quad_order=7):
    """Load vector b[i] = int f phi_i for a callable f(points) -> values."""
    qp, qw = _quad(quad_order)
    b = np.zeros(space.n_dofs)
    for t in range(len(space.triangles)):
        el = space.element(t)
        v = el.verts
        area = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0]]))) / 2.0
        pts = v[0] + qp[:, 0:1] * (v[1] - v[0]) + qp[:, 1:2] * (v[2] - v[0])
        N = el.basis(pts)
        np.add.at(b, space.local_dofs(t), area * (qw * np.asarray(f(pts))) @ N)
    return b


def boundary_dofs(space: C1Space, half_width, tol=1e-9):
    """Every DOF attached to a boundary vertex or a boundary edge of the square.

    Pinning ALL of them imposes psi = 0 together with all its derivatives, which
    is only correct when the true field is compactly supported inside the domain
    -- exactly the manufactured setting this module validates against. It is NOT
    the isolated-field far-field condition the BEM provides.
    """
    v = space.vertices
    on = (np.abs(np.abs(v[:, 0]) - half_width) < tol) | \
         (np.abs(np.abs(v[:, 1]) - half_width) < tol)
    nv = space.n_vert_dofs
    idx = [np.arange(i * nv, i * nv + nv) for i in np.where(on)[0]]
    base = space.n_vertices * nv
    edge_idx = [base + e for (a, b), e in space.edges.items() if on[a] and on[b]]
    out = np.concatenate(idx + [np.array(edge_idx, int)]) if idx else np.array(edge_idx, int)
    return np.unique(out.astype(int))


def solve_poisson_c1(space: C1Space, kappa_fn, half_width, quad_order=7):
    """Solve grad^2 psi = 2 kappa with psi (and its derivatives) pinned on the
    boundary, returning the global DOF vector.

    Weak form: int grad psi . grad w = -2 int kappa w, so K psi = -2 b[kappa].
    """
    K, _ = assemble_c1(space, quad_order)
    b = -2.0 * assemble_c1_load(space, kappa_fn, quad_order)

    bd = boundary_dofs(space, half_width)
    free = np.setdiff1d(np.arange(space.n_dofs), bd)
    psi = np.zeros(space.n_dofs)
    psi[free] = spla.spsolve(K[free][:, free].tocsc(), b[free])
    return psi


def c1_shear_at_vertices(space: C1Space, psi):
    """Shear at the mesh vertices, read STRAIGHT OFF the DOF vector for Argyris.

    Argyris carries {u_xx, u_xy, u_yy} as vertex DOFs, so
        gamma1 = 1/2 (u_xx - u_yy),  gamma2 = u_xy
    is a pure selection -- single-valued, no averaging over adjacent elements, no
    recovery step, no boundary ring to zero. This is the operator that replaces
    S1/S2 once the BEM coupling lands.

    HCT has no Hessian DOFs, so it falls back to evaluating in one adjacent
    element (and is multivalued at vertices, like P3).
    """
    nv = space.n_vert_dofs
    if nv == 6:
        u = np.asarray(psi).reshape(-1)[:space.n_vertices * 6].reshape(-1, 6)
        return 0.5 * (u[:, 3] - u[:, 5]), u[:, 4]

    g1 = np.zeros(space.n_vertices); g2 = np.zeros(space.n_vertices)
    seen = np.zeros(space.n_vertices, bool)
    for t, tri in enumerate(space.triangles):
        for lv, v in enumerate(tri):
            if seen[v]:
                continue
            a, b = space.eval_shear(psi, t, space.vertices[v:v + 1])
            g1[v], g2[v] = a[0], b[0]; seen[v] = True
    return g1, g2


def solved_shear_convergence(kind="argyris", nxs=(4, 6, 8, 12), half_width=2.5,
                             R=1.5, pw=6, quad_order=7):
    """Convergence of the shear obtained from an actual SOLVE (not interpolation).

    Uses the compactly-supported manufactured potential, for which pinning the
    whole boundary is exact, so the measured rate reflects the element and the
    solve rather than a boundary-condition mismatch.

    Returns dict with h, err (L2 over the domain), fitted and local orders.
    """
    from .experiments import manufactured_potential_derivs
    qp, qw = _quad(quad_order)
    f = lambda p, dx=0, dy=0: manufactured_potential_derivs(p, dx, dy, R=R, pw=pw)

    def kappa(points):
        return np.array([0.5 * (f(p, 2, 0) + f(p, 0, 2)) for p in points])

    hs, errs = [], []
    for nx in nxs:
        verts, tris = structured_triangulation(nx, half_width)
        S = C1Space(verts, tris, kind=kind)
        psi = solve_poisson_c1(S, kappa, half_width, quad_order)

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
    return dict(h=hs, err=errs, kind=kind,
                order=float(np.polyfit(np.log(hs), np.log(errs), 1)[0]),
                local=np.log(errs[1:] / errs[:-1]) / np.log(hs[1:] / hs[:-1]))
