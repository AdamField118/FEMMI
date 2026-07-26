"""
femmi/bem_hp.py
Degree-generalised BEM boundary assembly.

`femmi.bem` hardcodes P3 boundary elements -- four nodes per element, 4x4 local
blocks, `elems[:, 3]` for the far endpoint. That is fine while the FEM side is P3,
because the boundary trace of a P3 field is a cubic per edge and the BEM DOFs can
just BE the FEM boundary nodes.

A C^1 element breaks that. The trace of an Argyris field along a boundary edge is
determined by {u, u_t, u_tt} at each endpoint -- six conditions, i.e. a QUINTIC --
so coupling Argyris to the exterior needs degree-5 boundary elements. This module
provides the same three operators for any degree.

Validation strategy: at degree 3 these must reproduce `femmi.bem`'s tuned
assembly to machine precision, which is asserted in tests/test_bem_hp.py. That
keeps the trusted P3 path as the reference rather than re-deriving it.

The singular treatment is carried over unchanged: off-diagonal blocks by
Gauss-Legendre, the self-element block by Duffy decomposition with
log-Gauss-Jacobi for the log singularity.
"""

from __future__ import annotations
import numpy as np

from .bem import BoundaryMesh, log_gauss_jacobi_points, _gauss_legendre


def node_positions(degree, clustering="uniform"):
    """Parameter positions of the boundary-element nodes on [0, 1].

    clustering='uniform'   equispaced (the classical choice, and what degree 3
                           needs to reproduce bem._p3_boundary_basis exactly);
    clustering='chebyshev' Gauss-Lobatto points, clustered toward the element
                           ends.

    Equispaced Lagrange interpolation degrades badly with degree (Runge), and the
    boundary basis here already runs at degree 5 for the Argyris trace. Lobatto
    clustering keeps the Lebesgue constant small, and it also puts more resolution
    where a boundary singularity would sit -- the motivation for trying it against
    the square domain's corners.
    """
    if clustering == "uniform":
        return np.linspace(0.0, 1.0, degree + 1)
    if clustering == "chebyshev":
        k = np.arange(degree + 1)
        return 0.5 * (1.0 - np.cos(np.pi * k / degree))       # Gauss-Lobatto
    raise ValueError(f"unknown clustering {clustering!r}")


def boundary_basis(degree, t_arr, clustering="uniform"):
    """Lagrange basis of the given degree on [0, 1].

    Returns (n_points, degree + 1). degree=3 with uniform clustering reproduces
    bem._p3_boundary_basis.
    """
    t = np.asarray(t_arr, dtype=np.float64).ravel()
    nodes = node_positions(degree, clustering)
    phi = np.ones((len(t), degree + 1))
    for a in range(degree + 1):
        for b in range(degree + 1):
            if a == b:
                continue
            phi[:, a] *= (t - nodes[b]) / (nodes[a] - nodes[b])
    return phi


def arc_geometry(center, radius, theta0, theta1):
    """Exact circular-arc geometry for one boundary element.

    Returns (x(t), |dx/dt|(t), n(t)) callables for t in [0, 1]. Unlike a straight
    chord the Jacobian is constant but the NORMAL rotates along the element, and
    the nodes sit on the true circle rather than inside it.

    Why this matters: approximating a circle by chords leaves an O(h^2) geometry
    error on a method whose entire selling point is the exactness of the exterior
    condition. Curved elements remove it.
    """
    cx, cy = center
    dtheta = theta1 - theta0

    def x_of(t):
        th = theta0 + np.asarray(t) * dtheta
        return np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], -1)

    def jac_of(t):
        return np.full(np.shape(t), abs(radius * dtheta), dtype=float)

    def n_of(t):
        th = theta0 + np.asarray(t) * dtheta
        return np.stack([np.cos(th), np.sin(th)], -1)      # outward for CCW

    return x_of, jac_of, n_of


def build_circular_boundary_mesh(n_elements, degree, radius=2.5, center=(0.0, 0.0),
                                 clustering="uniform"):
    """A boundary mesh whose nodes lie EXACTLY on a circle, with per-element arc
    geometry attached (`.arcs`) for the curved assembly.

    The straight-chord builder places nodes on chords, so the discrete boundary is
    an inscribed polygon of area < pi R^2. Here every node is on the circle.
    """
    tn = node_positions(degree, clustering)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_elements + 1)

    nodes, elements, arcs = [], [], []
    for e in range(n_elements):
        th0, th1 = thetas[e], thetas[e + 1]
        x_of, jac_of, n_of = arc_geometry(center, radius, th0, th1)
        arcs.append((th0, th1))
        idx = []
        for k in range(degree + 1):
            if k == degree:
                idx.append(((e + 1) % n_elements) * degree)
                continue
            idx.append(e * degree + k)
            if len(nodes) <= e * degree + k:
                nodes.append(x_of(tn[k]))
        elements.append(idx)

    nodes = np.array(nodes, float)
    elements = np.array(elements, int)
    arc_len = np.full(n_elements, 2.0 * np.pi * radius / n_elements)
    mid = 0.5 * (thetas[:-1] + thetas[1:])
    normals = np.stack([np.cos(mid), np.sin(mid)], 1)

    bnd = BoundaryMesh(
        node_indices=np.arange(len(nodes)), nodes=nodes,
        edge_lengths=arc_len, normals=normals,
        n_boundary_dofs=len(nodes), elements=elements,
        element_lengths=arc_len, element_normals=normals,
        n_elements=n_elements)
    bnd.arcs = arcs
    bnd.arc_center = center
    bnd.arc_radius = radius
    return bnd


def build_boundary_mesh(points, degree, closed=True):
    """A BoundaryMesh of the given degree from an ordered CCW polygon of corner
    points.

    Each polygon segment becomes one boundary element carrying `degree + 1`
    equispaced Lagrange nodes; consecutive elements share an endpoint node, so the
    trace space is continuous around the loop.
    """
    corners = np.asarray(points, float)
    n_seg = len(corners) if closed else len(corners) - 1

    nodes = []
    elements = []
    for e in range(n_seg):
        a = corners[e]
        b = corners[(e + 1) % len(corners)]
        idx = []
        for k in range(degree + 1):
            if k == degree and closed:
                idx.append((e + 1) % n_seg * degree)     # shared with next element
                continue
            idx.append(e * degree + k)
            if len(nodes) <= e * degree + k:
                nodes.append(a + (k / degree) * (b - a))
        elements.append(idx)

    nodes = np.array(nodes, float)
    elements = np.array(elements, int)
    seg_a = corners[np.arange(n_seg)]
    seg_b = corners[(np.arange(n_seg) + 1) % len(corners)]
    tang = seg_b - seg_a
    lengths = np.linalg.norm(tang, axis=1)
    normals = np.stack([tang[:, 1], -tang[:, 0]], axis=1) / lengths[:, None]

    return BoundaryMesh(
        node_indices=np.arange(len(nodes)),
        nodes=nodes,
        edge_lengths=lengths,
        normals=normals,
        n_boundary_dofs=len(nodes),
        elements=elements,
        element_lengths=lengths,
        element_normals=normals,
        n_elements=n_seg,
    )


def assemble_single_layer_hp(bnd, degree, n_quad=25):
    """V_h[i,j] = int int G(x,y) phi_i(x) phi_j(y) ds ds, G = (1/2pi) log|x-y|.

    Off-diagonal blocks by Gauss-Legendre; the self block by Duffy decomposition
    with log-Gauss-Jacobi, exactly as in bem.assemble_single_layer.
    """
    nd = degree + 1
    N_b = bnd.n_boundary_dofs
    xi_gl, w_gl = _gauss_legendre(n_quad)
    xi_lj, w_lj = log_gauss_jacobi_points(n_quad)
    phi_gl = boundary_basis(degree, xi_gl)

    elems = bnd.elements
    p0 = bnd.nodes[elems[:, 0]]
    p1 = bnd.nodes[elems[:, -1]]
    L = bnd.element_lengths
    x_pts = p0[:, None, :] + xi_gl[None, :, None] * (p1 - p0)[:, None, :]

    V = np.zeros((N_b, N_b))
    for s in range(bnd.n_elements):
        L_s = L[s]
        xs = p0[s][None, :] + xi_gl[:, None] * (p1[s] - p0[s])[None, :]
        diff = xs[None, :, None, :] - x_pts[:, None, :, :]
        r2 = np.sum(diff**2, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.where(r2 > 1e-30, np.log(np.maximum(r2, 1e-300)) / (4.0 * np.pi), 0.0)
        kernel = (L_s * L[:, None, None] * G
                  * w_gl[None, :, None] * w_gl[None, None, :])
        kernel[s] = 0.0
        V_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_gl, phi_gl)

        V_diag = np.zeros((nd, nd))
        for q, (sigma, wq) in enumerate(zip(xi_gl, w_gl)):
            phi_s = phi_gl[q]
            log_Lsig = np.log(L_s * sigma)
            for v, wv in zip(xi_gl, w_gl):
                phi_t = boundary_basis(degree, np.array([sigma * (1.0 - v)]))[0]
                pre = L_s**2 / (2.0 * np.pi) * sigma * log_Lsig * wq * wv
                V_diag += pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))
            for v, wv_lj in zip(xi_lj, w_lj):
                phi_t = boundary_basis(degree, np.array([sigma * (1.0 - v)]))[0]
                pre = L_s**2 / (2.0 * np.pi) * sigma * wq * wv_lj
                V_diag -= pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))
        V_elem[s] = V_diag

        for a in range(nd):
            for b in range(nd):
                np.add.at(V, (elems[s][a], elems[:, b]), V_elem[:, a, b])

    return 0.5 * (V + V.T)


def assemble_double_layer_hp(bnd, degree, n_quad=8):
    """K_h[i,j] = int int dG/dn(y) phi_i(x) phi_j(y) ds ds.

    The self block vanishes for straight segments (the normal is orthogonal to
    the separation), so it is simply zeroed."""
    nd = degree + 1
    N_b = bnd.n_boundary_dofs
    xi_gl, w_gl = _gauss_legendre(n_quad)
    phi_gl = boundary_basis(degree, xi_gl)

    elems = bnd.elements
    p0 = bnd.nodes[elems[:, 0]]
    p1 = bnd.nodes[elems[:, -1]]
    L = bnd.element_lengths
    nrm = bnd.element_normals
    x_pts = p0[:, None, :] + xi_gl[None, :, None] * (p1 - p0)[:, None, :]

    K = np.zeros((N_b, N_b))
    for s in range(bnd.n_elements):
        xs = p0[s][None, :] + xi_gl[:, None] * (p1[s] - p0[s])[None, :]
        diff = xs[None, :, None, :] - x_pts[:, None, :, :]
        r2 = np.sum(diff**2, axis=-1)
        r2 = np.where(r2 < 1e-28, np.inf, r2)
        dGdn = np.sum(diff * nrm[:, None, None, :], axis=-1) / (2.0 * np.pi * r2)
        kernel = (L[s] * L[:, None, None] * dGdn
                  * w_gl[None, :, None] * w_gl[None, None, :])
        kernel[s] = 0.0
        K_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_gl, phi_gl)
        for a in range(nd):
            for b in range(nd):
                np.add.at(K, (elems[s][a], elems[:, b]), K_elem[:, a, b])
    return K


def assemble_boundary_mass_hp(bnd, degree, n_quad=None):
    """M_b[i,j] = int phi_i phi_j ds. Sanity check: M_b @ 1 sums to the perimeter."""
    nd = degree + 1
    n_quad = n_quad or (degree + 4)
    xi_gl, w_gl = _gauss_legendre(n_quad)
    phi_gl = boundary_basis(degree, xi_gl)
    Me_ref = np.einsum('q,qa,qb->ab', w_gl, phi_gl, phi_gl)

    M = np.zeros((bnd.n_boundary_dofs, bnd.n_boundary_dofs))
    for e in range(bnd.n_elements):
        Me = bnd.element_lengths[e] * Me_ref
        el = bnd.elements[e]
        for a in range(nd):
            for b in range(nd):
                M[el[a], el[b]] += Me[a, b]
    return M


def boundary_basis_deriv(degree, t_arr):
    """d/dt of the degree-`degree` Lagrange basis. (n_points, degree + 1)."""
    t = np.asarray(t_arr, dtype=np.float64).ravel()
    nodes = np.linspace(0.0, 1.0, degree + 1)
    d = np.zeros((len(t), degree + 1))
    for a in range(degree + 1):
        for m in range(degree + 1):
            if m == a:
                continue
            term = np.ones_like(t) / (nodes[a] - nodes[m])
            for b in range(degree + 1):
                if b in (a, m):
                    continue
                term = term * (t - nodes[b]) / (nodes[a] - nodes[b])
            d[:, a] += term
    return d


def assemble_hypersingular_hp(bnd, degree, n_quad=25):
    """The hypersingular operator W, via the Nedelec / Maue integration-by-parts
    identity: in 2D the hypersingular form reduces to the SINGLE-LAYER form
    applied to arc-length derivatives,

        <W u, v>  =  - int int  G(x, y)  du/ds(y)  dv/ds(x)  ds(y) ds(x).

    That is why W is cheap here -- it reuses the single-layer kernel and its
    singular treatment, with the basis replaced by its derivative. Assembling W
    directly from the |x-y|^-2 kernel would need a finite-part regularisation.

    Two consequences worth checking (and tested): W annihilates constants, since
    du/ds = 0 for a constant; and W is symmetric.
    """
    nd = degree + 1
    N_b = bnd.n_boundary_dofs
    xi_gl, w_gl = _gauss_legendre(n_quad)
    xi_lj, w_lj = log_gauss_jacobi_points(n_quad)
    dphi = boundary_basis_deriv(degree, xi_gl)

    elems = bnd.elements
    p0 = bnd.nodes[elems[:, 0]]
    p1 = bnd.nodes[elems[:, -1]]
    L = bnd.element_lengths
    x_pts = p0[:, None, :] + xi_gl[None, :, None] * (p1 - p0)[:, None, :]

    W = np.zeros((N_b, N_b))
    for s in range(bnd.n_elements):
        L_s = L[s]
        xs = p0[s][None, :] + xi_gl[:, None] * (p1[s] - p0[s])[None, :]
        diff = xs[None, :, None, :] - x_pts[:, None, :, :]
        r2 = np.sum(diff**2, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.where(r2 > 1e-30, np.log(np.maximum(r2, 1e-300)) / (4.0 * np.pi), 0.0)
        # d/ds = (1/L) d/dt, and the two 1/L factors cancel the two L Jacobians
        kernel = G * w_gl[None, :, None] * w_gl[None, None, :]
        kernel[s] = 0.0
        W_elem = -np.einsum('tqr,qa,rb->tab', kernel, dphi, dphi)

        Wd = np.zeros((nd, nd))
        for q, (sigma, wq) in enumerate(zip(xi_gl, w_gl)):
            d_s = dphi[q]
            log_Lsig = np.log(L_s * sigma)
            for v, wv in zip(xi_gl, w_gl):
                d_t = boundary_basis_deriv(degree, np.array([sigma * (1.0 - v)]))[0]
                pre = 1.0 / (2.0 * np.pi) * sigma * log_Lsig * wq * wv
                Wd += pre * (np.outer(d_s, d_t) + np.outer(d_t, d_s))
            for v, wv_lj in zip(xi_lj, w_lj):
                d_t = boundary_basis_deriv(degree, np.array([sigma * (1.0 - v)]))[0]
                pre = 1.0 / (2.0 * np.pi) * sigma * wq * wv_lj
                Wd -= pre * (np.outer(d_s, d_t) + np.outer(d_t, d_s))
        W_elem[s] = -Wd

        for a in range(nd):
            for b in range(nd):
                np.add.at(W, (elems[s][a], elems[:, b]), W_elem[:, a, b])

    return 0.5 * (W + W.T)


def assemble_bem_hp(bnd, degree, n_quad_sl=25, n_quad_dl=8):
    """(V_h, K_h, M_b) at the given boundary-element degree."""
    return (assemble_single_layer_hp(bnd, degree, n_quad=n_quad_sl),
            assemble_double_layer_hp(bnd, degree, n_quad=n_quad_dl),
            assemble_boundary_mass_hp(bnd, degree))
