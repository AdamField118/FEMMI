"""
femmi/bem.py
Boundary element matrices for FEM-BEM coupling.

Assembles V_h (single-layer), K_h (double-layer), M_b (boundary mass)
on the P3 boundary mesh, and builds the Calderon preconditioner
C = V_h^{-1}(0.5*M_b + K_h).
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import roots_genlaguerre, roots_legendre
from dataclasses import dataclass, field
from typing import Tuple

np.set_printoptions(precision=15)


@dataclass
class BoundaryMesh:
    """
    Ordered boundary node data for P3 BEM assembly.

    N_b boundary nodes in CCW order. Every 3 consecutive nodes [3i, 3i+1, 3i+2]
    together with node [3(i+1) % N_b] form one P3 boundary element.
    """
    node_indices    : np.ndarray
    nodes           : np.ndarray
    edge_lengths    : np.ndarray
    normals         : np.ndarray
    n_boundary_dofs : int
    elements        : np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=int))
    element_lengths : np.ndarray = field(default_factory=lambda: np.empty(0))
    element_normals : np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    n_elements      : int = 0


def _p3_boundary_basis(t_arr):
    """
    Evaluate the four P3 Lagrange basis functions at each point in t_arr.

    Nodes at t = 0, 1/3, 2/3, 1. Returns (n, 4) array.
    """
    t = np.asarray(t_arr, dtype=np.float64).ravel()
    phi = np.empty((len(t), 4), dtype=np.float64)
    phi[:, 0] = 0.5 * (1.0 - t) * (1.0 - 3.0*t) * (2.0 - 3.0*t)
    phi[:, 1] = 4.5 * t * (1.0 - t) * (2.0 - 3.0*t)
    phi[:, 2] = 4.5 * t * (1.0 - t) * (3.0*t - 1.0)
    phi[:, 3] = 0.5 * t * (3.0*t - 1.0) * (3.0*t - 2.0)
    return phi


def extract_boundary_edges(mesh):
    """
    Extract boundary nodes from a P3 mesh in CCW order.

    Returns a BoundaryMesh with P3 element groupings. Requires N_b % 3 == 0,
    which is guaranteed for structured rectangular P3 meshes.
    """
    nodes_all  = np.array(mesh.nodes, dtype=np.float64)
    bnd_idx    = np.array(mesh.boundary, dtype=np.int64)
    bnd_coords = nodes_all[bnd_idx]

    xmin = bnd_coords[:, 0].min(); xmax = bnd_coords[:, 0].max()
    ymin = bnd_coords[:, 1].min(); ymax = bnd_coords[:, 1].max()
    tol  = 1e-8 * max(xmax - xmin, ymax - ymin)

    on_bottom = np.abs(bnd_coords[:, 1] - ymin) < tol
    on_right  = np.abs(bnd_coords[:, 0] - xmax) < tol
    on_top    = np.abs(bnd_coords[:, 1] - ymax) < tol
    on_left   = np.abs(bnd_coords[:, 0] - xmin) < tol

    def _side(mask):
        idx = bnd_idx[mask]
        coords = nodes_all[idx]
        return list(zip(idx, coords))

    bottom = sorted(_side(on_bottom), key=lambda p:  p[1][0])
    right  = sorted(_side(on_right),  key=lambda p:  p[1][1])
    top    = sorted(_side(on_top),    key=lambda p: -p[1][0])
    left   = sorted(_side(on_left),   key=lambda p: -p[1][1])

    ordered = []
    for side in (bottom, right, top, left):
        ordered.extend(idx for idx, _ in side[:-1])

    ordered = np.array(ordered, dtype=np.int64)
    N_b = len(ordered)

    if N_b == 0:
        raise ValueError("No boundary nodes found.")
    if N_b % 3 != 0:
        raise ValueError(
            f"N_b={N_b} is not divisible by 3. "
            "P3 BEM requires 3 nodes per boundary edge."
        )

    ordered_coords = nodes_all[ordered]

    edge_lengths = np.empty(N_b)
    normals      = np.empty((N_b, 2))
    for i in range(N_b):
        j  = (i + 1) % N_b
        dx = ordered_coords[j, 0] - ordered_coords[i, 0]
        dy = ordered_coords[j, 1] - ordered_coords[i, 1]
        L  = np.hypot(dx, dy)
        if L < 1e-15:
            raise ValueError(f"Degenerate sub-segment at boundary node {ordered[i]}.")
        edge_lengths[i] = L
        normals[i]      = np.array([dy, -dx]) / L

    N_elem          = N_b // 3
    elements        = np.empty((N_elem, 4), dtype=np.int64)
    element_lengths = np.empty(N_elem)
    element_normals = np.empty((N_elem, 2))

    for e in range(N_elem):
        i0 = 3 * e
        i3 = (3 * e + 3) % N_b
        elements[e] = [i0, i0 + 1, i0 + 2, i3]
        p0 = ordered_coords[i0]
        p3 = ordered_coords[i3]
        dx = p3[0] - p0[0]; dy = p3[1] - p0[1]
        L  = np.hypot(dx, dy)
        element_lengths[e] = L
        element_normals[e] = np.array([dy, -dx]) / L

    return BoundaryMesh(
        node_indices    = ordered,
        nodes           = ordered_coords,
        edge_lengths    = edge_lengths,
        normals         = normals,
        n_boundary_dofs = N_b,
        elements        = elements,
        element_lengths = element_lengths,
        element_normals = element_normals,
        n_elements      = N_elem,
    )


def log_gauss_jacobi_points(n):
    """
    n-point quadrature for integrals of the form int_0^1 f(t)*(-ln t) dt.

    Uses generalized Gauss-Laguerre (alpha=1) with t = exp(-u).
    """
    u_nodes, weights = roots_genlaguerre(n, 1)
    return np.exp(-u_nodes), weights


def _gauss_legendre(n):
    """n-point Gauss-Legendre nodes and weights mapped to [0, 1]."""
    xi, wi = roots_legendre(n)
    return 0.5 * (xi + 1.0), 0.5 * wi


def assemble_single_layer(bnd, n_quad=25):
    """
    Assemble the single-layer BEM matrix V_h.

    V_h[i,j] = integral G(x,y) phi_i(x) phi_j(y) ds(x) ds(y)
    where G(x,y) = (1/2pi) ln|x-y|.

    Off-diagonal blocks use Gauss-Legendre; diagonal blocks use
    Duffy decomposition with log-Gauss-Laguerre for the log singularity.
    """
    N_b    = bnd.n_boundary_dofs
    N_elem = bnd.n_elements
    xi_gl, w_gl = _gauss_legendre(n_quad)
    xi_lj, w_lj = log_gauss_jacobi_points(n_quad)

    phi_gl = _p3_boundary_basis(xi_gl)

    elems  = bnd.elements
    p0_all = bnd.nodes[elems[:, 0]]
    p3_all = bnd.nodes[elems[:, 3]]
    L_all  = bnd.element_lengths

    x_pts = (p0_all[:, None, :]
             + xi_gl[None, :, None] * (p3_all - p0_all)[:, None, :])

    V = np.zeros((N_b, N_b))

    for s in range(N_elem):
        L_s     = L_all[s]
        p0_s    = p0_all[s]
        p3_s    = p3_all[s]
        s_nodes = elems[s]

        xs = p0_s[None, :] + xi_gl[:, None] * (p3_s - p0_s)[None, :]

        diff = xs[None, :, None, :] - x_pts[:, None, :, :]
        r2   = np.sum(diff**2, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            G_val = np.where(r2 > 1e-30,
                             np.log(np.maximum(r2, 1e-300)) / (4.0 * np.pi),
                             0.0)

        kernel      = (L_s * L_all[:, None, None] * G_val
                       * w_gl[None, :, None] * w_gl[None, None, :])
        kernel[s]   = 0.0

        V_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_gl, phi_gl)

        # Diagonal self-interaction via Duffy decomposition
        V_diag = np.zeros((4, 4))
        for q, (sigma, wq) in enumerate(zip(xi_gl, w_gl)):
            phi_s    = phi_gl[q]
            log_Lsig = np.log(L_s * sigma)

            for r, (v, wv) in enumerate(zip(xi_gl, w_gl)):
                tau   = sigma * (1.0 - v)
                phi_t = _p3_boundary_basis(np.array([tau]))[0]
                pre   = L_s**2 / (2.0 * np.pi) * sigma * log_Lsig * wq * wv
                V_diag += pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))

            for v, wv_lj in zip(xi_lj, w_lj):
                tau   = sigma * (1.0 - v)
                phi_t = _p3_boundary_basis(np.array([tau]))[0]
                pre   = L_s**2 / (2.0 * np.pi) * sigma * wq * wv_lj
                V_diag -= pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))

        V_elem[s] = V_diag

        t_nodes = elems
        for a in range(4):
            for b in range(4):
                np.add.at(V, (s_nodes[a], t_nodes[:, b]), V_elem[:, a, b])

    return 0.5 * (V + V.T)


def assemble_double_layer(bnd, n_quad=8):
    """
    Assemble the double-layer BEM matrix K_h.

    K_h[i,j] = integral (dG/dn(y))(x,y) phi_i(x) phi_j(y) ds(x) ds(y)
    where dG/dn = (1/2pi) (x-y).n(y) / |x-y|^2.

    Diagonal blocks are zero for straight boundary segments.
    """
    N_b    = bnd.n_boundary_dofs
    N_elem = bnd.n_elements
    xi_gl, w_gl = _gauss_legendre(n_quad)

    phi_gl = _p3_boundary_basis(xi_gl)

    elems  = bnd.elements
    p0_all = bnd.nodes[elems[:, 0]]
    p3_all = bnd.nodes[elems[:, 3]]
    L_all  = bnd.element_lengths
    n_all  = bnd.element_normals

    x_pts = (p0_all[:, None, :]
             + xi_gl[None, :, None] * (p3_all - p0_all)[:, None, :])

    K = np.zeros((N_b, N_b))

    for s in range(N_elem):
        L_s     = L_all[s]
        p0_s    = p0_all[s]; p3_s = p3_all[s]
        s_nodes = elems[s]

        xs   = p0_s[None, :] + xi_gl[:, None] * (p3_s - p0_s)[None, :]
        diff = xs[None, :, None, :] - x_pts[:, None, :, :]
        r2   = np.sum(diff**2, axis=-1)
        r2   = np.where(r2 < 1e-28, np.inf, r2)

        dGdn   = (np.sum(diff * n_all[:, None, None, :], axis=-1)
                  / (2.0 * np.pi * r2))
        kernel = (L_s * L_all[:, None, None] * dGdn
                  * w_gl[None, :, None] * w_gl[None, None, :])
        kernel[s] = 0.0

        K_elem  = np.einsum('tqr,qa,rb->tab', kernel, phi_gl, phi_gl)
        t_nodes = elems
        for a in range(4):
            for b in range(4):
                np.add.at(K, (s_nodes[a], t_nodes[:, b]), K_elem[:, a, b])

    return K


def assemble_boundary_mass(bnd):
    """
    Assemble the boundary Gram matrix M_b.

    M_b[i,j] = integral phi_i(s) phi_j(s) ds.
    Verification: M_b @ ones = perimeter.
    """
    N_b    = bnd.n_boundary_dofs
    N_elem = bnd.n_elements
    xi_gl, w_gl = _gauss_legendre(7)
    phi_gl = _p3_boundary_basis(xi_gl)

    Me_ref = np.einsum('q,qa,qb->ab', w_gl, phi_gl, phi_gl)

    M = np.zeros((N_b, N_b))
    for e in range(N_elem):
        L_e  = bnd.element_lengths[e]
        elem = bnd.elements[e]
        Me   = L_e * Me_ref
        for a in range(4):
            for b in range(4):
                M[elem[a], elem[b]] += Me[a, b]

    return M


def calderon_matrix(V_h, K_h, M_b):
    """
    Return C = V_h^{-1}(0.5*M_b + K_h) as a LinearOperator.

    LU-factorises V_h once; each matvec applies the factored solve.
    """
    import scipy.linalg as sla
    N_b = V_h.shape[0]
    half_Mb_plus_Kh = 0.5 * M_b + K_h
    V_lu = sla.lu_factor(V_h)

    def _matvec(x):
        return sla.lu_solve(V_lu, half_Mb_plus_Kh @ x)

    return spla.LinearOperator(shape=(N_b, N_b), matvec=_matvec, dtype=np.float64)


def assemble_bem_matrices(bnd, n_quad_sl=25, n_quad_dl=8):
    """Assemble and return (V_h, K_h, M_b) in one call."""
    V_h = assemble_single_layer(bnd, n_quad=n_quad_sl)
    K_h = assemble_double_layer(bnd, n_quad=n_quad_dl)
    M_b = assemble_boundary_mass(bnd)
    return V_h, K_h, M_b
    
def extract_boundary_edges_circular(mesh, center=(0.0, 0.0), radius=None):
    nodes_all  = np.array(mesh.nodes,    dtype=np.float64)
    bnd_idx    = np.array(mesh.boundary, dtype=np.int64)
    bnd_coords = nodes_all[bnd_idx]
    cx, cy     = center

    N_b = len(bnd_idx)
    if N_b % 3 != 0:
        raise ValueError(f"N_b={N_b} is not divisible by 3.")

    angles = np.arctan2(bnd_coords[:, 1] - cy, bnd_coords[:, 0] - cx)
    order  = np.argsort(angles)
    ordered        = bnd_idx[order]
    ordered_coords = nodes_all[ordered]

    edge_lengths = np.empty(N_b)
    normals      = np.empty((N_b, 2))
    for i in range(N_b):
        j  = (i + 1) % N_b
        dx = ordered_coords[j, 0] - ordered_coords[i, 0]
        dy = ordered_coords[j, 1] - ordered_coords[i, 1]
        L  = np.hypot(dx, dy)
        if L < 1e-15:
            raise ValueError(f"Degenerate boundary segment at position {i}.")
        edge_lengths[i] = L
        normals[i]      = np.array([dy, -dx]) / L

    N_elem          = N_b // 3
    elements        = np.empty((N_elem, 4), dtype=np.int64)
    element_lengths = np.empty(N_elem)
    element_normals = np.empty((N_elem, 2))

    for e in range(N_elem):
        i0 = 3 * e
        i3 = (3 * e + 3) % N_b
        elements[e] = [i0, i0 + 1, i0 + 2, i3]
        p0, p3 = ordered_coords[i0], ordered_coords[i3]
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
        L      = np.hypot(dx, dy)
        element_lengths[e] = L
        element_normals[e] = np.array([dy, -dx]) / L

    return BoundaryMesh(
        node_indices=ordered, nodes=ordered_coords,
        edge_lengths=edge_lengths, normals=normals,
        n_boundary_dofs=N_b, elements=elements,
        element_lengths=element_lengths, element_normals=element_normals,
        n_elements=N_elem,
    )