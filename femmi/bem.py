"""
femmi/bem.py
============
Boundary Element Method matrices for FEM-BEM coupling.

Uses P3 (cubic) boundary elements matching the FEM interior order.

The P3 mesh boundary nodes naturally sit at t = 0, 1/3, 2/3, 1 on each
original mesh edge, giving one 4-node P3 BEM element per FEM boundary edge.
No new nodes are needed; the BEM and FEM share the same DOFs on ∂Ω.

P3 BEM error in H^{-1/2}(∂Ω): O(h^4).
Coupled system (Costabel-Stephan):
    ‖ψ − ψ_h‖_{H^1(Ω)} = O(h^{min(3,4)}) = O(h^3)
    ‖ψ − ψ_h‖_{L^2(Ω)} = O(h^4)  via Aubin-Nitsche

Provides:
    V_h  – single-layer   (N_b × N_b)  symmetric
    K_h  – double-layer   (N_b × N_b)
    M_b  – boundary mass  (N_b × N_b)  symmetric positive definite
    C    = V_h⁻¹(½M_b + K_h)           as LinearOperator

Public API (unchanged from P1 version):
    BoundaryMesh
    extract_boundary_edges(mesh)           → BoundaryMesh
    log_gauss_jacobi_points(n)             → (nodes, weights)
    assemble_single_layer(bnd, n_quad=25)  → V_h
    assemble_double_layer(bnd, n_quad=8)   → K_h
    assemble_boundary_mass(bnd)            → M_b
    assemble_bem_matrices(bnd, ...)        → (V_h, K_h, M_b)
    calderon_matrix(V_h, K_h, M_b)        → LinearOperator
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import roots_genlaguerre, roots_legendre
from dataclasses import dataclass, field
from typing import Tuple

np.set_printoptions(precision=15)


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundaryMesh:
    """
    Ordered boundary node data for P3 BEM assembly.

    The N_b boundary nodes are CCW-ordered.  Every 3 consecutive nodes
    [3i, 3i+1, 3i+2] together with the next vertex node [3(i+1) % N_b]
    form one P3 boundary element (4 nodes at t = 0, 1/3, 2/3, 1).

    Attributes
    ----------
    node_indices    : (N_b,) int    – global FEM DOF indices
    nodes           : (N_b, 2)     – physical coordinates
    edge_lengths    : (N_b,)       – sub-segment lengths (consecutive nodes)
    normals         : (N_b, 2)     – outward normal of each sub-segment
    n_boundary_dofs : int          – N_b  (total boundary nodes)
    elements        : (N_elem, 4)  – indices into nodes[] for each P3 element
    element_lengths : (N_elem,)    – physical length of each P3 element
    element_normals : (N_elem, 2)  – outward unit normal of each P3 element
    n_elements      : int          – N_elem = N_b // 3
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


# ─────────────────────────────────────────────────────────────────────────────
# P3 Lagrange basis on [0, 1] at nodes {0, 1/3, 2/3, 1}
# ─────────────────────────────────────────────────────────────────────────────

def _p3_boundary_basis(t_arr: np.ndarray) -> np.ndarray:
    """
    Evaluate the four P3 Lagrange basis functions at each point in t_arr.

    Nodes at t = 0, 1/3, 2/3, 1 (matching P3 FEM edge nodes).
    Basis functions:
        φ_0(t) = ½(1−t)(1−3t)(2−3t)   [t=0 vertex]
        φ_1(t) = 9/2 · t(1−t)(2−3t)   [t=1/3 interior]
        φ_2(t) = 9/2 · t(1−t)(3t−1)   [t=2/3 interior]
        φ_3(t) = ½ · t(3t−1)(3t−2)    [t=1 vertex]

    Parameters
    ----------
    t_arr : (n,) array of parameter values in [0, 1]

    Returns
    -------
    phi : (n, 4) array
    """
    t = np.asarray(t_arr, dtype=np.float64).ravel()
    phi = np.empty((len(t), 4), dtype=np.float64)
    phi[:, 0] = 0.5 * (1.0 - t) * (1.0 - 3.0*t) * (2.0 - 3.0*t)
    phi[:, 1] = 4.5 * t * (1.0 - t) * (2.0 - 3.0*t)
    phi[:, 2] = 4.5 * t * (1.0 - t) * (3.0*t - 1.0)
    phi[:, 3] = 0.5 * t * (3.0*t - 1.0) * (3.0*t - 2.0)
    return phi


# ─────────────────────────────────────────────────────────────────────────────
# Boundary mesh extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary_edges(mesh) -> BoundaryMesh:
    """
    Extract boundary nodes from a P3 FEM mesh and return in CCW order.

    Builds P3 boundary elements: each original mesh edge on ∂Ω contributes
    one 4-node element [vertex, t=1/3 node, t=2/3 node, next vertex].

    Requires N_b % 3 == 0 (guaranteed for structured rectangular P3 meshes).
    """
    nodes_all = np.array(mesh.nodes, dtype=np.float64)
    bnd_idx   = np.array(mesh.boundary, dtype=np.int64)
    bnd_coords = nodes_all[bnd_idx]

    xmin = bnd_coords[:, 0].min(); xmax = bnd_coords[:, 0].max()
    ymin = bnd_coords[:, 1].min(); ymax = bnd_coords[:, 1].max()
    tol = 1e-8 * max(xmax - xmin, ymax - ymin)

    on_bottom = np.abs(bnd_coords[:, 1] - ymin) < tol
    on_right  = np.abs(bnd_coords[:, 0] - xmax) < tol
    on_top    = np.abs(bnd_coords[:, 1] - ymax) < tol
    on_left   = np.abs(bnd_coords[:, 0] - xmin) < tol

    def _side(mask):
        idx = bnd_idx[mask]
        coords = nodes_all[idx]
        return list(zip(idx, coords))

    bottom = sorted(_side(on_bottom), key=lambda p: +p[1][0])
    right  = sorted(_side(on_right),  key=lambda p: +p[1][1])
    top    = sorted(_side(on_top),    key=lambda p: -p[1][0])
    left   = sorted(_side(on_left),   key=lambda p: -p[1][1])

    ordered = []
    for side in (bottom, right, top, left):
        ordered.extend(idx for idx, _ in side[:-1])

    ordered = np.array(ordered, dtype=np.int64)
    N_b = len(ordered)

    if N_b == 0:
        raise ValueError("extract_boundary_edges: no boundary nodes found.")
    if N_b % 3 != 0:
        raise ValueError(
            f"extract_boundary_edges: N_b={N_b} is not divisible by 3. "
            "P3 BEM requires 3 nodes per original mesh edge on ∂Ω. "
            "Use a structured P3 mesh.")

    ordered_coords = nodes_all[ordered]  # (N_b, 2)

    # Sub-segment edge lengths and normals (between consecutive boundary nodes)
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
        normals[i] = np.array([dy, -dx]) / L

    # P3 elements: element i has nodes [3i, 3i+1, 3i+2, (3i+3) % N_b]
    N_elem = N_b // 3
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
        element_normals[e] = np.array([dy, -dx]) / L  # outward unit normal

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


# ─────────────────────────────────────────────────────────────────────────────
# Quadrature helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_gauss_jacobi_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    n-point quadrature (nodes, weights) for ∫₀¹ f(t)(−ln t) dt.

    Uses generalised Gauss-Laguerre (α=1) with substitution t = e^{−u}.
    """
    u_nodes, weights = roots_genlaguerre(n, 1)
    return np.exp(-u_nodes), weights


def _gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """n-point Gauss-Legendre nodes and weights on [0, 1]."""
    xi, wi = roots_legendre(n)
    return 0.5 * (xi + 1.0), 0.5 * wi


# ─────────────────────────────────────────────────────────────────────────────
# Single-layer matrix V_h  (P3 elements)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_single_layer(bnd: BoundaryMesh, n_quad: int = 25) -> np.ndarray:
    """
    Assemble the single-layer BEM matrix V_h with P3 boundary elements.

        V_h[i,j] = ∫_{∂Ω}∫_{∂Ω} G(x,y) φ_i(x) φ_j(y) ds(x) ds(y)
        G(x,y)   = (1/2π) ln|x−y|

    Off-diagonal element pairs: standard Gauss-Legendre.
    Diagonal (self-interaction): Duffy decomposition into two GL quadratures
        + one log-GL quadrature, generalised from P1 to P3 basis.

    Returns (N_b × N_b) symmetric matrix.
    """
    N_b    = bnd.n_boundary_dofs
    N_elem = bnd.n_elements
    xi_gl, w_gl = _gauss_legendre(n_quad)
    xi_lj, w_lj = log_gauss_jacobi_points(n_quad)

    phi_gl = _p3_boundary_basis(xi_gl)  # (nq, 4)

    # Precompute element first/last physical points and lengths
    elems = bnd.elements                            # (N_elem, 4) local node indices
    p0_all = bnd.nodes[elems[:, 0]]                 # (N_elem, 2)
    p3_all = bnd.nodes[elems[:, 3]]                 # (N_elem, 2)
    L_all  = bnd.element_lengths                    # (N_elem,)

    # Physical quadrature points on all elements
    # x_pts[e, q, :] = p0_e + t_q * (p3_e - p0_e)
    x_pts = (p0_all[:, None, :] +
             xi_gl[None, :, None] * (p3_all - p0_all)[:, None, :])  # (N_elem, nq, 2)

    V = np.zeros((N_b, N_b))

    for s in range(N_elem):
        L_s    = L_all[s]
        p0_s   = p0_all[s]
        p3_s   = p3_all[s]
        s_nodes = elems[s]   # 4 global local indices

        # Physical quadrature points on element s
        xs = (p0_s[None, :] +
              xi_gl[:, None] * (p3_s - p0_s)[None, :])  # (nq, 2)

        # ── Off-diagonal: vectorized over all t ──────────────────────────
        # diff[t, q, r, :] = xs[q] - x_pts[t, r]
        diff = xs[None, :, None, :] - x_pts[:, None, :, :]  # (N_elem, nq, nq, 2)
        r2   = np.sum(diff**2, axis=-1)                      # (N_elem, nq, nq)
        with np.errstate(divide='ignore', invalid='ignore'):
            G_val = np.where(r2 > 1e-30,
                             np.log(np.maximum(r2, 1e-300)) / (4.0 * np.pi),
                             0.0)  # (N_elem, nq, nq)

        # kernel[t, q, r] = L_s * L_t * G * w_q * w_r
        kernel = (L_s * L_all[:, None, None] * G_val
                  * w_gl[None, :, None] * w_gl[None, None, :])  # (N_elem, nq, nq)
        kernel[s] = 0.0  # diagonal handled below

        # V_elem[t, a, b] = Σ_{q,r} kernel[t,q,r] * phi_a[q] * phi_b[r]
        V_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_gl, phi_gl)  # (N_elem, 4, 4)

        # ── Diagonal: Duffy decomposition ────────────────────────────────
        # ln(L|s-t|) = ln(L*sigma) + ln(1 - v)  after substitution
        # Split into: sigma * ln(L*sigma) term (GL×GL) and -sigma*ln(v) term (GL×log-GL)
        V_diag = np.zeros((4, 4))
        for q, (sigma, wq) in enumerate(zip(xi_gl, w_gl)):
            phi_s = phi_gl[q]                     # (4,)  basis at sigma
            log_Lsig = np.log(L_s * sigma)

            # GL inner integral (ln(sigma) part)
            for r, (v, wv) in enumerate(zip(xi_gl, w_gl)):
                tau     = sigma * (1.0 - v)
                phi_t   = _p3_boundary_basis(np.array([tau]))[0]  # (4,)
                pre     = L_s**2 / (2.0 * np.pi) * sigma * log_Lsig * wq * wv
                V_diag += pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))

            # log-GL inner integral (-ln(v) part)
            for v, wv_lj in zip(xi_lj, w_lj):
                tau     = sigma * (1.0 - v)
                phi_t   = _p3_boundary_basis(np.array([tau]))[0]  # (4,)
                pre     = L_s**2 / (2.0 * np.pi) * sigma * wq * wv_lj
                V_diag -= pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))

        V_elem[s] = V_diag

        # Accumulate into global matrix
        t_nodes = elems  # (N_elem, 4)
        for a in range(4):
            for b in range(4):
                np.add.at(V, (s_nodes[a], t_nodes[:, b]), V_elem[:, a, b])

    return 0.5 * (V + V.T)


# ─────────────────────────────────────────────────────────────────────────────
# Double-layer matrix K_h  (P3 elements)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_double_layer(bnd: BoundaryMesh, n_quad: int = 8) -> np.ndarray:
    """
    Assemble the double-layer BEM matrix K_h with P3 boundary elements.

        K_h[i,j] = ∫_{∂Ω}∫_{∂Ω} (∂G/∂n(y))(x,y) φ_i(x) φ_j(y) ds(x) ds(y)
        ∂G/∂n(y) = (1/2π)(x−y)·n(y) / |x−y|²

    Diagonal element self-interaction is zero for straight boundary segments
    (the factor (x−y)·n(y) = 0 when x−y is parallel to the segment).
    """
    N_b    = bnd.n_boundary_dofs
    N_elem = bnd.n_elements
    xi_gl, w_gl = _gauss_legendre(n_quad)

    phi_gl = _p3_boundary_basis(xi_gl)  # (nq, 4)

    elems  = bnd.elements
    p0_all = bnd.nodes[elems[:, 0]]
    p3_all = bnd.nodes[elems[:, 3]]
    L_all  = bnd.element_lengths
    n_all  = bnd.element_normals          # (N_elem, 2) outward normals

    x_pts = (p0_all[:, None, :] +
             xi_gl[None, :, None] * (p3_all - p0_all)[:, None, :])  # (N_elem, nq, 2)

    K = np.zeros((N_b, N_b))

    for s in range(N_elem):
        L_s     = L_all[s]
        p0_s    = p0_all[s]; p3_s = p3_all[s]
        s_nodes = elems[s]

        xs = (p0_s[None, :] +
              xi_gl[:, None] * (p3_s - p0_s)[None, :])  # (nq, 2)

        # diff[t, q, r, :] = xs[q] - x_pts[t, r]
        diff = xs[None, :, None, :] - x_pts[:, None, :, :]  # (N_elem, nq, nq, 2)
        r2   = np.sum(diff**2, axis=-1)                      # (N_elem, nq, nq)
        r2   = np.where(r2 < 1e-28, np.inf, r2)

        # dGdn[t,q,r] = (x-y)·n_t / (2π|x-y|²)
        dGdn = (np.sum(diff * n_all[:, None, None, :], axis=-1)
                / (2.0 * np.pi * r2))                        # (N_elem, nq, nq)

        kernel = (L_s * L_all[:, None, None] * dGdn
                  * w_gl[None, :, None] * w_gl[None, None, :])
        kernel[s] = 0.0   # zero for straight segments

        K_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_gl, phi_gl)  # (N_elem, 4, 4)

        t_nodes = elems
        for a in range(4):
            for b in range(4):
                np.add.at(K, (s_nodes[a], t_nodes[:, b]), K_elem[:, a, b])

    return K


# ─────────────────────────────────────────────────────────────────────────────
# Boundary mass matrix M_b  (P3 elements)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_boundary_mass(bnd: BoundaryMesh) -> np.ndarray:
    """
    Assemble the boundary mass (Gram) matrix M_b with P3 elements.

        M_b[i,j] = ∫_{∂Ω} φ_i(s) φ_j(s) ds

    Uses 7-point Gauss-Legendre (exact for degree 13, sufficient for P3×P3).

    Verification: M_b @ ones = total boundary length.
    """
    N_b    = bnd.n_boundary_dofs
    N_elem = bnd.n_elements
    xi_gl, w_gl = _gauss_legendre(7)
    phi_gl = _p3_boundary_basis(xi_gl)   # (7, 4)

    # Element mass matrix (reference): Me_ref[a,b] = Σ_q w_q φ_a(t_q) φ_b(t_q)
    Me_ref = np.einsum('q,qa,qb->ab', w_gl, phi_gl, phi_gl)  # (4, 4)

    M = np.zeros((N_b, N_b))
    elems = bnd.elements

    for e in range(N_elem):
        L_e  = bnd.element_lengths[e]
        elem = elems[e]                    # 4 global node indices
        Me   = L_e * Me_ref                # scale by element length
        for a in range(4):
            for b in range(4):
                M[elem[a], elem[b]] += Me[a, b]

    return M


# ─────────────────────────────────────────────────────────────────────────────
# Calderon preconditioner  C = V_h⁻¹(½M_b + K_h)
# ─────────────────────────────────────────────────────────────────────────────

def calderon_matrix(V_h: np.ndarray,
                    K_h: np.ndarray,
                    M_b: np.ndarray) -> spla.LinearOperator:
    """
    C = V_h⁻¹(½M_b + K_h) as a LinearOperator.

    LU-factorises V_h once; matvec applies the factored solve.
    Complexity: O(N_b³) factorisation + O(N_b²) per matvec.
    """
    import scipy.linalg as sla
    N_b = V_h.shape[0]
    half_Mb_plus_Kh = 0.5 * M_b + K_h
    V_lu = sla.lu_factor(V_h)

    def _matvec(x: np.ndarray) -> np.ndarray:
        return sla.lu_solve(V_lu, half_Mb_plus_Kh @ x)

    return spla.LinearOperator(shape=(N_b, N_b), matvec=_matvec, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def assemble_bem_matrices(bnd: BoundaryMesh,
                          n_quad_sl: int = 25,
                          n_quad_dl: int = 8
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble V_h, K_h, M_b in one call."""
    V_h = assemble_single_layer(bnd, n_quad=n_quad_sl)
    K_h = assemble_double_layer(bnd, n_quad=n_quad_dl)
    M_b = assemble_boundary_mass(bnd)
    return V_h, K_h, M_b


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from femmi.p3_mesh_generator import generate_p3_structured_mesh

    print("=" * 60)
    print("femmi/bem.py — P3 BEM smoke test")
    print("=" * 60)

    mesh = generate_p3_structured_mesh(4, 4, xmin=0, xmax=1, ymin=0, ymax=1)
    bnd  = extract_boundary_edges(mesh)
    print(f"N_b = {bnd.n_boundary_dofs},  N_elem = {bnd.n_elements}")
    assert bnd.n_boundary_dofs % 3 == 0, "N_b not divisible by 3"

    # P3 basis partition of unity
    print("\nP3 basis partition of unity (should be 1.0):")
    for t_val in [0.0, 1/3, 0.5, 2/3, 1.0]:
        s = _p3_boundary_basis(np.array([t_val]))[0].sum()
        print(f"  t={t_val:.3f}: Σφ = {s:.15f}")

    # P3 basis Kronecker delta
    print("\nP3 basis Kronecker delta:")
    for k, t_val in enumerate([0.0, 1/3, 2/3, 1.0]):
        phi = _p3_boundary_basis(np.array([t_val]))[0]
        ok = abs(phi[k] - 1.0) < 1e-14 and np.abs(np.delete(phi, k)).max() < 1e-14
        print(f"  t={t_val:.3f}: φ[{k}]={phi[k]:.15f}  {'PASS' if ok else 'FAIL'}")

    # log-GL accuracy
    print("\nlog_gauss_jacobi_points (k=0, weight integral):")
    for n in [10, 25]:
        xi, wi = log_gauss_jacobi_points(n)
        err = abs(np.dot(wi, xi**0) - 1.0)
        print(f"  n={n:2d}: err = {err:.2e}")

    # Boundary mass
    print("\nM_b row sum (should equal perimeter):")
    M_b = assemble_boundary_mass(bnd)
    ones = np.ones(bnd.n_boundary_dofs)
    total     = float(ones @ M_b @ ones)
    perimeter = float(bnd.edge_lengths.sum())
    err = abs(total - perimeter)
    print(f"  M_b@1 = {total:.10f},  perimeter = {perimeter:.10f},  err = {err:.2e}")
    assert err < 1e-12 * perimeter, "M_b row sum FAILED"
    print("  PASS")

    # Single-layer symmetry and sign
    print("\nV_h symmetry and eigenvalue sign:")
    V_h = assemble_single_layer(bnd, n_quad=25)
    sym_err = np.linalg.norm(V_h - V_h.T) / np.linalg.norm(V_h)
    eigs    = np.linalg.eigvalsh(V_h)
    print(f"  symmetry error: {sym_err:.2e}  (threshold 1e-12)")
    print(f"  eigenvalues: [{eigs.min():.4f}, {eigs.max():.4f}]")
    print(f"  all same sign: {'PASS' if ((eigs > 0).all() or (eigs < 0).all()) else 'FAIL'}")

    # Double-layer Calderon identity
    print("\nCalderón identity ‖(½M_b+K_h)·1‖/‖½M_b·1‖:")
    K_h   = assemble_double_layer(bnd, n_quad=8)
    combo = 0.5 * M_b @ ones + K_h @ ones
    ratio = np.linalg.norm(combo) / np.linalg.norm(0.5 * M_b @ ones)
    print(f"  ratio = {ratio:.2e}  (threshold 5e-3)")
    print(f"  {'PASS' if ratio < 5e-3 else 'FAIL'}")

    print("\n✓ All P3 BEM smoke tests done.")