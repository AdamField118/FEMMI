"""
femmi/bem.py
============
Boundary Element Method matrices for FEM-BEM coupling.

Provides the three BEM matrices assembled on ∂Ω:

    V_h  – single-layer   (N_b × N_b)  symmetric positive definite
    K_h  – double-layer   (N_b × N_b)  compact perturbation of ½I
    M_b  – boundary mass  (N_b × N_b)  symmetric positive definite

and the Calderon preconditioner:

    C = V_h⁻¹ (½M_b + K_h)

The BEM uses P1 (piecewise-linear) basis functions on all N_b boundary
nodes (vertices + P3 edge nodes on ∂Ω).  The coupled stiffness is then

    A_coupled = K_FEM + Pᵀ C P

where P is the restriction operator (operators.py) that selects boundary
rows from the full FEM DOF vector.

Mathematical reference: MATH.md §5 and §6.
Notation follows Colton & Kress (2013) [C&K] §3.1–3.4.

Public API
----------
    BoundaryMesh                          – ordered boundary node data
    extract_boundary_edges(mesh)          → BoundaryMesh
    log_gauss_jacobi_points(n)            → (nodes, weights) for ∫₀¹ f(t)(−ln t) dt
    assemble_single_layer(bnd, n_quad=25) → V_h
    assemble_double_layer(bnd, n_quad=8)  → K_h
    assemble_boundary_mass(bnd)           → M_b
    calderon_matrix(V_h, K_h, M_b)       → scipy.sparse.linalg.LinearOperator
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import roots_genlaguerre, roots_legendre
from dataclasses import dataclass
from typing import Tuple

# Enforce 64-bit globally via numpy default
np.set_printoptions(precision=15)


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundaryMesh:
    """
    Ordered boundary node data for BEM assembly.

    All arrays are ordered counter-clockwise so that the domain interior is
    to the left of the direction of travel.

    Attributes
    ----------
    node_indices    : (N_b,) int    – global DOF indices into the FEM mesh
    nodes           : (N_b, 2)     – physical coordinates of boundary nodes
    edge_lengths    : (N_b,)       – length h_i of segment i → (i+1) % N_b
    normals         : (N_b, 2)     – outward unit normal of segment i → (i+1) % N_b
    n_boundary_dofs : int          – N_b
    """
    node_indices   : np.ndarray
    nodes          : np.ndarray
    edge_lengths   : np.ndarray
    normals        : np.ndarray
    n_boundary_dofs: int


# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Boundary mesh extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary_edges(mesh) -> BoundaryMesh:
    """
    Extract boundary nodes from a P3 FEM mesh and return them in CCW order.

    Works for rectangular domains (the current FEMMI domain shape).  Nodes
    are sorted onto four sides of the rectangle and ordered:

        bottom (y ≈ ymin): left → right
        right  (x ≈ xmax): bottom → top
        top    (y ≈ ymax): right → left
        left   (x ≈ xmin): top → bottom

    Parameters
    ----------
    mesh : femmi.fem_solver.Mesh
        P3 FEM mesh with `nodes` (n_nodes × 2) and `boundary` arrays.

    Returns
    -------
    BoundaryMesh
        Ordered boundary node data including CCW normals.

    Notes
    -----
    For a CCW-ordered boundary, the outward unit normal of segment i → i+1
    with tangent (dx, dy) is  n = (dy, −dx) / |(dx, dy)|.
    """
    nodes_all = np.array(mesh.nodes, dtype=np.float64)
    bnd_idx   = np.array(mesh.boundary, dtype=np.int64)

    bnd_coords = nodes_all[bnd_idx]

    xmin = bnd_coords[:, 0].min()
    xmax = bnd_coords[:, 0].max()
    ymin = bnd_coords[:, 1].min()
    ymax = bnd_coords[:, 1].max()

    # Tolerance for classifying nodes onto sides (1e-8 of domain width)
    tol = 1e-8 * max(xmax - xmin, ymax - ymin)

    # Classify each boundary node onto one or more sides (corner nodes → both)
    on_bottom = np.abs(bnd_coords[:, 1] - ymin) < tol
    on_right  = np.abs(bnd_coords[:, 0] - xmax) < tol
    on_top    = np.abs(bnd_coords[:, 1] - ymax) < tol
    on_left   = np.abs(bnd_coords[:, 0] - xmin) < tol

    # Collect each side as list of (global_idx, coord)
    def _side(mask, key):
        idx = bnd_idx[mask]
        coords = nodes_all[idx]
        return list(zip(idx, coords))

    bottom = _side(on_bottom, "bottom")
    right  = _side(on_right,  "right")
    top    = _side(on_top,    "top")
    left   = _side(on_left,   "left")

    # Sort each side for CCW travel
    bottom.sort(key=lambda p: +p[1][0])   # increasing x
    right .sort(key=lambda p: +p[1][1])   # increasing y
    top   .sort(key=lambda p: -p[1][0])   # decreasing x
    left  .sort(key=lambda p: -p[1][1])   # decreasing y

    # Concatenate, dropping the last node of each side (it is the first of
    # the next side – the four corners are each shared by two adjacent sides).
    ordered = []
    for side in (bottom, right, top, left):
        ordered.extend(idx for idx, _ in side[:-1])

    ordered = np.array(ordered, dtype=np.int64)
    N_b = len(ordered)

    if N_b == 0:
        raise ValueError("extract_boundary_edges: no boundary nodes found.")

    ordered_coords = nodes_all[ordered]   # (N_b, 2)

    # Edge lengths and outward normals (one per segment i → (i+1)%N_b)
    edge_lengths = np.empty(N_b)
    normals      = np.empty((N_b, 2))

    for i in range(N_b):
        j  = (i + 1) % N_b
        dx = ordered_coords[j, 0] - ordered_coords[i, 0]
        dy = ordered_coords[j, 1] - ordered_coords[i, 1]
        L  = np.hypot(dx, dy)
        if L < 1e-15:
            raise ValueError(
                f"extract_boundary_edges: degenerate segment at node {ordered[i]}.")
        edge_lengths[i] = L
        # Outward normal for CCW boundary: rotate tangent (dx,dy) by −90°
        normals[i] = np.array([dy, -dx]) / L

    return BoundaryMesh(
        node_indices   = ordered,
        nodes          = ordered_coords,
        edge_lengths   = edge_lengths,
        normals        = normals,
        n_boundary_dofs= N_b,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.2  Log-Gauss-Jacobi quadrature
# ─────────────────────────────────────────────────────────────────────────────

def log_gauss_jacobi_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return n-point quadrature (nodes, weights) for integrals of the form

        ∫₀¹ f(t) (−ln t) dt.

    Derivation
    ----------
    Substitute t = e^{−u}, dt = −e^{−u} du:

        ∫₀¹ f(t)(−ln t) dt = ∫₀^∞ f(e^{−u}) u e^{−u} du

    which is a Gauss-Laguerre integral with weight w(u) = u e^{−u}.
    Generalized Gauss-Laguerre with α = 1 gives exact integration of
    P(u) u e^{−u} for polynomials P of degree ≤ 2n−1.

    Parameters
    ----------
    n : int
        Number of quadrature points (n = 10 gives relative error < 1e-12
        for ∫₀¹ tᵏ (−ln t) dt, k = 0..4; n = 25 recommended for production).

    Returns
    -------
    nodes   : (n,) array of t-values in (0, 1)
    weights : (n,) array of positive weights summing to Γ(2) = 1

    Reference
    ---------
    Sauter & Schwab (2011) Table 5.3; MATH.md §5.2.
    """
    # GL(α=1) nodes u_i and weights w_i for ∫₀^∞ f(u) u e^{-u} du
    u_nodes, weights = roots_genlaguerre(n, 1)
    # Map back to [0, 1]: t_i = e^{-u_i}
    t_nodes = np.exp(-u_nodes)
    return t_nodes, weights


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return n-point Gauss-Legendre nodes and weights on [0, 1]."""
    xi, wi = roots_legendre(n)
    # roots_legendre gives nodes on [-1,1]; map to [0,1]
    return 0.5 * (xi + 1.0), 0.5 * wi


def _segment_point(bnd: BoundaryMesh, seg: int, xi: float) -> np.ndarray:
    """
    Physical point at parameter xi ∈ [0,1] on segment seg → (seg+1)%N_b.
    """
    N_b = bnd.n_boundary_dofs
    p0  = bnd.nodes[seg]
    p1  = bnd.nodes[(seg + 1) % N_b]
    return (1.0 - xi) * p0 + xi * p1


def _segment_points_batch(bnd: BoundaryMesh, seg: int,
                          xi_arr: np.ndarray) -> np.ndarray:
    """Return (len(xi_arr), 2) physical points on segment seg."""
    N_b = bnd.n_boundary_dofs
    p0  = bnd.nodes[seg]
    p1  = bnd.nodes[(seg + 1) % N_b]
    return np.outer(1.0 - xi_arr, p0) + np.outer(xi_arr, p1)


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  Single-layer matrix V_h
# ─────────────────────────────────────────────────────────────────────────────

def assemble_single_layer(bnd: BoundaryMesh, n_quad: int = 25) -> np.ndarray:
    """
    Assemble the single-layer BEM matrix V_h.

        V_h[i,j] = ∫_{∂Ω} ∫_{∂Ω} G(x,y) φᵢ(x) φⱼ(y) ds(x) ds(y)
        G(x,y)   = (1/2π) ln|x − y|   (2-D Laplacian fundamental solution)

    φᵢ are piecewise-linear (P1) hat functions on the boundary.

    Assembly strategy
    -----------------
    - Off-diagonal pairs (s ≠ t): fully vectorized numpy einsum.
      For each outer segment s, all N_b inner segments t are handled
      simultaneously: O(N_b × n_quad²) work per outer iteration.
    - Diagonal pairs (s = t): Duffy transformation (scalar loop over N_b).
      These are only N_b ≪ N_b² pairs, so cost is negligible.

    Total complexity: O(N_b × n_quad²) numpy operations vs O(N_b² × n_quad²)
    Python iterations in the scalar version — ~60× faster in practice.

    Parameters
    ----------
    bnd    : BoundaryMesh  (from extract_boundary_edges)
    n_quad : int           quadrature points per 1-D integral (default 25)

    Returns
    -------
    V_h : (N_b, N_b) symmetric negative-definite numpy array
          (negative-definite when domain capacity < 1, e.g. unit square)
    """
    N_b = bnd.n_boundary_dofs
    xi_gl, w_gl = _gauss_legendre(n_quad)
    xi_lj, w_lj = log_gauss_jacobi_points(n_quad)

    # Precompute quadrature points and basis functions on all segments
    p0 = bnd.nodes                                       # (N_b, 2)
    p1 = bnd.nodes[np.arange(1, N_b + 1) % N_b]        # (N_b, 2)
    L_all = bnd.edge_lengths                             # (N_b,)

    # y_pts[t, r, :] — quadrature points on segment t at parameter xi_gl[r]
    alpha = xi_gl[None, :]                                       # (1, nq)
    y_pts = (p0[:, None, :] * (1 - alpha[:, :, None])
           + p1[:, None, :] *      alpha[:, :, None])            # (N_b, nq, 2)

    # Shared basis functions (same for all segments)
    phi_q = np.stack([1 - xi_gl, xi_gl], axis=-1)               # (nq, 2)

    V = np.zeros((N_b, N_b))
    j0_all = np.arange(N_b)
    j1_all = (np.arange(N_b) + 1) % N_b

    for s in range(N_b):
        L_s   = L_all[s]
        x_pts = ((1 - xi_gl[:, None]) * bnd.nodes[s]
               +      xi_gl[:, None]  * bnd.nodes[(s + 1) % N_b])  # (nq, 2)
        phi_a = phi_q                                               # (nq, 2)

        # ── Off-diagonal: vectorized over all t ──────────────────────────
        # diff[t, q, r, :] = x_pts[q] - y_pts[t, r]
        diff  = (x_pts[None, :, None, :]                           # (1, nq, 1, 2)
               - y_pts[:, None, :, :])                             # (N_b, 1, nq, 2)
        r     = np.sqrt(np.sum(diff ** 2, axis=-1))                # (N_b, nq, nq)
        r_safe = np.where(r < 1e-15, np.nan, r)
        G_val = np.where(np.isnan(r_safe), 0.0,
                         np.log(r_safe) / (2.0 * np.pi))           # (N_b, nq, nq)

        # kernel[t,q,r] = L_s * L_t * G * w_q * w_r
        kernel = (L_s * L_all[:, None, None] * G_val
                  * w_gl[None, :, None] * w_gl[None, None, :])    # (N_b, nq, nq)
        kernel[s] = 0.0   # diagonal handled below

        # V_elem[t, a, b] = Σ_{q,r} kernel[t,q,r] * phi_a[q,a] * phi_b[r,b]
        V_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_a, phi_q)  # (N_b, 2, 2)

        # ── Diagonal: Duffy (scalar, only 1 block per outer iteration) ───
        V_diag = np.zeros((2, 2))
        for sigma, wq in zip(xi_gl, w_gl):
            if sigma <= 0:
                continue
            log_Lsig = np.log(L_s * sigma)
            phi_out  = np.array([1.0 - sigma, sigma])
            for v, wv in zip(xi_gl, w_gl):
                tau     = sigma * (1.0 - v)
                phi_in  = np.array([1.0 - tau, tau])
                pre     = L_s**2 / (2.0*np.pi) * sigma * log_Lsig * wq * wv
                V_diag += pre * (np.outer(phi_out, phi_in)
                               + np.outer(phi_in, phi_out))
            for v, wv_lj in zip(xi_lj, w_lj):
                tau     = sigma * (1.0 - v)
                phi_in  = np.array([1.0 - tau, tau])
                pre     = L_s**2 / (2.0*np.pi) * sigma * wq * wv_lj
                V_diag -= pre * (np.outer(phi_out, phi_in)
                               + np.outer(phi_in, phi_out))
        V_elem[s] = V_diag

        # Accumulate into global matrix
        i0, i1 = s, (s + 1) % N_b
        np.add.at(V, (i0, j0_all), V_elem[:, 0, 0])
        np.add.at(V, (i0, j1_all), V_elem[:, 0, 1])
        np.add.at(V, (i1, j0_all), V_elem[:, 1, 0])
        np.add.at(V, (i1, j1_all), V_elem[:, 1, 1])

    return 0.5 * (V + V.T)


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  Double-layer matrix K_h
# ─────────────────────────────────────────────────────────────────────────────

def assemble_double_layer(bnd: BoundaryMesh, n_quad: int = 8) -> np.ndarray:
    """
    Assemble the double-layer BEM matrix K_h.

        K_h[i,j] = ∫_{∂Ω} ∫_{∂Ω} (∂G/∂n(y))(x,y) φᵢ(x) φⱼ(y) ds(x) ds(y)
        ∂G/∂n(y) = (1/2π) (x − y)·n(y) / |x − y|²

    Assembly strategy
    -----------------
    Fully vectorized: for each outer segment s, all N_b inner segments t
    are handled simultaneously via numpy broadcasting.

    The diagonal block (s = t) contributes zero for straight boundary
    segments because (x−y) ∥ segment while n(y) ⊥ segment.
    """
    N_b = bnd.n_boundary_dofs
    xi_gl, w_gl = _gauss_legendre(n_quad)

    p0    = bnd.nodes
    p1    = bnd.nodes[np.arange(1, N_b + 1) % N_b]
    L_all = bnd.edge_lengths
    n_all = bnd.normals                                            # (N_b, 2)

    # Precompute quadrature points on all segments
    alpha = xi_gl[None, :]
    y_pts = (p0[:, None, :] * (1 - alpha[:, :, None])
           + p1[:, None, :] *      alpha[:, :, None])             # (N_b, nq, 2)

    phi_q = np.stack([1 - xi_gl, xi_gl], axis=-1)                 # (nq, 2)

    K = np.zeros((N_b, N_b))
    j0_all = np.arange(N_b)
    j1_all = (np.arange(N_b) + 1) % N_b

    for s in range(N_b):
        L_s   = L_all[s]
        x_pts = ((1 - xi_gl[:, None]) * bnd.nodes[s]
               +      xi_gl[:, None]  * bnd.nodes[(s + 1) % N_b])  # (nq, 2)

        # diff[t, q, r, :] = x_pts[q] - y_pts[t, r]
        diff  = (x_pts[None, :, None, :]                           # (1, nq, 1, 2)
               - y_pts[:, None, :, :])                             # (N_b, 1, nq, 2)
        r2    = np.sum(diff ** 2, axis=-1)                         # (N_b, nq, nq)
        r2    = np.where(r2 < 1e-28, np.inf, r2)

        # dGdn[t,q,r] = (x-y)·n(y) / (2π |x-y|²)
        dGdn  = (np.sum(diff * n_all[:, None, None, :], axis=-1)
                 / (2.0 * np.pi * r2))                            # (N_b, nq, nq)

        kernel = (L_s * L_all[:, None, None] * dGdn
                  * w_gl[None, :, None] * w_gl[None, None, :])   # (N_b, nq, nq)
        kernel[s] = 0.0   # diagonal is zero for straight segments

        # K_elem[t, a, b] = Σ_{q,r} kernel[t,q,r] * phi_a[q,a] * phi_b[r,b]
        K_elem = np.einsum('tqr,qa,rb->tab', kernel, phi_q, phi_q)  # (N_b, 2, 2)

        i0, i1 = s, (s + 1) % N_b
        np.add.at(K, (i0, j0_all), K_elem[:, 0, 0])
        np.add.at(K, (i0, j1_all), K_elem[:, 0, 1])
        np.add.at(K, (i1, j0_all), K_elem[:, 1, 0])
        np.add.at(K, (i1, j1_all), K_elem[:, 1, 1])

    return K


# ─────────────────────────────────────────────────────────────────────────────
# 1.5  Boundary mass matrix M_b
# ─────────────────────────────────────────────────────────────────────────────

def assemble_boundary_mass(bnd: BoundaryMesh) -> np.ndarray:
    """
    Assemble the boundary mass (Gram) matrix M_b.

        M_b[i,j] = ∫_{∂Ω} φᵢ(s) φⱼ(s) ds

    For P1 basis functions on segment k of length L_k, the 2×2 element
    contribution is the standard 1-D mass matrix:

        M^k = L_k × [[1/3, 1/6],
                     [1/6, 1/3]]

    This uses the exact formula; no numerical quadrature is needed.

    Verification: M_b @ ones = total boundary length  (MATH.md §1.5).
    """
    N_b = bnd.n_boundary_dofs
    M   = np.zeros((N_b, N_b))

    for k in range(N_b):
        L_k = bnd.edge_lengths[k]
        i0  = k
        i1  = (k + 1) % N_b
        M[i0, i0] += L_k / 3.0
        M[i1, i1] += L_k / 3.0
        M[i0, i1] += L_k / 6.0
        M[i1, i0] += L_k / 6.0

    return M


# ─────────────────────────────────────────────────────────────────────────────
# 1.6  Calderon preconditioner
# ─────────────────────────────────────────────────────────────────────────────

def calderon_matrix(V_h: np.ndarray,
                    K_h: np.ndarray,
                    M_b: np.ndarray) -> spla.LinearOperator:
    """
    Build the Calderon preconditioner  C = V_h⁻¹ (½M_b + K_h).

    Rather than forming the dense inverse V_h⁻¹, this function:
      1. Factorises V_h once via scipy LU (O(N_b³) cost, one-time).
      2. Returns a LinearOperator whose matvec applies the LU solve.

    Complexity: O(N_b³) factorisation + O(N_b²) per matvec.

    Parameters
    ----------
    V_h : (N_b, N_b) symmetric positive-definite single-layer matrix
    K_h : (N_b, N_b) double-layer matrix
    M_b : (N_b, N_b) boundary mass matrix

    Returns
    -------
    scipy.sparse.linalg.LinearOperator  (N_b × N_b)
        Applies  x ↦ V_h⁻¹ (½M_b + K_h) x  without forming the inverse.

    Notes
    -----
    The Schur complement derivation is in MATH.md §6.2:
        t = V_h⁻¹ (½M_b + K_h) P ψ
    which is substituted into the FEM equation to eliminate t and obtain
    A_coupled = K_FEM + Pᵀ C P.
    """
    N_b   = V_h.shape[0]
    half_Mb_plus_Kh = 0.5 * M_b + K_h   # (N_b, N_b) dense matrix

    # LU factorisation of V_h (one-time cost)
    # scipy.linalg.lu_factor gives a compact form; use splu-equivalent
    import scipy.linalg as sla
    V_lu = sla.lu_factor(V_h)

    def _matvec(x: np.ndarray) -> np.ndarray:
        """Apply C = V_h⁻¹ (½M_b + K_h) to vector x."""
        y = half_Mb_plus_Kh @ x
        return sla.lu_solve(V_lu, y)

    return spla.LinearOperator(
        shape    = (N_b, N_b),
        matvec   = _matvec,
        dtype    = np.float64,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: assemble all three matrices at once
# ─────────────────────────────────────────────────────────────────────────────

def assemble_bem_matrices(bnd: BoundaryMesh,
                          n_quad_sl: int = 25,
                          n_quad_dl: int = 8
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble V_h, K_h, M_b in one call.

    Parameters
    ----------
    bnd        : BoundaryMesh
    n_quad_sl  : quadrature order for single-layer (default 25)
    n_quad_dl  : quadrature order for double-layer (default 8)

    Returns
    -------
    (V_h, K_h, M_b)
    """
    V_h = assemble_single_layer(bnd, n_quad=n_quad_sl)
    K_h = assemble_double_layer(bnd, n_quad=n_quad_dl)
    M_b = assemble_boundary_mass(bnd)
    return V_h, K_h, M_b


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / self-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick self-test on a 4×4 mesh."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from femmi.p3_mesh_generator import generate_p3_structured_mesh

    print("=" * 60)
    print("femmi/bem.py — smoke test")
    print("=" * 60)

    mesh = generate_p3_structured_mesh(4, 4, xmin=-1, xmax=1, ymin=-1, ymax=1)
    bnd  = extract_boundary_edges(mesh)
    print(f"Boundary DOFs: N_b = {bnd.n_boundary_dofs}")

    # Log-Gauss-Jacobi check
    print("\nlog_gauss_jacobi_points accuracy:")
    for n in [5, 10, 25]:
        xi, wi = log_gauss_jacobi_points(n)
        for k in range(5):
            integral = np.dot(wi, xi**k)
            exact    = 1.0 / (k + 1)**2
            print(f"  n={n:2d}, k={k}: computed={integral:.12f}, "
                  f"exact={exact:.12f}, rel_err={abs(integral-exact)/exact:.2e}")

    # Boundary mass check
    print("\nM_b assembly:")
    M_b = assemble_boundary_mass(bnd)
    ones = np.ones(bnd.n_boundary_dofs)
    total = float(ones @ M_b @ ones)
    perimeter = float(np.sum(bnd.edge_lengths))
    print(f"  M_b @ ones: {total:.8f}  (expected perimeter = {perimeter:.8f})")
    assert abs(total - perimeter) < 1e-12 * perimeter, "M_b row-sum FAILED"
    print("  M_b check PASSED ✓")

    # Single-layer assembly (small mesh, low quadrature)
    print("\nV_h assembly (n_quad=10)...")
    V_h = assemble_single_layer(bnd, n_quad=10)
    sym_err = np.linalg.norm(V_h - V_h.T) / np.linalg.norm(V_h)
    eigs    = np.linalg.eigvalsh(V_h)
    print(f"  Symmetry error : {sym_err:.2e}  (threshold 1e-12)")
    print(f"  Min eigenvalue : {eigs.min():.6f}  (should be > 0)")
    assert sym_err < 1e-10, "V_h symmetry FAILED"
    assert eigs.min() > -1e-10, "V_h not positive semidefinite"
    print("  V_h checks PASSED ✓")

    # Double-layer row-sum check
    print("\nK_h assembly (n_quad=8)...")
    K_h  = assemble_double_layer(bnd, n_quad=8)
    rsum = K_h @ ones
    print(f"  Max |row sum|  : {np.abs(rsum).max():.2e}  (should be ≈ 0)")
    print("  K_h assembled ✓")

    print("\n✓ All smoke tests passed.")