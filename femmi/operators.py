"""
operators.py  --  P3 FEM operators for weak gravitational lensing.

Operators:
    K  : stiffness   K[i,j] = int grad(Ni).grad(Nj) dA
    M  : mass        M[i,j] = int Ni Nj dA
    S1 : shear-1     (S1 psi)[i] = 0.5*(psi_xx - psi_yy) at node i
    S2 : shear-2     (S2 psi)[i] = psi_xy at node i

Forward model:  psi = K^-1(-2 M kappa),  gamma = S psi

Two entry points:
    build_operators(nx, ny, ...)               -- uniform structured mesh
    build_operators_adaptive(nx, ny, ...,
        mask_center, mask_radius,
        refine_factor)                         -- locally refined mesh

Both return identical FEMOperators objects.

Regularizer helper:
    build_wiener_regularizer(ops, wiener_length)
        Returns R = M + l^2 * K  (Matern-like prior, replaces plain K).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional
import time

from .mesh import (
    generate_p3_structured_mesh,
    generate_p3_adaptive_mesh,
)
from .basis import (
    compute_p3_shape_functions,
    compute_p3_shape_gradients_reference,
)
from .assembly import (
    get_gauss_quadrature_triangle,
    compute_element_stiffness_p3,
    compute_element_load_p3,
    apply_boundary_conditions_p3,
)
from .types import Mesh
from .bem import extract_boundary_edges, assemble_bem_matrices


# =============================================================================
# Reference Hessians  (precomputed once at import)
# =============================================================================

_P3_REF_NODES = np.array([
    [0.0,      0.0     ],
    [1.0,      0.0     ],
    [0.0,      1.0     ],
    [1.0/3.0,  0.0     ],
    [2.0/3.0,  0.0     ],
    [2.0/3.0,  1.0/3.0 ],
    [1.0/3.0,  2.0/3.0 ],
    [0.0,      2.0/3.0 ],
    [0.0,      1.0/3.0 ],
    [1.0/3.0,  1.0/3.0 ],
])


def _build_ref_hessians() -> np.ndarray:
    """H_ref[eval_node, shape_fn, i, j] shape (10,10,2,2). JAX AD."""
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])
    hess_fn = jax.jacfwd(jax.jacrev(N_vec))
    return np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])


# =============================================================================
# Assembly routines
# =============================================================================

def _assemble_mass_p3(nodes, elements, quad_points, quad_weights):
    """M[i,j] = int Ni Nj dA."""
    n_nodes = len(nodes)
    max_nnz = len(elements) * 100
    I = np.zeros(max_nnz, dtype=np.int32)
    J = np.zeros(max_nnz, dtype=np.int32)
    D = np.zeros(max_nnz)
    idx = 0
    for elem in elements:
        x0, y0 = nodes[elem[0]]; x1, y1 = nodes[elem[1]]; x2, y2 = nodes[elem[2]]
        Jac  = np.array([[x1-x0, y1-y0], [x2-x0, y2-y0]])
        area = abs(np.linalg.det(Jac)) / 2.0
        Me   = np.zeros((10, 10))
        for q, (xi, eta) in enumerate(quad_points):
            N = np.array(compute_p3_shape_functions(xi, eta))
            Me += quad_weights[q] * area * np.outer(N, N)
        for i in range(10):
            for j in range(10):
                I[idx] = elem[i]; J[idx] = elem[j]; D[idx] = Me[i,j]; idx += 1
    return sp.coo_matrix((D[:idx], (I[:idx], J[:idx])),
                         shape=(n_nodes, n_nodes)).tocsr()


def _assemble_shear_ops(nodes, elements, H_ref):
    """S1, S2 sparse operators via nodal-averaged element Hessians."""
    n_nodes = len(nodes)
    max_nnz = len(elements) * 100
    I1 = np.zeros(max_nnz, dtype=np.int32); J1 = np.zeros(max_nnz, dtype=np.int32); D1 = np.zeros(max_nnz)
    I2 = np.zeros(max_nnz, dtype=np.int32); J2 = np.zeros(max_nnz, dtype=np.int32); D2 = np.zeros(max_nnz)
    idx    = 0
    counts = np.zeros(n_nodes, dtype=np.int32)
    for elem in elements:
        x0,y0=nodes[elem[0]]; x1,y1=nodes[elem[1]]; x2,y2=nodes[elem[2]]
        Jac = np.array([[x1-x0,y1-y0],[x2-x0,y2-y0]])
        A   = np.linalg.inv(Jac).T
        for li in range(10):
            H_phys = np.einsum('ja,kb,njk->nab', A, A, H_ref[li])
            row = elem[li]
            for lj in range(10):
                col = elem[lj]
                I1[idx]=row; J1[idx]=col; D1[idx]=0.5*(H_phys[lj,0,0]-H_phys[lj,1,1])
                I2[idx]=row; J2[idx]=col; D2[idx]=H_phys[lj,0,1]
                idx += 1
            counts[row] += 1
    S1r = sp.coo_matrix((D1[:idx],(I1[:idx],J1[:idx])),shape=(n_nodes,n_nodes)).tocsr()
    S2r = sp.coo_matrix((D2[:idx],(I2[:idx],J2[:idx])),shape=(n_nodes,n_nodes)).tocsr()
    sc  = sp.diags(1.0 / np.maximum(counts, 1))
    return (sc @ S1r).tocsr(), (sc @ S2r).tocsr()

def _precompute_reference_data(quad_pts_np: np.ndarray,
                               quad_wts_np: np.ndarray):
    """
    Precompute shape function values and reference gradients at all
    quadrature points.  Called once per assembly; shapes are fixed.
 
    Returns
    -------
    N_ref  : (nq, 10)    – N_j(xi_q, eta_q)
    dN_ref : (nq, 10, 2) – [dN_j/dxi, dN_j/deta] at each quad point
    """
    nq = len(quad_wts_np)
    N_ref  = np.zeros((nq, 10), dtype=np.float64)
    dN_ref = np.zeros((nq, 10, 2), dtype=np.float64)
 
    for q, (xi, eta) in enumerate(quad_pts_np):
        # Use JAX autodiff for accuracy, but only called nq=13 times total
        N_ref[q]  = np.array(compute_p3_shape_functions(xi, eta))
        dN_ref[q] = np.array(
            compute_p3_shape_gradients_reference(xi, eta))
 
    return N_ref, dN_ref

# =============================================================================
# FEMOperators
# =============================================================================

@dataclass
class FEMOperators:
    """
    All precomputed FEM-BEM operators for a fixed mesh.
 
    Attributes
    ----------
    mesh          : P3 Mesh (nodes, elements, boundary)
    K             : Neumann stiffness — NO Dirichlet rows (null space = span{1})
    M             : Full mass matrix — NO boundary row zeroing
    S1, S2        : shear operators
    A_coupled     : K_neumann + P^T C P  (FEM-BEM coupled stiffness)
    A_coupled_lu  : SuperLU factorization of A_coupled
    bnd_mesh      : BoundaryMesh from extract_boundary_edges
    C_dense       : (N_b × N_b) Calderon matrix V_h^{-1}(½M_b + K_h)
    n_nodes       : total node count
    boundary      : boundary node indices (from P3 mesh, unordered)
    interior      : bool mask (True = interior node)
    """
    mesh         : object
    K            : sp.csr_matrix
    M            : sp.csr_matrix
    S1           : sp.csr_matrix
    S2           : sp.csr_matrix
    A_coupled    : sp.csr_matrix
    A_coupled_lu : object
    bnd_mesh     : object
    C_dense      : np.ndarray
    n_nodes      : int
    boundary     : np.ndarray
    interior     : np.ndarray
 
    def psi_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        """Solve A_coupled ψ = −2Mκ (FEM-BEM coupled system)."""
        rhs = -2.0 * self.M @ kappa
        rhs[int(self.bnd_mesh.node_indices[0])] = 0.0   # gauge fix
        return self.A_coupled_lu.solve(rhs)
 
    def shear_from_psi(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.S1 @ psi, self.S2 @ psi
 
    def forward(self, kappa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.shear_from_psi(self.psi_from_kappa(kappa))
 
    def shear_magnitude(self, kappa: np.ndarray) -> np.ndarray:
        g1, g2 = self.forward(kappa)
        return np.sqrt(g1**2 + g2**2)
 
    def adjoint_rhs(self, dL_dg1: np.ndarray,
                    dL_dg2: np.ndarray) -> np.ndarray:
        """dL/dkappa = −2 Mᵀ A_coupled⁻¹ (S1ᵀ dL/dg1 + S2ᵀ dL/dg2)."""
        rhs = self.S1.T @ dL_dg1 + self.S2.T @ dL_dg2
        rhs[int(self.bnd_mesh.node_indices[0])] = 0.0   # gauge fix
        return -2.0 * self.M.T @ self.A_coupled_lu.solve(rhs, trans='T')


# =============================================================================
# Shared assembly backend
# =============================================================================

def _assemble_operators_from_mesh(mesh, verbose: bool = True,
                                   t0: Optional[float] = None) -> FEMOperators:
    """
    Assemble K, M, S1, S2 and factorise K for any P3 Mesh object.
    Used by both build_operators and build_operators_adaptive.
    """
    if t0 is None:
        t0 = time.perf_counter()

    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    boundary = np.array(mesh.boundary)
    n_nodes  = len(nodes)
    interior = np.ones(n_nodes, dtype=bool)
    interior[boundary] = False

    if verbose:
        print(f"       {n_nodes} nodes, {len(elements)} elements, "
              f"{len(boundary)} boundary DOFs")

    quad_pts, quad_wts = get_gauss_quadrature_triangle(order=5)
    quad_pts_np = np.array(quad_pts)
    quad_wts_np = np.array(quad_wts)

    # -- Precompute reference-space data (one-time, cheap) -------------------
    N_ref, dN_ref = _precompute_reference_data(quad_pts_np, quad_wts_np)
 
    # -- Stiffness K (Neumann — NO Dirichlet row modification) ---------------
    # Pure numpy einsum assembly: no JAX @jit overhead per element.
    # Ke[i,j] = area * sum_q w_q * dot(dN_phys[q,i], dN_phys[q,j])
    # where dN_phys = dN_ref @ J_inv.T
    if verbose: print("[femmi] Assembling stiffness matrix K (Neumann)...")
    t1 = time.perf_counter()
    max_nnz = len(elements) * 100
    I_k = np.zeros(max_nnz, dtype=np.int32)
    J_k = np.zeros(max_nnz, dtype=np.int32)
    K_d = np.zeros(max_nnz, dtype=np.float64)
    entry = 0
    for elem in elements:
        xy      = nodes[elem[:3]]                           # (3, 2) vertices
        Jac     = np.array([[xy[1,0]-xy[0,0], xy[1,1]-xy[0,1]],
                             [xy[2,0]-xy[0,0], xy[2,1]-xy[0,1]]])
        area    = abs(np.linalg.det(Jac)) / 2.0
        J_inv_T = np.linalg.inv(Jac).T                     # (2, 2)
        dN_phys = dN_ref @ J_inv_T                          # (nq, 10, 2)
        Ke      = area * np.einsum('q,qia,qja->ij',
                                   quad_wts_np, dN_phys, dN_phys)  # (10,10)
        I_k[entry:entry+100] = np.repeat(elem, 10)
        J_k[entry:entry+100] = np.tile(elem, 10)
        K_d[entry:entry+100] = Ke.ravel()
        entry += 100
    K = sp.coo_matrix((K_d[:entry], (I_k[:entry], J_k[:entry])),
                      shape=(n_nodes, n_nodes)).tocsr()
    if verbose:
        print(f"       K assembled (Neumann): {K.shape}, nnz={K.nnz}  "
              f"({time.perf_counter()-t1:.1f}s)")

    # -- Mass M (full — NO boundary row zeroing) -----------------------------
    # Pure numpy assembly using the same precomputed N_ref.
    # Me[i,j] = area * sum_q w_q * N_ref[q,i] * N_ref[q,j]
    if verbose: print("[femmi] Assembling mass matrix M (full)...")
    t2 = time.perf_counter()
    I_m = np.zeros(max_nnz, dtype=np.int32)
    J_m = np.zeros(max_nnz, dtype=np.int32)
    M_d = np.zeros(max_nnz, dtype=np.float64)
    entry = 0
    for elem in elements:
        xy   = nodes[elem[:3]]
        Jac  = np.array([[xy[1,0]-xy[0,0], xy[1,1]-xy[0,1]],
                          [xy[2,0]-xy[0,0], xy[2,1]-xy[0,1]]])
        area = abs(np.linalg.det(Jac)) / 2.0
        Me   = area * np.einsum('q,qi,qj->ij', quad_wts_np, N_ref, N_ref)
        I_m[entry:entry+100] = np.repeat(elem, 10)
        J_m[entry:entry+100] = np.tile(elem, 10)
        M_d[entry:entry+100] = Me.ravel()
        entry += 100
    M = sp.coo_matrix((M_d[:entry], (I_m[:entry], J_m[:entry])),
                      shape=(n_nodes, n_nodes)).tocsr()
    if verbose:
        print(f"       M assembled (full): {M.shape}, nnz={M.nnz}  "
              f"({time.perf_counter()-t2:.1f}s)")

    # -- Reference Hessians ---------------------------------------------------
    if verbose: print("[femmi] Precomputing P3 reference Hessians (JAX AD)...")
    t3 = time.perf_counter()
    H_ref = _build_ref_hessians()
    if verbose: print(f"       H_ref built  ({time.perf_counter()-t3:.1f}s)")

    # -- Shear operators ------------------------------------------------------
    if verbose: print("[femmi] Assembling shear operators S1, S2...")
    t4 = time.perf_counter()
    S1, S2 = _assemble_shear_ops(nodes, elements, H_ref)
    if verbose:
        print(f"       S1, S2 assembled: nnz={S1.nnz}, {S2.nnz}  ({time.perf_counter()-t4:.1f}s)")
    
    # Zero out shear at boundary nodes. P3 nodal averaging is unreliable
    # there (1-2 element contributions vs 6 for interior), producing
    # large spurious spikes that dominate the MAP loss.
    S1_lil = S1.tolil(); S1_lil[boundary, :] = 0; S1 = S1_lil.tocsr()
    S2_lil = S2.tolil(); S2_lil[boundary, :] = 0; S2 = S2_lil.tocsr()
 
    # -- BEM matrices ---------------------------------------------------------
    # Assemble V_h, K_h, M_b on ∂Ω; form Calderon operator C.
    # MATH.md §5–6; bem.py §1.3–1.6.
    if verbose: print("[femmi] Assembling BEM matrices (V_h, K_h, M_b)...")
    t_bem = time.perf_counter()
    bnd_mesh = extract_boundary_edges(mesh)
    N_b      = bnd_mesh.n_boundary_dofs
    V_h, K_h, M_b = assemble_bem_matrices(bnd_mesh, n_quad_sl=25, n_quad_dl=8)
    if verbose:
        print(f"       BEM assembled: N_b={N_b}  ({time.perf_counter()-t_bem:.1f}s)")
 
    # C = V_h^{-1} (½M_b + K_h)   (N_b × N_b dense)
    C_dense = np.linalg.solve(V_h, 0.5 * M_b + K_h)
 
    # -- Coupled stiffness A_coupled = K_neumann + P^T C P -------------------
    # P is the restriction to boundary DOFs (in CCW order from bnd_mesh).
    # P^T C P adds C[a,b] to A[bnd[a], bnd[b]] for all a, b in 0..N_b-1.
    # MATH.md §6.2.
    if verbose: print("[femmi] Assembling A_coupled = K + P^T C P...")
    t_ac = time.perf_counter()
    bnd_idx  = bnd_mesh.node_indices          # (N_b,) global DOF indices
    A_lil    = K.tolil()
    A_lil[np.ix_(bnd_idx, bnd_idx)] += C_dense
    # Gauge fix: pin one boundary node to break the constant null space.
    # ψ → 0 at ∞ means ψ ≈ 0 on ∂Ω for localized κ; we enforce this at
    # one node to remove the translation ambiguity.  The shear γ = ∂²ψ is
    # constant-independent so this gauge choice doesn't affect γ.
    idx_gauge = int(bnd_idx[0])
    A_lil[idx_gauge, :] = 0.0
    A_lil[idx_gauge, idx_gauge] = 1.0
    A_coupled = A_lil.tocsr()
    if verbose:
        print(f"       A_coupled assembled: {A_coupled.shape}  "
              f"({time.perf_counter()-t_ac:.1f}s)")
 
    # -- Factorize A_coupled --------------------------------------------------
    if verbose: print("[femmi] Factorizing A_coupled (SuperLU)...")
    t5 = time.perf_counter()
    A_coupled_lu = spla.splu(A_coupled.tocsc())
    if verbose:
        print(f"       A_coupled LU done  ({time.perf_counter()-t5:.1f}s)")
        print(f"[femmi] All operators ready  (total {time.perf_counter()-t0:.1f}s)\n")
 
    return FEMOperators(
        mesh=mesh, K=K, M=M, S1=S1, S2=S2,
        A_coupled=A_coupled,
        A_coupled_lu=A_coupled_lu,
        bnd_mesh=bnd_mesh,
        C_dense=C_dense,
        n_nodes=n_nodes, boundary=boundary, interior=interior,
    )


# =============================================================================
# Public factory functions
# =============================================================================

def build_operators(nx: int, ny: int,
                    xmin: float = -2.5, xmax: float = 2.5,
                    ymin: float = -2.5, ymax: float = 2.5,
                    verbose: bool = True) -> FEMOperators:
    """Build FEM operators on a uniform P3 structured mesh."""
    t0 = time.perf_counter()
    if verbose:
        print(f"[femmi] Building P3 mesh: {nx}x{ny} cells...")
    mesh = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0)


def build_operators_adaptive(nx: int, ny: int,
                              xmin: float = -2.5, xmax: float = 2.5,
                              ymin: float = -2.5, ymax: float = 2.5,
                              mask_center: Tuple[float, float] = (0.0, 0.0),
                              mask_radius: float = 0.5,
                              refine_factor: int = 3,
                              verbose: bool = True) -> FEMOperators:
    """
    Build FEM operators on a locally refined P3 mesh.

    Args:
        nx, ny        : background grid resolution
        xmin..ymax    : domain bounds
        mask_center   : (cx, cy) centre of the circular mask
        mask_radius   : radius of the circular mask
        refine_factor : mesh density multiplier near mask (3 = 3x finer)
        verbose       : print timing info
    """
    t0 = time.perf_counter()
    if verbose:
        print(f"[femmi] Building adaptive P3 mesh: {nx}x{ny} background, "
              f"x{refine_factor} near mask (r={mask_radius:.2f})...")
    mesh = generate_p3_adaptive_mesh(
        nx, ny, xmin, xmax, ymin, ymax,
        mask_center=mask_center,
        mask_radius=mask_radius,
        refine_factor=refine_factor,
        verbose=verbose,
    )
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0)


# =============================================================================
# Regularizer helpers
# =============================================================================

def build_laplacian(ops: FEMOperators) -> sp.csr_matrix:
    """H1 regularizer: kappa^T K kappa = ||grad kappa||^2."""
    return ops.K.copy()


def build_wiener_regularizer(ops: FEMOperators,
                              wiener_length: float) -> sp.csr_matrix:
    """
    Matern-like regularizer  R = M + l^2 * K.

    The MAP cost term  lambda * kappa^T R kappa  corresponds to a Gaussian
    prior whose covariance is the Green's function of (I - l^2 nabla^2) --
    a Matern-1/2 field with correlation length l.

    Setting l = sigma_lens (the lens scale) makes the prior match the
    expected spatial structure of kappa, penalising high-frequency noise
    much more strongly than the plain H1 prior (R = K) while allowing
    smooth structure at scale l to pass through.

    Args:
        ops           : FEMOperators (provides M and K)
        wiener_length : correlation length l  (recommend l = sigma_lens ~ 0.5)

    Returns:
        R : (n_nodes, n_nodes) sparse CSR matrix
    """
    return (ops.M + wiener_length**2 * ops.K).tocsr()