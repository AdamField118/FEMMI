"""
fem.py  --  P3 FEM operators for weak gravitational lensing.

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

New regularizer helper:
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

from .p3_mesh_generator import (
    generate_p3_structured_mesh,
    generate_p3_adaptive_mesh,
)
from .p3_shape_functions import (
    compute_p3_shape_functions,
    compute_p3_shape_gradients_reference,
)
from .p3_assembly import (
    get_gauss_quadrature_triangle,
    compute_element_stiffness_p3,
    compute_element_load_p3,
    apply_boundary_conditions_p3,
)
from .fem_solver import Mesh


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


# =============================================================================
# FEMOperators
# =============================================================================

@dataclass
class FEMOperators:
    """
    All precomputed FEM operators for a fixed mesh.

    Attributes
    ----------
    mesh      : P3 Mesh (nodes, elements, boundary)
    K         : stiffness matrix (Dirichlet BCs applied)
    M         : mass matrix (boundary rows zeroed)
    S1, S2    : shear operators
    K_lu      : SuperLU factorization of K
    n_nodes   : total node count
    boundary  : boundary node indices
    interior  : bool mask (True = interior node)
    """
    mesh     : object
    K        : sp.csr_matrix
    M        : sp.csr_matrix
    S1       : sp.csr_matrix
    S2       : sp.csr_matrix
    K_lu     : object
    n_nodes  : int
    boundary : np.ndarray
    interior : np.ndarray

    def psi_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        """Solve K psi = -2 M kappa."""
        rhs = -2.0 * self.M @ kappa
        rhs[self.boundary] = 0.0
        return self.K_lu.solve(rhs)

    def shear_from_psi(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.S1 @ psi, self.S2 @ psi

    def forward(self, kappa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.shear_from_psi(self.psi_from_kappa(kappa))

    def shear_magnitude(self, kappa: np.ndarray) -> np.ndarray:
        g1, g2 = self.forward(kappa)
        return np.sqrt(g1**2 + g2**2)

    def adjoint_rhs(self, dL_dg1: np.ndarray,
                    dL_dg2: np.ndarray) -> np.ndarray:
        """dL/dkappa = -2 M^T K^{-1} (S1^T dL/dg1 + S2^T dL/dg2)."""
        rhs = self.S1.T @ dL_dg1 + self.S2.T @ dL_dg2
        rhs[self.boundary] = 0.0
        return -2.0 * self.M.T @ self.K_lu.solve(rhs)


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

    # -- Stiffness K ----------------------------------------------------------
    if verbose: print("[fem] Assembling stiffness matrix K...")
    t1 = time.perf_counter()
    max_nnz = len(elements) * 100
    I_k = np.zeros(max_nnz, dtype=np.int32)
    J_k = np.zeros(max_nnz, dtype=np.int32)
    K_d = np.zeros(max_nnz)
    entry = 0
    for elem in elements:
        Ke = np.array(compute_element_stiffness_p3(
            jnp.array(nodes[elem]),
            jnp.array(quad_pts_np),
            jnp.array(quad_wts_np)))
        for i in range(10):
            for j in range(10):
                I_k[entry]=elem[i]; J_k[entry]=elem[j]; K_d[entry]=Ke[i,j]; entry+=1
    K_raw = sp.coo_matrix((K_d[:entry],(I_k[:entry],J_k[:entry])),
                          shape=(n_nodes,n_nodes)).tocsr()
    K_lil = K_raw.tolil()
    for b in boundary:
        K_lil[b,:] = 0; K_lil[b,b] = 1.0
    K = K_lil.tocsr()
    if verbose:
        print(f"       K assembled: {K.shape}, nnz={K.nnz}  ({time.perf_counter()-t1:.1f}s)")

    # -- Mass M ---------------------------------------------------------------
    if verbose: print("[fem] Assembling mass matrix M...")
    t2 = time.perf_counter()
    M_raw = _assemble_mass_p3(nodes, elements, quad_pts_np, quad_wts_np)
    M_lil = M_raw.tolil()
    for b in boundary:
        M_lil[b,:] = 0
    M = M_lil.tocsr()
    if verbose:
        print(f"       M assembled: {M.shape}, nnz={M.nnz}  ({time.perf_counter()-t2:.1f}s)")

    # -- Reference Hessians ---------------------------------------------------
    if verbose: print("[fem] Precomputing P3 reference Hessians (JAX AD)...")
    t3 = time.perf_counter()
    H_ref = _build_ref_hessians()
    if verbose: print(f"       H_ref built  ({time.perf_counter()-t3:.1f}s)")

    # -- Shear operators ------------------------------------------------------
    if verbose: print("[fem] Assembling shear operators S1, S2...")
    t4 = time.perf_counter()
    S1, S2 = _assemble_shear_ops(nodes, elements, H_ref)
    if verbose:
        print(f"       S1, S2 assembled: nnz={S1.nnz}, {S2.nnz}  ({time.perf_counter()-t4:.1f}s)")

    # -- SuperLU factorise ----------------------------------------------------
    if verbose: print("[fem] Factorizing K (SuperLU)...")
    t5 = time.perf_counter()
    K_lu = spla.splu(K.tocsc())
    if verbose:
        print(f"       LU factorization done  ({time.perf_counter()-t5:.1f}s)")
        print(f"[fem] All operators ready  (total {time.perf_counter()-t0:.1f}s)\n")

    return FEMOperators(
        mesh=mesh, K=K, M=M, S1=S1, S2=S2, K_lu=K_lu,
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
        print(f"[fem] Building P3 mesh: {nx}x{ny} cells...")
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

    Identical interface to build_operators; the returned FEMOperators is
    fully compatible with MAPReconstructor.

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
        print(f"[fem] Building adaptive P3 mesh: {nx}x{ny} background, "
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

    The gradient term in the adjoint becomes:
        d/dkappa [ lambda * kappa^T R kappa ] = 2 * lambda * R * kappa

    Args:
        ops           : FEMOperators (provides M and K)
        wiener_length : correlation length l  (recommend l = sigma_lens ~ 0.5)

    Returns:
        R : (n_nodes, n_nodes) sparse CSR matrix
    """
    return (ops.M + wiener_length**2 * ops.K).tocsr()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("fem.py -- smoke test")

    ops = build_operators(6, 6, verbose=True)
    nodes = np.array(ops.mesh.nodes)
    kappa = np.exp(-(nodes[:,0]**2 + nodes[:,1]**2) / (2*0.5**2))
    g1, g2 = ops.forward(kappa)
    print(f"max|g1|={np.abs(g1).max():.4f}  max|g2|={np.abs(g2).max():.4f}")

    print("\n--- Adaptive 6x6 (mask r=0.5) ---")
    ops2 = build_operators_adaptive(6, 6, mask_center=(0.,0.),
                                    mask_radius=0.5, refine_factor=3)
    nodes2 = np.array(ops2.mesh.nodes)
    kappa2 = np.exp(-(nodes2[:,0]**2 + nodes2[:,1]**2) / (2*0.5**2))
    g1b, g2b = ops2.forward(kappa2)
    print(f"max|g1|={np.abs(g1b).max():.4f}  max|g2|={np.abs(g2b).max():.4f}")
    print("\nfem.py OK")