"""
femmi/operators.py
P3 FEM operators for weak gravitational lensing.

Assembles K (Neumann stiffness), M (mass), S1/S2 (shear), and the
FEM-BEM coupled stiffness A_coupled = K + P^T C P.
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

from .mesh import generate_p3_structured_mesh, generate_p3_adaptive_mesh
from .basis import compute_p3_shape_functions, compute_p3_shape_gradients_reference
from .assembly import get_gauss_quadrature_triangle
from .types import Mesh
from .bem import extract_boundary_edges, assemble_bem_matrices


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


def _build_ref_hessians():
    """Precompute H_ref[eval_node, shape_fn, i, j] shape (10,10,2,2) via JAX AD."""
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])
    hess_fn = jax.jacfwd(jax.jacrev(N_vec))
    return np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])


def _assemble_shear_ops(nodes, elements, H_ref):
    """Build sparse shear operators S1 and S2 via nodal-averaged element Hessians."""
    n_nodes = len(nodes)
    max_nnz = len(elements) * 100
    I1 = np.zeros(max_nnz, dtype=np.int32); J1 = np.zeros(max_nnz, dtype=np.int32); D1 = np.zeros(max_nnz)
    I2 = np.zeros(max_nnz, dtype=np.int32); J2 = np.zeros(max_nnz, dtype=np.int32); D2 = np.zeros(max_nnz)
    idx    = 0
    counts = np.zeros(n_nodes, dtype=np.int32)
    for elem in elements:
        x0,y0=nodes[elem[0]]; x1,y1=nodes[elem[1]]; x2,y2=nodes[elem[2]]
        Jac = np.array([[x1-x0, y1-y0], [x2-x0, y2-y0]])
        A   = np.linalg.inv(Jac).T
        for li in range(10):
            H_phys = np.einsum('ja,kb,njk->nab', A, A, H_ref[li])
            row    = elem[li]
            for lj in range(10):
                col       = elem[lj]
                I1[idx]   = row; J1[idx] = col; D1[idx] = 0.5*(H_phys[lj,0,0]-H_phys[lj,1,1])
                I2[idx]   = row; J2[idx] = col; D2[idx] = H_phys[lj,0,1]
                idx += 1
            counts[row] += 1
    S1r = sp.coo_matrix((D1[:idx], (I1[:idx], J1[:idx])), shape=(n_nodes, n_nodes)).tocsr()
    S2r = sp.coo_matrix((D2[:idx], (I2[:idx], J2[:idx])), shape=(n_nodes, n_nodes)).tocsr()
    sc  = sp.diags(1.0 / np.maximum(counts, 1))
    return (sc @ S1r).tocsr(), (sc @ S2r).tocsr()


def _precompute_reference_data(quad_pts_np, quad_wts_np):
    """Precompute N_ref (nq, 10) and dN_ref (nq, 10, 2) at all quadrature points."""
    nq     = len(quad_wts_np)
    N_ref  = np.zeros((nq, 10), dtype=np.float64)
    dN_ref = np.zeros((nq, 10, 2), dtype=np.float64)
    for q, (xi, eta) in enumerate(quad_pts_np):
        N_ref[q]  = np.array(compute_p3_shape_functions(xi, eta))
        dN_ref[q] = np.array(compute_p3_shape_gradients_reference(xi, eta))
    return N_ref, dN_ref


@dataclass
class FEMOperators:
    """
    All precomputed FEM-BEM operators for a fixed mesh.

    A_coupled = K_neumann + P^T C P where C = V_h^{-1}(0.5*M_b + K_h).
    K has no Dirichlet row modifications; its constant null space is
    removed by the BEM coupling and gauge fix.
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

    def psi_from_kappa(self, kappa):
        rhs = -2.0 * self.M @ kappa
        rhs[int(self.bnd_mesh.node_indices[0])] = 0.0
        return self.A_coupled_lu.solve(rhs)

    def shear_from_psi(self, psi):
        return self.S1 @ psi, self.S2 @ psi

    def forward(self, kappa):
        return self.shear_from_psi(self.psi_from_kappa(kappa))

    def shear_magnitude(self, kappa):
        g1, g2 = self.forward(kappa)
        return np.sqrt(g1**2 + g2**2)

    def adjoint_rhs(self, dL_dg1, dL_dg2):
        rhs = self.S1.T @ dL_dg1 + self.S2.T @ dL_dg2
        rhs[int(self.bnd_mesh.node_indices[0])] = 0.0
        return -2.0 * self.M.T @ self.A_coupled_lu.solve(rhs, trans='T')


def _assemble_operators_from_mesh(mesh, verbose=True, t0=None):
    """Assemble K, M, S1, S2, BEM matrices, and A_coupled for any P3 mesh."""
    if t0 is None:
        t0 = time.perf_counter()

    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    boundary = np.array(mesh.boundary)
    n_nodes  = len(nodes)
    interior = np.ones(n_nodes, dtype=bool)
    interior[boundary] = False

    if verbose:
        print(f"  {n_nodes} nodes, {len(elements)} elements, {len(boundary)} boundary DOFs")

    quad_pts, quad_wts = get_gauss_quadrature_triangle(order=5)
    quad_pts_np = np.array(quad_pts)
    quad_wts_np = np.array(quad_wts)
    N_ref, dN_ref = _precompute_reference_data(quad_pts_np, quad_wts_np)

    # Neumann stiffness - no Dirichlet row modifications
    if verbose:
        print("  assembling K (Neumann)...")
    t1      = time.perf_counter()
    max_nnz = len(elements) * 100
    I_k = np.zeros(max_nnz, dtype=np.int32)
    J_k = np.zeros(max_nnz, dtype=np.int32)
    K_d = np.zeros(max_nnz, dtype=np.float64)
    entry = 0
    for elem in elements:
        xy      = nodes[elem[:3]]
        Jac     = np.array([[xy[1,0]-xy[0,0], xy[1,1]-xy[0,1]],
                             [xy[2,0]-xy[0,0], xy[2,1]-xy[0,1]]])
        area    = abs(np.linalg.det(Jac)) / 2.0
        J_inv_T = np.linalg.inv(Jac).T
        dN_phys = dN_ref @ J_inv_T
        Ke      = area * np.einsum('q,qia,qja->ij', quad_wts_np, dN_phys, dN_phys)
        I_k[entry:entry+100] = np.repeat(elem, 10)
        J_k[entry:entry+100] = np.tile(elem, 10)
        K_d[entry:entry+100] = Ke.ravel()
        entry += 100
    K = sp.coo_matrix((K_d[:entry], (I_k[:entry], J_k[:entry])),
                      shape=(n_nodes, n_nodes)).tocsr()
    if verbose:
        print(f"  K: {K.shape}, nnz={K.nnz}  ({time.perf_counter()-t1:.1f}s)")

    # Full mass matrix - no boundary row zeroing
    if verbose:
        print("  assembling M...")
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
        print(f"  M: {M.shape}, nnz={M.nnz}  ({time.perf_counter()-t2:.1f}s)")

    if verbose:
        print("  computing reference Hessians...")
    t3    = time.perf_counter()
    H_ref = _build_ref_hessians()
    if verbose:
        print(f"  H_ref done  ({time.perf_counter()-t3:.1f}s)")

    if verbose:
        print("  assembling S1, S2...")
    t4      = time.perf_counter()
    S1, S2  = _assemble_shear_ops(nodes, elements, H_ref)
    # Zero shear at boundary nodes - P3 nodal averaging is unreliable there
    S1_lil  = S1.tolil(); S1_lil[boundary, :] = 0; S1 = S1_lil.tocsr()
    S2_lil  = S2.tolil(); S2_lil[boundary, :] = 0; S2 = S2_lil.tocsr()
    if verbose:
        print(f"  S1, S2 done  ({time.perf_counter()-t4:.1f}s)")

    if verbose:
        print("  assembling BEM matrices...")
    t_bem    = time.perf_counter()
    bnd_mesh = extract_boundary_edges(mesh)
    N_b      = bnd_mesh.n_boundary_dofs
    V_h, K_h, M_b = assemble_bem_matrices(bnd_mesh, n_quad_sl=25, n_quad_dl=8)
    if verbose:
        print(f"  BEM done: N_b={N_b}  ({time.perf_counter()-t_bem:.1f}s)")

    C_dense = np.linalg.solve(V_h, 0.5 * M_b + K_h)

    if verbose:
        print("  assembling A_coupled...")
    t_ac     = time.perf_counter()
    bnd_idx  = bnd_mesh.node_indices
    A_lil    = K.tolil()
    A_lil[np.ix_(bnd_idx, bnd_idx)] += C_dense
    # Gauge fix: pin one boundary node to remove the constant null space
    idx_gauge = int(bnd_idx[0])
    A_lil[idx_gauge, :] = 0.0
    A_lil[idx_gauge, idx_gauge] = 1.0
    A_coupled = A_lil.tocsr()
    if verbose:
        print(f"  A_coupled: {A_coupled.shape}  ({time.perf_counter()-t_ac:.1f}s)")

    if verbose:
        print("  factorizing A_coupled (SuperLU)...")
    t5           = time.perf_counter()
    A_coupled_lu = spla.splu(A_coupled.tocsc())
    if verbose:
        print(f"  LU done  ({time.perf_counter()-t5:.1f}s)")
        print(f"  total: {time.perf_counter()-t0:.1f}s\n")

    return FEMOperators(
        mesh=mesh, K=K, M=M, S1=S1, S2=S2,
        A_coupled=A_coupled,
        A_coupled_lu=A_coupled_lu,
        bnd_mesh=bnd_mesh,
        C_dense=C_dense,
        n_nodes=n_nodes,
        boundary=boundary,
        interior=interior,
    )


def build_operators(nx, ny, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5, verbose=True):
    """Build FEM operators on a uniform P3 structured mesh."""
    t0 = time.perf_counter()
    if verbose:
        print(f"Building P3 mesh: {nx}x{ny} cells...")
    mesh = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0)


def build_operators_adaptive(nx, ny, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
                              mask_center=(0.0, 0.0), mask_radius=0.5,
                              refine_factor=3, verbose=True):
    """Build FEM operators on a locally refined P3 mesh near a circular mask."""
    t0 = time.perf_counter()
    if verbose:
        print(f"Building adaptive P3 mesh: {nx}x{ny}, x{refine_factor} near mask...")
    mesh = generate_p3_adaptive_mesh(
        nx, ny, xmin, xmax, ymin, ymax,
        mask_center=mask_center,
        mask_radius=mask_radius,
        refine_factor=refine_factor,
        verbose=verbose,
    )
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0)


def build_laplacian(ops):
    """Return K as the H1 regularizer (kappa^T K kappa = ||grad kappa||^2)."""
    return ops.K.copy()


def build_wiener_regularizer(ops, wiener_length):
    """
    Return R = M + l^2 * K (Matern-like prior).

    The MAP cost lambda * kappa^T R kappa corresponds to a Gaussian prior
    whose covariance is the Green's function of (I - l^2 nabla^2).
    Setting l = sigma_lens matches the prior to the expected lens scale.
    """
    return (ops.M + wiener_length**2 * ops.K).tocsr()