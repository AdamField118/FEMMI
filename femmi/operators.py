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
from functools import lru_cache
from typing import Tuple, Optional
import time

from .mesh import generate_p3_structured_mesh, generate_p3_adaptive_mesh
from .basis import compute_p3_shape_functions, compute_p3_shape_gradients_reference
from .assembly import get_gauss_quadrature_triangle
from .types import Mesh
from .bem import (extract_boundary_edges,
                  extract_boundary_edges_circular,
                  assemble_bem_matrices)


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


@lru_cache(maxsize=None)
def _build_ref_hessians():
    """Precompute H_ref[eval_node, shape_fn, i, j] shape (10,10,2,2) via JAX AD.

    Cached: this depends only on the P3 reference element, so it is the same
    (10,10,2,2) array on every mesh, but the JAX jacfwd/jacrev tracing costs
    ~1.4s each time. Rebuilding it per operator build made every convergence
    sweep pay it once per resolution for nothing."""
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])
    hess_fn = jax.jacfwd(jax.jacrev(N_vec))
    return np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])


@lru_cache(maxsize=None)
def _reference_data(order):
    """(quad_pts, quad_wts, N_ref, dN_ref) for a quadrature order -- also purely
    a property of the reference element, so cached for the same reason."""
    quad_pts, quad_wts = get_gauss_quadrature_triangle(order=order)
    quad_pts_np = np.array(quad_pts)
    quad_wts_np = np.array(quad_wts)
    N_ref, dN_ref = _precompute_reference_data(quad_pts_np, quad_wts_np)
    return quad_pts_np, quad_wts_np, N_ref, dN_ref


def _element_jacobians(nodes, elements):
    """Batched (area, A) for every element, where A = inv(Jac).T maps reference
    gradients to physical ones. Jac rows are the two edge vectors from vertex 0,
    matching the per-element convention this module has always used."""
    xy = nodes[elements[:, :3]]                        # (ne, 3, 2)
    Jac = np.stack([xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]], axis=1)   # (ne,2,2)
    det = Jac[:, 0, 0] * Jac[:, 1, 1] - Jac[:, 0, 1] * Jac[:, 1, 0]
    area = np.abs(det) / 2.0
    A = np.linalg.inv(Jac).transpose(0, 2, 1)          # (ne,2,2) == inv(Jac).T
    return area, A


def _coo_from_element_blocks(vals, elements, n_nodes):
    """Scatter per-element (ne, 10, 10) blocks into a sparse matrix."""
    ne = len(elements)
    I = np.repeat(elements, 10, axis=1).ravel()        # row index varies slowest
    J = np.tile(elements, (1, 10)).ravel()
    return sp.coo_matrix((vals.ravel(), (I, J)), shape=(n_nodes, n_nodes)).tocsr()


def _assemble_shear_ops(nodes, elements, H_ref):
    """Build sparse shear operators S1 and S2 via nodal-averaged element Hessians.

    Vectorised over elements: the old form ran a 10x10 Python loop per element,
    which dominated mesh setup. H_phys[e,n,j] is the physical Hessian of shape
    function j at evaluation node n of element e."""
    n_nodes = len(nodes)
    _, A = _element_jacobians(nodes, elements)

    # H_phys[e,n,j,a,b] = sum_kl A[e,k,a] A[e,l,b] H_ref[n,j,k,l]
    H_phys = np.einsum('eka,elb,njkl->enjab', A, A, H_ref, optimize=True)
    D1 = 0.5 * (H_phys[..., 0, 0] - H_phys[..., 1, 1])      # (ne, 10, 10)
    D2 = H_phys[..., 0, 1]

    S1r = _coo_from_element_blocks(D1, elements, n_nodes)
    S2r = _coo_from_element_blocks(D2, elements, n_nodes)
    counts = np.bincount(elements.ravel(), minlength=n_nodes)
    sc = sp.diags(1.0 / np.maximum(counts, 1))
    return (sc @ S1r).tocsr(), (sc @ S2r).tocsr()


def _assemble_recovery_blocks(nodes, elements, quad_wts_np, dN_ref):
    """Assemble Bx, By, Bxy with Bx[i,j] = int N_i,x N_j,x, By[i,j] = int N_i,y N_j,y,
    Bxy[i,j] = int N_i,x N_j,y -- the pieces of the variational (weak-form) shear
    recovery in build_recovered_shear_ops."""
    n_nodes = len(nodes)
    area, A = _element_jacobians(nodes, elements)
    # physical shape-function gradients at every quadrature point of every element
    dN = np.einsum('qlb,eba->eqla', dN_ref, A, optimize=True)      # (ne, nq, 10, 2)
    w = quad_wts_np
    blk = lambda a, b: (area[:, None, None]
                        * np.einsum('q,eqi,eqj->eij', w, dN[..., a], dN[..., b],
                                    optimize=True))
    mk = lambda v: _coo_from_element_blocks(v, elements, n_nodes)
    return mk(blk(0, 0)), mk(blk(1, 1)), mk(blk(0, 1))


class RecoveredShear:
    """Variationally recovered shear gamma = (1/2(psi_xx - psi_yy), psi_xy).

    The default S1/S2 operators sample the element Hessian AT THE P3 NODES and
    average over the elements meeting there. Nodes are exactly where a C^0 P3
    element's second derivative jumps, so that estimate loses an order: it is
    only ~O(h^1.3) in practice, well short of the O(h^2) the approximation theory
    allows for P3.

    This recovers the shear variationally instead. Integrating by parts once,

        int N_i gamma1 = 1/2 [ -int N_i,x psi_,x + int N_i,y psi_,y ]  + boundary
        int N_i gamma2 =     -int N_i,y psi_,x                          + boundary

    so only FIRST derivatives of psi_h are ever evaluated (those converge at
    O(h^3) for P3, and are continuous across elements), and the result is the L2
    projection of the recovered field onto the continuous P3 space:

        M gamma1_rec = 1/2 (By - Bx) psi,     M gamma2_rec = -1/2 (Bxy + Bxy^T) psi.

    The boundary term int_dOmega N_i psi_,x n ds is dropped, so recovered values
    on the boundary ring are not meaningful -- use `interior_mask` (or a compactly
    supported field, as in experiments.shear_convergence) when measuring error.
    """

    def __init__(self, ops):
        from scipy.sparse.linalg import splu
        mesh = ops.mesh
        nodes = np.asarray(mesh.nodes, np.float64)
        elements = np.asarray(mesh.elements)
        _, quad_wts_np, _, dN_ref = _reference_data(5)
        Bx, By, Bxy = _assemble_recovery_blocks(nodes, elements,
                                                quad_wts_np, dN_ref)
        self.ops = ops
        self.A1 = (0.5 * (By - Bx)).tocsc()
        self.A2 = (-0.5 * (Bxy + Bxy.T)).tocsc()
        self._M_lu = splu(ops.M.tocsc())
        self.interior_mask = np.asarray(ops.interior, bool)

    def __call__(self, psi):
        psi = np.asarray(psi, np.float64)
        return (self._M_lu.solve(self.A1 @ psi),
                self._M_lu.solve(self.A2 @ psi))


def build_recovered_shear_ops(ops):
    """Convenience wrapper: returns a RecoveredShear callable psi -> (g1, g2)."""
    return RecoveredShear(ops)


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
    # When set (Dirichlet BC), the RHS is zeroed on ALL these nodes so the solve
    # enforces psi=0 on the boundary. Default None -> BEM: zero only the single
    # gauge node (the constant null space is otherwise removed by the coupling).
    dirichlet_nodes : Optional[np.ndarray] = None

    def _rhs_zero_nodes(self):
        if self.dirichlet_nodes is not None:
            return self.dirichlet_nodes
        return np.array([int(self.bnd_mesh.node_indices[0])])

    def _solve_psi(self, rhs: np.ndarray) -> np.ndarray:
        """Forward solve: zero gauge/Dirichlet nodes in RHS, then apply A_lu."""
        rhs = rhs.copy()
        rhs[self._rhs_zero_nodes()] = 0.0
        return self.A_coupled_lu.solve(rhs)

    def _solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Adjoint solve: zero gauge/Dirichlet nodes in RHS, then apply A_lu^T."""
        rhs = rhs.copy()
        rhs[self._rhs_zero_nodes()] = 0.0
        return self.A_coupled_lu.solve(rhs, trans='T')

    def psi_from_kappa(self, kappa):
        return self._solve_psi(-2.0 * self.M @ kappa)

    def shear_from_psi(self, psi):
        return self.S1 @ psi, self.S2 @ psi

    def forward(self, kappa):
        return self.shear_from_psi(self.psi_from_kappa(kappa))

    def shear_magnitude(self, kappa):
        g1, g2 = self.forward(kappa)
        return np.sqrt(g1**2 + g2**2)

    def adjoint_rhs(self, dL_dg1, dL_dg2):
        rhs = self.S1.T @ dL_dg1 + self.S2.T @ dL_dg2
        return -2.0 * self.M.T @ self._solve_adjoint(rhs)


def _assemble_operators_from_mesh(mesh, verbose=True, t0=None, boundary_extractor=None,
                                  coupling='steinbach', sigma_scale=1.0):
    """Assemble K, M, S1, S2, BEM matrices, and A_coupled for any P3 mesh.

    coupling : selects the FEM-BEM boundary coupling C added to the stiffness
        boundary block (A_coupled = K + P^T C P):

        'steinbach' (default) : C = -M_b V_sigma^{-1}(0.5 M_b - K_h), the
            symmetric Steinbach coupling. It uses the physically correct
            exterior sign (0.5 M_b - K_h), the Galerkin M_b pairing, and a
            sigma-scaled single layer
            V_sigma = V_h - (ln sigma / 2pi) w w^T,  w = M_b 1,  sigma = diam(Gamma),
            which repairs the n=0 logarithmic-capacity mode (Steinbach 2008).
            The result is scale- and translation-invariant and more accurate
            than a Dirichlet truncation when the boundary is near the mass.

        'deflation' : EXPERIMENTAL, undocumented. Same coupling but the n=0 mode
            is removed by explicit constant-mode deflation of V_h instead of
            sigma-scaling. Provided for numerical experiments only.

    sigma_scale : multiplies the characteristic length sigma = diam(Gamma) used by
        the 'steinbach' coupling (sigma -> sigma_scale * diam). Default 1.0.
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
        print(f"  {n_nodes} nodes, {len(elements)} elements, {len(boundary)} boundary DOFs")

    quad_pts_np, quad_wts_np, N_ref, dN_ref = _reference_data(5)

    # Neumann stiffness - no Dirichlet row modifications
    if verbose:
        print("  assembling K (Neumann)...")
    t1 = time.perf_counter()
    area, A = _element_jacobians(nodes, elements)
    dN_phys = np.einsum('qlb,eba->eqla', dN_ref, A, optimize=True)   # (ne,nq,10,2)
    Ke = area[:, None, None] * np.einsum('q,eqia,eqja->eij', quad_wts_np,
                                         dN_phys, dN_phys, optimize=True)
    K = _coo_from_element_blocks(Ke, elements, n_nodes)
    if verbose:
        print(f"  K: {K.shape}, nnz={K.nnz}  ({time.perf_counter()-t1:.1f}s)")

    # Full mass matrix - no boundary row zeroing
    if verbose:
        print("  assembling M...")
    t2 = time.perf_counter()
    # N_ref does not depend on the element, so the reference block is assembled
    # once and every element block is just area * M_ref.
    M_ref = np.einsum('q,qi,qj->ij', quad_wts_np, N_ref, N_ref, optimize=True)
    Me = area[:, None, None] * M_ref[None, :, :]
    M = _coo_from_element_blocks(Me, elements, n_nodes)
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
    if boundary_extractor is None:
        boundary_extractor = extract_boundary_edges
    bnd_mesh = boundary_extractor(mesh)
    N_b      = bnd_mesh.n_boundary_dofs
    V_h, K_h, M_b = assemble_bem_matrices(bnd_mesh, n_quad_sl=25, n_quad_dl=8)
    if verbose:
        print(f"  BEM done: N_b={N_b}  ({time.perf_counter()-t_bem:.1f}s)")

    if coupling in ('steinbach', 'deflation'):
        # Symmetric Steinbach coupling: physically correct exterior sign
        # (0.5 M_b - K_h), Galerkin M_b pairing, n=0 log-capacity mode repaired.
        w  = M_b @ np.ones(N_b)
        Xm = 0.5 * M_b - K_h
        if coupling == 'steinbach':
            from scipy.spatial.distance import pdist
            sigma = float(sigma_scale) * float(pdist(bnd_mesh.nodes).max())  # diam(Gamma)
            V_eff = V_h - (np.log(sigma) / (2.0 * np.pi)) * np.outer(w, w)
            C_dense = -(M_b @ np.linalg.solve(V_eff, Xm))
            if verbose:
                print(f"  Steinbach coupling: sigma=diam={sigma:.3g}")
        else:  # 'deflation' (experimental)
            P0 = np.outer(w, w) / float(w @ w)
            I  = np.eye(N_b)
            V_eff = (I - P0) @ V_h @ (I - P0) + P0
            C_dense = -(M_b @ ((I - P0) @ np.linalg.solve(V_eff, (I - P0) @ Xm)))
    else:
        raise ValueError(f"unknown coupling={coupling!r}; "
                         "use 'steinbach' (default) or 'deflation'")

    if verbose:
        print("  assembling A_coupled...")
    t_ac     = time.perf_counter()
    bnd_idx  = bnd_mesh.node_indices
    A_lil    = K.tolil()
    A_lil[np.ix_(bnd_idx, bnd_idx)] += C_dense
    # Gauge fix: pin one boundary node to remove the constant null space.
    # The gauge node is placed at 3pi/4 (upper-left diagonal) where gamma1=0
    # for a centred lens, minimising visible artifact in shear plots.
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

    # Zero the gauge-node column of S1, S2.
    # The gauge fix forces psi[gauge]=0 while the true value is nonzero;
    # adjacent nodes' shear would be corrupted via those columns without this.
    # Row-zeroing (done above for all boundary nodes) is not sufficient —
    # the leak is through the columns of interior nodes adjacent to the gauge.
    S1_lil = S1.tolil(); S1_lil[:, idx_gauge] = 0; S1 = S1_lil.tocsr()
    S2_lil = S2.tolil(); S2_lil[:, idx_gauge] = 0; S2 = S2_lil.tocsr()

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


def build_operators(nx=20, ny=20, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
                    verbose=True, geometry='square',
                    radius=2.5, n_boundary=252,
                    coupling='steinbach', sigma_scale=1.0):
    """Build FEM operators on a square or circular P3 mesh.

    Parameters
    ----------
    geometry : 'square' (default) or 'circular'
        Domain shape.  'square' uses nx, ny, xmin, xmax, ymin, ymax.
        'circular' uses radius and n_boundary; nx/ny are ignored.
    radius : float
        Radius of the circular domain (only used when geometry='circular').
    n_boundary : int
        Number of boundary nodes for the circular domain.
    coupling : 'steinbach' (default); see _assemble_operators_from_mesh. The
        symmetric, scale- and translation-invariant BEM coupling.
    sigma_scale : length scale multiplier for the 'steinbach' coupling (default 1.0).
    """
    if geometry == 'circular':
        return build_operators_circular(
            radius=radius, n_boundary=n_boundary, verbose=verbose,
            coupling=coupling, sigma_scale=sigma_scale)
    t0 = time.perf_counter()
    if verbose:
        print(f"Building P3 mesh: {nx}x{ny} cells...")
    mesh = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0,
                                         coupling=coupling, sigma_scale=sigma_scale)


def dirichlet_from_operators(ops, verbose=False):
    """
    Build a Dirichlet-BC counterpart of an existing FEMOperators, reusing its
    assembled K, M, S1, S2.

    Replaces the BEM-coupled A with the truncated-domain operator that imposes
    psi = 0 on the whole boundary (the standard, and per this project's thesis
    *incorrect*, finite-domain treatment). The returned operator plugs into
    MAPReconstructor exactly like the BEM one, so a reconstruction can be run
    with only the boundary condition changed -- everything else held fixed.
    """
    bnd  = np.asarray(ops.boundary)
    A    = ops.K.tolil()
    A[bnd, :] = 0.0
    A[bnd, bnd] = 1.0                      # psi_b = 0 rows
    A_csr = A.tocsr()
    if verbose:
        print(f"  Dirichlet A: {A_csr.shape}, pinning {len(bnd)} boundary nodes")
    A_lu = spla.splu(A_csr.tocsc())

    return FEMOperators(
        mesh=ops.mesh, K=ops.K, M=ops.M, S1=ops.S1, S2=ops.S2,
        A_coupled=A_csr, A_coupled_lu=A_lu, bnd_mesh=ops.bnd_mesh,
        C_dense=ops.C_dense, n_nodes=ops.n_nodes, boundary=ops.boundary,
        interior=ops.interior, dirichlet_nodes=bnd,
    )


def build_operators_dirichlet(nx=20, ny=20, xmin=-2.5, xmax=2.5,
                              ymin=-2.5, ymax=2.5, verbose=True):
    """FEM operators on a square P3 mesh with psi=0 Dirichlet boundary."""
    ops = build_operators(nx, ny, xmin, xmax, ymin, ymax, verbose=verbose)
    return dirichlet_from_operators(ops, verbose=verbose)


def build_operators_adaptive(nx, ny, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
                              mask_center=(0.0, 0.0), mask_radius=0.5,
                              refine_factor=3, verbose=True,
                              coupling='steinbach', sigma_scale=1.0):
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
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0,
                                         coupling=coupling, sigma_scale=sigma_scale)


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

def build_operators_circular(radius=2.5, n_boundary=60, n_rings=None, center=(0.0, 0.0),
                             verbose=True,
                             coupling='steinbach', sigma_scale=1.0):
    from .mesh import generate_p3_circular_mesh
    from .bem  import extract_boundary_edges_circular
    t0 = time.perf_counter()
    if verbose:
        print(f"Building circular P3 mesh: R={radius:.2f}  n_boundary={n_boundary}...")
    mesh = generate_p3_circular_mesh(radius=radius, n_boundary=n_boundary, n_rings=n_rings, center=center, verbose=verbose)
    return _assemble_operators_from_mesh(mesh, verbose=verbose, t0=t0,
                                         boundary_extractor=lambda m: extract_boundary_edges_circular(m, center=center, radius=radius),
                                         coupling=coupling, sigma_scale=sigma_scale)

def build_operators_catalog(x, y, center=None, radius=None, pad=0.15,
                            n_boundary=96, dedup_radius=None, guard_ring=True,
                            verbose=True, coupling='steinbach', sigma_scale=1.0):
    """
    Build FEM-BEM operators on a catalog-native P3 mesh (nodes at galaxy
    positions, clean circular far-field boundary).

    Returns (ops, catalog_mesh): FEMOperators plus the CatalogMesh carrying
    galaxy_nodes / source_index so observed shear can be placed on the right
    nodes. For a FlatCatalog `flat`, call with (flat.x, flat.y, center=(0., 0.)).
    """
    from .mesh import generate_p3_catalog_mesh

    t0 = time.perf_counter()
    cm = generate_p3_catalog_mesh(
        x, y, center=center, radius=radius, pad=pad, n_boundary=n_boundary,
        dedup_radius=dedup_radius, guard_ring=guard_ring, verbose=verbose)
    ctr, rad = cm.center, cm.radius
    ops = _assemble_operators_from_mesh(
        cm.mesh, verbose=verbose, t0=t0,
        boundary_extractor=lambda m: extract_boundary_edges_circular(
            m, center=ctr, radius=rad),
        coupling=coupling, sigma_scale=sigma_scale)
    return ops, cm
