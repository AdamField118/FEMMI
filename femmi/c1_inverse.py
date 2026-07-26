"""
femmi/c1_inverse.py
MAP reconstruction on the C^1 spaces -- the inverse half of the Argyris path.

`c1_coupling` gives the forward map kappa -> psi with the exact exterior
condition. This closes the loop: shear observations -> kappa, by minimising

    L(kappa) = || W (S psi - gamma_obs) ||^2  +  lam * kappa^T R kappa,
    psi      = A^{-1} (-2 M kappa),

with A the coupled FEM-BEM operator, M the C^1 mass matrix, and S the shear
observation operator.

WHY S IS TRIVIAL HERE
---------------------
On P3, extracting shear needs `_assemble_shear_ops`: element Hessians sampled at
the nodes, averaged over adjacent elements, with the boundary ring zeroed because
the estimate is unreliable there. Argyris carries {u_xx, u_xy, u_yy} as VERTEX
DOFs, so

    gamma_1 = 1/2 (u_xx - u_yy),    gamma_2 = u_xy

is a pure SELECTION out of the DOF vector -- two sparse matrices with two and one
entry per row. No quadrature, no averaging, no boundary special case. The
operator that took a page of assembly on P3 is three lines here.

The gradient follows the same adjoint structure as the P3 path,

    dL/dkappa = -4 M^T A^{-T} ( S1^T W r1 + S2^T W r2 ) + 2 lam R kappa,

and A^{-T} is available from the same LU factorisation (the coupled operator is
NOT symmetric -- the BEM double layer is not -- so the transpose solve is
required, not optional).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.optimize as sopt

from .c1_coupling import C1CoupledOperators
from .c1_assembly import assemble_c1_load


def shear_selection_operators(space):
    """(S1, S2) mapping the C^1 DOF vector to shear at the mesh vertices.

    Argyris only. The Hessian components are DOFs 3, 4, 5 of each vertex block,
    so this is a selection: gamma1 = 1/2(u_xx - u_yy), gamma2 = u_xy.
    """
    if space.n_vert_dofs != 6:
        raise ValueError("shear selection needs Hessian vertex DOFs (Argyris); "
                         f"got n_vert_dofs={space.n_vert_dofs}")
    nv = space.n_vertices
    rows1 = np.repeat(np.arange(nv), 2)
    cols1 = np.stack([np.arange(nv) * 6 + 3, np.arange(nv) * 6 + 5], 1).ravel()
    vals1 = np.tile([0.5, -0.5], nv)
    S1 = sp.coo_matrix((vals1, (rows1, cols1)), shape=(nv, space.n_dofs)).tocsr()
    S2 = sp.coo_matrix((np.ones(nv), (np.arange(nv), np.arange(nv) * 6 + 4)),
                       shape=(nv, space.n_dofs)).tocsr()
    return S1, S2


def c1_gradient_operator(space, quad_order=7):
    """(Gx, Gy, area): element-centroid gradients of a C^1 field, and areas.

    The P3 version (priors.build_gradient_operator) reads reference-element shape
    gradients; a C^1 element has no reference pullback (its DOFs are physical
    derivatives), so the gradients are evaluated directly at each element's
    centroid. This is what lets TV and gradient-sparsity priors act on the C^1
    DOF vector at all.
    """
    verts, tris = space.vertices, space.triangles
    rows, cols, gx, gy = [], [], [], []
    area = np.zeros(len(tris))
    for t in range(len(tris)):
        el = space.element(t)
        v = el.verts
        area[t] = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0]]))) / 2.0
        c = v.mean(axis=0)[None, :]
        bx = el.basis(c, 1, 0)[0]
        by = el.basis(c, 0, 1)[0]
        idx = space.local_dofs(t)
        rows.extend([t] * len(idx)); cols.extend(idx.tolist())
        gx.extend(bx.tolist()); gy.extend(by.tolist())
    shape = (len(tris), space.n_dofs)
    return (sp.coo_matrix((gx, (rows, cols)), shape=shape).tocsr(),
            sp.coo_matrix((gy, (rows, cols)), shape=shape).tocsr(),
            area)


class C1TotalVariation:
    """Smoothed total variation on a C^1 space: phi = sum_e a_e |grad k|_eps."""
    is_quadratic = False

    def __init__(self, space, eps=1e-3, quad_order=7):
        self.name = f"C1-TV(eps={eps:g})"
        self.eps = float(eps)
        self.Gx, self.Gy, self.area = c1_gradient_operator(space, quad_order)

    def value_grad(self, kappa):
        gx = self.Gx @ kappa; gy = self.Gy @ kappa
        mag = np.sqrt(gx**2 + gy**2 + self.eps**2)
        w = self.area / mag
        return (float(np.dot(self.area, mag)),
                self.Gx.T @ (w * gx) + self.Gy.T @ (w * gy))


class C1Sparsity:
    """Smoothed-L1 on the kappa VALUE DOFs, weighted by the lumped mass.

    Only the value DOFs are penalised: an L1 penalty on derivative DOFs would
    suppress curvature rather than amplitude, which is not what a compact-source
    prior means.
    """
    is_quadratic = False

    def __init__(self, space, ops, eps=1e-3):
        self.name = f"C1-Sparse(eps={eps:g})"
        self.eps = float(eps)
        nv = space.n_vert_dofs
        self.idx = np.arange(0, space.n_vertices * nv, nv)
        self.w = np.asarray(ops.M @ np.ones(space.n_dofs)).ravel()[self.idx]
        self.w = np.abs(self.w)
        self.n = space.n_dofs

    def value_grad(self, kappa):
        t = np.asarray(kappa)[self.idx]
        mag = np.sqrt(t**2 + self.eps**2)
        g = np.zeros(self.n)
        g[self.idx] = self.w * t / mag
        return float(np.dot(self.w, mag)), g


def make_c1_prior(kind, space, ops, **kw):
    """Build a non-Gaussian prior against a C^1 space ('tv' or 'sparse')."""
    if kind == "tv":
        return C1TotalVariation(space, **kw)
    if kind == "sparse":
        return C1Sparsity(space, ops, **kw)
    raise ValueError(f"unknown C^1 prior {kind!r}; expected 'tv' or 'sparse'")


class C1MAPReconstructor:
    """MAP reconstruction of kappa on a C^1 space with the BEM far-field.

    kappa is represented in the SAME C^1 space as psi, so the load vector is
    -2 M kappa exactly as in the P3 path.
    """

    def __init__(self, space, coupled=None, lam=1e-2, wiener_length=1.0,
                 data_weight=None, maxiter=300, degree=5, quad_order=7,
                 prior=None, prior_kw=None):
        self.space = space
        self.ops = coupled or C1CoupledOperators(space, degree=degree,
                                                 quad_order=quad_order)
        self.S1, self.S2 = shear_selection_operators(space)
        self.lam = float(lam)
        self.maxiter = int(maxiter)
        self.w = (np.ones(space.n_vertices) if data_weight is None
                  else np.asarray(data_weight, float))
        # Wiener/Matern-style regulariser on the C^1 DOF vector, matching the
        # P3 path's R = M + l^2 K. A non-Gaussian prior (string or object)
        # replaces it; see make_c1_prior.
        self.R = (self.ops.M + (wiener_length ** 2) * self.ops.K).tocsr()
        self.prior = (make_c1_prior(prior, space, self.ops, **(prior_kw or {}))
                      if isinstance(prior, str) else prior)

    # -- forward / adjoint ------------------------------------------------- #
    def psi_of(self, kappa):
        return self.ops.solve_psi(-2.0 * (self.ops.M @ kappa))

    def shear_of(self, kappa):
        psi = self.psi_of(kappa)
        return self.S1 @ psi, self.S2 @ psi

    def _obj_grad(self, kappa, g1_obs, g2_obs):
        psi = self.psi_of(kappa)
        r1 = self.S1 @ psi - g1_obs
        r2 = self.S2 @ psi - g2_obs
        wr1, wr2 = self.w * r1, self.w * r2
        data = float(np.dot(wr1, r1) + np.dot(wr2, r2))

        # psi = A^{-1}(P b) with P the gauge projector (zeroes component g of the
        # RHS) and b = -2 M kappa, so
        #     dL/dkappa = -2 M^T P (A^{-T} dL/dpsi).
        # The projector therefore acts AFTER the transpose solve. Applying it to
        # the adjoint RHS beforehand is a different operator and gets some DOFs
        # badly wrong (34% on the worst one) while leaving most of them fine --
        # which is exactly why the finite-difference check samples several.
        adj = self.ops.A_lu.solve(self.S1.T @ wr1 + self.S2.T @ wr2, trans='T')
        adj[self.ops.idx_gauge] = 0.0
        grad = -4.0 * (self.ops.M.T @ adj)

        if self.prior is None:
            Rk = self.R @ kappa
            loss = data + self.lam * float(np.dot(kappa, Rk))
            grad = grad + 2.0 * self.lam * Rk
        else:
            phi, gphi = self.prior.value_grad(kappa)
            loss = data + self.lam * float(phi)
            grad = grad + self.lam * gphi
        return loss, grad

    def reconstruct(self, g1_obs, g2_obs, kappa_init=None, verbose=False):
        g1_obs = np.asarray(g1_obs, float); g2_obs = np.asarray(g2_obs, float)
        x0 = (np.zeros(self.space.n_dofs) if kappa_init is None
              else np.asarray(kappa_init, float).copy())
        res = sopt.minimize(
            lambda k: self._obj_grad(k, g1_obs, g2_obs),
            x0, jac=True, method="L-BFGS-B",
            options=dict(maxiter=self.maxiter, ftol=1e-14, gtol=1e-10))
        return res.x, res

    # -- convenience ------------------------------------------------------- #
    def kappa_at_vertices(self, kappa):
        """The value DOFs of the kappa field -- kappa sampled at the vertices."""
        nv = self.space.n_vert_dofs
        return np.asarray(kappa)[:self.space.n_vertices * nv:nv]

    def synthesise_data(self, kappa_fn, noise_std=0.0, seed=0, quad_order=7):
        """Shear at the vertices from FEMMI's own C^1 forward.

        For validating the inverse machinery only -- an inverse crime by
        construction. Score reconstructions against femmi.truth instead.
        """
        b = -2.0 * assemble_c1_load(self.space, kappa_fn, quad_order)
        psi = self.ops.solve_psi(b)
        g1, g2 = self.S1 @ psi, self.S2 @ psi
        if noise_std:
            rng = np.random.default_rng(seed)
            g1 = g1 + rng.normal(0, noise_std, len(g1))
            g2 = g2 + rng.normal(0, noise_std, len(g2))
        return g1, g2
