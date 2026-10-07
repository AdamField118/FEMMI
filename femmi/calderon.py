"""Dual-grid boundary operators and spectral preconditioning experiments.

The mixed Gram is well-conditioned, but that fact alone says nothing about
conditioning of a preconditioned single-layer operator. dual_boundary_operators
now assembles the actual low-order dual V and primal-hat W and uses BOTH Gram
inverse factors, P=G^{-T} W_stabilized G^{-1}. dual_conditioning measures P V
through a symmetric similarity transform. These polygonal P0/P1 experiments
are separate from the production high-order P3/P5 FEM-BEM coupling.

The earlier calderon_conditioning function is retained as a historical
same-space/Gram diagnostic, not an end-to-end dual-preconditioner test.
"""

from __future__ import annotations
import numpy as np


def barycentric_refine(nodes):
    """Split every element of a closed 1-D boundary loop at its midpoint.

    nodes : (n, 2) vertices of the closed polygon, in order (no repeat of the
    first point). Returns (fine_nodes, parent) where fine_nodes has 2n points --
    the originals at even indices, the new midpoints at odd -- and parent[k] is
    the coarse element that fine element k came from.

    Even/odd interleaving is deliberate: it makes "the two fine elements of
    coarse element e" simply 2e and 2e+1, and "the two fine elements meeting at
    coarse node i" simply 2i-1 and 2i (mod 2n), which is what the dual basis
    needs and what an appended-midpoint ordering would obscure.
    """
    nodes = np.asarray(nodes, float)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 3:
        raise ValueError("nodes must be (n, 2) with n >= 3")
    n = len(nodes)
    mids = 0.5 * (nodes + np.roll(nodes, -1, axis=0))
    fine = np.empty((2 * n, 2))
    fine[0::2] = nodes
    fine[1::2] = mids
    parent = np.repeat(np.arange(n), 2)
    return fine, parent


def _lengths(nodes):
    d = np.roll(nodes, -1, axis=0) - nodes
    return np.hypot(d[:, 0], d[:, 1])


def dual_constant_basis(nodes):
    """Buffa-Christiansen piecewise constants, as coefficients on fine elements.

    Returns D of shape (n_coarse, 2 n_coarse): row i is the dual function
    associated with coarse NODE i, expressed in the fine piecewise-constant
    basis. It is supported on the two fine elements touching that node, which is
    half of each of the two adjacent coarse elements.

    The weights are chosen so that each dual function has unit integral, which is
    what makes the pairing against the coarse hats a scaled identity on the
    constant mode and keeps <psi_i, 1> independent of the local element size --
    the property that a same-mesh Gram matrix does not have.
    """
    nodes = np.asarray(nodes, float)
    n = len(nodes)
    fine, _ = barycentric_refine(nodes)
    Lf = _lengths(fine)                       # 2n fine element lengths

    D = np.zeros((n, 2 * n))
    for i in range(n):
        a = (2 * i - 1) % (2 * n)             # second half of coarse element i-1
        b = (2 * i) % (2 * n)                 # first half of coarse element i
        tot = Lf[a] + Lf[b]
        D[i, a] = 1.0 / tot
        D[i, b] = 1.0 / tot
    return D


def primal_constant_on_fine(nodes):
    """Coarse piecewise constants expressed on the fine mesh: (n, 2n)."""
    n = len(nodes)
    P = np.zeros((n, 2 * n))
    for e in range(n):
        P[e, 2 * e] = 1.0
        P[e, 2 * e + 1] = 1.0
    return P


def mixed_gram(nodes):
    """<psi_i, lambda_j>: dual constants against the coarse HAT functions.

    Square (n, n), indexed node x node.

    THE PAIRING HAS TO CROSS SPACES, and getting this wrong is the whole trap.
    Pairing the node-indexed dual constants against the ELEMENT-indexed coarse
    constants gives a bidiagonal circulant whose rows sum to 1 -- eigenvalues
    (1 + w^k)/2 over the n-th roots of unity, which vanishes at w = -1. It is
    exactly SINGULAR for every even n, and measuring it returns cond ~ 1e16
    rather than anything mesh-independent. (This module was first written that
    way; tests/test_calderon.py keeps the broken pairing as a regression.)

    The correct partner is the space on the other side of the Calderon identity:
    V acts on densities in H^{-1/2} (piecewise constants, element-indexed) while
    W acts on traces in H^{+1/2} (continuous piecewise linears, node-indexed).
    The BC duals are node-indexed, so they pair with the hats.

    That pairing is computable in closed form. On the fine half-element from a
    node to a midpoint the coarse hat at that node runs 1 -> 1/2, so its mean is
    3/4, and the neighbouring hat runs 0 -> 1/2 with mean 1/4. With the dual
    normalised to unit integral over its two half-elements this gives

        G[i, i]     = 3/4                       exactly, for ANY mesh
        G[i, i+-1]  = (1/4) * L_half / L_total,  summing to 1/4

    so the matrix is diagonally dominant with a diagonal that does not depend on
    the element sizes at all. For a uniform mesh its eigenvalues are
    3/4 + (1/4) cos(theta) in [1/2, 1], i.e. cond <= 2 INDEPENDENT OF N -- which
    is the mesh independence a same-mesh Gram matrix cannot give.
    """
    nodes = np.asarray(nodes, float)
    n = len(nodes)
    fine, _ = barycentric_refine(nodes)
    Lf = _lengths(fine)

    G = np.zeros((n, n))
    for i in range(n):
        a = (2 * i - 1) % (2 * n)      # midpoint -> node i (second half of el i-1)
        b = (2 * i) % (2 * n)          # node i -> midpoint (first half of el i)
        tot = Lf[a] + Lf[b]
        # hat_i has mean 3/4 on both halves touching node i
        G[i, i] = 0.75 * (Lf[a] + Lf[b]) / tot
        # the far neighbour of each half carries mean 1/4
        G[i, (i - 1) % n] += 0.25 * Lf[a] / tot
        G[i, (i + 1) % n] += 0.25 * Lf[b] / tot
    return G


def calderon_conditioning(n_bs=(48, 96, 192, 384), degree=3, radius=1.0,
                          sigma_scale=1.0, verbose=True):
    """Measure conditioning growth: raw V, same-mesh pairing, and BC pairing.

    Returns a list of dicts. This is a MEASUREMENT, not a claim -- the point of
    the module is to find out whether the dual basis removes the O(N) growth that
    MATH.md 18.3.9 reports for the same-mesh pairing, and the answer belongs in
    the table it produces rather than in a docstring.
    """
    from .bem_hp import (build_circular_boundary_mesh, assemble_single_layer_hp,
                         assemble_hypersingular_hp, assemble_boundary_mass_hp)

    out = []
    for n_b in n_bs:
        n_el = max(4, int(n_b) // degree)
        bnd = build_circular_boundary_mesh(n_el, radius=radius, degree=degree)
        N = bnd.n_boundary_dofs

        V = assemble_single_layer_hp(bnd, degree)
        W = assemble_hypersingular_hp(bnd, degree)
        M = assemble_boundary_mass_hp(bnd, degree)

        w = M @ np.ones(N)
        sigma = float(sigma_scale) * 2.0 * radius
        V_eff = V - (np.log(sigma) / (2.0 * np.pi)) * np.outer(w, w)
        # W annihilates constants; stabilise the same mode so cond() is finite
        W_eff = W + np.outer(w, w) / max(float(np.dot(w, w)), 1e-300)

        c_raw = float(np.linalg.cond(V_eff))
        c_same = float(np.linalg.cond(V_eff @ np.linalg.solve(M, W_eff)))

        # BC pairing on the element polygon: the dual Gram replaces M
        poly = np.asarray(bnd.nodes)[::degree][:n_el]
        G = mixed_gram(poly)

        rec = dict(n_b=N, cond_V=c_raw, cond_same_mesh=c_same,
                   ratio_same=c_raw / c_same,
                   cond_gram_same_mesh=float(np.linalg.cond(M)),
                   cond_gram_bc=float(np.linalg.cond(G)))
        out.append(rec)
        if verbose:
            print(f"  N_b={N:5d}  cond(V)={c_raw:.3e}  "
                  f"cond(V M^-1 W)={c_same:.3e}  ratio={rec['ratio_same']:.2f}x  "
                  f"cond(Gram same)={rec['cond_gram_same_mesh']:.3e}  "
                  f"cond(Gram BC)={rec['cond_gram_bc']:.3e}", flush=True)
    return out


def dual_boundary_operators(nodes, quadrature_order=16, sigma=None):
    """Low-order dual-constant V and primal-hat W on a closed polygon.

    V uses the positive kernel -log(r/sigma)/(2*pi), opposite to FEMMI's
    potential-kernel sign. W = E.T V_fine E uses tangential derivatives of
    continuous hats; discontinuous dual constants are NOT a valid W space.
    Self integrals are analytic. Shared-endpoint integrals use a Duffy split
    with the radial log integrated analytically; separated panels use Gauss.
    This is an experimental low-order preconditioner, not a drop-in for P3/P5.
    """
    from numpy.polynomial.legendre import leggauss
    nodes=np.asarray(nodes,float)
    fine,_=barycentric_refine(nodes);length=_lengths(fine);n=len(nodes);nf=2*n
    if np.any(length<=0) or not np.all(np.isfinite(fine)):
        raise ValueError('finite nondegenerate polygon required')
    if quadrature_order<2:raise ValueError('quadrature_order must be >=2')
    diameter=float(np.max(np.linalg.norm(nodes[:,None]-nodes[None,:],axis=2)))
    sigma=2*diameter if sigma is None else float(sigma)
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('sigma must be positive')
    z,w=leggauss(quadrature_order);z=(z+1)/2;w=w/2
    vector=np.roll(fine,-1,axis=0)-fine
    points=fine[:,None,:]+z[None,:,None]*vector[:,None,:]
    V=np.empty((nf,nf))
    for i in range(nf):
        V[i,i]=length[i]**2*(1.5+np.log(sigma/length[i]))/(2*np.pi)
        for j in range(i):
            if (i-j)%nf in (1,nf-1):
                if i==(j+1)%nf:u,v=-vector[j],vector[i]
                else:u,v=-vector[i],vector[j]
                a=np.linalg.norm(u[None,:]-z[:,None]*v[None,:],axis=1)
                b=np.linalg.norm(z[:,None]*u[None,:]-v[None,:],axis=1)
                integral=-.5+.5*np.dot(w,np.log(a)+np.log(b))-np.log(sigma)
            else:
                distance=np.linalg.norm(points[i,:,None,:]-points[j,None,:,:],axis=2)
                integral=w@np.log(distance/sigma)@w
            V[i,j]=V[j,i]=-length[i]*length[j]*integral/(2*np.pi)
    D=dual_constant_basis(nodes);G=mixed_gram(nodes)
    coarse_length=_lengths(nodes)
    E=np.zeros((nf,n))
    for i in range(n):
        E[2*i:2*i+2,i]=-1/coarse_length[i]
        E[2*i:2*i+2,(i+1)%n]=1/coarse_length[i]
    W=E.T@V@E;Vd=D@V@D.T
    # Stabilise only the constant trace mode; W itself retains its exact null.
    mass=(coarse_length+np.roll(coarse_length,1))/2
    Ws=W+np.outer(mass,mass)/mass.sum()**2
    Gi=np.linalg.solve(G,np.eye(n))
    P=Gi.T@Ws@Gi
    return dict(V=Vd,W=W,G=G,preconditioner=P,fine_V=V,sigma=sigma)


def dual_conditioning(nodes,quadrature_order=16):
    """Spectral condition of P V, evaluated through a symmetric congruence."""
    from scipy.linalg import eigvalsh,cholesky
    op=dual_boundary_operators(nodes,quadrature_order)
    L=cholesky(op['preconditioner'],lower=True)
    eigen=eigvalsh(L.T@op['V']@L)
    raw=eigvalsh(op['V'])
    if min(eigen)<=0 or min(raw)<=0:raise ValueError('operators are not positive definite')
    return dict(n_boundary=len(nodes),cond_V=float(raw[-1]/raw[0]),
                cond_preconditioned=float(eigen[-1]/eigen[0]),
                eigen_min=float(eigen[0]),eigen_max=float(eigen[-1]),
                constant_residual=float(np.linalg.norm(op['W']@np.ones(len(nodes)))))
