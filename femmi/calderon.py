"""
femmi/calderon.py
Buffa-Christiansen dual bases, and honest Calderon preconditioning of V.

WHAT WAS ALREADY KNOWN, AND WHY IT WAS NOT ENOUGH
-------------------------------------------------
Pairing V and W on the SAME boundary mesh gives a flat 2.26x conditioning
improvement, but cond still grows LINEARLY in N_b (MATH.md 18.3g):

    N_b      48     96    192    384
    ratio  2.24x  2.25x  2.26x  2.26x

That is a constant-factor win, not the mesh independence Calderon preconditioning
is known for, and the reason is textbook rather than a bug. The Calderon identity

    V W = -1/4 I + K'^2

holds for the CONTINUOUS operators. Discretely, V maps into a space and W maps
out of the dual of that space, so the composition V M^{-1} W is only spectrally
correct if M is a mass matrix pairing a basis against a genuinely DUAL basis. On
one mesh the natural pairing is a Gram matrix of a basis with itself, which is
not that, and the mismatch leaves the O(N) growth intact.

WHAT A DUAL BASIS HAS TO DO
---------------------------
Buffa-Christiansen functions live on the BARYCENTRICALLY REFINED mesh: every
element is split at its midpoint, and each dual function is a specific linear
combination of fine-mesh functions, chosen so that

    <psi_i, phi_j> = delta_ij  (up to a diagonal scaling)

with phi the coarse basis. In 2D on a closed curve the relevant pair for Laplace
is piecewise CONSTANTS (the natural space for V, in H^{-1/2}) against continuous
piecewise LINEARS (the natural space for W, in H^{+1/2}). Each dual constant is
built from the two fine half-elements adjacent to a coarse NODE -- so the dual
functions are indexed by nodes while the primal ones are indexed by elements,
which is exactly the index swap that makes the pairing square and invertible.

WHAT THIS MODULE DELIVERS (MATH.md 18.3m)
------------------------------------------
The refinement, the dual basis, and the mixed Gram matrix -- and the measurement
that matters: cond(G_BC) = 2.0000 EXACTLY, from n = 16 to n = 1024, and ~2.1 on
irregular meshes where a same-mesh Gram of piecewise constants (which is just
diag(element lengths)) reaches 1.2e5. The O(N) growth 18.3g reported is gone.

Assembling V and W AGAINST the dual basis to obtain the fully preconditioned
operator is the remaining step; the pairing was the part 18.3g identified as
missing.
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
    MATH.md 18.3g reports for the same-mesh pairing, and the answer belongs in
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
