"""
femmi/aca.py
Hierarchical (H-matrix) compression of the dense BEM operators via Adaptive
Cross Approximation.

WHY
---
`bem.assemble_single_layer` builds a DENSE N_b x N_b matrix, and every entry is a
double boundary integral. That is O(N_b^2) memory and work, and it is the wall
this project keeps hitting: profiling a build at nx=20 puts 2.3s of a 2.6s total
in BEM assembly, and circular-domain runs were abandoned after an hour.

The single-layer kernel G(x, y) = (1/2pi) log|x - y| is ASYMPTOTICALLY SMOOTH:
for two well-separated pieces of boundary the interaction block is numerically
low rank. ACA finds that low-rank factorisation adaptively from a handful of
sampled rows and columns, without ever forming the block -- so an admissible
block of size m x n costs O(k(m+n)) instead of O(mn), with k the numerical rank.

Structure: a binary cluster tree over boundary DOFs, the standard admissibility
test min(diam(s), diam(t)) <= eta * dist(s, t), ACA with partial pivoting on
admissible blocks, dense assembly on the rest.

SCOPE
-----
This module compresses and applies the operator. It does NOT yet replace
`assemble_single_layer` inside `build_operators`, because the coupled solve
currently LU-factorises a dense A_coupled; consuming an H-matrix needs the
iterative solver + Calderon preconditioner (the next item). What is delivered
here is the compressed operator, its matvec, and measured accuracy/compression.
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# cluster tree
# --------------------------------------------------------------------------- #
class Cluster:
    """A node of the binary cluster tree over a set of points."""

    __slots__ = ("idx", "centre", "radius", "left", "right")

    def __init__(self, idx, points):
        self.idx = np.asarray(idx, int)
        p = points[self.idx]
        self.centre = p.mean(axis=0)
        self.radius = float(np.max(np.linalg.norm(p - self.centre, axis=1))) if len(p) else 0.0
        self.left = self.right = None

    @property
    def is_leaf(self):
        return self.left is None


def build_cluster_tree(points, min_size=32):
    """Binary space partition: split each cluster along its widest extent at the
    median, until leaves hold at most `min_size` points."""
    points = np.asarray(points, float)

    def rec(idx):
        c = Cluster(idx, points)
        if len(idx) <= min_size:
            return c
        p = points[idx]
        axis = int(np.argmax(p.max(axis=0) - p.min(axis=0)))
        order = np.argsort(p[:, axis])
        half = len(idx) // 2
        c.left = rec(idx[order[:half]])
        c.right = rec(idx[order[half:]])
        return c

    return rec(np.arange(len(points)))


def is_admissible(s: Cluster, t: Cluster, eta=1.0):
    """Standard admissibility: min(diam s, diam t) <= eta * dist(s, t).

    Admissible blocks are the ones where the kernel is smooth over the whole
    block and therefore numerically low rank."""
    dist = np.linalg.norm(s.centre - t.centre) - s.radius - t.radius
    return dist > 0 and min(2 * s.radius, 2 * t.radius) <= eta * dist


# --------------------------------------------------------------------------- #
# ACA with partial pivoting
# --------------------------------------------------------------------------- #
def aca_partial(block_row, block_col, m, n, tol=1e-6, max_rank=None):
    """Approximate an m x n block as U @ V.T without forming it.

    block_row(i) -> length-n row i;  block_col(j) -> length-m column j.

    Classic partially-pivoted ACA: pick a pivot row, subtract what is already
    captured, take its largest residual entry as the pivot column, and repeat.
    Stops when the newest rank-1 update is small relative to the running estimate
    of the block's Frobenius norm.

    Returns (U, V) with shapes (m, k) and (n, k), or None if the block did not
    compress below max_rank (caller should then store it densely).
    """
    max_rank = max_rank or max(1, min(m, n) // 2)
    U = np.zeros((m, max_rank)); V = np.zeros((n, max_rank))
    used_rows, used_cols = set(), set()
    i = 0
    norm2 = 0.0
    k = 0

    for _ in range(max_rank):
        # residual of row i
        row = np.asarray(block_row(i), float).copy()
        if k:
            row -= U[i, :k] @ V[:, :k].T
        row[list(used_cols)] = 0.0
        j = int(np.argmax(np.abs(row)))
        if abs(row[j]) < 1e-300:
            break
        row /= row[j]

        col = np.asarray(block_col(j), float).copy()
        if k:
            col -= V[j, :k] @ U[:, :k].T

        U[:, k] = col; V[:, k] = row
        used_rows.add(i); used_cols.add(j)

        # incremental Frobenius-norm update
        cross = 0.0
        if k:
            cross = 2.0 * np.sum((U[:, :k].T @ col) * (V[:, :k].T @ row))
        nk2 = float(np.dot(col, col) * np.dot(row, row))
        norm2 = max(norm2 + cross + nk2, 0.0)
        k += 1

        if nk2 <= (tol ** 2) * norm2:
            break

        # next pivot row: largest residual entry of the new column
        c = np.abs(col).copy()
        if used_rows:
            c[list(used_rows)] = -1.0
        i = int(np.argmax(c))
        if c[i] < 0:
            break

    if k == 0:
        return None
    return U[:, :k].copy(), V[:, :k].copy()


# --------------------------------------------------------------------------- #
# H-matrix
# --------------------------------------------------------------------------- #
class HMatrix:
    """Block-partitioned matrix: low-rank (U, V) on admissible blocks, dense
    elsewhere. Supports matvec and reports its own compression."""

    def __init__(self, blocks, shape):
        self.blocks = blocks          # list of (rows, cols, kind, payload)
        self.shape = shape

    def matvec(self, x):
        x = np.asarray(x, float)
        y = np.zeros(self.shape[0])
        for r, c, kind, payload in self.blocks:
            if kind == "lr":
                U, V = payload
                y[r] += U @ (V.T @ x[c])
            else:
                y[r] += payload @ x[c]
        return y

    def __matmul__(self, x):
        return self.matvec(x)

    @property
    def nnz_stored(self):
        tot = 0
        for _, _, kind, payload in self.blocks:
            if kind == "lr":
                U, V = payload
                tot += U.size + V.size
            else:
                tot += payload.size
        return tot

    @property
    def compression(self):
        """Stored entries as a fraction of the dense matrix."""
        return self.nnz_stored / (self.shape[0] * self.shape[1])

    def to_dense(self):
        A = np.zeros(self.shape)
        for r, c, kind, payload in self.blocks:
            sub = (payload[0] @ payload[1].T) if kind == "lr" else payload
            A[np.ix_(r, c)] += sub
        return A


def build_hmatrix(points, entry_block, min_size=32, eta=1.0, tol=1e-6,
                  max_rank_frac=0.5, near_block=None):
    """Compress the operator defined by `entry_block(rows, cols) -> dense subblock`.

    points     : (N, 2) geometric location of each DOF, used for clustering.
    entry_block: far-field evaluator, used on ADMISSIBLE blocks. ACA only ever
                 asks it for single rows and columns, so it is never called on
                 anything large.
    near_block : evaluator for INADMISSIBLE blocks. Defaults to `entry_block`,
                 which is only correct when that evaluator is also valid at close
                 range.

                 For the BEM single layer it is NOT: `single_layer_entry_fn` uses
                 plain Gauss-Legendre with no singular treatment, which is fine
                 across separated clusters and wrong on the near/self blocks where
                 the log singularity needs Duffy/log-Gauss handling. Feeding the
                 far-field evaluator to the near field silently builds a different
                 operator -- it cost a 69% error in the coupled solve before this
                 argument existed. Pass a near-field-accurate evaluator (e.g. a
                 lookup into the exactly assembled matrix) for the BEM.
    """
    near_block = near_block or entry_block
    points = np.asarray(points, float)
    root = build_cluster_tree(points, min_size=min_size)
    blocks = []

    def rec(s, t):
        if is_admissible(s, t, eta):
            m, n = len(s.idx), len(t.idx)
            res = aca_partial(
                lambda i: entry_block(s.idx[i:i + 1], t.idx)[0],
                lambda j: entry_block(s.idx, t.idx[j:j + 1])[:, 0],
                m, n, tol=tol, max_rank=max(1, int(max_rank_frac * min(m, n))))
            if res is not None and res[0].shape[1] * (m + n) < m * n:
                blocks.append((s.idx, t.idx, "lr", res))
                return
        if s.is_leaf or t.is_leaf:
            blocks.append((s.idx, t.idx, "dense", near_block(s.idx, t.idx)))
            return
        for a in (s.left, s.right):
            for b in (t.left, t.right):
                rec(a, b)

    rec(root, root)
    return HMatrix(blocks, (len(points), len(points)))


# --------------------------------------------------------------------------- #
# single-layer entries at DOF level (the thing ACA needs)
# --------------------------------------------------------------------------- #
def single_layer_entry_fn(bnd, n_quad=12):
    """Return (points, entry_block) for the Galerkin single-layer operator.

    entry_block(rows, cols) computes V[rows, cols] directly:

        V[i,j] = sum_{e ni i} sum_{f ni j} L_e L_f
                 sum_{q,r} w_q w_r G(x_e(q), x_f(r)) phi_a(q) phi_b(r)

    summing over the boundary elements that carry DOF i (local index a) and DOF j
    (local index b). No singular treatment is applied, so this is valid for
    ADMISSIBLE blocks -- exactly where ACA uses it. Near-field and self blocks
    keep the tuned Duffy/log-Gauss handling in bem.assemble_single_layer.
    """
    from .bem import _p3_boundary_basis, _gauss_legendre

    elems = np.asarray(bnd.elements)
    nodes = np.asarray(bnd.nodes)
    L = np.asarray(bnd.element_lengths)
    N_b = bnd.n_boundary_dofs

    xi, w = _gauss_legendre(n_quad)
    phi = _p3_boundary_basis(xi)                       # (nq, 4)
    p0 = nodes[elems[:, 0]]; p3 = nodes[elems[:, 3]]
    xq = p0[:, None, :] + xi[None, :, None] * (p3 - p0)[:, None, :]   # (ne,nq,2)

    # DOF -> list of (element, local index)
    owner = [[] for _ in range(N_b)]
    for e, el in enumerate(elems):
        for a, d in enumerate(el):
            owner[int(d)].append((e, a))

    # geometric location of each DOF, for clustering
    tnodes = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    pts = np.zeros((N_b, 2))
    for d in range(N_b):
        e, a = owner[d][0]
        pts[d] = p0[e] + tnodes[a] * (p3[e] - p0[e])

    def entry_block(rows, cols):
        rows = np.atleast_1d(rows); cols = np.atleast_1d(cols)
        out = np.zeros((len(rows), len(cols)))
        for ri, i in enumerate(rows):
            for e, a in owner[int(i)]:
                wi = L[e] * w * phi[:, a]                       # (nq,)
                for cj, j in enumerate(cols):
                    for f, b in owner[int(j)]:
                        d2 = np.sum((xq[e][:, None, :] - xq[f][None, :, :]) ** 2,
                                    axis=-1)
                        G = np.log(np.maximum(d2, 1e-300)) / (4.0 * np.pi)
                        wj = L[f] * w * phi[:, b]
                        out[ri, cj] += wi @ G @ wj
        return out

    return pts, entry_block


def near_field_entry_fn(bnd, degree=3, n_quad=25):
    """Return (points, near_block) evaluating the single layer with the CORRECT
    singular treatment, for near/self blocks.

    `single_layer_entry_fn` uses plain Gauss-Legendre and is only valid across
    separated clusters; feeding it to inadmissible blocks builds a different
    operator (it cost a 69% error in the coupled solve). Until now the only fix
    was to look entries up in an exactly assembled dense V_h -- which defeats the
    purpose, since the dense assembly is exactly what ACA is meant to avoid.

    This computes near blocks directly: element pairs that share no support use
    the same smooth quadrature, and pairs on the SAME element use the Duffy /
    log-Gauss-Jacobi decomposition from bem_hp. Only the O(N) near blocks are ever
    touched, so no dense N_b x N_b matrix is ever formed.
    """
    from .bem_hp import boundary_basis, assemble_single_layer_hp
    from .bem import _gauss_legendre

    elems = np.asarray(bnd.elements)
    nodes = np.asarray(bnd.nodes)
    L = np.asarray(bnd.element_lengths)
    N_b = bnd.n_boundary_dofs
    nd = degree + 1

    xi, w = _gauss_legendre(n_quad)
    phi = boundary_basis(degree, xi)
    p0 = nodes[elems[:, 0]]; p1 = nodes[elems[:, -1]]
    xq = p0[:, None, :] + xi[None, :, None] * (p1 - p0)[:, None, :]

    owner = [[] for _ in range(N_b)]
    for e, el in enumerate(elems):
        for a, d in enumerate(el):
            owner[int(d)].append((e, a))

    tnodes = np.linspace(0.0, 1.0, nd)
    pts = np.zeros((N_b, 2))
    for d in range(N_b):
        e, a = owner[d][0]
        pts[d] = p0[e] + tnodes[a] * (p1[e] - p0[e])

    # Per-element self blocks, computed once. Assembling a single-element boundary
    # mesh would be circular, so reuse the same Duffy/log-Gauss expression.
    from .bem import log_gauss_jacobi_points
    xi_lj, w_lj = log_gauss_jacobi_points(n_quad)
    self_blocks = np.zeros((bnd.n_elements, nd, nd))
    for e in range(bnd.n_elements):
        L_e = L[e]
        Vd = np.zeros((nd, nd))
        for q, (sig, wq) in enumerate(zip(xi, w)):
            phi_s = phi[q]
            logs = np.log(L_e * sig)
            for v, wv in zip(xi, w):
                phi_t = boundary_basis(degree, np.array([sig * (1.0 - v)]))[0]
                pre = L_e**2 / (2.0 * np.pi) * sig * logs * wq * wv
                Vd += pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))
            for v, wv_lj in zip(xi_lj, w_lj):
                phi_t = boundary_basis(degree, np.array([sig * (1.0 - v)]))[0]
                pre = L_e**2 / (2.0 * np.pi) * sig * wq * wv_lj
                Vd -= pre * (np.outer(phi_s, phi_t) + np.outer(phi_t, phi_s))
        self_blocks[e] = Vd

    def near_block(rows, cols):
        rows = np.atleast_1d(rows); cols = np.atleast_1d(cols)
        out = np.zeros((len(rows), len(cols)))
        for ri, i in enumerate(rows):
            for e, a in owner[int(i)]:
                wi = L[e] * w * phi[:, a]
                for cj, j in enumerate(cols):
                    for f, b in owner[int(j)]:
                        if e == f:
                            out[ri, cj] += self_blocks[e][a, b]
                            continue
                        d2 = np.sum((xq[e][:, None, :] - xq[f][None, :, :]) ** 2,
                                    axis=-1)
                        G = np.log(np.maximum(d2, 1e-300)) / (4.0 * np.pi)
                        out[ri, cj] += wi @ G @ (L[f] * w * phi[:, b])
        return out

    # the assembled operator is symmetrised, so mirror that here
    def near_block_sym(rows, cols):
        rows = np.atleast_1d(rows); cols = np.atleast_1d(cols)
        return 0.5 * (near_block(rows, cols) + near_block(cols, rows).T)

    return pts, near_block_sym
