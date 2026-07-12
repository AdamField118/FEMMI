"""
femmi/priors.py
Pluggable priors (regularisers) for the MAP mass reconstruction.

The MAP cost is

    J(kappa) = || F kappa - gamma_obs ||^2_w  +  lambda * phi(kappa)

and every prior exposes the SAME contract: given the current kappa it returns
the penalty value phi and its gradient grad_phi, so the reconstructor forms

    loss += lambda * phi ,      grad += lambda * grad_phi .

The default is WienerPrior (a Gaussian / Matern 2-point prior), which is the
standard weak-lensing choice and reproduces the historical behaviour exactly.
The remaining priors capture NON-Gaussian structure that the 2-point Gaussian
cannot -- the regime where a data-driven or edge/sparsity model helps:

  WienerPrior       Gaussian / Matern:  phi = kappa^T (M + l^2 K) kappa   (default)
  TotalVariationPrior  edge-preserving:  phi = sum_e a_e |grad kappa|_e   (smoothed L1
                    of the gradient; piecewise-smooth maps, sharp cluster edges)
  SparsityPrior     compact-source L1:   phi = sum_i w_i |T kappa|_i      (smoothed;
                    T = identity for peaks/clusters a la GLIMPSE, or the graph of
                    a chosen linear operator)
  MaxEntropyPrior   positive maps:       phi = sum_i w_i [k ln(k/m) - k + m]
                    (Marshall et al. 2002 classic WL maximum-entropy)
  ScorePrior        learned / arbitrary: grad_phi = -score(kappa), where score is
                    grad log p(kappa) -- the hook for a neural non-Gaussian prior
                    (Remy et al. 2020), exploiting FEMMI's differentiable forward.

References
----------
Marshall, Hobson, Gull & Bridle 2002, MNRAS 335, 1037 (maximum entropy WL).
Lanusse, Starck, Leonard & Pires 2016, A&A 591, A2 (GLIMPSE, sparse/starlet).
Jeffrey et al. 2018, MNRAS 479, 2871 (sparsity + Gaussian, DES SV).
Remy et al. 2020, arXiv:2011.08271 (neural score prior; ScorePrior is the hook).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp


class Prior:
    """Base class. Subclasses implement value_grad(kappa) -> (phi, grad_phi)."""
    name = "prior"
    is_quadratic = False           # only quadratic priors support Morozov lambda-selection

    def value_grad(self, kappa):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _lumped_mass(ops):
    """Per-node integration weights w_i = (M 1)_i (lumped mass), all positive."""
    return np.asarray(ops.M @ np.ones(ops.n_nodes)).ravel()


def build_gradient_operator(ops):
    """Assemble the element-centroid gradient operator of the P3 mesh.

    Returns (Gx, Gy, area) where Gx, Gy are (n_elem x n_nodes) sparse matrices
    giving the x/y components of grad(kappa) at each element centroid, and area
    is the per-element area. Used by the total-variation / gradient-sparsity
    priors. Built once and cached on ops (`_grad_ops`)."""
    cached = getattr(ops, "_grad_ops", None)
    if cached is not None:
        return cached

    from .basis import compute_p3_shape_gradients_reference
    nodes    = np.asarray(ops.mesh.nodes)
    elements = np.asarray(ops.mesh.elements)
    n_nodes  = len(nodes)
    n_elem   = len(elements)
    dN_c = np.asarray(compute_p3_shape_gradients_reference(1.0 / 3.0, 1.0 / 3.0))  # (10,2)

    rows = np.repeat(np.arange(n_elem), 10)
    cols = elements.ravel()
    gx_d = np.zeros(n_elem * 10)
    gy_d = np.zeros(n_elem * 10)
    area = np.zeros(n_elem)
    for e, elem in enumerate(elements):
        xy  = nodes[elem[:3]]
        Jac = np.array([[xy[1, 0] - xy[0, 0], xy[1, 1] - xy[0, 1]],
                        [xy[2, 0] - xy[0, 0], xy[2, 1] - xy[0, 1]]])
        area[e] = abs(np.linalg.det(Jac)) / 2.0
        dN_phys = dN_c @ np.linalg.inv(Jac).T          # (10,2) physical gradients
        gx_d[e * 10:(e + 1) * 10] = dN_phys[:, 0]
        gy_d[e * 10:(e + 1) * 10] = dN_phys[:, 1]
    Gx = sp.coo_matrix((gx_d, (rows, cols)), shape=(n_elem, n_nodes)).tocsr()
    Gy = sp.coo_matrix((gy_d, (rows, cols)), shape=(n_elem, n_nodes)).tocsr()
    ops._grad_ops = (Gx, Gy, area)
    return ops._grad_ops


# --------------------------------------------------------------------------- #
# priors
# --------------------------------------------------------------------------- #
class WienerPrior(Prior):
    """Gaussian / Matern 2-point prior: phi = kappa^T R kappa, R = M + l^2 K.

    l = wiener_length is the Matern-1/2 correlation length; l = 0 recovers the
    H1 (gradient-energy) prior R = K. This is the default and reproduces the
    historical FEMMI regulariser exactly."""
    is_quadratic = True

    def __init__(self, ops, wiener_length=0.0):
        self.name = f"Wiener(l={wiener_length:.2f})" if wiener_length > 0 else "H1"
        self.wiener_length = float(wiener_length)
        if wiener_length > 0.0:
            self.R = (ops.M + wiener_length**2 * ops.K).tocsr()
        else:
            self.R = ops.K

    def value_grad(self, kappa):
        Rk = self.R @ kappa
        return float(np.dot(kappa, Rk)), 2.0 * Rk


class TotalVariationPrior(Prior):
    """Edge-preserving total variation: phi = sum_e a_e sqrt(|grad k|_e^2 + eps^2).

    The smoothed (Huber-like) TV of the reconstruction gradient. Promotes
    piecewise-smooth maps with sharp boundaries -- suited to compact clusters
    with steep edges, where a Gaussian prior over-smooths the core."""

    def __init__(self, ops, eps=1e-3):
        self.name = f"TV(eps={eps:g})"
        self.eps = float(eps)
        self.Gx, self.Gy, self.area = build_gradient_operator(ops)

    def value_grad(self, kappa):
        gx = self.Gx @ kappa
        gy = self.Gy @ kappa
        mag = np.sqrt(gx**2 + gy**2 + self.eps**2)
        phi = float(np.dot(self.area, mag))
        w = self.area / mag                       # d mag / d(gx,gy) weight
        grad = self.Gx.T @ (w * gx) + self.Gy.T @ (w * gy)
        return phi, grad


class SparsityPrior(Prior):
    """Smoothed-L1 (compact-source) sparsity: phi = sum_i w_i sqrt((T k)_i^2 + eps^2).

    transform='field' (default) penalises |kappa| directly -- a GLIMPSE-style
    prior favouring a few compact peaks/clusters on an empty background.
    transform='laplacian' penalises |K kappa|, a higher-order sparse prior that,
    like TV, favours piecewise-smooth structure. w_i are lumped-mass weights so
    the penalty is a proper surface integral on the irregular mesh."""

    def __init__(self, ops, transform="field", eps=1e-3):
        self.name = f"Sparse({transform},eps={eps:g})"
        self.eps = float(eps)
        self.transform = transform
        self.w = _lumped_mass(ops)
        if transform == "field":
            self.T = None
        elif transform == "laplacian":
            self.T = ops.K
        else:
            raise ValueError("transform must be 'field' or 'laplacian'")

    def value_grad(self, kappa):
        t = kappa if self.T is None else self.T @ kappa
        mag = np.sqrt(t**2 + self.eps**2)
        phi = float(np.dot(self.w, mag))
        d = self.w * t / mag
        grad = d if self.T is None else self.T.T @ d
        return phi, grad


class MaxEntropyPrior(Prior):
    """Maximum-entropy prior (Marshall et al. 2002): a positive-map cross-entropy
    relative to a flat model m,  phi = sum_i w_i [k ln(k/m) - k + m].

    Requires kappa > 0 (the convergence is clipped to `floor`), so it is intended
    for reconstructions of a positive mass field (e.g. a single cluster). w_i are
    lumped-mass integration weights."""

    def __init__(self, ops, model=1e-2, floor=1e-6):
        self.name = f"MaxEnt(m={model:g})"
        self.model = float(model)
        self.floor = float(floor)
        self.w = _lumped_mass(ops)

    def value_grad(self, kappa):
        k = np.clip(kappa, self.floor, None)
        phi = float(np.dot(self.w, k * np.log(k / self.model) - k + self.model))
        grad = self.w * np.log(k / self.model)
        grad[kappa <= self.floor] = 0.0          # clip is flat below the floor
        return phi, grad


class ScorePrior(Prior):
    """Arbitrary / learned prior defined by its score s(kappa) = grad log p(kappa).

    The MAP wants grad of phi = -log p, so grad_phi = -score(kappa). This is the
    hook for a neural non-Gaussian prior (Remy et al. 2020): pass a callable that
    returns the (differentiable) score field, optionally with neg_logp for the
    line-search value. Because FEMMI's forward is differentiable, the same score
    can later drive Langevin / HMC posterior sampling, not just MAP.

    score_fn   : kappa -> grad log p(kappa)  (array, same shape as kappa)
    neg_logp   : optional kappa -> -log p(kappa) (float). If None, phi is 0.0,
                 which is a valid gradient-consistent proxy for L-BFGS but makes
                 the reported loss omit the prior term."""

    def __init__(self, score_fn, neg_logp=None, name="score"):
        self.name = name
        self.score_fn = score_fn
        self.neg_logp = neg_logp

    def value_grad(self, kappa):
        grad = -np.asarray(self.score_fn(kappa)).ravel()
        phi = 0.0 if self.neg_logp is None else float(self.neg_logp(kappa))
        return phi, grad


def make_prior(kind, ops, **kw):
    """Factory: make_prior('wiener'|'tv'|'sparse'|'maxent', ops, **kw).

    'wiener' accepts wiener_length; 'tv' accepts eps; 'sparse' accepts transform,
    eps; 'maxent' accepts model, floor. ScorePrior is constructed directly (it
    needs a callable, not a keyword)."""
    kind = kind.lower()
    if kind in ("wiener", "gaussian", "matern", "h1"):
        return WienerPrior(ops, **kw)
    if kind in ("tv", "total_variation", "totalvariation"):
        return TotalVariationPrior(ops, **kw)
    if kind in ("sparse", "sparsity", "l1"):
        return SparsityPrior(ops, **kw)
    if kind in ("maxent", "maximum_entropy", "entropy"):
        return MaxEntropyPrior(ops, **kw)
    if kind in ("neural", "score", "nn"):
        from .neural_prior.prior import NeuralScorePrior   # lazy: needs flax/optax
        return NeuralScorePrior(ops, **kw)
    raise ValueError(f"unknown prior kind {kind!r}; use 'wiener', 'tv', 'sparse', "
                     "'maxent', or 'neural' (ScorePrior directly)")
