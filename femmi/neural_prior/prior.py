"""
femmi/neural_prior/prior.py
NeuralScorePrior: a learned non-Gaussian prior for the FEMMI MAP / sampler.

FEMMI reconstructs kappa on an irregular P3 mesh; the score network lives on a
regular grid (where convolutions make sense). This class is the bridge:

    kappa(nodes) --bin--> kappa(grid) --score net--> s(grid) --interp--> s(nodes)

and returns grad_phi = -s(nodes), so it plugs straight into MAPReconstructor and
the Langevin/HMC sampler exactly like any other Prior. It exposes the score, so
it is most natural with the score-based sampler (femmi.sampling), but also works
for MAP.

'One flag away': constructing it with no checkpoint trains a small default model
on synthetic non-Gaussian maps (train.get_or_train) and caches it -- no external
data, no extra steps.

Scope note: the shipped default is trained on synthetic shifted-log-normal fields
(data.py), a template of the mechanism, not a calibrated cosmological prior. The
learned score is amplitude/scale dependent; lambda absorbs the overall scaling.
Replace the checkpoint with one trained on real simulations (e.g. MassiveNu) for
science use -- the interface is unchanged.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp

from ..priors import Prior
from .train import get_or_train


def _binning_operators(nodes, n_pix, extent):
    """Sparse (bin, gather): bin averages nodes->grid; gather is bilinear grid->nodes."""
    x0, x1, y0, y1 = extent
    gx = (nodes[:, 0] - x0) / (x1 - x0 + 1e-12) * (n_pix - 1)
    gy = (nodes[:, 1] - y0) / (y1 - y0 + 1e-12) * (n_pix - 1)
    gx = np.clip(gx, 0, n_pix - 1 - 1e-6); gy = np.clip(gy, 0, n_pix - 1 - 1e-6)
    ix = np.floor(gx).astype(int); iy = np.floor(gy).astype(int)
    fx = gx - ix; fy = gy - iy
    n = len(nodes); npix2 = n_pix * n_pix

    def pix(i, j):
        return j * n_pix + i                        # row-major (iy, ix) -> flat

    # bilinear gather: 4 neighbours per node (interleaved per node)
    rows = np.repeat(np.arange(n), 4)
    cols = np.stack([pix(ix, iy), pix(np.minimum(ix + 1, n_pix - 1), iy),
                     pix(ix, np.minimum(iy + 1, n_pix - 1)),
                     pix(np.minimum(ix + 1, n_pix - 1), np.minimum(iy + 1, n_pix - 1))], axis=1).ravel()
    wts = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                    (1 - fx) * fy, fx * fy], axis=1).ravel()
    gather = sp.coo_matrix((wts, (rows, cols)), shape=(n, npix2)).tocsr()

    # bin = row-normalised transpose of gather: each node splats bilinearly onto
    # its 4 pixels, and every pixel is the weight-average of the nodes that reach
    # it. Consistent with the bilinear gather (so gather@bin ~ smoothing, exact on
    # linear fields where the grid is populated).
    splat = gather.T.tocsr()
    wsum = np.asarray(splat @ np.ones(n)).ravel()
    empty = wsum <= 1e-12
    inv = np.where(empty, 0.0, 1.0 / np.maximum(wsum, 1e-12))
    bin_op = sp.diags(inv) @ splat

    # fill operator for empty pixels: nearest-populated-pixel dilation, so the
    # score net sees a dense image instead of holes. Precomputed as a sparse map.
    fill = _nearest_fill_operator(n_pix, ~empty)
    return (fill @ bin_op).tocsr(), gather


def _nearest_fill_operator(n_pix, populated):
    """Sparse (npix2 x npix2) map: populated pixels pass through; empty pixels take
    the value of their nearest populated pixel (grid Chebyshev distance)."""
    npix2 = n_pix * n_pix
    pop_idx = np.where(populated)[0]
    rows = np.arange(npix2)
    cols = np.arange(npix2)
    if pop_idx.size and (~populated).any():
        py, px = np.divmod(pop_idx, n_pix)
        ey, ex = np.divmod(np.where(~populated)[0], n_pix)
        # nearest populated pixel for each empty pixel (small grids -> direct)
        d2 = (ex[:, None] - px[None, :])**2 + (ey[:, None] - py[None, :])**2
        nearest = pop_idx[np.argmin(d2, axis=1)]
        cols[~populated] = nearest
    return sp.coo_matrix((np.ones(npix2), (rows, cols)), shape=(npix2, npix2)).tocsr()


class NeuralScorePrior(Prior):
    """Learned score prior. grad_phi(kappa) = -score_net(kappa), bridged mesh<->grid."""

    def __init__(self, ops, n_pix=32, base=16, sigma_eval=0.1, steps=8000,
                 ckpt=None, verbose=True):
        # If a checkpoint path is given, the architecture is read from its name,
        # so you can point at a run by filename and the grid matches the model.
        if ckpt is not None:
            from .train import parse_ckpt_arch
            fn_pix, fn_base = parse_ckpt_arch(ckpt)
            if fn_pix is not None:
                n_pix, base = fn_pix, fn_base
        self.name = f"Neural(score,n_pix={n_pix})"
        self.n_pix = n_pix
        self.sigma_eval = float(sigma_eval)
        self.model, self.params = get_or_train(n_pix=n_pix, base=base, steps=steps,
                                               verbose=verbose, path=ckpt)
        nodes = np.asarray(ops.mesh.nodes)
        pad = 0.02 * (np.ptp(nodes[:, 0]) + np.ptp(nodes[:, 1]))
        self.extent = (nodes[:, 0].min() - pad, nodes[:, 0].max() + pad,
                       nodes[:, 1].min() - pad, nodes[:, 1].max() + pad)
        self.bin_op, self.gather = _binning_operators(nodes, n_pix, self.extent)
        self._sig = jnp.ones((1,)) * self.sigma_eval

    def score(self, kappa, sigma=None):
        """The learned score grad log p_sigma(kappa) at the mesh nodes.

        sigma is the noise level the score is conditioned on. None uses the
        default sigma_eval (for MAP); the annealed sampler passes the current
        annealing level so the network supplies the correctly-tempered prior
        score at each temperature."""
        sig = self._sig if sigma is None else jnp.ones((1,)) * float(sigma)
        grid = np.asarray(self.bin_op @ kappa).reshape(self.n_pix, self.n_pix)
        s_grid = np.asarray(self.model.apply(self.params,
                                             jnp.asarray(grid)[None, ..., None], sig))
        return self.gather @ s_grid.reshape(-1)

    def value_grad(self, kappa):
        # grad of phi = -log p is -score; phi itself has no closed form -> 0.0 proxy
        # (valid for the score-based sampler and for gradient descent; the reported
        # MAP loss then omits the prior term, as documented for ScorePrior).
        return 0.0, -np.asarray(self.score(kappa), dtype=np.float64)
