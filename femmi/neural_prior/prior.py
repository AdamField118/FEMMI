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


def _boundary_taper(nodes, frac):
    """Per-node weight in [0, 1] that -> 0 at the domain edge and -> 1 in the
    interior over a band of width `frac` x (domain size). The mesh->grid bridge is
    unreliable at the boundary -- edge grid pixels are extrapolated (nearest-fill)
    and the CNN has border effects -- so the learned score there injects spurious
    edge structure. Tapering the score to zero near the boundary removes that
    artifact; the reconstruction there is then driven by the (well-behaved) data
    term alone, exactly like the Wiener prior. frac<=0 disables (returns 1.0)."""
    if not frac or frac <= 0:
        return 1.0
    x = nodes[:, 0]; y = nodes[:, 1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    bx = frac * (x1 - x0) + 1e-12; by = frac * (y1 - y0) + 1e-12
    tx = np.clip(np.minimum(x - x0, x1 - x) / bx, 0.0, 1.0)
    ty = np.clip(np.minimum(y - y0, y1 - y) / by, 0.0, 1.0)
    ss = lambda t: t * t * (3.0 - 2.0 * t)              # smoothstep
    return ss(tx) * ss(ty)


class NeuralScorePrior(Prior):
    """Learned score prior. grad_phi(kappa) = -score_net(kappa), bridged mesh<->grid."""

    def __init__(self, ops, n_pix=32, base=16, sigma_eval=0.1, steps=8000,
                 ckpt=None, hybrid=False, boundary_taper=0.08,
                 train_data="synthetic", data_dir=None, map_glob=None, pool_maps=512,
                 verbose=True):
        # If a checkpoint path is given, the architecture is read from its name,
        # so you can point at a run by filename and the grid matches the model.
        if ckpt is not None:
            from .train import parse_ckpt_arch
            fn_pix, fn_base = parse_ckpt_arch(ckpt)
            if fn_pix is not None:
                n_pix, base = fn_pix, fn_base
        self.n_pix = n_pix
        self.sigma_eval = float(sigma_eval)
        self.model, self.params, ckpt_used = get_or_train(
            n_pix=n_pix, base=base, steps=steps, verbose=verbose, path=ckpt, hybrid=hybrid,
            train_data=train_data, data_dir=data_dir, map_glob=map_glob, pool_maps=pool_maps)

        # HYBRID prior (Remy et al. 2020 eq. 6): the network learns only the
        # non-Gaussian residual and the analytic Gaussian score p_th is added
        # back. Whether a checkpoint is hybrid is decided by its Gaussian sidecar
        # (not the requested flag), so pointing at a plain checkpoint always works.
        from .train import load_gauss_power
        power = load_gauss_power(ckpt_used)
        self.hybrid = power is not None
        if hybrid and not self.hybrid and verbose:
            print("  [warn] hybrid requested but checkpoint has no Gaussian sidecar; "
                  "using the full-score prior")
        self.name = f"Neural({'hybrid' if self.hybrid else 'score'},n_pix={n_pix})"
        nodes = np.asarray(ops.mesh.nodes)
        pad = 0.02 * (np.ptp(nodes[:, 0]) + np.ptp(nodes[:, 1]))
        self.extent = (nodes[:, 0].min() - pad, nodes[:, 0].max() + pad,
                       nodes[:, 1].min() - pad, nodes[:, 1].max() + pad)
        self.bin_op, self.gather = _binning_operators(nodes, n_pix, self.extent)
        # taper the prior score to zero near the domain boundary (see _boundary_taper)
        self.taper = _boundary_taper(nodes, boundary_taper)

        # PERFORMANCE: the score net is called tens of thousands of times inside
        # the sampler. Two things make that fast instead of a hang:
        #  (1) JIT the whole U-Net forward so it compiles ONCE and runs as fused
        #      kernels (not one eager op-dispatch per layer, per call), and
        #  (2) run it in float32. FEMMI enables x64 globally, which would make the
        #      convolutions run in float64 -- 2-8x slower on a GPU and double the
        #      memory. Score precision does not need x64.
        import jax
        self.params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32),
                                             self.params)
        _model, _params = self.model, self.params
        if self.hybrid:
            from .gaussian_score import GridGaussianScore
            self.gscore = GridGaussianScore(power)
            _g = self.gscore
            self._apply = jax.jit(lambda grid, sig:
                                  _model.apply(_params, grid, sig) + _g.score(grid, sig))
        else:
            self._apply = jax.jit(lambda grid, sig: _model.apply(_params, grid, sig))
        # warm up (pay the one-time compile now, not mid-sampling)
        z = jnp.zeros((1, n_pix, n_pix, 1), jnp.float32)
        self._apply(z, jnp.asarray([sigma_eval], jnp.float32)).block_until_ready()

    def score(self, kappa, sigma=None):
        """The learned score grad log p_sigma(kappa) at the mesh nodes.

        sigma is the noise level the score is conditioned on. None uses the
        default sigma_eval (for MAP); the annealed sampler passes the current
        annealing level so the network supplies the correctly-tempered prior
        score at each temperature."""
        s = self.sigma_eval if sigma is None else float(sigma)
        grid = np.asarray(self.bin_op @ kappa, np.float32).reshape(1, self.n_pix, self.n_pix, 1)
        s_grid = np.asarray(self._apply(jnp.asarray(grid),
                                        jnp.asarray([s], jnp.float32)))
        return self.taper * (self.gather @ s_grid.reshape(-1).astype(np.float64))

    def value_grad(self, kappa):
        # grad of phi = -log p is -score; phi itself has no closed form -> 0.0 proxy
        # (valid for the score-based sampler and for gradient descent; the reported
        # MAP loss then omits the prior term, as documented for ScorePrior).
        return 0.0, -np.asarray(self.score(kappa), dtype=np.float64)
