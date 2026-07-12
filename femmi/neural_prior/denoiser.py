"""
femmi/neural_prior/denoiser.py
A small noise-conditional U-Net (Flax) that models the score of the convergence
prior,  r_theta(x, sigma) ~ grad_x log p_sigma(x),  trained by Denoising Score
Matching (Remy et al. 2020, arXiv:2011.08271, eq. 3; Song & Ermon 2019).

Deliberately small so it trains in a couple of minutes on CPU with no external
data -- everything the neural prior needs ships with FEMMI.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn


def _sigma_embedding(log_sigma, dim):
    """Sinusoidal embedding of the (log) noise level, like a diffusion timestep."""
    freqs = jnp.exp(jnp.linspace(0.0, 4.0, dim // 2))
    ang = log_sigma[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class _Block(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x, semb):
        h = nn.Conv(self.ch, (3, 3))(x)
        # FiLM: shift the features by a projection of the noise embedding
        h = h + nn.Dense(self.ch)(semb)[:, None, None, :]
        h = nn.gelu(h)
        h = nn.Conv(self.ch, (3, 3))(h)
        h = nn.gelu(h)
        if x.shape[-1] == self.ch:
            h = h + x                                    # residual
        return h


class ScoreUNet(nn.Module):
    """3-scale U-Net. Input (N,H,W,1) map + (N,) noise level; output (N,H,W,1)
    score. Channels kept small (base=32) for fast CPU training."""
    base: int = 32
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x, sigma):
        semb = _sigma_embedding(jnp.log(sigma + 1e-8), self.emb_dim)
        semb = nn.gelu(nn.Dense(self.emb_dim)(semb))

        c = self.base
        h0 = _Block(c)(x, semb)
        h1 = _Block(2 * c)(nn.avg_pool(h0, (2, 2), (2, 2)), semb)
        h2 = _Block(4 * c)(nn.avg_pool(h1, (2, 2), (2, 2)), semb)

        def up(h):
            n, hh, ww, ch = h.shape
            return jax.image.resize(h, (n, hh * 2, ww * 2, ch), method="nearest")

        u1 = _Block(2 * c)(jnp.concatenate([up(h2), h1], axis=-1), semb)
        u0 = _Block(c)(jnp.concatenate([up(u1), h0], axis=-1), semb)
        # scale the raw output by 1/sigma: the DSM target u/sigma has that scale
        out = nn.Conv(1, (1, 1))(u0)
        return out / (sigma[:, None, None, None] + 1e-8)


def init_params(rng, n_pix, base=32):
    model = ScoreUNet(base=base)
    x = jnp.zeros((1, n_pix, n_pix, 1))
    s = jnp.ones((1,))
    params = model.init(rng, x, s)
    return model, params
