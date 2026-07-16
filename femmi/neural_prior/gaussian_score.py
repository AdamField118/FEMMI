"""
femmi/neural_prior/gaussian_score.py
Analytic Gaussian ("theory") prior score on the regular grid -- the p_th term of
the HYBRID neural prior (Remy et al. 2020 eq. 6):

    grad log p(kappa) = grad log p_th(kappa) + r_theta(kappa, sigma)

p_th is a stationary Gaussian with the SAME 2-point statistics (power spectrum)
as the training field, so the network r_theta only has to learn the non-Gaussian
residual on top of it. For a Gaussian field with power spectrum P(k), the score
at noise level sigma is diagonal in Fourier:

    grad log p_sigma(x) = -F^{-1}[ F[x] / (P(k) + sigma^2) ]

We use the orthonormal FFT so the signal power P(k) and the noise power sigma^2
are both "per mode" and directly comparable (Parseval). P(k) is estimated once
from a batch of training maps and stored alongside the checkpoint.
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp


class GridGaussianScore:
    """Fixed (non-trainable) analytic Gaussian score on an n_pix x n_pix grid."""

    def __init__(self, power):
        self.power = jnp.asarray(np.asarray(power, np.float32))     # (n_pix, n_pix)
        self.n_pix = int(self.power.shape[0])

    @staticmethod
    def power_from_maps(maps):
        """Sample power spectrum P(k) (orthonormal FFT) averaged over a batch.

        maps: (n, n_pix, n_pix) real. Returns (n_pix, n_pix) float32."""
        m = np.asarray(maps, np.float64)
        F = np.fft.fft2(m, norm="ortho")
        return (np.abs(F) ** 2).mean(0).astype(np.float32)

    def score(self, x, sigma):
        """grad log p_sigma(x) on the grid.

        x: (B, n_pix, n_pix, 1); sigma: scalar or (B,). Returns (B, n_pix, n_pix, 1)."""
        xg = x[..., 0]                                             # (B, H, W)
        sig = jnp.asarray(sigma, xg.dtype).reshape(-1, 1, 1)       # (B, 1, 1)
        X = jnp.fft.fft2(xg, norm="ortho")
        S = -X / (self.power[None] + sig ** 2)
        s = jnp.real(jnp.fft.ifft2(S, norm="ortho"))
        return s[..., None].astype(xg.dtype)
