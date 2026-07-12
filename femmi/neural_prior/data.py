"""
femmi/neural_prior/data.py
Self-contained synthetic training maps for the neural prior. No external data:
we draw non-Gaussian convergence-like fields as SHIFTED LOG-NORMAL random fields,
the standard simple non-Gaussian model for weak-lensing convergence (Clerkin et
al. 2017; Hilbert et al. 2011). A Gaussian random field with a power-law power
spectrum is exponentiated, giving the skewed, peaked one-point statistics (bright
compact peaks on a near-empty background) that a Gaussian 2-point prior cannot
capture -- exactly the structure the neural prior should learn.
"""

from __future__ import annotations
import numpy as np


def lognormal_kappa_maps(n, n_pix, slope=2.5, sigma_g=0.9, kappa_std=0.35,
                         seed=0):
    """Draw `n` non-Gaussian kappa maps of size (n_pix, n_pix).

    slope    : power-law index of the Gaussian field power spectrum P(k) ~ k^-slope
    sigma_g  : std of the underlying Gaussian field (controls non-Gaussianity)
    kappa_std: target per-map std of the output kappa (amplitude normalisation)
    Returns (n, n_pix, n_pix) float32.
    """
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n_pix)[:, None]
    kx = np.fft.fftfreq(n_pix)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    pk = k**(-slope)
    pk[0, 0] = 0.0                                   # zero mean
    amp = np.sqrt(pk)

    out = np.empty((n, n_pix, n_pix), np.float32)
    for i in range(n):
        wr = rng.standard_normal((n_pix, n_pix))
        wi = rng.standard_normal((n_pix, n_pix))
        field = np.fft.ifft2((wr + 1j * wi) * amp).real
        field *= sigma_g / (field.std() + 1e-8)      # set Gaussian field std
        kap = np.exp(field - 0.5 * sigma_g**2) - 1.0 # shifted log-normal, zero mean
        kap *= kappa_std / (kap.std() + 1e-8)
        out[i] = kap.astype(np.float32)
    return out


def batch_generator(n_pix, batch=32, slope=2.5, sigma_g=0.9, kappa_std=0.35,
                    seed=0):
    """Infinite stream of fresh synthetic map batches (never repeats -> no
    overfitting a fixed dataset). Yields (batch, n_pix, n_pix, 1) float32."""
    s = seed
    while True:
        maps = lognormal_kappa_maps(batch, n_pix, slope, sigma_g, kappa_std, seed=s)
        s += 1
        yield maps[..., None]
