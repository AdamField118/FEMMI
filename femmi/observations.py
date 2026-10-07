"""Shared diagonal-noise observation conventions.

For each active position and shear component, Var(n_i) = noise_std**2 / w_i.
Zero weight means missing data. ``mask=True`` also means missing, never a
measurement of zero shear. Weights are dimensionless relative precisions;
their normalization and the supplied noise scale must be chosen together.
"""

import numpy as np


def observation_weights(n, data_weight=None, mask=None):
    w = np.ones(n) if data_weight is None else np.array(data_weight, dtype=float, copy=True)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("data_weight must be a finite nonnegative vector of observation length")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean vector of observation length")
        w[mask] = 0.0
    if not np.any(w > 0):
        raise ValueError("at least one observation must have positive weight")
    return w


def prepare_observations(g1, g2, n, data_weight=None, mask=None):
    """Validate active observations and replace missing placeholders safely."""
    w = observation_weights(n, data_weight, mask)
    out = []
    for g in (g1, g2):
        g = np.array(g, dtype=float, copy=True)
        if g.shape != (n,):
            raise ValueError("shear components must be vectors of observation length")
        if not np.all(np.isfinite(g[w > 0])):
            raise ValueError("active shear observations must be finite")
        g[w == 0] = 0.0
        out.append(g)
    return out[0], out[1], w


def weighted_rms(r1, r2, w):
    """RMS in whitened shear units, normalized by active component count."""
    active = w > 0
    a, b = np.asarray(r1)[active], np.asarray(r2)[active]
    return float(np.sqrt(np.sum(w[active] * (a*a + b*b)) / (2 * active.sum())))


def noise_scale(g1, g2, w):
    """MAD of whitened active components; signal can bias this estimate."""
    active = w > 0
    g = np.concatenate([np.sqrt(w[active]) * np.asarray(v)[active] for v in (g1, g2)])
    return 1.4826 * float(np.median(np.abs(g - np.median(g))))
