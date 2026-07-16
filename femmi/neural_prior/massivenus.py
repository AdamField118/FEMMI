"""
femmi/neural_prior/massivenus.py
Train the neural prior on MassiveNuS convergence maps (Liu et al. 2018; Columbia
Lensing) -- the exact simulation suite used by Remy et al. 2020 -- instead of the
shipped synthetic shifted-log-normal maps, for an apples-to-apples reproduction.

MassiveNuS kappa maps are distributed by the Columbia Lensing group
(http://columbialensing.org). Download the convergence maps you want (e.g. the
massless-neutrino fiducial cosmology at a source redshift) into a directory and
point `data_dir` at it. This module reads .npy / .npz / .fits maps and serves
random n_pix patches in the SAME interface as data.batch_generator, so training
is `femmi train-prior ... --set prior.neural.train_data=massivenus
--set prior.neural.data_dir=/path/to/maps`.
"""

from __future__ import annotations
import glob
import os
import numpy as np


def _load_map(path):
    if path.endswith(".npy"):
        a = np.load(path)
    elif path.endswith(".npz"):
        d = np.load(path)
        a = d["kappa"] if "kappa" in d.files else d[d.files[0]]
    elif path.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.io import fits          # optional dep (extras: io / paper)
        with fits.open(path) as h:
            a = next(hdu.data for hdu in h if hdu.data is not None)
    else:
        raise ValueError(f"unsupported convergence-map file: {path}")
    return np.asarray(a, np.float32)


def find_maps(data_dir):
    """All convergence-map files under data_dir (recursive)."""
    files = []
    for ext in ("*.npy", "*.npz", "*.fits", "*.fit", "*.fits.gz"):
        files += glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"no .npy/.npz/.fits convergence maps under {data_dir!r} -- download "
            "MassiveNuS maps from http://columbialensing.org")
    return sorted(files)


def massivenus_maps(data_dir, n, n_pix, kappa_std=0.35, seed=0, _cache=None):
    """Return (n, n_pix, n_pix) random patches from the MassiveNuS maps in data_dir.

    Each patch is zero-meaned; if kappa_std is set it is renormalised to that
    per-patch std (matching the synthetic-data amplitude convention), else the
    native amplitude is kept."""
    files = find_maps(data_dir)
    rng = np.random.default_rng(seed)
    cache = {} if _cache is None else _cache
    out = np.empty((n, n_pix, n_pix), np.float32)
    for i in range(n):
        f = files[rng.integers(len(files))]
        m = cache.get(f)
        if m is None:
            m = _load_map(f); cache[f] = m
        H, W = m.shape[-2:]
        if H < n_pix or W < n_pix:
            raise ValueError(f"map {f} ({H}x{W}) is smaller than n_pix={n_pix}")
        iy = int(rng.integers(0, H - n_pix + 1)); ix = int(rng.integers(0, W - n_pix + 1))
        patch = np.asarray(m[..., iy:iy + n_pix, ix:ix + n_pix], np.float32)
        patch = patch - patch.mean()
        if kappa_std:
            patch = patch * (kappa_std / (patch.std() + 1e-8))
        out[i] = patch
    return out


def massivenus_batch_generator(data_dir, batch=32, n_pix=64, kappa_std=0.35, seed=0):
    """Infinite stream of (batch, n_pix, n_pix, 1) patches -- drop-in for
    data.batch_generator, so the DSM trainer is unchanged."""
    find_maps(data_dir)                       # fail fast if the directory is empty
    cache: dict = {}
    s = seed
    while True:
        yield massivenus_maps(data_dir, batch, n_pix, kappa_std=kappa_std,
                              seed=s, _cache=cache)[..., None]
        s += 1
