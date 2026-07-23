"""
femmi/neural_prior/massivenus.py
Train the neural prior on MassiveNuS convergence maps (Liu et al. 2018; Columbia
Lensing) -- the exact simulation suite used by Remy et al. 2020 -- instead of the
shipped synthetic shifted-log-normal maps, for an apples-to-apples reproduction.

MassiveNuS kappa maps are distributed by the Columbia Lensing group
(http://columbialensing.org). Download the fiducial galaxy-lensing maps
(`convergence_gal_mnv0.00000_om0.30000_As2.1000.tar`), extract one source
redshift (e.g. `Maps10/` = z_s=1, the paper's choice), and point `data_dir` at
that folder. This module reads .npy / .npz / .fits maps and serves random n_pix
patches in the SAME interface as data.batch_generator.

Scale note: a redshift folder holds ~10,000 maps. To stay memory-bounded and fast,
the loader globs the directory ONCE and holds a fixed random POOL of `pool_size`
maps in RAM (drawing patches from those), rather than caching every map it ever
touches. `map_glob` filters filenames (e.g. '*z1.00*') so you can point at a
folder that mixes redshifts and still train on one.
"""

from __future__ import annotations
import fnmatch
import glob
import os
import numpy as np

_EXTS = (".npy", ".npz", ".fits", ".fit", ".fits.gz")


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


def find_maps(data_dir, map_glob=None):
    """All convergence-map files under data_dir (recursive), optionally filtered by
    a filename glob such as '*z1.00*' (matched against the basename)."""
    files = []
    for ext in _EXTS:
        files += glob.glob(os.path.join(data_dir, "**", "*" + ext), recursive=True)
    if map_glob:
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), map_glob)]
    if not files:
        raise FileNotFoundError(
            f"no {'/'.join(_EXTS)} maps under {data_dir!r}"
            + (f" matching {map_glob!r}" if map_glob else "")
            + " -- download MassiveNuS maps from http://columbialensing.org")
    return sorted(files)


class MassiveNuSMaps:
    """A fixed, memory-bounded pool of convergence maps that serves random n_pix
    patches. The directory is globbed ONCE; up to `pool_size` maps are loaded into
    RAM (a random subset if there are more on disk) and every patch is drawn from
    that pool -- so memory is bounded to ~pool_size maps no matter how many are on
    disk, with no per-batch globbing or unbounded caching."""

    def __init__(self, data_dir, n_pix, kappa_std=0.35, map_glob=None,
                 pool_size=512, seed=0, verbose=False):
        files = find_maps(data_dir, map_glob)
        self.n_disk = len(files)
        rng = np.random.default_rng(seed)
        if len(files) > pool_size:
            files = [files[i] for i in sorted(rng.choice(len(files), pool_size, replace=False))]
        self.pool = [_load_map(f) for f in files]
        self.n_pix = int(n_pix)
        self.kappa_std = kappa_std
        if verbose:
            H, W = self.pool[0].shape[-2:]
            print(f"  MassiveNuS: pool of {len(self.pool)}/{self.n_disk} maps "
                  f"({H}x{W}) from {data_dir}")

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        out = np.empty((n, self.n_pix, self.n_pix), np.float32)
        for i in range(n):
            m = self.pool[int(rng.integers(len(self.pool)))]
            H, W = m.shape[-2:]
            if H < self.n_pix or W < self.n_pix:
                raise ValueError(f"map ({H}x{W}) smaller than n_pix={self.n_pix}")
            iy = int(rng.integers(0, H - self.n_pix + 1))
            ix = int(rng.integers(0, W - self.n_pix + 1))
            p = np.asarray(m[..., iy:iy + self.n_pix, ix:ix + self.n_pix], np.float32)
            p = p - p.mean()
            if self.kappa_std:
                p = p * (self.kappa_std / (p.std() + 1e-8))
            out[i] = p
        return out

    def generator(self, batch, seed=0):
        """Infinite stream of (batch, n_pix, n_pix, 1) patches -- drop-in for
        data.batch_generator."""
        s = seed
        while True:
            yield self.sample(batch, s)[..., None]
            s += 1


# --- thin backward-compatible helper ---------------------------------------- #
def massivenus_maps(data_dir, n, n_pix, kappa_std=0.35, seed=0, map_glob=None,
                    pool_size=512):
    """Return (n, n_pix, n_pix) random patches from the maps in data_dir."""
    return MassiveNuSMaps(data_dir, n_pix, kappa_std=kappa_std, map_glob=map_glob,
                          pool_size=pool_size, seed=seed).sample(n, seed)
