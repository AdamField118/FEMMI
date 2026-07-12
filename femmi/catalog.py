"""
femmi/catalog.py
Catalog-native mass reconstruction and an apples-to-apples Kaiser-Squires
(Fourier-gridding) comparison, both driven directly by a galaxy shear catalog.

Two reconstruction paths that each "do their own thing" with the same catalog:

  reconstruct_catalog(...)      FEMMI: nodes placed AT galaxy positions, FEM-BEM
                                MAP solve with the data term restricted to those
                                nodes (guard/boundary nodes carry no shear).
  kaiser_squires_binned(...)    KS/SMPy style: bin the catalog shear onto a
                                regular grid (weighted mean per pixel), optional
                                Gaussian smoothing, FFT inversion.

Plus analytic_gaussian_catalog(...), a self-consistent E-mode synthetic catalog
(Gaussian convergence with its exact tangential shear) for tests and demos --
independent of the FEMMI forward model, so it is a fair ground truth for both.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .operators      import build_operators_catalog
from .forward        import DifferentiableForward
from .inverse        import MAPReconstructor, ReconstructionResult
from .regularization import estimate_noise_level


# ---------------------------------------------------------------------------
# FEMMI catalog-native reconstruction
# ---------------------------------------------------------------------------

@dataclass
class CatalogReconstruction:
    kappa_nodes  : np.ndarray            # kappa at every mesh node
    kappa_gal    : np.ndarray            # kappa at each input galaxy (source order)
    ops          : object                # FEMOperators on the catalog mesh
    catalog_mesh : object                # CatalogMesh (nodes, galaxy_nodes, ...)
    data_weight  : np.ndarray            # per-node data weight actually used
    lam_reg      : float
    noise_std    : float
    result       : ReconstructionResult

    @property
    def nodes(self):
        return np.array(self.ops.mesh.nodes)


def reconstruct_catalog(x, y, g1, g2, weight=None, center=(0.0, 0.0),
                        radius=None, n_boundary=96, lam_reg=1e-2,
                        wiener_length=None, noise_std=None, noise_source='mad',
                        use_morozov=True, use_weights=False, maxiter=500,
                        prior=None, prior_kw=None, verbose=True, **mesh_kw):
    """
    Reconstruct kappa directly from a galaxy shear catalog (catalog-native).

    x, y      : galaxy positions in the flat frame (e.g. arcmin). For a
                FlatCatalog, pass flat.x, flat.y with center=(0., 0.).
    g1, g2    : observed shear on the (x, y) axes.
    weight    : optional per-galaxy weight (used only if use_weights=True).
    lam_reg   : regularisation strength (starting value; overridden by Morozov).
    wiener_length : Matern-1/2 prior length; defaults to 0.2 * ring radius.
    noise_std : per-component shear noise for Morozov. If None, estimated per
                noise_source.
    noise_source : how to estimate noise_std when it is None:
                'mad'   -> MAD on the galaxy shear (biased high by the signal,
                           which can saturate Morozov at lam_max);
                'bmode' -> the B-mode noise floor (delta_noise), signal-free and
                           usually smaller, giving a better-scaled Morozov solve
                           at the cost of one extra fixed-lambda E/B solve.
    prior     : optional non-Gaussian prior (femmi.priors). Either a Prior
                instance or a string kind ('tv', 'sparse', 'maxent') built here
                against the catalog mesh. None -> the default Wiener/Matern prior.
                Morozov lambda-selection applies only to the default Wiener prior;
                custom priors use the fixed lam_reg.
    prior_kw  : keyword dict forwarded to make_prior when `prior` is a string.
    use_morozov : select lambda automatically via the discrepancy principle.
    use_weights : fold per-galaxy inverse-variance weights into the data term.
                  Off by default (binary galaxy selection), which keeps noise_std
                  a plain per-component shear std.

    Returns
    -------
    CatalogReconstruction
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    g1 = np.asarray(g1, np.float64); g2 = np.asarray(g2, np.float64)

    ops, cm = build_operators_catalog(
        x, y, center=center, radius=radius, n_boundary=n_boundary,
        verbose=verbose, **mesh_kw)

    n  = ops.n_nodes
    gn = cm.galaxy_nodes
    si = cm.source_index

    g1n = np.zeros(n); g1n[gn] = g1[si]
    g2n = np.zeros(n); g2n[gn] = g2[si]

    dw = np.zeros(n)
    if use_weights and weight is not None:
        w = np.asarray(weight, np.float64)[si]
        dw[gn] = w / (np.mean(w) + 1e-30)
    else:
        dw[gn] = 1.0

    if wiener_length is None:
        wiener_length = 0.2 * cm.radius

    # Optional non-Gaussian prior: a string kind ('tv','sparse','maxent',...) is
    # built here now that ops exists; a Prior instance is used as-is. None -> the
    # default Wiener/Matern prior parameterised by wiener_length.
    prior_obj = prior
    if isinstance(prior, str):
        from .priors import make_prior
        prior_obj = make_prior(prior, ops, **(prior_kw or {}))

    fwd = DifferentiableForward(ops, lam_reg=lam_reg)
    rec = MAPReconstructor(
        fwd, maxiter=maxiter, gtol=1e-8, callback_every=0,
        wiener_length=wiener_length, data_weight=dw, prior=prior_obj)

    if noise_std is None:
        if noise_source == 'bmode':
            if verbose:
                print("Estimating noise from the B-mode floor (delta_noise)...")
            noise_std = rec.estimate_noise_bmode(
                g1n, g2n, maxiter=min(200, maxiter))
            if verbose:
                print(f"  delta_noise = {noise_std:.4e}")
        else:
            noise_std = estimate_noise_level(
                np.concatenate([g1n[gn], g2n[gn]]), method='mad')

    rec.noise_std = noise_std if use_morozov else None
    kappa, result = rec.reconstruct(g1n, g2n, verbose=verbose)

    kappa_gal = np.full(len(x), np.nan)
    kappa_gal[si] = kappa[gn]

    return CatalogReconstruction(
        kappa_nodes=kappa, kappa_gal=kappa_gal, ops=ops, catalog_mesh=cm,
        data_weight=dw, lam_reg=float(fwd.lam_reg), noise_std=float(noise_std),
        result=result)


# ---------------------------------------------------------------------------
# Kaiser-Squires on a Fourier grid (the SMPy-style path)
# ---------------------------------------------------------------------------

def bin_shear_to_grid(x, y, g1, g2, weight=None, grid_size=64, extent=None):
    """
    Bin a shear catalog onto a regular grid by weighted mean per pixel.

    Returns (g1_grid, g2_grid, counts, extent) where extent = (xmin,xmax,ymin,ymax).
    Empty pixels are zero (standard KS handling).
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    g1 = np.asarray(g1, np.float64); g2 = np.asarray(g2, np.float64)
    w  = np.ones_like(x) if weight is None else np.asarray(weight, np.float64)

    if extent is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = extent

    ix = np.clip(((x - xmin) / (xmax - xmin) * grid_size).astype(int), 0, grid_size - 1)
    iy = np.clip(((y - ymin) / (ymax - ymin) * grid_size).astype(int), 0, grid_size - 1)
    flat = iy * grid_size + ix
    npix = grid_size * grid_size

    wsum  = np.bincount(flat, weights=w,        minlength=npix)
    wg1   = np.bincount(flat, weights=w * g1,   minlength=npix)
    wg2   = np.bincount(flat, weights=w * g2,   minlength=npix)
    nz    = wsum > 0
    g1g   = np.zeros(npix); g2g = np.zeros(npix)
    g1g[nz] = wg1[nz] / wsum[nz]
    g2g[nz] = wg2[nz] / wsum[nz]

    return (g1g.reshape(grid_size, grid_size),
            g2g.reshape(grid_size, grid_size),
            wsum.reshape(grid_size, grid_size),
            (xmin, xmax, ymin, ymax))


def kaiser_squires_binned(x, y, g1, g2, weight=None, grid_size=64,
                          smoothing_px=1.0, extent=None, eval_pts=None,
                          return_bmode=False):
    """
    Catalog -> grid -> KS FFT convergence (the SMPy-style Fourier path).

    Bins the shear (bin_shear_to_grid), optionally Gaussian-smooths, inverts with
    the Kaiser-Squires kernel, and (if eval_pts given) samples kappa at those
    points by nearest-pixel lookup.

    Returns kappa on the grid (grid_size x grid_size) with extent, or, if
    eval_pts is given, kappa sampled at eval_pts. With return_bmode, also
    returns the B-mode map (KS on 45-deg-rotated shear).
    """
    import scipy.fft as sfft

    g1g, g2g, counts, ext = bin_shear_to_grid(
        x, y, g1, g2, weight=weight, grid_size=grid_size, extent=extent)
    xmin, xmax, ymin, ymax = ext

    if smoothing_px and smoothing_px > 0:
        g1g = _gaussian_smooth(g1g, smoothing_px)
        g2g = _gaussian_smooth(g2g, smoothing_px)

    G1 = sfft.fft2(g1g); G2 = sfft.fft2(g2g)
    kx = sfft.fftfreq(grid_size, d=(xmax - xmin) / grid_size) * 2 * np.pi
    ky = sfft.fftfreq(grid_size, d=(ymax - ymin) / grid_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2; k2[0, 0] = 1.0
    cos2 = (KX**2 - KY**2) / k2
    sin2 = 2.0 * KX * KY / k2

    kE = np.real(sfft.ifft2(cos2 * G1 + sin2 * G2))
    kB = np.real(sfft.ifft2(cos2 * G2 - sin2 * G1))

    def _sample(grid):
        if eval_pts is None:
            return grid
        ex = np.asarray(eval_pts, np.float64)
        ix = np.clip(((ex[:, 0] - xmin) / (xmax - xmin) * grid_size).astype(int), 0, grid_size - 1)
        iy = np.clip(((ex[:, 1] - ymin) / (ymax - ymin) * grid_size).astype(int), 0, grid_size - 1)
        return grid[iy, ix]

    kE_out = _sample(kE)
    if not return_bmode:
        return kE_out if eval_pts is not None else (kE, ext)
    kB_out = _sample(kB)
    if eval_pts is not None:
        return kE_out, kB_out
    return (kE, kB, ext)


def _gaussian_smooth(a, sigma_px):
    """Separable FFT Gaussian blur (avoids a scipy.ndimage dependency)."""
    import scipy.fft as sfft
    n = a.shape[0]
    ky = sfft.fftfreq(n)[:, None]; kx = sfft.fftfreq(n)[None, :]
    ker = np.exp(-2.0 * (np.pi * sigma_px)**2 * (kx**2 + ky**2))
    return np.real(sfft.ifft2(sfft.fft2(a) * ker))


# ---------------------------------------------------------------------------
# Analytic synthetic catalog (self-consistent E-mode ground truth)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Frontier Fields lens-model maps (CATS deliverables) -> shear field -> catalog
# ---------------------------------------------------------------------------

def _pixscale_arcsec_from_header(hdr, default=1.0):
    """Best-effort pixel scale (arcsec/pixel) from a FITS header."""
    for key in ("CDELT1", "CD1_1"):
        if key in hdr and float(hdr[key]) != 0.0:
            return abs(float(hdr[key])) * 3600.0     # deg -> arcsec
    if "CDELT1" in hdr:
        return abs(float(hdr["CDELT1"])) * 3600.0
    return float(default)


def _shear_from_kappa_fft(kappa):
    """Exact E-mode shear (g1, g2) of a convergence map via the KS forward FFT."""
    import scipy.fft as sfft
    ny, nx = kappa.shape
    K = sfft.fft2(kappa)
    ky = sfft.fftfreq(ny)[:, None]; kx = sfft.fftfreq(nx)[None, :]
    k2 = kx**2 + ky**2; k2[0, 0] = 1.0
    G = ((kx**2 - ky**2) + 2j * kx * ky) / k2 * K
    g = sfft.ifft2(G)
    return np.real(g), np.imag(g)


def load_frontier_model(data_dir, source="psi", pixscale_arcsec=None,
                        downsample=4, verbose=True):
    """
    Load a Frontier Fields / CATS lens-MODEL map set and build a shear field
    with a known convergence ground truth. These are FITS *images* (kappa, psi,
    gamma, ...), not a galaxy catalog.

    source='psi'    : derive (g1, g2) from the lensing potential psi via its
                      Hessian, self-calibrated to the provided kappa map (the
                      unknown pixel-to-angle scale cancels in kappa/kappa_pix).
    source='deflect': derive (g1, g2) from the deflection field alpha = grad psi
                      (the *-arcsec-deflect maps) via first derivatives. This is
                      an independent, less noise-amplifying cross-check on the
                      psi-Hessian shear (one differentiation instead of two).
    source='kappa'  : synthesize the exact E-mode shear of the kappa map by FFT
                      (use if psi is missing; imposes periodic boundaries).

    downsample      : integer stride to shrink the (often huge) native maps.

    Returns dict: X, Y (arcmin grids, centred), g1, g2, kappa_true, plus meta.
    """
    from astropy.io import fits
    import glob

    def _find(kind):
        cands = sorted(glob.glob(os.path.join(data_dir, f"*_{kind}.fits")))
        best  = [c for c in cands if "-map" not in os.path.basename(c)]
        return (best or cands or [None])[0]

    kfile = _find("kappa")
    if kfile is None:
        raise FileNotFoundError(f"no *_kappa.fits in {data_dir}")
    kappa = np.asarray(fits.getdata(kfile), dtype=np.float64)
    hdr   = fits.getheader(kfile)
    if pixscale_arcsec is None:
        pixscale_arcsec = _pixscale_arcsec_from_header(hdr, default=1.0)

    d = max(1, int(downsample))
    kappa = kappa[::d, ::d]
    pix_arcmin = pixscale_arcsec * d / 60.0

    def _selfcal(kap_pix, g1_pix, g2_pix):
        # unknown pixel-to-angle scale cancels via kappa/kappa_pix on the core
        hi = kappa > np.percentile(kappa, 90)
        scale = float(np.median(kappa[hi] / (kap_pix[hi] + 1e-30)))
        return scale * g1_pix, scale * g2_pix, scale

    pfile = _find("psi")
    ax_file = _find("x-arcsec-deflect")
    ay_file = _find("y-arcsec-deflect")

    if source == "psi" and pfile is not None:
        psi = np.asarray(fits.getdata(pfile), dtype=np.float64)[::d, ::d]
        # Hessian in pixel units (array axis 0 = y/row, axis 1 = x/col)
        py, px   = np.gradient(psi)
        pyy, _   = np.gradient(py)
        pxy, pxx = np.gradient(px)
        g1, g2, scale = _selfcal(0.5 * (pxx + pyy), 0.5 * (pxx - pyy), pxy)
        used = f"psi Hessian (kappa self-calibration scale={scale:.3g})"
    elif source == "deflect" and ax_file is not None and ay_file is not None:
        ax = np.asarray(fits.getdata(ax_file), dtype=np.float64)[::d, ::d]
        ay = np.asarray(fits.getdata(ay_file), dtype=np.float64)[::d, ::d]
        # alpha = grad psi; first derivatives give the Hessian components
        axy, axx = np.gradient(ax)          # d alpha_x /dy, /dx
        ayy, ayx = np.gradient(ay)          # d alpha_y /dy, /dx
        kap_pix  = 0.5 * (axx + ayy)
        g1_pix   = 0.5 * (axx - ayy)
        g2_pix   = 0.5 * (axy + ayx)        # both approximate psi_xy; average
        g1, g2, scale = _selfcal(kap_pix, g1_pix, g2_pix)
        used = f"deflection grad (kappa self-calibration scale={scale:.3g})"
    else:
        g1, g2 = _shear_from_kappa_fft(kappa)
        used = "kappa FFT forward (periodic)"

    ny, nx = kappa.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    X = (xs - nx / 2.0) * pix_arcmin
    Y = (ys - ny / 2.0) * pix_arcmin

    if verbose:
        print(f"  frontier model: {os.path.basename(kfile)}  {ny}x{nx} "
              f"(downsample {d}), pixscale={pixscale_arcsec:.3g}\"/pix, "
              f"field ~{nx*pix_arcmin:.1f}x{ny*pix_arcmin:.1f} arcmin")
        print(f"  shear from {used};  kappa range [{kappa.min():.3f}, {kappa.max():.3f}]")

    return dict(X=X, Y=Y, g1=g1, g2=g2, kappa_true=kappa,
                pixscale_arcsec=pixscale_arcsec, downsample=d,
                name=os.path.basename(kfile))


def field_to_catalog(field, n_gal=3000, shape_noise=0.05, rmax_arcmin=None,
                     kappa_max=1.0, reduced_shear=False, seed=0):
    """
    Sample a gridded shear field (from load_frontier_model) at random points to
    emulate a galaxy catalog: irregular positions, interpolated shear, optional
    shape noise, and the local convergence truth. Returns a dict compatible with
    reconstruct_catalog / run_head_to_head (x, y, g1, g2, weight, kappa_true).

    kappa_max     : drop galaxies where the model kappa exceeds this (the
        strong-lensing / multiple-image core, where the weak-shear approximation
        that both FEMMI and KS assume is invalid). None keeps everything.
        Real background sources are also not observed through the cluster core.
    reduced_shear : emit the observable reduced shear g = gamma / (1 - kappa)
        instead of gamma. More physical, but only meaningful once the kappa>~1
        core is excluded (it diverges at kappa = 1).
    rmax_arcmin   : also restrict to a central radius.
    """
    rng = np.random.default_rng(seed)
    X, Y = field["X"].ravel(), field["Y"].ravel()
    g1, g2 = field["g1"].ravel(), field["g2"].ravel()
    kap = field["kappa_true"].ravel()

    keep = np.isfinite(g1) & np.isfinite(g2) & np.isfinite(kap)

    if kappa_max is not None:
        keep &= kap < kappa_max
    if rmax_arcmin is not None:
        keep &= np.hypot(X, Y) <= rmax_arcmin
    idx_pool = np.where(keep)[0]
    take = rng.choice(idx_pool, size=min(n_gal, idx_pool.size), replace=False)

    x, y = X[take], Y[take]
    e1, e2 = g1[take].copy(), g2[take].copy()

    kt = kap[take]
    if reduced_shear:
        denom = np.clip(1.0 - kt, 1e-3, None)
        e1, e2 = e1 / denom, e2 / denom
    if shape_noise > 0:
        e1 += rng.normal(0, shape_noise, x.size)
        e2 += rng.normal(0, shape_noise, x.size)

    return dict(x=x, y=y, g1=e1, g2=e2, weight=np.ones(x.size),
                kappa_true=kt, center=(0.0, 0.0),
                field_radius=float(np.hypot(x, y).max()))


def analytic_gaussian_shear(points, sigma=0.4, amp=1.0, center=(0.0, 0.0)):
    """
    Exact convergence and (infinite-domain) shear of a Gaussian lens at the
    given points. kappa(r) = amp*exp(-r^2/2sigma^2); the tangential shear is
    gamma_t = kappabar(<r) - kappa(r) with the correct far-field decay, so this
    is the right ground-truth input for comparing boundary conditions.

    Returns (kappa, g1, g2) arrays aligned with `points`.
    """
    p  = np.asarray(points, np.float64)
    dx = p[:, 0] - center[0]
    dy = p[:, 1] - center[1]
    r2 = dx**2 + dy**2
    phi = np.arctan2(dy, dx)
    kappa = amp * np.exp(-r2 / (2 * sigma**2))
    kbar = np.where(r2 > 1e-12,
                    2 * amp * sigma**2 / np.where(r2 > 1e-12, r2, 1.0)
                    * (1 - np.exp(-r2 / (2 * sigma**2))),
                    amp)
    gamma_t = kbar - kappa
    return kappa, -gamma_t * np.cos(2 * phi), -gamma_t * np.sin(2 * phi)


def analytic_gaussian_catalog(n_gal=1500, sigma=0.5, amp=1.0, field_radius=2.5,
                              shape_noise=0.0, center=(0.0, 0.0), seed=0):
    """
    A galaxy catalog whose shear is the EXACT reduced-shear-free tangential
    shear of a Gaussian convergence kappa(r) = amp * exp(-r^2 / 2 sigma^2).

    For an axisymmetric kappa the tangential shear is
        gamma_t(r) = kappabar(<r) - kappa(r),
        kappabar(<r) = (2 amp sigma^2 / r^2) (1 - exp(-r^2 / 2 sigma^2)),
    and (g1, g2) = -gamma_t (cos 2phi, sin 2phi). This is a pure E-mode field
    built WITHOUT the FEMMI operator, so it is a fair truth for both methods.

    Returns dict: x, y, g1, g2, weight, kappa_true (at the galaxies), and the
    callable kappa_of_r for evaluating truth anywhere.
    """
    rng = np.random.default_rng(seed)
    cx, cy = center

    # uniform galaxies in a disk
    th = rng.uniform(0, 2 * np.pi, n_gal)
    rr = field_radius * np.sqrt(rng.uniform(0, 1, n_gal))
    x = cx + rr * np.cos(th)
    y = cy + rr * np.sin(th)

    dx, dy = x - cx, y - cy
    r   = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    r2  = r**2

    def kappa_of_r(rq):
        return amp * np.exp(-rq**2 / (2 * sigma**2))

    kappa_true = kappa_of_r(r)
    kbar = np.where(r2 > 1e-12,
                    2 * amp * sigma**2 / np.where(r2 > 1e-12, r2, 1.0)
                    * (1 - np.exp(-r2 / (2 * sigma**2))),
                    amp)  # limit kbar(0) = amp
    gamma_t = kbar - kappa_true
    g1 = -gamma_t * np.cos(2 * phi)
    g2 = -gamma_t * np.sin(2 * phi)

    if shape_noise > 0:
        g1 = g1 + rng.normal(0, shape_noise, n_gal)
        g2 = g2 + rng.normal(0, shape_noise, n_gal)

    return dict(x=x, y=y, g1=g1, g2=g2, weight=np.ones(n_gal),
                kappa_true=kappa_true, kappa_of_r=kappa_of_r,
                center=center, field_radius=field_radius)
