"""
femmi/truth.py
INDEPENDENT ground-truth convergence/shear fields -- truth that is generated
without ever touching FEMMI's forward operator, and without the periodic FFT
that Kaiser-Squires inverts.

Why this module exists
----------------------
The headline FEMMI claim is that it recovers the absolute convergence
normalisation (the DC / mass-sheet mode) that KS structurally cannot. If the
test shear is produced by FEMMI's OWN forward (`experiments.femmi_forward_shear`)
that claim is an inverse crime: FEMMI is being asked to invert an operator it
also generated the data with, and a referee will say so immediately. Likewise,
generating truth with the periodic KS FFT is an inverse crime in KS's favour --
and worse, that forward annihilates the DC mode outright, so the very quantity
under test is destroyed before either method sees it.

Both generators here are neutral third parties:

  galsim_nfw_truth      exact ANALYTIC convergence and shear of one or more NFW
                        halos, evaluated in closed form by GalSim. No mesh, no
                        grid, no FFT -- there is no discretisation in the truth
                        at all. This is the primary independent truth.

  massivenus_truth      a real MassiveNuS simulated convergence map (Liu et al.
                        2018), with its shear obtained by APERIODIC real-space
                        convolution with the continuum lensing kernel
                        (`aperiodic_shear_from_kappa`) -- not the periodic FFT.

Convention note: FEMMI's shear sign convention is gamma1 = 1/2(psi_xx - psi_yy),
gamma2 = psi_xy, which for an axisymmetric lens gives (g1, g2) =
-gamma_t (cos 2phi, sin 2phi) -- the standard weak-lensing convention, and the
same one `catalog.analytic_gaussian_shear` uses. GalSim's `getShear` already
matches it, so no sign flip is applied anywhere in this module. This is checked
directly in tests/test_truth.py rather than taken on faith.
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# analytic NFW truth (GalSim) -- no discretisation at all
# --------------------------------------------------------------------------- #
def galsim_nfw_truth(nodes, halos=((2.0e14, 4.0, (0.0, 0.0)),), z_l=0.3, z_s=1.0,
                     omega_m=0.3, omega_lam=0.7, arcmin_per_unit=1.0):
    """Exact analytic NFW convergence and shear at `nodes`, from GalSim.

    nodes  : (N, 2) positions in mesh units (interpreted as arcmin by default;
             set arcmin_per_unit to rescale).
    halos  : iterable of (mass_Msun, concentration, (cx, cy)) in mesh units.
             Several halos superpose linearly in the weak-lensing regime, which
             is the regime both FEMMI and KS assume, so summing kappa and gamma
             over halos is exact here.

    Returns (kappa, g1, g2) aligned with `nodes`.

    This is the cleanest possible independent truth: closed-form, continuum, and
    produced by neither method's forward model.
    """
    import warnings
    import galsim

    p = np.asarray(nodes, np.float64)
    kappa = np.zeros(len(p)); g1 = np.zeros(len(p)); g2 = np.zeros(len(p))
    scale = arcmin_per_unit * 60.0                     # mesh units -> arcsec

    for mass, conc, (cx, cy) in halos:
        hx, hy = float(cx) * scale, float(cy) * scale
        h = galsim.NFWHalo(mass=float(mass), conc=float(conc), redshift=float(z_l),
                           omega_m=omega_m, omega_lam=omega_lam,
                           halo_pos=galsim.PositionD(hx, hy))
        # GalSim takes arcsec; positions are vectorised through its array API.
        xs = p[:, 0] * scale
        ys = p[:, 1] * scale
        # NFW kappa/gamma are singular exactly at the halo centre (and GalSim
        # reads uninitialised memory there); nudge any coincident node by a
        # micro-arcsec, far below any mesh scale, so the field stays finite.
        coincident = (xs - hx) ** 2 + (ys - hy) ** 2 == 0.0
        if np.any(coincident):
            xs = np.where(coincident, xs + 1e-6, xs)
        with warnings.catch_warnings():
            # GalSim calls np.divide(..., where=...) without out=, which numpy
            # always warns about; the nudge above is what actually keeps the
            # centre finite.
            warnings.filterwarnings("ignore", message=".*where.*without.*out.*")
            kappa += np.asarray(h.getConvergence((xs, ys), z_s=z_s), np.float64)
            s1, s2 = h.getShear((xs, ys), z_s=z_s, reduced=False)
        g1 += np.asarray(s1, np.float64); g2 += np.asarray(s2, np.float64)

    return kappa, g1, g2


# --------------------------------------------------------------------------- #
# aperiodic (infinite-domain) shear of a gridded convergence map
# --------------------------------------------------------------------------- #
def aperiodic_shear_from_kappa(kappa, pix=1.0):
    """Infinite-domain E-mode shear of a convergence map, by zero-padded
    real-space convolution with the continuum lensing kernel

        gamma(theta) = (1/pi) \\int D(theta - theta') kappa(theta') d^2theta',
        D(theta)     = -(theta_1 + i theta_2)^2 / |theta|^4.

    Unlike the Kaiser-Squires FFT this imposes NO periodicity: the map is padded
    to twice its size so the convolution is genuinely aperiodic, and kappa is
    treated as zero outside the map (an isolated field). That makes it a valid
    independent truth for testing an inverter's boundary behaviour, which the
    periodic forward cannot be.

    kappa : (n, n) convergence map.
    pix   : pixel size in the same angular units the mesh uses.

    Returns (g1, g2) maps with the same shape as `kappa`.
    """
    k = np.asarray(kappa, np.float64)
    ny, nx = k.shape
    py, px = 2 * ny, 2 * nx

    # kernel sampled on the padded grid, centred so that FFT wraparound puts the
    # singular point at index 0 (where the kernel is set to zero)
    iy = np.fft.fftfreq(py, d=1.0 / py)[:, None]      # ..., -2, -1, 0, 1, 2, ...
    ix = np.fft.fftfreq(px, d=1.0 / px)[None, :]
    t1 = ix * pix
    t2 = iy * pix
    r2 = t1**2 + t2**2
    r2_safe = np.where(r2 > 0, r2, 1.0)
    D = -((t1 + 1j * t2) ** 2) / r2_safe**2
    D[r2 == 0] = 0.0                                   # self-pixel: zero by symmetry

    kp = np.zeros((py, px)); kp[:ny, :nx] = k
    g = np.fft.ifft2(np.fft.fft2(D) * np.fft.fft2(kp)) * (pix**2 / np.pi)
    g = g[:ny, :nx]
    return np.real(g), np.imag(g)


def massivenus_truth(nodes, data_dir, half_width, n_pix=256, kappa_std=None,
                     map_glob=None, seed=0, subtract_mean=False):
    """A real MassiveNuS simulated convergence patch as ground truth, with its
    APERIODIC shear (see aperiodic_shear_from_kappa) sampled at `nodes`.

    nodes        : (N, 2) mesh nodes, assumed to span [-half_width, half_width]^2.
    data_dir     : folder of MassiveNuS convergence maps (see neural_prior.massivenus).
    n_pix        : patch size drawn from the simulation maps.
    kappa_std    : if set, rescale the patch to this std (the loader's default
                   behaviour also removes the patch mean, which would destroy the
                   very DC mode under test -- so we restore it here unless
                   subtract_mean=True).
    subtract_mean: leave the patch mean-subtracted (kills the mass-sheet signal;
                   only useful for shape-only comparisons).

    Returns (kappa, g1, g2) at the nodes.
    """
    from .neural_prior.massivenus import MassiveNuSMaps
    from .catalog import _grid_to_nodes_bilinear

    pool = MassiveNuSMaps(data_dir, n_pix, kappa_std=(kappa_std or 0.0),
                          map_glob=map_glob, pool_size=32, seed=seed)
    patch = np.asarray(pool.sample(1, seed)[0], np.float64)

    if not subtract_mean:
        # MassiveNuSMaps.sample removes the patch mean; a mass-sheet test needs a
        # genuine nonzero mean, so put a realistic one back (the simulation's own
        # patch-to-patch mean is what KS cannot see).
        patch = patch - patch.min()

    hw = float(half_width)
    pix = 2.0 * hw / n_pix
    g1g, g2g = aperiodic_shear_from_kappa(patch, pix=pix)

    ext = (-hw, hw, -hw, hw)
    nodes = np.asarray(nodes, np.float64)
    return (_grid_to_nodes_bilinear(patch, nodes, ext),
            _grid_to_nodes_bilinear(g1g, nodes, ext),
            _grid_to_nodes_bilinear(g2g, nodes, ext))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def lognormal_truth(nodes, half_width, n_pix=256, kappa_std=0.35, slope=2.5,
                    sigma_g=0.9, seed=0, shift=True, taper=0.65):
    """A shifted-log-normal convergence field with APERIODIC shear -- a
    non-Gaussian independent truth.

    catalog.lognormal_shear already builds this field but takes its shear from
    FEMMI's own forward, which is an inverse crime. Here the shear comes from the
    continuum kernel (aperiodic_shear_from_kappa), so neither method generated it.

    Why it matters for the benchmark: a single smooth NFW halo is precisely the
    case a Gaussian/Wiener prior is designed to win, so ranking priors on it says
    little. Log-normal fields are peaked and non-Gaussian -- the regime where TV,
    sparsity and learned priors are supposed to pay off -- so this is the fair
    field for discriminating between them.
    """
    from .neural_prior.data import lognormal_kappa_maps      # pure numpy
    from .catalog import _grid_to_nodes_bilinear

    m = lognormal_kappa_maps(1, n_pix, slope=slope, sigma_g=sigma_g,
                             kappa_std=kappa_std, seed=int(seed))[0]
    patch = np.asarray(m, np.float64)
    if shift:
        # give it a genuine nonzero mean, as a real convergence field has
        patch = patch - patch.min()

    hw = float(half_width)
    if taper:
        # A raw log-normal map fills the whole box, so kappa is NOT zero at the
        # boundary and FEMMI's isolated-field (far-field-zero) assumption is
        # violated outright -- Morozov then drives lambda to its ceiling and every
        # prior collapses to the same over-smoothed answer, which says nothing
        # about the priors. A smooth radial taper to compact support keeps the
        # peaked, non-Gaussian structure that actually discriminates priors while
        # respecting the assumption both methods are entitled to.
        c = (np.arange(n_pix) + 0.5) * (2.0 * hw / n_pix) - hw
        X, Y = np.meshgrid(c, c)
        r = np.hypot(X, Y) / hw
        w = np.clip((1.0 - r / taper) / (1.0 - 0.0), 0.0, None)
        w = np.where(r < taper, 0.5 * (1.0 - np.cos(np.pi * np.clip(w, 0, 1))), 0.0)
        patch = patch * w
    g1g, g2g = aperiodic_shear_from_kappa(patch, pix=2.0 * hw / n_pix)
    ext = (-hw, hw, -hw, hw)
    nodes = np.asarray(nodes, np.float64)
    return (_grid_to_nodes_bilinear(patch, nodes, ext),
            _grid_to_nodes_bilinear(g1g, nodes, ext),
            _grid_to_nodes_bilinear(g2g, nodes, ext))


def independent_truth(nodes, source="nfw", half_width=2.5, seed=0, **kw):
    """Return (kappa, g1, g2) from an independent (non-FEMMI, non-FFT) truth.

    source='nfw'        analytic GalSim NFW halo(s)      [default; always available]
    source='lognormal'  non-Gaussian log-normal field + aperiodic shear
    source='massivenus' a MassiveNuS map + aperiodic shear  [needs data_dir=...]
    """
    if source == "nfw":
        return galsim_nfw_truth(nodes, **kw)
    if source == "lognormal":
        return lognormal_truth(nodes, half_width=half_width, seed=seed, **kw)
    if source == "massivenus":
        return massivenus_truth(nodes, half_width=half_width, seed=seed, **kw)
    raise ValueError(f"unknown independent-truth source {source!r} "
                     "(expected 'nfw', 'lognormal' or 'massivenus')")
