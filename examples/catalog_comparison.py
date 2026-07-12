"""
examples/catalog_comparison.py
Apples-to-apples head-to-head on a SINGLE galaxy shear catalog:

  FEMMI   -- catalog-native: FEM nodes AT galaxy positions, FEM-BEM MAP solve
             with the data term restricted to those nodes (no gridding).
  KS/SMPy -- Fourier path: bin the catalog shear onto a regular grid, smooth,
             invert with the Kaiser-Squires kernel.

Both consume the same (x, y, g1, g2, weight); neither is handed the other's
intermediate product. Each also reports its B-mode map as a systematics null.

Usage
-----
    # synthetic Gaussian catalog with analytic (ground-truth) shear
    python examples/catalog_comparison.py

    # a real shear catalog (auto-detects ra/dec/g1/g2/weight columns)
    python examples/catalog_comparison.py --fits path/to/shear_catalog.fits

    # Abell 2744 layout from the download script (data/abell2744/cats_v4.1)
    python examples/catalog_comparison.py --fits data/abell2744/cats_v4.1/<file>.fits

Writes outputs/fig_catalog_comparison.png.
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.catalog import (
    analytic_gaussian_catalog, reconstruct_catalog, kaiser_squires_binned,
    load_frontier_model, field_to_catalog,
)


def _load_fits(path, hdu=1, flip_g2=False, max_gal=6000):
    """Read a real shear catalog and project to the flat (arcmin) frame."""
    from femmi.io import read_fits_catalog
    cat  = read_fits_catalog(path, hdu=hdu)
    flat = cat.to_tangent_plane(units="arcmin", flip_g2=flip_g2)
    x, y, g1, g2, w = flat.x, flat.y, flat.g1, flat.g2, flat.weight
    if len(x) > max_gal:                      # thin for a quick run
        idx = np.random.default_rng(0).choice(len(x), max_gal, replace=False)
        x, y, g1, g2, w = x[idx], y[idx], g1[idx], g2[idx], w[idx]
    print(f"  loaded {len(x)} galaxies from {os.path.basename(path)}  "
          f"field ~{x.ptp():.1f} x {y.ptp():.1f} arcmin")
    return dict(x=x, y=y, g1=g1, g2=g2, weight=w, kappa_true=None)


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def run_head_to_head(cat, wiener_length=None, grid_size=64, ks_smoothing_px=2.0,
                     n_boundary=96, use_morozov=True, lam_reg=1e-2, maxiter=400,
                     noise_source='mad'):
    x, y, g1, g2 = cat['x'], cat['y'], cat['g1'], cat['g2']
    weight = cat.get('weight')
    truth  = cat.get('kappa_true')
    center = cat.get('center', (float(np.mean(x)), float(np.mean(y))))
    R      = float(np.hypot(x - center[0], y - center[1]).max())
    if wiener_length is None:
        wiener_length = 0.2 * R

    # ---- FEMMI catalog-native (E-mode), plus its B-mode null (rotated shear) --
    print(f"\n[FEMMI] catalog-native FEM-BEM MAP (noise_source={noise_source}) ...")
    fem = reconstruct_catalog(x, y, g1, g2, weight=weight, center=center,
                              n_boundary=n_boundary, wiener_length=wiener_length,
                              use_morozov=use_morozov, lam_reg=lam_reg,
                              noise_source=noise_source,
                              maxiter=maxiter, verbose=True)
    print("[FEMMI] B-mode null (45-deg-rotated shear) ...")
    femB = reconstruct_catalog(x, y, g2, -np.asarray(g1), weight=weight, center=center,
                               n_boundary=n_boundary, wiener_length=wiener_length,
                               use_morozov=False, lam_reg=fem.lam_reg,
                               maxiter=maxiter, verbose=False)
    kfem, kfemB = fem.kappa_gal, femB.kappa_gal

    # ---- KS / SMPy Fourier path at the same galaxy positions -----------------
    print("[KS] bin -> smooth -> FFT ...")
    eval_pts = np.column_stack([x, y])
    ext = (x.min(), x.max(), y.min(), y.max())
    kks, kksB = kaiser_squires_binned(x, y, g1, g2, weight=weight,
                                      grid_size=grid_size, smoothing_px=ks_smoothing_px,
                                      extent=ext, eval_pts=eval_pts, return_bmode=True)

    # ---- metrics -------------------------------------------------------------
    inner = np.hypot(x - center[0], y - center[1]) < 0.6 * R
    both  = inner & np.isfinite(kfem)
    print("\n================ head-to-head ================")
    if truth is not None:
        def score(k):
            m = both & np.isfinite(k)
            l2 = np.linalg.norm(k[m] - truth[m]) / (np.linalg.norm(truth[m]) + 1e-30)
            cc = np.corrcoef(k[m], truth[m])[0, 1]
            return l2, cc
        l2f, ccf = score(kfem); l2k, cck = score(kks)
        kt_peak = float(np.nanmax(truth[both]))
        print(f"  region: {int(both.sum())} galaxies, kappa_true peak={kt_peak:.2f}")
        print(f"  FEMMI : L2={l2f:.3f}  corr={ccf:+.3f}  peak_rec={np.nanmax(kfem[both]):.2f}")
        print(f"  KS    : L2={l2k:.3f}  corr={cck:+.3f}  peak_rec={np.nanmax(kks[both]):.2f}")
        winner = "FEMMI" if l2f < l2k else "KS"
        print(f"  -> lower-L2 winner: {winner}")
    agree = np.corrcoef(kfem[both], kks[both])[0, 1]
    print(f"  FEMMI vs KS agreement (inner corr) = {agree:+.3f}")
    print(f"  B-mode RMS (inner)   FEMMI={_rms(kfemB[both]):.4e}  "
          f"KS={_rms(kksB[both]):.4e}   (small => clean null)")
    print("==============================================")

    return dict(x=x, y=y, center=center, R=R, truth=truth,
                kfem=kfem, kfemB=kfemB, kks=kks, kksB=kksB,
                fem=fem, grid_size=grid_size, ks_smoothing_px=ks_smoothing_px, ext=ext)


def make_figure(res, path):
    x, y = res['x'], res['y']
    tri  = mtri.Triangulation(x, y)
    e_panels = [("FEMMI E (catalog-native)", res['kfem']),
                ("KS E (Fourier grid)",      res['kks'])]
    if res['truth'] is not None:
        e_panels.insert(0, ("truth kappa", res['truth']))
    b_panels = [("FEMMI B (null)", res['kfemB']), ("KS B (null)", res['kksB'])]

    # Shared scales so amplitude is comparable across panels. The E scale is
    # driven by the truth (or reconstructions) at the 98th percentile, so a
    # residual strong-lensing cusp does not wash out the whole colour range.
    e_ref = res['truth'] if res['truth'] is not None else res['kfem']
    e_vmax = float(np.nanpercentile(np.abs(e_ref), 98)) or 1.0
    b_vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.nan_to_num(res['kfemB']), np.nan_to_num(res['kksB'])])), 98)) or 1.0

    panels = ([(t, d, "hot", 0.0, e_vmax) for t, d in e_panels] +
              [(t, d, "RdBu_r", -b_vmax, b_vmax) for t, d in b_panels])

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), facecolor="#1a1a1a")
    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        ax.set_facecolor("#1a1a1a")
        d = np.nan_to_num(np.asarray(data))
        tc = ax.tripcolor(tri, d, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_aspect("equal"); ax.tick_params(colors="#888", labelsize=7)
    fig.suptitle("Catalog-native FEMMI  vs  Fourier-grid Kaiser-Squires "
                 "(shared kappa scale)", color="white", fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"\nwrote {os.path.normpath(path)}")


def main():
    ap = argparse.ArgumentParser(description="FEMMI vs KS on a shear catalog")
    ap.add_argument("--fits", type=str, default=None, help="real shear catalog FITS")
    ap.add_argument("--frontier", type=str, default=None,
                    help="Frontier Fields / CATS model-map directory "
                         "(e.g. data/abell2744/cats_v4.1)")
    ap.add_argument("--source", choices=["psi", "deflect", "kappa"], default="psi",
                    help="frontier: derive shear from psi Hessian, the deflection "
                         "field (independent cross-check), or kappa FFT")
    ap.add_argument("--downsample", type=int, default=6, help="frontier map stride")
    ap.add_argument("--pixscale", type=float, default=None,
                    help="frontier pixel scale (arcsec/pixel) override")
    ap.add_argument("--hdu", type=int, default=1)
    ap.add_argument("--flip-g2", action="store_true")
    ap.add_argument("--n-gal", type=int, default=1500, help="galaxy count")
    ap.add_argument("--shape-noise", type=float, default=0.10)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--no-morozov", action="store_true")
    ap.add_argument("--noise-source", choices=["mad", "bmode"], default="bmode",
                    help="Morozov noise level: MAD (biased high by the signal) "
                         "or the B-mode floor (recommended, less over-smoothing)")
    ap.add_argument("--kappa-max", type=float, default=1.0,
                    help="frontier: drop sources where model kappa exceeds this "
                         "(the strong-lensing core, invalid for weak shear)")
    ap.add_argument("--reduced-shear", action="store_true",
                    help="frontier: emit observable reduced shear g=gamma/(1-kappa)")
    ap.add_argument("--rmax", type=float, default=None,
                    help="frontier: restrict sources to this radius (arcmin)")
    ap.add_argument("--maxiter", type=int, default=400,
                    help="L-BFGS iterations for the FEMMI MAP solve; raise for "
                         "large catalogs that have not converged (e.g. 2000-4000)")
    args = ap.parse_args()

    if args.frontier:
        field = load_frontier_model(args.frontier, source=args.source,
                                    pixscale_arcsec=args.pixscale,
                                    downsample=args.downsample)
        cat = field_to_catalog(field, n_gal=args.n_gal,
                               shape_noise=args.shape_noise,
                               kappa_max=args.kappa_max,
                               reduced_shear=args.reduced_shear,
                               rmax_arcmin=args.rmax, seed=1)
        print(f"  sampled {len(cat['x'])} galaxies from the model field "
              f"(shape_noise={args.shape_noise}, kappa_max={args.kappa_max}, "
              f"reduced_shear={args.reduced_shear})")
    elif args.fits:
        cat = _load_fits(args.fits, hdu=args.hdu, flip_g2=args.flip_g2)
    else:
        print(f"No --fits/--frontier given; synthetic Gaussian catalog "
              f"(n={args.n_gal}, shape_noise={args.shape_noise}).")
        cat = analytic_gaussian_catalog(n_gal=args.n_gal, sigma=0.5,
                                        shape_noise=args.shape_noise, seed=1)

    res = run_head_to_head(cat, grid_size=args.grid_size,
                           use_morozov=not args.no_morozov,
                           noise_source=args.noise_source,
                           maxiter=args.maxiter)
    out = os.path.join(os.path.dirname(__file__), "..", "outputs",
                       "fig_catalog_comparison.png")
    make_figure(res, out)


if __name__ == "__main__":
    main()
