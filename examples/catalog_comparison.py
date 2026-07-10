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
                     n_boundary=96, use_morozov=True, lam_reg=1e-2, maxiter=400):
    x, y, g1, g2 = cat['x'], cat['y'], cat['g1'], cat['g2']
    weight = cat.get('weight')
    truth  = cat.get('kappa_true')
    center = cat.get('center', (float(np.mean(x)), float(np.mean(y))))
    R      = float(np.hypot(x - center[0], y - center[1]).max())
    if wiener_length is None:
        wiener_length = 0.2 * R

    # ---- FEMMI catalog-native (E-mode), plus its B-mode null (rotated shear) --
    print("\n[FEMMI] catalog-native FEM-BEM MAP ...")
    fem = reconstruct_catalog(x, y, g1, g2, weight=weight, center=center,
                              n_boundary=n_boundary, wiener_length=wiener_length,
                              use_morozov=use_morozov, lam_reg=lam_reg,
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
        print(f"  FEMMI : L2={l2f:.3f}  corr={ccf:+.3f}")
        print(f"  KS    : L2={l2k:.3f}  corr={cck:+.3f}")
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
    panels = [("FEMMI E (catalog-native)", res['kfem'], "hot"),
              ("KS E (Fourier grid)",      res['kks'], "hot"),
              ("FEMMI B (null)",           res['kfemB'], "RdBu_r"),
              ("KS B (null)",              res['kksB'], "RdBu_r")]
    if res['truth'] is not None:
        panels.insert(0, ("truth kappa", res['truth'], "hot"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), facecolor="#1a1a1a")
    for ax, (title, data, cmap) in zip(axes, panels):
        ax.set_facecolor("#1a1a1a")
        d = np.nan_to_num(np.asarray(data))
        if cmap == "RdBu_r":
            v = np.percentile(np.abs(d), 99); vmin, vmax = -v, v
        else:
            vmax = np.percentile(d, 99); vmin = 0.0
        tc = ax.tripcolor(tri, d, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_aspect("equal"); ax.tick_params(colors="#888", labelsize=7)
    fig.suptitle("Catalog-native FEMMI  vs  Fourier-grid Kaiser-Squires",
                 color="white", fontsize=13, y=1.03)
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
    ap.add_argument("--source", choices=["psi", "kappa"], default="psi",
                    help="frontier: derive shear from psi Hessian or kappa FFT")
    ap.add_argument("--downsample", type=int, default=6, help="frontier map stride")
    ap.add_argument("--pixscale", type=float, default=None,
                    help="frontier pixel scale (arcsec/pixel) override")
    ap.add_argument("--hdu", type=int, default=1)
    ap.add_argument("--flip-g2", action="store_true")
    ap.add_argument("--n-gal", type=int, default=1500, help="galaxy count")
    ap.add_argument("--shape-noise", type=float, default=0.10)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--no-morozov", action="store_true")
    args = ap.parse_args()

    if args.frontier:
        field = load_frontier_model(args.frontier, source=args.source,
                                    pixscale_arcsec=args.pixscale,
                                    downsample=args.downsample)
        cat = field_to_catalog(field, n_gal=args.n_gal,
                               shape_noise=args.shape_noise, seed=1)
        print(f"  sampled {len(cat['x'])} galaxies from the model field "
              f"(shape_noise={args.shape_noise})")
    elif args.fits:
        cat = _load_fits(args.fits, hdu=args.hdu, flip_g2=args.flip_g2)
    else:
        print(f"No --fits/--frontier given; synthetic Gaussian catalog "
              f"(n={args.n_gal}, shape_noise={args.shape_noise}).")
        cat = analytic_gaussian_catalog(n_gal=args.n_gal, sigma=0.5,
                                        shape_noise=args.shape_noise, seed=1)

    res = run_head_to_head(cat, grid_size=args.grid_size,
                           use_morozov=not args.no_morozov)
    out = os.path.join(os.path.dirname(__file__), "..", "outputs",
                       "fig_catalog_comparison.png")
    make_figure(res, out)


if __name__ == "__main__":
    main()
