"""
examples/prior_comparison.py
Reconstruct the SAME catalog with each of FEMMI's priors and compare, so the
effect of the prior is isolated (identical mesh, forward, data, lambda scheme).

Priors (femmi/priors.py):
  wiener  Gaussian / Matern 2-point prior            (default; smooth)
  tv      total variation                            (edge-preserving)
  sparse  smoothed-L1 on the field                   (compact peaks / clusters)
  maxent  maximum entropy (Marshall et al. 2002)     (positive maps)

The Wiener prior is the reference and uses Morozov lambda-selection; the
non-Gaussian priors use a fixed lambda (Morozov's discrepancy principle is only
defined for the quadratic prior). This demo is about the qualitative behaviour
of each prior, not a tuned bake-off -- each non-Gaussian prior has a lambda/eps
that would need tuning per problem.

Run:
    python examples/prior_comparison.py
"""

from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.catalog import reconstruct_catalog, analytic_gaussian_catalog


def run(n_gal=1200, sigma=0.5, shape_noise=0.06, seed=1, maxiter=300):
    cat = analytic_gaussian_catalog(n_gal=n_gal, sigma=sigma,
                                    shape_noise=shape_noise, seed=seed)
    x, y = cat["x"], cat["y"]
    truth = cat["kappa_true"]
    center = cat.get("center", (0.0, 0.0))
    inner = np.hypot(x - center[0], y - center[1]) < 0.6 * cat["field_radius"]

    def l2(k):
        m = inner & np.isfinite(k)
        return float(np.linalg.norm(k[m] - truth[m]) / (np.linalg.norm(truth[m]) + 1e-30))

    configs = [
        ("wiener", None, dict(use_morozov=True, noise_source="bmode")),
        ("tv",     "tv", dict(use_morozov=False, lam_reg=3e-2)),
        ("sparse", "sparse", dict(use_morozov=False, lam_reg=3e-2, prior_kw={"transform": "field"})),
        ("maxent", "maxent", dict(use_morozov=False, lam_reg=1e-2, prior_kw={"model": 1e-2})),
    ]

    print(f"Gaussian catalog: {n_gal} galaxies, sigma={sigma}, shape_noise={shape_noise}")
    print(f"{'prior':>8} | {'L2(kappa)':>10} | {'peak_rec':>9}  (truth peak {truth.max():.2f})")
    recs = {}
    for label, kind, kw in configs:
        prior = kw.pop("prior_kw", None)
        rec = reconstruct_catalog(x, y, cat["g1"], cat["g2"], center=center,
                                  n_boundary=96, maxiter=maxiter, verbose=False,
                                  prior=kind, prior_kw=prior, **kw)
        k = rec.kappa_gal
        recs[label] = k
        print(f"{label:>8} | {l2(k):10.3f} | {np.nanmax(k[inner]):9.2f}")

    _figure(x, y, truth, recs)
    return recs


def _figure(x, y, truth, recs):
    tri = mtri.Triangulation(x, y)
    xt, yt = x[tri.triangles], y[tri.triangles]
    edges = np.hypot(np.diff(xt[:, [0, 1, 2, 0]], axis=1),
                     np.diff(yt[:, [0, 1, 2, 0]], axis=1)).max(axis=1)
    tri.set_mask(edges > 6.0 * np.median(edges))
    vmax = float(np.nanpercentile(truth, 99)) or 1.0

    panels = [("truth", truth)] + [(k, recs[k]) for k in recs]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.2),
                             facecolor="#1a1a1a")
    for ax, (title, d) in zip(axes, panels):
        ax.set_facecolor("#111111")
        tc = ax.tripcolor(tri, np.nan_to_num(d), cmap="hot", vmin=0.0, vmax=vmax,
                          shading="gouraud")
        ax.set_title(title, color="white"); ax.set_aspect("equal"); ax.tick_params(colors="#aaa")
        fig.colorbar(tc, ax=ax, fraction=0.046)
    fig.suptitle("Prior comparison (same mesh / forward / data)", color="white", y=1.02)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "fig_prior_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"\nwrote {os.path.normpath(out)}")


if __name__ == "__main__":
    run()
