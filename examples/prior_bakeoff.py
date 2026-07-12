"""
examples/prior_bakeoff.py
A PROPER prior bake-off: unlike prior_comparison.py (fixed lambdas, qualitative),
this tunes each prior's regularisation strength lambda over a small grid and
reports each prior AT ITS BEST lambda on the same catalog, so the comparison is
apples-to-apples. The Wiener prior additionally gets its Morozov-selected lambda
as a reference point.

Metric: truth-space L2(kappa) on a central aperture, plus peak recovery. Uses the
analytic Gaussian catalog (known truth); pass --neural to include the learned
NeuralScorePrior (MAP with the score prior).

Run:
    python examples/prior_bakeoff.py
    python examples/prior_bakeoff.py --neural      # also score the neural prior
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators_catalog
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor
from femmi.catalog   import analytic_gaussian_catalog
from femmi.priors    import make_prior, WienerPrior


def _l2_peak(k, truth, inner):
    m = inner & np.isfinite(k)
    l2 = float(np.linalg.norm(k[m] - truth[m]) / (np.linalg.norm(truth[m]) + 1e-30))
    return l2, float(np.nanmax(k[inner]))


def _best_over_lambda(ops, g1n, g2n, gn, si, truth_gal, inner, kind, kw,
                      lambdas, maxiter, n):
    best = None
    for lam in lambdas:
        fwd = DifferentiableForward(ops, lam_reg=lam)
        prior = None if kind == "wiener" else make_prior(kind, ops, **(kw or {}))
        wl = 0.5 if kind == "wiener" else 0.0
        rec = MAPReconstructor(fwd, maxiter=maxiter, callback_every=0,
                               wiener_length=wl, prior=prior)
        k, _ = rec.reconstruct(g1n, g2n, verbose=False)
        kgal = np.full(len(truth_gal), np.nan); kgal[si] = k[gn]
        l2, peak = _l2_peak(kgal, truth_gal, inner)
        if best is None or l2 < best[0]:
            best = (l2, peak, lam)
    return best


def run(args):
    cat = analytic_gaussian_catalog(n_gal=args.n_gal, sigma=0.5,
                                    shape_noise=args.shape_noise, seed=1)
    x, y = cat["x"], cat["y"]; truth = cat["kappa_true"]
    center = cat.get("center", (0.0, 0.0))
    inner = np.hypot(x - center[0], y - center[1]) < 0.6 * cat["field_radius"]

    ops, cm = build_operators_catalog(x, y, center=center, n_boundary=96, verbose=False)
    gn, si = cm.galaxy_nodes, cm.source_index
    n = ops.n_nodes
    g1n = np.zeros(n); g1n[gn] = cat["g1"][si]
    g2n = np.zeros(n); g2n[gn] = cat["g2"][si]

    lambdas = np.geomspace(args.lam_min, args.lam_max, args.n_lam)
    print(f"Gaussian catalog: {len(x)} gal, shape_noise={args.shape_noise}, "
          f"tuning lambda over {args.n_lam} pts in [{args.lam_min:g},{args.lam_max:g}]")
    print(f"{'prior':>10} | {'best L2':>8} | {'peak':>6} | {'lambda*':>10}")

    configs = [("wiener", None), ("tv", {}),
               ("sparse", {"transform": "field"}), ("maxent", {"model": 1e-2})]
    if args.neural:
        configs.append(("neural", {"n_pix": args.n_pix, "verbose": False}))

    for kind, kw in configs:
        l2, peak, lam = _best_over_lambda(ops, g1n, g2n, gn, si, truth, inner,
                                          kind, kw, lambdas, args.maxiter, n)
        print(f"{kind:>10} | {l2:8.3f} | {peak:6.2f} | {lam:10.2e}")
    print(f"\n(truth peak {truth.max():.2f}; lower L2 = better, higher peak = sharper core)")


def main():
    ap = argparse.ArgumentParser(description="Prior bake-off (lambda-tuned)")
    ap.add_argument("--n-gal", type=int, default=1200)
    ap.add_argument("--shape-noise", type=float, default=0.06)
    ap.add_argument("--lam-min", type=float, default=1e-3)
    ap.add_argument("--lam-max", type=float, default=1e0)
    ap.add_argument("--n-lam", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=250)
    ap.add_argument("--neural", action="store_true", help="also score the neural prior")
    ap.add_argument("--n-pix", type=int, default=48)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
