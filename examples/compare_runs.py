"""
examples/compare_runs.py
Head-to-head of several FEMMI runs on the SAME field -- e.g. Wiener vs neural vs
hybrid, each produced by a separate `femmi run` (one sbatch job per prior). It
just aggregates the saved .npz files (no recompute): a shared truth panel, each
run's posterior mean with its relative-L2-vs-truth in the title, and -- if the
runs saved `samples` -- each run's appearance-frequency map P(kappa > tau).

It also writes <out>_stats.png with the two metrics the paper actually uses (and
where a non-Gaussian prior should beat Wiener, unlike pixel-L2): the power
spectrum and the one-point PDF, plus a printed small-scale-power recovery number.

    python examples/compare_runs.py runs/wiener.npz runs/neural.npz runs/hybrid.npz \
        --labels Wiener Neural Hybrid

Works for the log-normal synthetic (configs/lognormal.yaml) and for real data
(data.source: frontier) alike -- any runs that share nodes and a truth field.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from femmi.plotstyle import use_paper_style


def _l2(k, t):
    m = np.isfinite(k) & np.isfinite(t)
    dm = lambda a: a - a[m].mean()                 # remove the mass-sheet DC mode
    return np.linalg.norm(dm(k)[m] - dm(t)[m]) / (np.linalg.norm(dm(t)[m]) + 1e-30)


def _rasterize(nodes, field, n=96):
    """Interpolate a node field onto a regular n x n grid (for the power spectrum)."""
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    interp = mtri.LinearTriInterpolator(tri, np.nan_to_num(field))
    xs = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), n)
    ys = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), n)
    g = interp(*np.meshgrid(xs, ys))
    return np.where(np.isfinite(g), g, 0.0)


def _radial_power(grid):
    """Radially averaged power spectrum P(k) of a 2D map (DC removed)."""
    n = grid.shape[0]
    P = np.abs(np.fft.fftshift(np.fft.fft2(grid - grid.mean()))) ** 2
    ky, kx = np.indices(grid.shape)
    kr = np.hypot(kx - n // 2, ky - n // 2).astype(int)
    prof = np.bincount(kr.ravel(), P.ravel()) / np.maximum(np.bincount(kr.ravel()), 1)
    return np.arange(1, n // 2), prof[1:n // 2]


def _stats_figure(runs, labels, truth, nodes, out):
    """Two evaluation metrics the paper actually uses (and where a non-Gaussian
    prior should beat Wiener, unlike pixel-L2): the power spectrum and the
    one-point PDF. Prints each run's small-scale-power recovery vs truth."""
    fig, (axp, axh) = plt.subplots(1, 2, figsize=(12, 4.6))
    kt = Pt = None
    if truth is not None:
        kt, Pt = _radial_power(_rasterize(nodes, truth))
        axp.loglog(kt, Pt, "k--", lw=2, label="truth")
        print("small-scale power recovery (mean P_run/P_truth over high k; 1.0 = perfect):")
    for r, lab in zip(runs, labels):
        k, P = _radial_power(_rasterize(nodes, r["kappa"]))
        axp.loglog(k, P, label=lab)
        if Pt is not None:
            hi = slice(len(k) // 2, None)              # small scales
            print(f"  {lab:12s} {np.mean(P[hi] / (Pt[hi] + 1e-30)):.3f}")
    axp.set_xlabel("k [1/pixel]"); axp.set_ylabel("P(k)")
    axp.set_title("power spectrum"); axp.legend(fontsize=8)

    kv = [truth] if truth is not None else []
    kv += [r["kappa"] for r in runs]
    allv = np.concatenate([a[np.isfinite(a)] for a in kv])
    bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 60)
    if truth is not None:
        axh.hist(truth[np.isfinite(truth)], bins=bins, density=True, histtype="step",
                 color="k", ls="--", lw=2, label="truth")
    for r, lab in zip(runs, labels):
        v = r["kappa"][np.isfinite(r["kappa"])]
        axh.hist(v, bins=bins, density=True, histtype="step", label=lab)
    axh.set_yscale("log"); axh.set_xlabel("kappa"); axh.set_ylabel("pdf")
    axh.set_title("one-point PDF"); axh.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Compare FEMMI runs on one field")
    ap.add_argument("npz", nargs="+", help="run .npz files (same field / nodes)")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per run")
    ap.add_argument("-o", "--out", default="compare.png")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--detect-threshold", type=float, default=None)
    args = ap.parse_args()
    use_paper_style()

    runs = [np.load(p) for p in args.npz]
    labels = args.labels or [p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in args.npz]
    nodes = runs[0]["nodes"]
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    truth = runs[0]["truth"] if "truth" in runs[0].files else None
    vmax = float(np.nanpercentile(truth if truth is not None else runs[0]["kappa"], 99)) or 1.0
    tau = args.detect_threshold if args.detect_threshold is not None else 0.5 * vmax

    any_samples = any("samples" in r.files for r in runs)
    nrows = 2 if any_samples else 1
    ncols = len(runs) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows), squeeze=False)

    def panel(ax, field, title, cmap=args.cmap, lo=0, hi=vmax):
        tc = ax.tripcolor(tri, np.nan_to_num(field), cmap=cmap, shading="gouraud", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        fig.colorbar(tc, ax=ax, fraction=0.046)

    # row 0: truth + each posterior mean (L2 in title)
    if truth is not None:
        panel(axes[0][0], truth, "truth")
    else:
        axes[0][0].axis("off")
    print("relative L2 vs truth (DC-removed):")
    for j, (r, lab) in enumerate(zip(runs, labels), start=1):
        t = f"{lab}"
        if truth is not None:
            l2 = _l2(r["kappa"], truth); t += f"  L2={l2:.3f}"; print(f"  {lab:12s} {l2:.3f}")
        panel(axes[0][j], r["kappa"], t)

    # row 1: appearance frequency P(kappa > tau) per run
    if any_samples:
        axes[1][0].axis("off")
        axes[1][0].text(0.5, 0.5, f"P(kappa > {tau:.2g})", ha="center", va="center",
                        transform=axes[1][0].transAxes, fontsize=11)
        for j, (r, lab) in enumerate(zip(runs, labels), start=1):
            if "samples" in r.files:
                freq = (r["samples"] > tau).mean(axis=0)
                panel(axes[1][j], freq, f"{lab}: P(kappa>{tau:.2g})", cmap="magma", lo=0, hi=1)
            else:
                axes[1][j].axis("off")

    fig.tight_layout(); fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    # power spectrum + one-point PDF -- the metrics where a non-Gaussian prior is
    # supposed to help (pixel-L2 structurally favours the smooth Wiener estimator).
    _stats_figure(runs, labels, truth, nodes, args.out.rsplit(".", 1)[0] + "_stats.png")


if __name__ == "__main__":
    main()
