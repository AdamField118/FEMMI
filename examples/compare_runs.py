"""
examples/compare_runs.py
Head-to-head of several FEMMI runs on the SAME field -- e.g. Wiener vs neural vs
hybrid, each produced by a separate `femmi run` (one sbatch job per prior). It
just aggregates the saved .npz files (no recompute): a shared truth panel, each
run's posterior mean with its relative-L2-vs-truth in the title, and -- if the
runs saved `samples` -- each run's appearance-frequency map P(kappa > tau).

    python examples/compare_runs.py runs/wiener.npz runs/neural.npz runs/hybrid.npz \
        --labels Wiener Neural Hybrid

Works for the log-normal synthetic (configs/lognormal.yaml) and for real data
(data.source: frontier) alike -- any runs that share nodes and a truth field.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _l2(k, t):
    m = np.isfinite(k) & np.isfinite(t)
    dm = lambda a: a - a[m].mean()                 # remove the mass-sheet DC mode
    return np.linalg.norm(dm(k)[m] - dm(t)[m]) / (np.linalg.norm(dm(t)[m]) + 1e-30)


def main():
    ap = argparse.ArgumentParser(description="Compare FEMMI runs on one field")
    ap.add_argument("npz", nargs="+", help="run .npz files (same field / nodes)")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per run")
    ap.add_argument("-o", "--out", default="compare.png")
    ap.add_argument("--cmap", default="hot")
    ap.add_argument("--detect-threshold", type=float, default=None)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
