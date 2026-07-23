"""
examples/plot_npz.py
Plot a FEMMI output .npz (written by `femmi run ...` / pipeline._save).

The .npz holds node-aligned fields -- kappa (the point estimate), optionally
std (posterior uncertainty), truth, and (for sampling runs) `samples`, the
individual posterior draws -- plus `nodes`, the (N, 2) mesh coordinates.

Summary panels (truth / kappa / std) are always drawn. When `samples` is present
the figure also gets the Figure-2 material from Remy et al. 2020:
  * a row of N individual posterior samples (sharp, non-Gaussian -- the mean blurs
    these out), and
  * an appearance-frequency map  P(kappa > tau)  -- the fraction of samples in
    which each location exceeds a threshold, i.e. how often a structure shows up
    across the posterior (the paper's bi-modal cluster test at the field level).

    python examples/plot_npz.py runs/run.npz
    python examples/plot_npz.py runs/run.npz -o paper.png --cmap inferno
    python examples/plot_npz.py runs/run.npz --n-samples 4 --detect-threshold 0.3
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from femmi.plotstyle import use_paper_style, SEQ_CMAP


def _panel(fig, ax, tri, field, title, cmap, vmin=None, vmax=None):
    tc = ax.tripcolor(tri, np.nan_to_num(field), cmap=cmap, shading="gouraud",
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    fig.colorbar(tc, ax=ax, fraction=0.046)


def main():
    ap = argparse.ArgumentParser(description="Plot a FEMMI output .npz")
    ap.add_argument("npz", help="path to the saved .npz")
    ap.add_argument("-o", "--out", default=None, help="output image (default: <npz>.png)")
    ap.add_argument("--cmap", default="viridis", help="colormap for kappa/truth (default: viridis)")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="how many individual posterior samples to show (if present)")
    ap.add_argument("--detect-threshold", type=float, default=None,
                    help="kappa threshold tau for the appearance-frequency map "
                         "P(kappa>tau); default = 0.5 x the 99th-pct amplitude")
    args = ap.parse_args()
    use_paper_style()

    d = np.load(args.npz)
    nodes = d["nodes"]
    if len(d["kappa"]) != len(nodes):
        raise SystemExit(
            f"kappa has {len(d['kappa'])} values but nodes has {len(nodes)}; this "
            "looks like a galaxy-space (catalog) save without matching coordinates.")
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])

    # shared kappa/truth colour scale so panels are directly comparable
    kfields = [d[k] for k in ("truth", "kappa") if k in d.files]
    vmax = float(np.nanpercentile(np.concatenate(kfields), 99)) or 1.0

    has_samples = "samples" in d.files and d["samples"].ndim == 2 \
        and d["samples"].shape[1] == len(nodes)

    # --- top row: summaries (truth / kappa / std) + appearance frequency -------
    top = [(k, args.cmap, 0, vmax) for k in ("truth", "kappa") if k in d.files]
    if "std" in d.files:
        top.append(("std", "viridis", None, None))
    freq = None
    if has_samples:
        tau = args.detect_threshold if args.detect_threshold is not None else 0.5 * vmax
        freq = (d["samples"] > tau).mean(axis=0)         # per-node appearance frequency
        top.append((f"P(kappa>{tau:.2g})", "magma", 0.0, 1.0))

    n_bot = min(args.n_samples, d["samples"].shape[0]) if has_samples else 0
    ncols = max(len(top), n_bot)
    nrows = 2 if n_bot else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows),
                             squeeze=False)

    for j in range(ncols):
        ax = axes[0][j]
        if j < len(top):
            key, cmap, lo, hi = top[j]
            field = freq if key.startswith("P(") else d[key]
            _panel(fig, ax, tri, field, key, cmap, lo, hi)
        else:
            ax.axis("off")

    # --- bottom row: individual posterior draws --------------------------------
    if n_bot:
        idx = np.linspace(0, d["samples"].shape[0] - 1, n_bot).astype(int)
        for j in range(ncols):
            ax = axes[1][j]
            if j < n_bot:
                _panel(fig, ax, tri, d["samples"][idx[j]],
                       f"posterior sample {j + 1}", args.cmap, 0, vmax)
            else:
                ax.axis("off")

    out = args.out or (args.npz.rsplit(".", 1)[0] + ".png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    if has_samples:
        print(f"  {d['samples'].shape[0]} posterior samples; appearance map at "
              f"tau={args.detect_threshold if args.detect_threshold is not None else 0.5*vmax:.3g}")


if __name__ == "__main__":
    main()
