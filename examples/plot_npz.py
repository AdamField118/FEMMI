"""
examples/plot_npz.py
Plot a FEMMI output .npz (written by `femmi run ...` / pipeline._save).

The .npz holds node-aligned fields -- kappa (the point estimate), optionally
std (posterior uncertainty) and truth -- plus `nodes`, the (N, 2) mesh
coordinates. Each field is drawn as a tripcolor over the mesh triangulation.

    python examples/plot_npz.py runs/run.npz
    python examples/plot_npz.py runs/run.npz -o paper.png --cmap inferno
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def main():
    ap = argparse.ArgumentParser(description="Plot a FEMMI output .npz")
    ap.add_argument("npz", help="path to the saved .npz")
    ap.add_argument("-o", "--out", default=None, help="output image (default: <npz>.png)")
    ap.add_argument("--cmap", default="hot", help="colormap for kappa/truth (default: hot)")
    args = ap.parse_args()

    d = np.load(args.npz)
    nodes = d["nodes"]
    if len(d["kappa"]) != len(nodes):
        raise SystemExit(
            f"kappa has {len(d['kappa'])} values but nodes has {len(nodes)}; this "
            "looks like a galaxy-space (catalog) save without matching coordinates.")
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])

    # Show whichever fields are present: truth, kappa, std.
    panels = [(k, args.cmap) for k in ("truth", "kappa") if k in d.files]
    panels += [("std", "viridis")] if "std" in d.files else []

    # Shared color scale for truth vs kappa so they're directly comparable.
    kfields = [d[k] for k, _ in panels if k in ("truth", "kappa")]
    vmax = float(np.nanpercentile(np.concatenate(kfields), 99)) or 1.0

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, cmap) in zip(axes, panels):
        lim = dict(vmin=0, vmax=vmax) if key in ("truth", "kappa") else {}
        tc = ax.tripcolor(tri, np.nan_to_num(d[key]), cmap=cmap, shading="gouraud", **lim)
        ax.set_title(key); ax.set_aspect("equal")
        fig.colorbar(tc, ax=ax, fraction=0.046)

    out = args.out or (args.npz.rsplit(".", 1)[0] + ".png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
