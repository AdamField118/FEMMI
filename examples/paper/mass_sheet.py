"""
examples/paper/mass_sheet.py   [P0.1 -- the paper's central result]
Absolute-normalisation / mass-sheet recovery: FEMMI recovers the DC (mean) mode
that Kaiser-Squires structurally cannot.

A compact convergence field with a known nonzero mean is reconstructed with FEMMI
and KS from the same (self-consistent, isolated-field) shear. KS floats by an
unconstrained additive constant -> its radial profile sits below truth by the mean;
FEMMI recovers the absolute level.

    python examples/paper/mass_sheet.py

Caveat (printed): this is the far-field-zero regime FEMMI assumes. With
infinite-domain shear (--no-self-consistent) the boundary assumption is violated
and FEMMI's DC recovery degrades -- the honest scope of the claim.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import mass_sheet_recovery
from femmi.plotstyle import use_paper_style, SEQ_CMAP, PALETTE


def main():
    ap = argparse.ArgumentParser(description="Mass-sheet / absolute-normalisation recovery")
    ap.add_argument("--nx", type=int, default=18)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--no-self-consistent", action="store_true",
                    help="use infinite-domain analytic shear (violates FEMMI's boundary assumption)")
    ap.add_argument("-o", "--out", default="mass_sheet.png")
    args = ap.parse_args()
    use_paper_style()

    d = mass_sheet_recovery(nx=args.nx, noise_std=args.noise,
                            self_consistent=not args.no_self_consistent)
    print(f"mean kappa   truth={d['mean_truth']:+.4f}")
    print(f"  FEMMI={d['mean_femmi']:+.4f}  (err {d['err_femmi']:.4f})")
    print(f"  KS   ={d['mean_ks']:+.4f}  (err {d['err_ks']:.4f})   "
          f"-> FEMMI recovers the absolute level {d['err_ks']/max(d['err_femmi'],1e-6):.0f}x better")

    tri = mtri.Triangulation(d["nodes"][:, 0], d["nodes"][:, 1])
    vmax = float(np.nanpercentile(d["truth"], 99)) or 1.0

    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.55)
    for j, (key, title) in enumerate([("truth", "truth $\\kappa$"),
                                       ("femmi", "FEMMI"), ("ks", "Kaiser-Squires")]):
        ax = fig.add_subplot(gs[0, j])
        tc = ax.tripcolor(tri, np.nan_to_num(d[key]), cmap=SEQ_CMAP, shading="gouraud",
                          vmin=0, vmax=vmax)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tc, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(d["r"], d["prof_truth"], color="k", lw=2.2, label="truth")
    ax.plot(d["r"], d["prof_femmi"], color=PALETTE[0], lw=2, label=f"FEMMI (err {d['err_femmi']:.3f})")
    ax.plot(d["r"], d["prof_ks"], color=PALETTE[1], lw=2, ls="--", label=f"KS (err {d['err_ks']:.3f})")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("radius"); ax.set_ylabel("azimuthally-averaged $\\kappa$")
    ax.set_title("absolute normalisation"); ax.legend(frameon=False)

    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
