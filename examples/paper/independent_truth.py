"""
examples/paper/independent_truth.py   [P0.1 + P1.4 -- the claims on NEUTRAL truth]
The mass-sheet / absolute-normalisation and boundary-bias comparisons, run on
ground truth that NEITHER method produced.

examples/paper/mass_sheet.py generates its test shear with FEMMI's OWN forward
operator. That is an inverse crime: FEMMI is asked to invert an operator that
also made the data, and its apparent DC-mode recovery partly reflects that. Here
the truth comes from femmi.truth -- an analytic GalSim NFW halo field (closed
form, no mesh and no FFT anywhere), or a MassiveNuS simulation map whose shear is
obtained by aperiodic real-space convolution.

Both paper claims share all the setup, so both are measured and plotted at once:
  top row     truth / FEMMI / KS convergence maps
  bottom left absolute normalisation -- azimuthally-averaged kappa vs truth
  bottom right boundary bias -- DC-removed error vs radius

    python examples/paper/independent_truth.py
    python examples/paper/independent_truth.py --source massivenus --data-dir DIR
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import independent_truth_recovery
from femmi.plotstyle import use_paper_style, SEQ_CMAP, PALETTE


def main():
    ap = argparse.ArgumentParser(description="FEMMI vs KS on independent ground truth")
    ap.add_argument("--nx", type=int, default=18)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--source", default="nfw", choices=["nfw", "massivenus"])
    ap.add_argument("--data-dir", default=None, help="MassiveNuS map folder (--source massivenus)")
    ap.add_argument("-o", "--out", default="independent_truth.png")
    args = ap.parse_args()
    use_paper_style()

    tkw = {}
    if args.source == "massivenus":
        if not args.data_dir:
            raise SystemExit("--source massivenus needs --data-dir")
        tkw["data_dir"] = args.data_dir

    d = independent_truth_recovery(nx=args.nx, noise_std=args.noise,
                                   source=args.source, truth_kw=tkw)

    print(f"independent truth source: {d['source']}")
    print(f"mean kappa   truth={d['mean_truth']:+.4f}")
    print(f"  FEMMI={d['mean_femmi']:+.4f}  (err {d['err_femmi']:.4f})")
    print(f"  KS   ={d['mean_ks']:+.4f}  (err {d['err_ks']:.4f})")
    print(f"edge error (outermost bin)  FEMMI={d['err_femmi_r'][-1]:.4f}  "
          f"KS={d['err_ks_r'][-1]:.4f}")

    tri = mtri.Triangulation(d["nodes"][:, 0], d["nodes"][:, 1])
    vmax = float(np.nanpercentile(d["truth"], 99)) or 1.0
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=1.1)

    for j, (key, title) in enumerate([("truth", "truth $\\kappa$ (independent)"),
                                      ("femmi", "FEMMI"), ("ks", "Kaiser-Squires")]):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        tc = ax.tripcolor(tri, np.nan_to_num(d[key]), cmap=SEQ_CMAP,
                          shading="gouraud", vmin=0, vmax=vmax)
        ax.set_title(title); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tc, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0:3])
    ax.plot(d["r"], d["prof_truth"], color="k", lw=2.2, label="truth")
    ax.plot(d["r"], d["prof_femmi"], color=PALETTE[0], lw=2,
            label=f"FEMMI (mean err {d['err_femmi']:.3f})")
    ax.plot(d["r"], d["prof_ks"], color=PALETTE[1], lw=2, ls="--",
            label=f"KS (mean err {d['err_ks']:.3f})")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("radius"); ax.set_ylabel("azimuthally-averaged $\\kappa$")
    ax.set_title("absolute normalisation (DC / mass-sheet mode)")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 3:6])
    ax.plot(d["r_err"], d["err_femmi_r"], color=PALETTE[0], lw=2, marker="o", ms=4,
            label="FEMMI")
    ax.plot(d["r_err"], d["err_ks_r"], color=PALETTE[1], lw=2, ls="--", marker="s",
            ms=4, label="Kaiser-Squires")
    ax.axvline(d["half_width"], color="#666666", lw=0.9, ls=":")
    ax.text(d["half_width"], ax.get_ylim()[1], " domain edge", va="top",
            fontsize=8, color="#666666")
    ax.set_xlabel("radius"); ax.set_ylabel("DC-removed $|\\kappa_{rec}-\\kappa_{true}|$")
    ax.set_title("boundary bias (shape error vs radius)")
    ax.legend(frameon=False)

    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
