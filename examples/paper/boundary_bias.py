"""
examples/paper/boundary_bias.py   [P0.2]
Reconstruction error vs distance-from-centre, FEMMI vs Kaiser-Squires. FEMMI's
exact exterior (BEM) boundary condition should keep the error flat toward the
domain edge, where KS's periodic/Dirichlet truncation biases the reconstruction.
The DC mode is removed first, so this isolates the *shape* error (not the
mass-sheet offset, which P0.1 covers).

    python examples/paper/boundary_bias.py
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import boundary_error_profile
from femmi.plotstyle import use_paper_style, PALETTE


def main():
    ap = argparse.ArgumentParser(description="Boundary-bias: FEMMI vs KS error vs radius")
    ap.add_argument("--nx", type=int, default=22)
    ap.add_argument("-o", "--out", default="boundary_bias.png")
    args = ap.parse_args()
    use_paper_style()

    d = boundary_error_profile(nx=args.nx)
    print(f"edge error  FEMMI={d['err_femmi'][-1]:.3f}  KS={d['err_ks'][-1]:.3f}  "
          f"(FEMMI {d['err_ks'][-1]/max(d['err_femmi'][-1],1e-6):.1f}x better at the edge)")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(d["r"], d["err_femmi"], color=PALETTE[0], lw=2, marker="o", ms=4, label="FEMMI (exact BEM BC)")
    ax.plot(d["r"], d["err_ks"], color=PALETTE[1], lw=2, ls="--", marker="s", ms=4, label="Kaiser-Squires")
    ax.axvline(d["half_width"], color="#888", lw=0.8, ls=":")
    ax.text(d["half_width"], ax.get_ylim()[1] * 0.95, " domain edge", color="#555", va="top", fontsize=9)
    ax.set_xlabel("distance from centre"); ax.set_ylabel("|$\\kappa_{\\rm rec}-\\kappa_{\\rm true}$| (DC-removed)")
    ax.set_title("boundary bias: FEMMI vs Kaiser-Squires"); ax.legend(frameon=False)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
