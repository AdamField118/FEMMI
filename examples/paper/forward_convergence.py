"""
examples/paper/forward_convergence.py   [P1.5]
Convergence of FEMMI's forward shear toward the analytic Gaussian shear on interior
nodes, at increasing resolution -- the credibility plot for the forward operator.

Caveat: the reference is the *infinite-domain* analytic shear, while FEMMI assumes
a finite far-field-zero boundary, so the error floors out once the discretisation
error drops below the boundary-truncation systematic. The convergence *rate* in
the pre-floor regime is what the fit reports; a fully clean order study needs a
manufactured solution consistent with FEMMI's BC.

    python examples/paper/forward_convergence.py
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import forward_convergence
from femmi.plotstyle import use_paper_style, PALETTE


def main():
    ap = argparse.ArgumentParser(description="Forward-operator shear convergence")
    ap.add_argument("--nxs", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    ap.add_argument("-o", "--out", default="forward_convergence.png")
    args = ap.parse_args()
    use_paper_style()

    h, err, order = forward_convergence(nxs=tuple(args.nxs))
    print(f"fitted convergence order ~ {order:.2f}")
    for hi, ei in zip(h, err):
        print(f"  h={hi:.3f}  rel_L2={ei:.4e}")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.loglog(h, err, color=PALETTE[0], lw=2, marker="o", ms=5, label="FEMMI forward shear")
    ref = err[0] * (h / h[0]) ** 2
    ax.loglog(h, ref, color="#888", lw=1.2, ls="--", label="$O(h^2)$ reference")
    ax.set_xlabel("mesh spacing $h$"); ax.set_ylabel("relative $L^2$ shear error (interior)")
    ax.set_title(f"forward convergence (fitted order $\\approx$ {order:.2f})")
    ax.legend(frameon=False)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
