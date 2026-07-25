"""
examples/paper/forward_convergence.py   [P1.5 -- forward-operator validation]
Convergence of FEMMI's recovered lensing potential psi = F-solve(kappa) toward a
COMPACTLY-SUPPORTED manufactured psi (zero at the boundary, so no finite-vs-
infinite-domain floor), at increasing resolution.

This validates the forward operator F. For P3 elements the theory rate is O(h^4)
in L2, and that is what this shows -- clean, stable local orders ~4. The manufactured
potential is smooth (C^5) so the rate is not regularity-limited, and the additive
gauge (FEMMI pins one node) is removed before comparing.

(The shear is a well-defined post-hoc derivative of psi, sampled at nodes; its
lower, nodal-sampling-limited rate is a property of that derivative operator, not
of F, so it is not the operator-validation quantity.)

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
    ap = argparse.ArgumentParser(description="Forward-operator (potential) convergence")
    ap.add_argument("--nxs", type=int, nargs="+", default=[8, 12, 16, 24, 32, 40])
    ap.add_argument("-o", "--out", default="forward_convergence.png")
    args = ap.parse_args()
    use_paper_style()

    h, err, order = forward_convergence(nxs=tuple(args.nxs))

    print(f"fitted convergence order ~ {order:.2f}  (theory O(h^4) for P3)")
    for hi, ei in zip(h, err):
        print(f"  h={hi:.3f}  rel_L2(psi)={ei:.4e}")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.loglog(h, err, color=PALETTE[0], lw=2, marker="o", ms=5, label="FEMMI potential $\\psi$")
    ref = err[0] * (h / h[0]) ** 4
    ax.loglog(h, ref, color="#666666", lw=1.2, ls="--", label="$O(h^4)$ reference")
    ax.set_xlabel("mesh spacing $h$"); ax.set_ylabel("relative $L^2$ error in $\\psi$")
    ax.set_title(f"forward-operator convergence (fitted order $\\approx$ {order:.2f})")
    ax.legend(frameon=False)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
