"""
examples/paper/injectivity.py   [P1.6]
The mass-sheet degeneracy at the operator level: how strongly does the forward
respond to a UNIFORM mass sheet (the DC mode)?

  Kaiser-Squires / FFT forward:  F.1 = 0  exactly  -> the mean is unrecoverable.
  FEMMI (BEM far-field):         ||F.1||/sqrt(N) ~ O(sigma_max) -> the DC mode is
                                 observable, so the absolute level IS constrained.

This is the mathematical backing for P0.1. Note FEMMI does NOT make shear->kappa
well-posed in general (it's a compact operator with a decaying singular spectrum,
like KS -- that's what regularisation handles); the difference is specifically at
the DC mode.

    python examples/paper/injectivity.py
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import square_ops, constant_mode_response, ks_constant_mode_response
from femmi.plotstyle import use_paper_style, PALETTE


def main():
    ap = argparse.ArgumentParser(description="DC-mode identifiability: FEMMI vs KS")
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("-o", "--out", default="injectivity.png")
    args = ap.parse_args()
    use_paper_style()

    femmi = constant_mode_response(square_ops(args.nx))
    ks = ks_constant_mode_response()
    print(f"DC-mode response  FEMMI={femmi:.3f}   KS={ks:.2e}")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    bars = ax.bar(["Kaiser-Squires", "FEMMI"], [ks, femmi],
                  color=[PALETTE[1], PALETTE[0]], width=0.6)
    ax.bar_label(bars, labels=[f"{ks:.1e}", f"{femmi:.2f}"], padding=3)
    ax.set_ylabel(r"forward response to a uniform sheet  $\|F\,\mathbf{1}\|/\sqrt{N}$")
    ax.set_title("mass-sheet (DC) mode: annihilated by KS, observed by FEMMI")
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
