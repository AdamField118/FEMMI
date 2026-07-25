"""
examples/paper/shear_recovery.py   [P0.3 -- shear extraction at the theory rate]
The shear is a SECOND derivative of the lensing potential, and how it is taken
out of the P3 field matters twice over.

Left panel  -- convergence of the extracted shear against a compactly-supported
manufactured solution, comparing FEMMI's default NODAL extraction (element
Hessians sampled at the P3 nodes and averaged) with a VARIATIONAL recovery
(operators.RecoveredShear) that integrates by parts and never differentiates
twice. Both reach the O(h^2) rate that approximation theory gives for P3; the
variational route gets there with a markedly smaller constant.

Right panel -- why that rate is not what a real catalog delivers. A second
derivative amplifies any perturbation in psi by h^-2, so with fixed noise the
error goes like C h^2 + sigma/h^2: refining helps only down to an optimal h and
then actively hurts. O(h^2) is the right theory and the wrong expectation for
catalog-native data.

    python examples/paper/shear_recovery.py
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import shear_convergence, shear_noise_amplification
from femmi.plotstyle import use_paper_style, PALETTE


def main():
    ap = argparse.ArgumentParser(description="Shear extraction: nodal vs variational recovery")
    ap.add_argument("--nxs", type=int, nargs="+", default=[8, 12, 16, 24, 32, 40])
    ap.add_argument("--noise", type=float, default=1e-4,
                    help="noise std injected into psi for the right-hand panel")
    ap.add_argument("-o", "--out", default="shear_recovery.png")
    args = ap.parse_args()
    use_paper_style()

    d = shear_convergence(nxs=tuple(args.nxs))
    print(f"fitted order  nodal      = {d['order_nodal']:.2f}")
    print(f"fitted order  recovered  = {d['order_recovered']:.2f}   (theory O(h^2) for P3)")
    print("  h        nodal        recovered     local orders (nodal / recovered)")
    for i, h in enumerate(d["h"]):
        loc = ("      --  " if i == 0 else
               f"  {d['local_nodal'][i-1]:.2f} / {d['local_recovered'][i-1]:.2f}")
        print(f"  {h:.4f}  {d['err_nodal'][i]:.4e}  {d['err_recovered'][i]:.4e}{loc}")

    n = shear_noise_amplification(nxs=tuple(args.nxs), noise_std=args.noise)
    print(f"\nwith noise_std={args.noise:g} in psi, error is minimised at h={n['h_opt']:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # the fitted order over the whole range is dragged down by the coarse,
    # pre-asymptotic meshes; the LOCAL order on the finest pair is the rate
    # actually being approached, so label with that.
    ax1.loglog(d["h"], d["err_nodal"], color=PALETTE[1], lw=2, marker="s", ms=5,
               label=f"nodal sampling (local order {d['local_nodal'][-1]:.2f})")
    ax1.loglog(d["h"], d["err_recovered"], color=PALETTE[0], lw=2, marker="o", ms=5,
               label=f"variational recovery (local order {d['local_recovered'][-1]:.2f})")
    ref = d["err_recovered"][-1] * (d["h"] / d["h"][-1]) ** 2
    ax1.loglog(d["h"], ref, color="#666666", lw=1.2, ls="--", label="$O(h^2)$ reference")
    ratio = d["err_nodal"][-1] / d["err_recovered"][-1]
    ax1.annotate(f"recovery: {ratio:.1f}$\\times$ smaller error\nat the same $h$",
                 xy=(d["h"][-1], d["err_recovered"][-1]), xytext=(0.06, 0.72),
                 textcoords="axes fraction", fontsize=9, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
    ax1.set_xlabel("mesh spacing $h$")
    ax1.set_ylabel("relative $L^2$ error in $\\gamma$")
    ax1.set_title("shear extraction reaches the P3 theory rate")
    ax1.legend(frameon=False)

    ax2.loglog(n["h"], n["err_clean"], color=PALETTE[0], lw=2, marker="o", ms=5,
               label="noiseless $\\psi$  ($\\propto h^{2}$)")
    ax2.loglog(n["h"], n["err_noisy"], color=PALETTE[3], lw=2, marker="^", ms=5,
               label=f"$\\psi$ + noise {args.noise:g}  ($\\propto h^{{-2}}$)")
    ax2.axvline(n["h_opt"], color="#666666", lw=0.9, ls=":")
    ax2.text(n["h_opt"], ax2.get_ylim()[1], " optimal $h$", va="top", fontsize=8,
             color="#666666")
    ax2.set_xlabel("mesh spacing $h$")
    ax2.set_ylabel("relative $L^2$ error in $\\gamma$")
    ax2.set_title("noise amplification: refining past $h_{opt}$ hurts")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
