"""
examples/paper/element_comparison.py
How much does the ELEMENT choice buy you for shear?

Shear is the traceless Hessian of psi, so the element's ability to carry a second
derivative is the whole story. This compares, on the same compactly-supported
manufactured potential and in the same L2 norm:

  P3 nodal        FEMMI's current path -- element Hessians sampled at the P3
                  nodes and averaged over adjacent elements. C^0, so the Hessian
                  jumps exactly where it is sampled.
  P3 recovered    operators.RecoveredShear -- variational recovery, integrating
                  by parts so only first derivatives are taken. Same rate, better
                  constant.
  HCT             C^1 cubic macro-element, 12 DOF. Same O(h^2) rate as P3, but
                  psi_h is in H^2.
  Argyris         C^1 quintic, 21 DOF. O(h^4) -- two orders better -- and its
                  second derivatives are vertex DOFs, so the Hessian at a node is
                  single-valued and needs no averaging at all.

Note on the norms: the P3 curves use the exact L2 norm of the P3 error field via
the mass matrix; the C^1 curves use element quadrature. Both are genuine L2
norms of the same quantity, so the comparison is apples-to-apples.

    python examples/paper/element_comparison.py
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.experiments import shear_convergence, element_shear_convergence
from femmi.plotstyle import use_paper_style, PALETTE


def main():
    ap = argparse.ArgumentParser(description="Element comparison for shear accuracy")
    ap.add_argument("--nxs", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    ap.add_argument("-o", "--out", default="element_comparison.png")
    args = ap.parse_args()
    use_paper_style()
    nxs = tuple(args.nxs)

    p3 = shear_convergence(nxs=nxs)
    hct = element_shear_convergence(kind="hct", nxs=nxs)
    arg = element_shear_convergence(kind="argyris", nxs=nxs)

    curves = [
        ("P3 nodal (current)", p3["h"], p3["err_nodal"], p3["local_nodal"], PALETTE[1], "s", "-"),
        ("P3 variational recovery", p3["h"], p3["err_recovered"], p3["local_recovered"], PALETTE[3], "v", "-"),
        ("HCT ($C^1$ cubic, 12 DOF)", hct["h"], hct["err"], hct["local"], PALETTE[2], "d", "-"),
        ("Argyris ($C^1$ quintic, 21 DOF)", arg["h"], arg["err"], arg["local"], PALETTE[0], "o", "-"),
    ]

    for name, h, e, loc, *_ in curves:
        print(f"{name:34s} local order -> {loc[-1]:.2f}   err(h={h[-1]:.3f}) = {e[-1]:.3e}")

    # Cost, fairly counted. "21 DOF per element" overstates Argyris badly: its
    # DOFs live on vertices and edges and are shared, so the GLOBAL count lands
    # within a few percent of P3's on the same mesh.
    from femmi.elements import C1Space, structured_triangulation
    from femmi.experiments import square_ops
    nx = nxs[-1]
    v, t = structured_triangulation(nx, 2.5)
    n_arg = C1Space(v, t, kind="argyris").n_dofs
    n_hct = C1Space(v, t, kind="hct").n_dofs
    n_p3 = square_ops(nx, 2.5).n_nodes
    print(f"\nglobal DOFs at nx={nx}:  P3={n_p3}   Argyris={n_arg} ({n_arg/n_p3:.2f}x)"
          f"   HCT={n_hct} ({n_hct/n_p3:.2f}x)")
    print(f"Argyris is {p3['err_nodal'][-1] / arg['err'][-1]:.0f}x more accurate than "
          f"P3 nodal at h={arg['h'][-1]:.3f} for {n_arg/n_p3:.2f}x the DOFs, and the "
          f"gap widens with refinement.")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, h, e, loc, col, mk, ls in curves:
        ax.loglog(h, e, color=col, lw=2, marker=mk, ms=5, ls=ls,
                  label=f"{name} — order {loc[-1]:.2f}")

    h = arg["h"]
    ax.loglog(h, p3["err_nodal"][-1] * (h / h[-1]) ** 2, color="#999999", lw=1.1,
              ls="--", label="$O(h^2)$")
    ax.loglog(h, arg["err"][-1] * (h / h[-1]) ** 4, color="#555555", lw=1.1,
              ls=":", label="$O(h^4)$")

    ax.set_xlabel("mesh spacing $h$")
    ax.set_ylabel("relative $L^2$ error in $\\gamma$")
    ax.set_title("shear accuracy by element")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
