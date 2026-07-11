"""
examples/bem_scaling_diagnostic.py
Investigation of the BEM forward-accuracy gap surfaced by the boundary-condition
ablation (examples/bc_ablation.py), where the FEM-BEM forward was consistently
LESS accurate than a plain Dirichlet truncation.

FINDING
-------
The lensing forward map kappa -> gamma is scale-invariant (rescaling all
coordinates leaves the shear unchanged). Dirichlet respects this exactly; the
BEM coupling does NOT -- its forward error grows with the absolute coordinate
scale and can exceed 100% while Dirichlet stays flat.

ROOT CAUSE
----------
The Johnson-Nedelec coupling C = V_h^{-1}(0.5 M_b + K_h) has the wrong
dimensional scaling. With domain size L:
    V_h ~ L^2 (log kernel),  M_b ~ L,  K_h ~ L   =>   C ~ 1/L,
whereas the FEM stiffness K is scale-free (O(1)). So on a domain of size L the
boundary coupling is under-weighted by ~1/L relative to the interior stiffness;
as L grows the far-field condition is effectively lost and A_coupled -> the bare
(singular) Neumann stiffness. Measured: ||C|| falls exactly as 1/L.

FIX (experimental)
------------------
Multiply C by a characteristic length so the coupling is O(1) and scale-free.
build_operators(..., couple_scale='auto') uses the boundary diameter; this makes
the forward error scale-invariant AND brings it down to the Dirichlet level (it
even improves the unit-scale case). The exact length constant deserves a proper
derivation from the symmetric Steklov-Poincare operator -- the scalar factor here
is empirical -- but the scaling law and the direction of the fix are unambiguous.

Run:
    python examples/bem_scaling_diagnostic.py
"""

from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators, dirichlet_from_operators
from femmi.catalog   import analytic_gaussian_shear


def _fwd_err(ops, kt, g1a, g2a, sel):
    g1, g2 = (np.asarray(a) for a in ops.forward(kt))
    return float(np.linalg.norm(np.hypot(g1[sel] - g1a[sel], g2[sel] - g2a[sel])) /
                 np.linalg.norm(np.hypot(g1a[sel], g2a[sel])))


def run(scales=(1.0, 3.0, 10.0, 30.0, 100.0), nx=24):
    rows = []
    for s in scales:
        L, sig = 1.5 * s, 0.4 * s
        o_def = build_operators(nx, nx, -L, L, -L, L, verbose=False)
        o_fix = build_operators(nx, nx, -L, L, -L, L, verbose=False, couple_scale='auto')
        o_dir = dirichlet_from_operators(o_def)
        nodes = np.array(o_def.mesh.nodes)
        sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 0.5 * L
        kt, g1, g2 = analytic_gaussian_shear(nodes, sigma=sig, amp=1.0)
        row = dict(scale=s,
                   c_norm=float(np.linalg.norm(o_def.C_dense)),
                   bem=_fwd_err(o_def, kt, g1, g2, sel),
                   fix=_fwd_err(o_fix, kt, g1, g2, sel),
                   dir=_fwd_err(o_dir, kt, g1, g2, sel))
        rows.append(row)
        print(f"  scale={s:6.1f}: ||C||={row['c_norm']:.3e}  "
              f"BEM(default)={row['bem']:.4f}  BEM(couple_scale=auto)={row['fix']:.4f}  "
              f"Dirichlet={row['dir']:.4f}")
    return rows


def make_figure(rows, path):
    s = [r["scale"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#1a1a1a")
    for ax in (a1, a2):
        ax.set_facecolor("#111111"); ax.tick_params(colors="#aaa"); ax.grid(True, alpha=0.2)
        ax.set_xscale("log"); ax.set_xlabel("coordinate scale (x -> s*x)", color="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")

    a1.plot(s, [r["bem"] for r in rows], "o-", color="#ff5252", label="BEM (default)")
    a1.plot(s, [r["fix"] for r in rows], "o-", color="#00e676", label="BEM (couple_scale=auto)")
    a1.plot(s, [r["dir"] for r in rows], "s-", color="#4488ff", label="Dirichlet")
    a1.set_yscale("log"); a1.set_ylabel("forward error vs analytic shear", color="white")
    a1.set_title("Scale-invariance: BEM breaks, fix restores it", color="white")
    a1.legend(labelcolor="white", framealpha=0.2)

    a2.plot(s, [r["c_norm"] for r in rows], "o-", color="#ffab40", label="||C|| (coupling)")
    a2.plot(s, [rows[0]["c_norm"] * rows[0]["scale"] / si for si in s], "--",
            color="#888", label="1/L reference")
    a2.set_yscale("log"); a2.set_ylabel("||C_dense||", color="white")
    a2.set_title("Coupling strength falls as 1/L (the bug)", color="white")
    a2.legend(labelcolor="white", framealpha=0.2)

    fig.suptitle("BEM coupling scaling diagnostic", color="white", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"\nwrote {os.path.normpath(path)}")


def main():
    print("BEM coupling scaling diagnostic (lensing forward is scale-invariant)")
    rows = run()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "fig_bem_scaling.png")
    make_figure(rows, out)
    worst = max(r["bem"] for r in rows) / min(r["bem"] for r in rows)
    fixspread = max(r["fix"] for r in rows) / min(r["fix"] for r in rows)
    print(f"\nBEM(default) error varies {worst:.1f}x across scale; "
          f"BEM(couple_scale=auto) varies {fixspread:.2f}x (scale-invariant).")


if __name__ == "__main__":
    main()
