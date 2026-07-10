"""
examples/eb_modes_demo.py
E/B-mode decomposition of a FEMMI reconstruction.

The physical (gravitational-lensing) signal lives entirely in the E-mode. The
B-mode -- the same estimator applied to the shear rotated by 45 degrees -- is a
systematics null test: for a clean signal it carries no coherent structure.

Produces outputs/fig_eb_modes.png:  truth | E-mode | B-mode (null) | KS B-mode

Run:
    python examples/eb_modes_demo.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators      import build_operators
from femmi.forward        import DifferentiableForward
from femmi.inverse        import MAPReconstructor, kaiser_squires
from femmi.regularization import estimate_noise_level


def main(nx=20, noise_level=0.10, seed=42):
    ops   = build_operators(nx, nx, -2.5, 2.5, -2.5, 2.5, verbose=False)
    nodes = np.array(ops.mesh.nodes)

    kappa_true = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2 * 0.5**2))
    g1, g2 = (np.asarray(a) for a in ops.forward(kappa_true))

    rng   = np.random.default_rng(seed)
    noise = noise_level * np.std(np.hypot(g1, g2))
    g1o   = g1 + rng.normal(0, noise, g1.shape)
    g2o   = g2 + rng.normal(0, noise, g2.shape)

    ns  = estimate_noise_level(np.concatenate([g1o, g2o]), method="mad")
    fwd = DifferentiableForward(ops, lam_reg=1e-3)
    rec = MAPReconstructor(fwd, maxiter=400, wiener_length=0.5, noise_std=ns)

    diag, kE, kB = rec.bmode_diagnostics(g1o, g2o, verbose=True)
    _, ksB       = kaiser_squires(g1o, g2o, nodes, return_bmode=True)
    print("\n" + diag.summary())
    print(f"\nMorozov delta fed in (MAD)={ns:.4e};  B-channel noise floor="
          f"{diag.delta_noise:.4e}  -> cross-check ratio {diag.delta_consistency:.2f}")

    interior = np.hypot(nodes[:, 0], nodes[:, 1]) < 1.5
    corr_E = np.corrcoef(kE[interior], kappa_true[interior])[0, 1]
    corr_B = np.corrcoef(kB[interior], kappa_true[interior])[0, 1]
    print(f"\ncorr(E-mode, truth) = {corr_E:+.3f}   (signal recovered)")
    print(f"corr(B-mode, truth) = {corr_B:+.3f}   (null: ~0 => no systematics)")

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    panels = [
        (kappa_true, "kappa truth",          "hot"),
        (kE,         "FEMMI E-mode (signal)", "hot"),
        (kB,         "FEMMI B-mode (null)",   "RdBu_r"),
        (ksB,        "Kaiser-Squires B-mode", "RdBu_r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#1a1a1a")
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.set_facecolor("#1a1a1a")
        if cmap == "RdBu_r":
            vmax = np.percentile(np.abs(data), 99); vmin = -vmax
        else:
            vmax = np.percentile(np.abs(data), 99); vmin = 0
        tc = ax.tripcolor(triang, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
        plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_aspect("equal")
        ax.tick_params(colors="white")

    fig.suptitle(f"E/B decomposition  |  noise={noise_level*100:.0f}%  |  "
                 f"corr(E,truth)={corr_E:.2f}, corr(B,truth)={corr_B:.2f}",
                 color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "fig_eb_modes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"\nwrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
