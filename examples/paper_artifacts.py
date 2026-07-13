"""
examples/paper_artifacts.py
Reproduce the Remy et al. 2020 (arXiv:2011.08271) figure structure with FEMMI:
a probabilistic mass map on masked, noisy data. FEMMI plays the role of their
posterior sampler; the differentiable FEM-BEM forward is what makes the sampling
possible.

Figure 1 layout (their Fig. 1): truth kappa | survey mask | Kaiser-Squires |
posterior mean, over a masked/noisy simulated field.
Figure 2 layout (their Fig. 2): posterior std (uncertainty) + a row of individual
posterior samples -- the samples vary most where the data constrain least, the
qualitative statement their bimodal-cluster cutouts make.

Sampler: exact perturb-and-MAP (RTO) with the Gaussian/Wiener prior by default;
--neural switches to the learned score prior with annealed HMC (the paper's
tempered sampler). Truth is a self-contained two-peak analytic field (no external
data); swap in a GalSim NFW field for a publication run.

Run:
    python examples/paper_artifacts.py
    python examples/paper_artifacts.py --neural
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.forward   import DifferentiableForward
from femmi.inverse   import MAPReconstructor, kaiser_squires
from femmi.sampling  import sample_posterior
from femmi.catalog   import analytic_gaussian_shear


def _two_peak_truth(nodes):
    """A two-peak convergence field: a main clump + a lighter offset clump."""
    k1, g1a, g2a = analytic_gaussian_shear(nodes, sigma=0.45, amp=1.0, center=(-0.6, 0.0))
    k2, g1b, g2b = analytic_gaussian_shear(nodes, sigma=0.30, amp=0.5, center=(0.9, 0.5))
    return (np.asarray(k1) + np.asarray(k2),
            np.asarray(g1a) + np.asarray(g1b), np.asarray(g2a) + np.asarray(g2b))


def run(args):
    ops = build_operators(args.nx, args.nx, -2.5, 2.5, -2.5, 2.5, verbose=False)
    nodes = np.array(ops.mesh.nodes)
    kt, g1t, g2t = _two_peak_truth(nodes)

    rng = np.random.default_rng(0)
    g1 = g1t + rng.normal(0, args.shape_noise, len(g1t))
    g2 = g2t + rng.normal(0, args.shape_noise, len(g2t))

    # survey mask: knock out a circular hole (missing data), like their COSMOS mask
    r_hole = np.hypot(nodes[:, 0] - 0.3, nodes[:, 1] + 0.4)
    masked = r_hole < args.mask_radius
    weight = np.ones(len(nodes)); weight[masked] = 0.0        # no data in the hole
    g1[masked] = 0.0; g2[masked] = 0.0

    # Kaiser-Squires (Fourier) reference
    kks = np.asarray(kaiser_squires(g1, g2, nodes))

    # FEMMI posterior
    fwd = DifferentiableForward(ops, lam_reg=args.lam)
    if args.neural:
        from femmi.priors import make_prior
        prior = make_prior("neural", ops, n_pix=args.n_pix, base=args.base, verbose=True)
        ps = sample_posterior(fwd, g1, g2, noise_std=args.shape_noise, prior=prior,
                              data_weight=weight, method="annealed_hmc",
                              n_levels=args.n_levels, steps_per_level=args.steps_per_level,
                              n_chains=args.n_chains, keep_final=4, seed=1, verbose=True)
    else:
        ps = sample_posterior(fwd, g1, g2, noise_std=args.shape_noise, wiener_length=0.5,
                              data_weight=weight, method="rto", n_samples=args.n_samples,
                              seed=1, verbose=True)
    # For the Gaussian (RTO) posterior the mean equals the MAP exactly, and the MAP
    # is the smooth regularised estimate; use it for the "mean" panel so it is not
    # speckled by the Monte-Carlo noise of a finite sample set. The samples (Fig. 2)
    # deliberately keep that speckle -- it IS the uncertainty.
    mean_display = ps.map_kappa if ps.method == "rto" else ps.mean

    sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 1.8
    l2 = lambda k: float(np.linalg.norm(k[sel] - kt[sel]) / np.linalg.norm(kt[sel]))
    print(f"\nmethod={ps.method}")
    print(f"  L2(kappa): KS={l2(kks):.3f}  FEMMI={l2(mean_display):.3f}")
    print(f"  mean posterior std (inner) = {ps.std[sel].mean():.3e}")

    _figures(nodes, kt, masked, kks, ps, mean_display, args)


def _figures(nodes, kt, masked, kks, ps, mean_display, args):
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    vmax = float(np.nanpercentile(kt, 99)) or 1.0
    tag = "neural + annealed HMC" if args.neural else "Wiener + RTO (exact)"

    # Figure 1: truth | mask | KS | posterior mean
    maskfield = np.where(masked, 1.0, 0.0)
    row1 = [("truth kappa", kt, "hot", 0, vmax), ("survey mask", maskfield, "gray", 0, 1),
            ("Kaiser-Squires", kks, "hot", 0, vmax), ("FEMMI posterior mean", mean_display, "hot", 0, vmax)]
    _panelrow(tri, row1, f"FEMMI probabilistic mass map -- {tag}  [Fig. 1]",
              _out("fig_paper_1"))

    # Figure 2: posterior std + 3 individual samples
    idx = np.linspace(0, len(ps.samples) - 1, 3).astype(int)
    row2 = [("posterior std (uncertainty)", ps.std, "viridis", None, None)]
    row2 += [(f"posterior sample {i+1}", ps.samples[j], "hot", 0, vmax) for i, j in enumerate(idx)]
    _panelrow(tri, row2, f"Uncertainty + posterior samples -- {tag}  [Fig. 2]",
              _out("fig_paper_2"))


def _panelrow(tri, panels, title, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), facecolor="#1a1a1a")
    for ax, (t, d, cmap, vmin, vmax) in zip(axes, panels):
        ax.set_facecolor("#111111")
        tc = ax.tripcolor(tri, np.nan_to_num(d), cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax)
        ax.set_title(t, color="white"); ax.set_aspect("equal"); ax.tick_params(colors="#aaa")
        fig.colorbar(tc, ax=ax, fraction=0.046)
    fig.suptitle(title, color="white", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig); print(f"wrote {os.path.normpath(out)}")


def _out(name):
    return os.path.join(os.path.dirname(__file__), "..", "outputs", f"{name}.png")


def main():
    ap = argparse.ArgumentParser(description="Reproduce Remy et al. 2020 figure structure with FEMMI")
    ap.add_argument("--nx", type=int, default=18)
    ap.add_argument("--shape-noise", type=float, default=0.06)
    ap.add_argument("--mask-radius", type=float, default=0.6)
    ap.add_argument("--lam", type=float, default=30.0,
                    help="prior precision in the Bayesian posterior (data term /2sigma^2); "
                         "~ lambda_MAP / (2 sigma_n^2), so larger than a MAPReconstructor lambda")
    ap.add_argument("--n-samples", type=int, default=200, help="RTO samples (Wiener)")
    ap.add_argument("--neural", action="store_true", help="neural score prior + annealed HMC")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--steps-per-level", type=int, default=12)
    ap.add_argument("--n-chains", type=int, default=40)
    ap.add_argument("--n-pix", type=int, default=32)
    ap.add_argument("--base", type=int, default=16)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
