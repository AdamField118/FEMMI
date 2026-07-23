"""
examples/uncertainty_demo.py
Posterior uncertainty quantification on a FEMMI reconstruction, built on the
DIFFERENTIABLE forward. Shows the posterior mean and per-pixel standard-deviation
(uncertainty) map next to the truth and the MAP.

Default uses exact perturb-and-MAP (RTO) with the Gaussian/Wiener prior -- each
sample is an independent linear solve on perturbed data, so the uncertainty map
is exact for the linear-Gaussian posterior. With --neural it switches to the
score-based Langevin sampler driven by the learned NeuralScorePrior (the mode
that only needs the prior score).

Run:
    python examples/uncertainty_demo.py
    python examples/uncertainty_demo.py --neural
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators
from femmi.forward   import DifferentiableForward
from femmi.sampling  import sample_posterior
from femmi.catalog   import analytic_gaussian_shear


def run(args):
    ops = build_operators(args.nx, args.nx, -2.5, 2.5, -2.5, 2.5, verbose=False)
    nodes = np.array(ops.mesh.nodes)
    kt, g1a, g2a = analytic_gaussian_shear(nodes, sigma=0.5)
    rng = np.random.default_rng(0)
    g1 = g1a + rng.normal(0, args.shape_noise, len(g1a))
    g2 = g2a + rng.normal(0, args.shape_noise, len(g2a))

    fwd = DifferentiableForward(ops, lam_reg=args.lam)
    if args.neural:
        from femmi.priors import make_prior
        prior = make_prior("neural", ops, n_pix=args.n_pix, base=args.base, verbose=True)
        ps = sample_posterior(fwd, g1, g2, noise_std=args.shape_noise, prior=prior,
                              method="langevin", n_steps=args.n_steps, burnin=args.n_steps // 3,
                              thin=5, verbose=True)
    else:
        ps = sample_posterior(fwd, g1, g2, noise_std=args.shape_noise, wiener_length=0.5,
                              method="rto", n_samples=args.n_samples, verbose=True)

    sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 1.5
    l2 = lambda k: float(np.linalg.norm(k[sel] - kt[sel]) / np.linalg.norm(kt[sel]))
    print(f"\nmethod={ps.method}")
    print(f"  MAP L2={l2(ps.map_kappa):.3f}   post-mean L2={l2(ps.mean):.3f}")
    print(f"  mean posterior std (inner) = {ps.std[sel].mean():.3e}")

    _figure(nodes, kt, ps, args)


def _figure(nodes, kt, ps, args):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    panels = [("truth kappa", kt, "hot"), ("MAP", ps.map_kappa, "hot"),
              ("posterior mean", ps.mean, "hot"),
              ("posterior std (uncertainty)", ps.std, "viridis")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), facecolor="white")
    vmax = float(np.nanpercentile(kt, 99)) or 1.0
    for ax, (title, d, cmap) in zip(axes, panels):
        ax.set_facecolor("white")
        vm = None if cmap == "viridis" else vmax
        tc = ax.tripcolor(tri, d, cmap=cmap, shading="gouraud",
                          vmin=(0 if cmap == "hot" else None), vmax=vm)
        ax.set_title(title, color="#111111"); ax.set_aspect("equal"); ax.tick_params(colors="#555555")
        fig.colorbar(tc, ax=ax, fraction=0.046)
    tag = "neural (Langevin)" if args.neural else "Wiener (RTO exact)"
    fig.suptitle(f"FEMMI posterior UQ via the differentiable forward -- {tag}",
                 color="#111111", y=1.02)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs",
                       f"fig_uncertainty_{'neural' if args.neural else 'wiener'}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.normpath(out)}")


def main():
    ap = argparse.ArgumentParser(description="FEMMI posterior UQ demo")
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--shape-noise", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=40.0,
                    help="prior precision in the Bayesian posterior (data term /2sigma^2); "
                         "~ lambda_MAP / (2 sigma_n^2)")
    ap.add_argument("--n-samples", type=int, default=300, help="RTO samples")
    ap.add_argument("--neural", action="store_true", help="use the neural prior + Langevin")
    ap.add_argument("--n-steps", type=int, default=1500, help="Langevin steps (neural)")
    ap.add_argument("--n-pix", type=int, default=32, help="neural: score-grid size (must match the trained checkpoint)")
    ap.add_argument("--base", type=int, default=16, help="neural: U-Net base channels (must match the trained checkpoint)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
