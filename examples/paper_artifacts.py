"""
examples/paper_artifacts.py
Reproduce the Remy et al. 2020 (arXiv:2011.08271) figure structure with FEMMI:
a probabilistic mass map on masked, noisy data.

This is a SPECIFIC recreation script that reuses the SAME config schema and the
SAME pipeline builders as `femmi run` (femmi.pipeline / femmi.config) -- it just
adds Kaiser-Squires and the two paper-style figures on top. So one config
describes the run; this script turns it into the figures.

Figure 1: truth kappa | survey mask | Kaiser-Squires | posterior mean.
Figure 2: posterior std (uncertainty) | individual posterior samples.

The prior kind selects the sampler (Gaussian -> exact RTO; neural/other ->
annealed HMC). Truth is a self-contained two-peak field unless the config points
data.source at a FITS/frontier catalog.

Run:
    python examples/paper_artifacts.py --config configs/paper_artifacts.yaml
    python examples/paper_artifacts.py --config configs/paper_artifacts.yaml --set prior.kind=neural
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.config   import load_config
from femmi.pipeline import build_forward_and_data, build_prior
from femmi.forward  import DifferentiableForward
from femmi.inverse  import kaiser_squires
from femmi.sampling import sample_posterior


def run(cfg):
    d = build_forward_and_data(cfg)                 # ops, node-placed shear, truth, mask
    ops, g1n, g2n, weight = d["ops"], d["g1n"], d["g2n"], d["weight"]
    nodes, kt = d["nodes"], d["truth_nodes"]
    masked = weight <= 0

    kks = np.asarray(kaiser_squires(g1n, g2n, nodes))

    prior = build_prior(cfg, ops)
    lam = cfg.get("inverse.lam")
    fwd = DifferentiableForward(ops, lam_reg=(1e-2 if lam is None else lam))
    s = cfg.section("sampler")
    ps = sample_posterior(
        fwd, g1n, g2n, noise_std=d["noise_std"], prior=prior,
        wiener_length=cfg.get("inverse.wiener_length"), data_weight=weight,
        method=s.get("method", "auto"), n_samples=s.get("n_samples"),
        n_levels=s.get("n_levels"), steps_per_level=s.get("steps_per_level"),
        n_chains=s.get("n_chains"), n_leapfrog=s.get("n_leapfrog"),
        keep_final=s.get("keep_final"), sigma_max=s.get("sigma_max"),
        sigma_min=s.get("sigma_min"), n_delta_logp=s.get("n_delta_logp"),
        seed=s.get("seed"), verbose=True)

    # RTO posterior mean == MAP (smooth); use it so the mean panel is not speckled
    # by finite-sample Monte-Carlo noise. Samples (Fig. 2) keep the speckle -- it
    # IS the uncertainty.
    mean_display = ps.map_kappa if ps.method == "rto" else ps.mean

    sel = np.hypot(nodes[:, 0], nodes[:, 1]) < 0.7 * cfg.get("forward.half_width")
    l2 = lambda k: float(np.linalg.norm(k[sel] - kt[sel]) / np.linalg.norm(kt[sel]))
    tag = f"{cfg.get('prior.kind')} + {ps.method}"
    print(f"\nmethod={ps.method}  prior={cfg.get('prior.kind')}")
    print(f"  L2(kappa): KS={l2(kks):.3f}  FEMMI={l2(mean_display):.3f}")
    print(f"  mean posterior std (inner) = {ps.std[sel].mean():.3e}")

    _figures(nodes, kt, masked, kks, ps, mean_display, tag, cfg)


def _figures(nodes, kt, masked, kks, ps, mean_display, tag, cfg):
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    vmax = float(np.nanpercentile(kt, 99)) or 1.0
    from femmi.pipeline import _resolve_out_dir
    out_dir = _resolve_out_dir(cfg)
    out = lambda n: os.path.join(out_dir, f"{cfg.get('output.name')}_{n}.png")

    maskfield = np.where(masked, 1.0, 0.0)
    row1 = [("truth kappa", kt, "hot", 0, vmax), ("survey mask", maskfield, "gray", 0, 1),
            ("Kaiser-Squires", kks, "hot", 0, vmax), ("FEMMI posterior mean", mean_display, "hot", 0, vmax)]
    _panelrow(tri, row1, f"FEMMI probabilistic mass map -- {tag}  [Fig. 1]", out("fig1"))

    idx = np.linspace(0, len(ps.samples) - 1, 3).astype(int)
    row2 = [("posterior std (uncertainty)", ps.std, "viridis", None, None)]
    row2 += [(f"posterior sample {i+1}", ps.samples[j], "hot", 0, vmax) for i, j in enumerate(idx)]
    _panelrow(tri, row2, f"Uncertainty + posterior samples -- {tag}  [Fig. 2]", out("fig2"))


def _panelrow(tri, panels, title, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), facecolor="#1a1a1a")
    for ax, (t, dd, cmap, vmin, vmax) in zip(axes, panels):
        ax.set_facecolor("#111111")
        tc = ax.tripcolor(tri, np.nan_to_num(dd), cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax)
        ax.set_title(t, color="white"); ax.set_aspect("equal"); ax.tick_params(colors="#aaa")
        fig.colorbar(tc, ax=ax, fraction=0.046)
    fig.suptitle(title, color="white", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig); print(f"wrote {os.path.normpath(out)}")


def main():
    ap = argparse.ArgumentParser(description="Reproduce Remy et al. 2020 figures with FEMMI (config-driven)")
    ap.add_argument("--config", type=str, default=None,
                    help="YAML config (see configs/paper_artifacts.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value, e.g. --set prior.kind=neural")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from femmi.cli import _apply_overrides
    _apply_overrides(cfg, args.set)
    # this recreation uses the structured grid + posterior sampling
    cfg.set("forward.geometry", "square")
    cfg.set("inverse.method", "sample")
    run(cfg)


if __name__ == "__main__":
    main()
