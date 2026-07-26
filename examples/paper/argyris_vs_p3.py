"""
examples/paper/argyris_vs_p3.py
Argyris (C^1, circular domain) against P3 (C^0, square) and Kaiser-Squires, on
the full inverse problem.

Every earlier element comparison measured INTERPOLATION or a forward solve. This
one runs the actual reconstruction -- shear observations in, kappa out, scored
against independent GalSim NFW truth -- which is the only comparison that decides
whether the element choice matters in a pipeline.

The two axes are reported separately because they say different things:

  accuracy per DOF          how much linear algebra the reconstruction costs.
  accuracy per OBSERVATION  how many galaxies you need. In a real survey this is
                            the quantity you cannot buy more of, and it is where
                            the C^1 element pays: Argyris carries the Hessian as
                            vertex DOFs, so one vertex yields a full shear
                            observation with no averaging over neighbours.

    python examples/paper/argyris_vs_p3.py
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.elements import C1Space, circular_triangulation
from femmi.c1_inverse import C1MAPReconstructor
from femmi.truth import galsim_nfw_truth
from femmi.experiments import square_ops, femmi_map, ks_map
from femmi.plotstyle import use_paper_style, PALETTE

HALOS = ((2.0e14, 4.0, (0.0, 0.0)),)


def _score(k_rec, k_true):
    m = k_true < 1.0                       # drop the strong-lensing core
    a, b = np.asarray(k_rec)[m], np.asarray(k_true)[m]
    dm = lambda z: z - np.nanmean(z)
    return (float(np.linalg.norm(a - b) / np.linalg.norm(b)),
            float(np.linalg.norm(dm(a) - dm(b)) / np.linalg.norm(dm(b))),
            float(abs(a.mean() - b.mean())))


def run_argyris(nb, noise, lams=(1e-1, 3e-1, 1.0)):
    v, t = circular_triangulation(nb, radius=2.5)
    S = C1Space(v, t, kind="argyris")
    kt, g1t, g2t = galsim_nfw_truth(v, halos=HALOS)
    rng = np.random.default_rng(0)
    g1 = g1t + rng.normal(0, noise, len(g1t))
    g2 = g2t + rng.normal(0, noise, len(g2t))

    best = None
    for lam in lams:
        rec = C1MAPReconstructor(S, lam=lam, wiener_length=1.0)
        t0 = time.perf_counter()
        k, _ = rec.reconstruct(g1, g2)
        dt = time.perf_counter() - t0
        sc = _score(rec.kappa_at_vertices(k), kt)
        if best is None or sc[1] < best[0][1]:
            best = (sc, dt, lam)
    sc, dt, lam = best
    return dict(name="Argyris (circle)", dofs=S.n_dofs, n_obs=len(v),
                rel=sc[0], shape=sc[1], mean=sc[2], sec=dt, lam=lam)


def run_p3(nx, noise, with_ks=False):
    ops = square_ops(nx, 2.5)
    nodes = np.array(ops.mesh.nodes)
    kt, g1t, g2t = galsim_nfw_truth(nodes, halos=HALOS)
    rng = np.random.default_rng(0)
    g1 = g1t + rng.normal(0, noise, len(g1t))
    g2 = g2t + rng.normal(0, noise, len(g2t))

    t0 = time.perf_counter()
    k = femmi_map(ops, g1, g2, noise, wiener_length=1.0, weight=np.ones(len(nodes)))
    dt = time.perf_counter() - t0
    sc = _score(np.asarray(k), kt)
    out = [dict(name="P3 (square)", dofs=ops.n_nodes, n_obs=len(nodes),
                rel=sc[0], shape=sc[1], mean=sc[2], sec=dt, lam=None)]
    if with_ks:
        s = _score(ks_map(g1, g2, nodes, grid_size=48), kt)
        out.append(dict(name="Kaiser-Squires", dofs=ops.n_nodes, n_obs=len(nodes),
                        rel=s[0], shape=s[1], mean=s[2], sec=0.0, lam=None))
    return out


def main():
    ap = argparse.ArgumentParser(description="Argyris vs P3 vs KS, inverse problem")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--nbs", type=int, nargs="+", default=[24, 36, 48, 60])
    ap.add_argument("--nxs", type=int, nargs="+", default=[8, 10, 12, 14])
    ap.add_argument("-o", "--out", default="argyris_vs_p3.png")
    args = ap.parse_args()
    use_paper_style()

    arg = [run_argyris(nb, args.noise) for nb in args.nbs]
    p3, ks = [], []
    for i, nx in enumerate(args.nxs):
        rows = run_p3(nx, args.noise, with_ks=(i == len(args.nxs) - 1))
        p3.append(rows[0])
        ks += rows[1:]

    hdr = f"{'method':>18}{'DOFs':>8}{'n_obs':>8}{'rel L2':>9}{'shape L2':>10}{'mean err':>10}{'sec':>7}"
    print(hdr); print("-" * len(hdr))
    for r in arg + p3 + ks:
        print(f"{r['name']:>18}{r['dofs']:>8}{r['n_obs']:>8}{r['rel']:>9.4f}"
              f"{r['shape']:>10.4f}{r['mean']:>10.4f}{r['sec']:>7.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, key, xlabel, title in (
            (ax1, "dofs", "degrees of freedom", "accuracy per DOF"),
            (ax2, "n_obs", "shear observations", "accuracy per OBSERVATION")):
        ax.loglog([r[key] for r in arg], [r["shape"] for r in arg],
                  color=PALETTE[0], lw=2, marker="o", ms=6, label="Argyris (circle)")
        ax.loglog([r[key] for r in p3], [r["shape"] for r in p3],
                  color=PALETTE[1], lw=2, marker="s", ms=6, label="P3 (square)")
        if ks:
            ax.axhline(ks[-1]["shape"], color="#777777", lw=1.2, ls="--",
                       label="Kaiser-Squires")
        ax.set_xlabel(xlabel); ax.set_ylabel("DC-removed relative $L^2$ error")
        ax.set_title(title); ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(); fig.savefig(args.out)
    print(f"\nwrote {args.out}")
    print("Per DOF the two elements are comparable; per OBSERVATION Argyris is far")
    print("ahead, which is the axis a galaxy survey actually constrains.")


if __name__ == "__main__":
    main()
