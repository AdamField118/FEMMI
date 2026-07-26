"""
examples/paper/galaxy_density.py   [candidate paper claim]
What SOURCE DENSITY does each method need?

The mass-sheet line is closed: the DC mode's whole observable signature is an
edge effect on any domain, so no element or mesh recovers it (MATH.md 6.3a). What
replaces it is a claim about DATA EFFICIENCY, and this is the experiment behind
it.

The x-axis is the effective source density n_eff in gal/arcmin^2, not a raw
galaxy count, because that is how weak-lensing surveys are specified: DES Y3
reaches 5.6, HSC Y3 19.9, Euclid ~30. A count means nothing without the field
area, and n_eff is the one number a survey cannot simply buy more of -- it is set
by depth, seeing and shape-measurement success. The reference lines mark real
surveys so the measured densities land on a scale a reader recognises.

Everything is catalog-native: vertices sit AT galaxy positions for both FEM
methods, the same galaxies go to all three, and the truth is an analytic GalSim
NFW field that none of them generated. Accuracy is the DC-removed shape error,
since that is what survives the mass-sheet limitation.

Read the second panel: it converts the error curves into "the density the other
method needs to match Argyris, divided by Argyris's density", which is the
number a survey proposal would quote.

EVERYTHING IS AVERAGED OVER SEEDS, and that is not decoration. On one
realisation the equivalence factor is a ratio of interpolated densities on
curves that are themselves noisy, and it swings from 0.58x to 2.9x across seeds
0/1/2 -- P3 even beats Argyris at n_eff = 20 in two of the three. Quoting a
single seed here would be quoting a draw from that spread. The error bars are
standard errors on the mean of `--seeds` realisations.

    python examples/paper/galaxy_density.py
    python examples/paper/galaxy_density.py --n-eff 5 10 20 30 --seeds 0 1 2 3 4
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from femmi.density import (density_sweep, average_over_seeds, to_table,
                           SURVEY_NEFF)
from femmi.plotstyle import use_paper_style, PALETTE


def _curve(rows, name):
    """Measured density, DC-removed error and its standard error, coarse to fine."""
    r = sorted([x for x in rows if x.get("method", "").startswith(name)
                and "error" not in x], key=lambda z: z["n_eff"])
    return (np.array([x["n_eff"] for x in r], float),
            np.array([x["shape_l2"] for x in r]),
            np.array([x.get("shape_l2_std", 0.0) for x in r]))


def _equivalent_density(target_err, n_eff, err):
    """Source density this method needs to reach `target_err`, log-log interpolated.

    Returns nan when the target lies OUTSIDE the method's measured range: np.interp
    clamps at the endpoints, which silently reports a factor of exactly 1.00x and
    reads as "the methods converged" when it only means the sweep ran out of
    points. Those are dropped rather than plotted.
    """
    if target_err < err.min() or target_err > err.max():
        return np.nan
    return np.exp(np.interp(np.log(target_err),
                            np.log(err[::-1]), np.log(n_eff[::-1])))


def main():
    ap = argparse.ArgumentParser(description="Accuracy vs source density")
    ap.add_argument("--n-eff", type=float, nargs="+", default=[5.0, 10.0, 20.0, 30.0],
                    help="effective source densities, gal/arcmin^2")
    ap.add_argument("--radius", type=float, default=3.0, help="field radius, arcmin")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="catalog realisations to average over")
    ap.add_argument("-o", "--out", default="galaxy_density.png")
    args = ap.parse_args()
    use_paper_style()

    raw = density_sweep(n_effs=tuple(args.n_eff), noise_std=args.noise,
                        seeds=tuple(args.seeds), radius=args.radius)
    rows = average_over_seeds(raw)
    print("\nper realisation:")
    print(to_table(raw))
    print(f"\naveraged over {len(args.seeds)} realisations:")
    print(to_table(rows))

    na, ea, sa = _curve(rows, "Argyris")
    np_, ep, sp = _curve(rows, "P3")
    nk, ek, sk = _curve(rows, "Kaiser")

    print(f"\n{'Argyris at':>14}{'its error':>11}{'P3 needs':>11}{'factor':>9}"
          f"{'KS needs':>11}{'factor':>9}")
    print(f"{'gal/arcmin2':>14}{'':>11}{'gal/arcmin2':>11}{'':>9}"
          f"{'gal/arcmin2':>11}")
    for n, e in zip(na, ea):
        qp = _equivalent_density(e, np_, ep)
        qk = _equivalent_density(e, nk, ek)
        f = lambda q: ("      --       --" if not np.isfinite(q)
                       else f"{q:>11.1f}{q / n:>8.2f}x")
        print(f"{n:>14.1f}{e:>11.4f}{f(qp)}{f(qk)}")

    for r in rows:
        if r.get("method", "").startswith("Argyris"):
            print(f"  mesh at {r['n_eff']:>5.1f} gal/arcmin^2 ({r['n_gal']} gal): "
                  f"min angle {r['mesh_min_angle']:.2f} deg, worst cond "
                  f"{r['mesh_max_cond']:.1e}, "
                  f"{r['mesh_n_ill']:.1f}/{r['mesh_n_elements']:.0f} "
                  f"ill-conditioned (seed mean)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # 2% slack so a survey sitting exactly at a swept density still gets drawn --
    # the measured n_eff is a hair below nominal (whole galaxies, dropped ring).
    lo = min(na.min(), np_.min(), nk.min()) * 0.98
    hi = max(na.max(), np_.max(), nk.max()) * 1.02
    for ax in (ax1, ax2):
        for label, v in SURVEY_NEFF.items():
            if lo <= v <= hi:
                ax.axvline(v, color="#bbbbbb", lw=0.8, zorder=0)
                ax.text(v, 0.985, label, rotation=90, fontsize=7, color="#777777",
                        ha="right", va="top", transform=ax.get_xaxis_transform())

    for n, e, s, c, m, lab in ((na, ea, sa, PALETTE[0], "o", "Argyris (catalog)"),
                               (np_, ep, sp, PALETTE[1], "s", "P3 (catalog)"),
                               (nk, ek, sk, PALETTE[3], "^", "Kaiser-Squires")):
        ax1.errorbar(n, e, yerr=s, color=c, lw=2, marker=m, ms=6, capsize=3,
                     label=lab)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("effective source density $n_{\\mathrm{eff}}$ "
                   "[gal arcmin$^{-2}$]")
    ax1.set_ylabel("DC-removed relative $L^2$ error")
    ax1.set_title(f"accuracy vs source density ({len(args.seeds)} realisations)")
    ax1.legend(frameon=False, fontsize=9)

    fp = np.array([_equivalent_density(e, np_, ep) / n for n, e in zip(na, ea)])
    fk = np.array([_equivalent_density(e, nk, ek) / n for n, e in zip(na, ea)])
    mp, mk = np.isfinite(fp), np.isfinite(fk)
    ax2.semilogx(na[mp], fp[mp], color=PALETTE[1], lw=2, marker="s", ms=6,
                 label="vs P3")
    ax2.semilogx(na[mk], fk[mk], color=PALETTE[3], lw=2, marker="^", ms=6,
                 label="vs KS")
    ax2.axhline(1.0, color="#777777", lw=1.0, ls="--")
    ax2.set_xlabel("Argyris $n_{\\mathrm{eff}}$ [gal arcmin$^{-2}$]")
    ax2.set_ylabel("density the other method needs $\\div$ Argyris")
    ax2.set_title("source-density equivalence factor")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout(); fig.savefig(args.out)
    print(f"\nwrote {args.out}")
    print("Caveat that belongs with any quote of these numbers: random galaxy")
    print("positions make sliver triangles, and Argyris inverts a 21x21 Vandermonde")
    print("per element -- see the mesh lines above and MATH.md 18.3i.")


if __name__ == "__main__":
    main()
