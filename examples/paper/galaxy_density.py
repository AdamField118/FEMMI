"""Legacy density plotter (historical selection rules).

For publication use calibrated_comparison.py and report_calibration.py:
shared catalogues, independently tuned arms, held-out seeds and paired errors.
Legacy cached rows cannot be treated as the corrected experiment.
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
    ap.add_argument("--cache", default=None, metavar="FILE.json",
                    help="reuse results from FILE if it exists, else write them "
                         "there. The sweep is ~35 min at 6 seeds, so redrawing "
                         "the figure should not re-run the physics.")
    args = ap.parse_args()
    use_paper_style()

    if args.cache and os.path.exists(args.cache):
        import json
        with open(args.cache) as fh:
            raw = json.load(fh)
        print(f"loaded {len(raw)} results from {args.cache}")
    else:
        raw = density_sweep(n_effs=tuple(args.n_eff), noise_std=args.noise,
                            seeds=tuple(args.seeds), radius=args.radius)
        if args.cache:
            import json
            with open(args.cache, "w") as fh:
                json.dump(raw, fh, indent=1)
            print(f"wrote {len(raw)} results to {args.cache}")
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

    # Most of these are LOWER BOUNDS, not missing data: once Argyris's error
    # drops below anything the baseline reaches anywhere in the sweep, all we
    # know is that the baseline needs more than the densest catalog measured.
    # Plotting only the finite points would leave the panel nearly empty and
    # read as a failed measurement, when it is actually the strongest part of
    # the result -- so the bounds are drawn as up-arrows at n_max / n.
    fp = np.array([_equivalent_density(e, np_, ep) / n for n, e in zip(na, ea)])
    fk = np.array([_equivalent_density(e, nk, ek) / n for n, e in zip(na, ea)])
    for f, nb, eb, c, m, lab, jit in ((fp, np_, ep, PALETTE[1], "s", "vs P3", 0.97),
                                      (fk, nk, ek, PALETTE[3], "^", "vs KS", 1.03)):
        ok = np.isfinite(f)
        ax2.semilogx(na[ok], f[ok], color=c, lw=2, marker=m, ms=7, ls="none",
                     label=lab)
        # A bound applies where Argyris's error is below the baseline's best.
        # Require it to be worth stating: at the densest Argyris point the bound
        # is n_max/n_max = 1x, which is true and says nothing.
        lo = nb.max() / na
        bound = ~ok & (ea < eb.min()) & (lo > 1.05)
        if bound.any():
            ax2.errorbar(na[bound] * jit, lo[bound], yerr=0.30, lolims=True,
                         color=c, marker=m, ms=7, ls="none", capsize=4,
                         elinewidth=1.6)
    ax2.axhline(1.0, color="#777777", lw=1.0, ls="--")
    ax2.set_xlabel("Argyris $n_{\\mathrm{eff}}$ [gal arcmin$^{-2}$]")
    ax2.set_ylabel("density the other method needs $\\div$ Argyris")
    ax2.set_title("source-density equivalence factor\n(arrows: lower bounds, "
                  "sweep ran out of density)", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.set_ylim(bottom=0.9)

    fig.tight_layout(); fig.savefig(args.out)
    print(f"\nwrote {args.out}")
    print("Caveat that belongs with any quote of these numbers: random galaxy")
    print("positions make sliver triangles, and Argyris inverts a 21x21 Vandermonde")
    print("per element -- see the mesh lines above and MATH.md 18.3.10.")


if __name__ == "__main__":
    main()
