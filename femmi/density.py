"""
femmi/density.py
Accuracy versus GALAXY DENSITY -- the candidate paper claim.

Everything here is parameterised by the EFFECTIVE SOURCE DENSITY

    n_eff  [galaxies per square arcminute]

because that, not a raw galaxy count, is the number weak-lensing surveys are
specified and compared by. A count is meaningless without the field area; a
density is directly comparable across surveys and is what a survey proposal,
a forecast, or a referee will ask for. Field geometry is in arcmin throughout
(femmi.truth.galsim_nfw_truth already treats mesh units as arcmin), so
n_gal = n_eff * pi * R^2 and the conversion is exact rather than nominal.

WHY THIS AND NOT THE MASS SHEET
-------------------------------
The mass-sheet line is closed: the DC mode's entire observable signature is an
edge effect on ANY domain (an infinite uniform sheet produces zero shear by
symmetry), so no element, mesh or geometry rescues it -- see MATH.md 6.3a.

What replaced it came out of the element comparison. On a structured mesh Argyris
matched P3's reconstruction accuracy from 8.5x fewer shear observations
(MATH.md 18.3h). The reason is structural: a P3 node contributes a shear estimate
only through an average over the elements meeting there, while an Argyris vertex
carries {u_xx, u_xy, u_yy} outright. If that survives on real catalog geometry, it
is a survey-relevant claim in a way the mass-sheet result never was --

    the same convergence accuracy at LOWER SOURCE DENSITY,

and n_eff is the one quantity a survey cannot simply buy more of: it is set by
depth, seeing and shape-measurement success.

This module runs that test honestly: vertices AT galaxy positions (no structured
grid), truth from femmi.truth (neither method's forward), the same galaxies given
to every method, and MULTIPLE CATALOG REALISATIONS -- see caveat 1, which turned
out to be the largest effect in the experiment.

WHAT IT ACTUALLY FOUND (6 realisations, MATH.md 18.3i)
------------------------------------------------------
Against KAISER-SQUIRES the advantage is solid: Argyris matches KS at 1.4x-4.1x
lower source density, separated by 4-10 sigma of catalog scatter across the whole
DES-through-HSC range, with the advantage largest at low density and gone by
Euclid density (KS's problem is estimator noise and boundary truncation, and
density buys both down).

Against catalog-native P3 the advantage is real in the mean at every density but
only resolved above the scatter at ONE of the four densities (4.1 sigma at
n_eff = 10, 0.9-1.8 sigma elsewhere), with a mean factor of 1.4x-2.0x. That is
SUGGESTIVE, NOT ESTABLISHED, and it should be quoted that way. The structured-mesh
8.5x does not survive contact with catalog geometry either way: the element helps
by a factor of order two against the FEM baseline, not of order ten.

THE CAVEATS THIS FOUND, WHICH ARE PART OF THE RESULT
----------------------------------------------------
1. CATALOG NOISE DOMINATES A SINGLE RUN. The equivalence factor is a ratio of
interpolated densities on curves that are themselves noisy, and across seeds
0/1/2 the Argyris-vs-P3 factor swings from 0.58x to 2.9x -- P3 beats Argyris
outright at n_eff = 20 in two of the three. The 3-seed and 6-seed averages of
that factor also disagree (4.11/2.04/1.08 against 1.95/--/1.43), so the factor is
not a stable estimator at this sample size even after averaging; the per-density
error comparison is. One realisation is a draw from that spread, not a
measurement, which is why `density_sweep` takes `seeds` rather than `seed`, and
why `average_over_seeds` carries a standard error.

2. MESH CONDITIONING. Random galaxy positions make sliver triangles, and Argyris
inverts a 21x21 Vandermonde per element. At survey densities the median
conditioning is a benign ~1e5 but the worst element reaches 1e11-1e14 -- three
surviving digits or fewer -- and the number of such elements grows with density,
because more points means more chances to draw a near-degenerate triple.
`mesh_quality` reports this, and any claim from this module has to be read next to
it: catalog-native C^1 needs mesh conditioning, it is not free.

THE OPEN LEAD
-------------
Argyris barely improves between n_eff = 10 and 20 (0.5159 -> 0.5224, flat within
the scatter) while P3 and KS keep improving, so something other than source
density is limiting it there. `lam` is pinned at 0.3 across the whole sweep, so
the likely culprit is over-regularisation of the denser catalogs -- more sources
need less smoothing. If so, the P3 comparison above is a LOWER BOUND on the
element's advantage. Re-running with lam chosen per catalog by
femmi.lambda_selection (Morozov / L-curve) is the test.
"""

from __future__ import annotations
import numpy as np


def mesh_quality(space):
    """Diagnostics for a C^1 space on scattered data.

    Returns min/median triangle angle and the per-element Vandermonde condition
    number, which is what actually degrades on slivers.
    """
    verts, tris = space.vertices, space.triangles
    angs, conds = [], []
    for i, tri in enumerate(tris):
        P = verts[tri]
        a = []
        for k in range(3):
            u1 = P[(k + 1) % 3] - P[k]; u2 = P[(k + 2) % 3] - P[k]
            a.append(np.degrees(np.arccos(np.clip(
                u1 @ u2 / (np.linalg.norm(u1) * np.linalg.norm(u2)), -1, 1))))
        angs.append(min(a))
        el = space.element(i)
        conds.append(float(np.linalg.cond(el._V)) if hasattr(el, "_V") else np.nan)
    angs = np.asarray(angs); conds = np.asarray(conds)
    return dict(min_angle=float(angs.min()), median_angle=float(np.median(angs)),
                median_cond=float(np.nanmedian(conds)),
                max_cond=float(np.nanmax(conds)),
                n_ill=int(np.nansum(conds > 1e10)), n_elements=len(tris))


# Effective source densities of real weak-lensing surveys, gal/arcmin^2, for
# putting any measured density on a scale a reader recognises.
SURVEY_NEFF = {"DES Y3": 5.6, "KiDS-1000": 6.2, "CFHTLenS": 11.0,
               "HSC Y3": 19.9, "LSST Y10": 27.0, "Euclid": 30.0}


def n_gal_for_density(n_eff, radius_arcmin):
    """Galaxies in a circular field of the given radius at density n_eff."""
    return int(round(float(n_eff) * np.pi * float(radius_arcmin) ** 2))


def density_for_n_gal(n_gal, radius_arcmin):
    """Inverse of n_gal_for_density -- gal/arcmin^2."""
    return float(n_gal) / (np.pi * float(radius_arcmin) ** 2)


def sample_catalog(n_gal, radius=3.0, seed=0):
    """Uniform galaxy positions in a disk of the given radius (arcmin)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n_gal)
    rr = radius * np.sqrt(rng.uniform(0, 1, n_gal))
    return rr * np.cos(th), rr * np.sin(th)


def _score(k_rec, k_true, mask=None):
    m = (k_true < 1.0) if mask is None else (mask & (k_true < 1.0))
    a, b = np.asarray(k_rec)[m], np.asarray(k_true)[m]
    dm = lambda z: z - np.nanmean(z)
    return dict(rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                shape_l2=float(np.linalg.norm(dm(a) - dm(b))
                               / np.linalg.norm(dm(b))),
                mean_err=float(abs(np.nanmean(a) - np.nanmean(b))))


def argyris_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0, lam=0.3,
                        wiener_length=1.0, halos=((2.0e14, 4.0, (0.0, 0.0)),)):
    """Catalog-native Argyris: one vertex per galaxy, ring vertices carry no data."""
    from .elements import C1Space, catalog_triangulation
    from .c1_inverse import C1MAPReconstructor
    from .truth import galsim_nfw_truth
    import time

    n_gal = n_gal_for_density(n_eff, radius)
    x, y = sample_catalog(n_gal, radius=radius, seed=seed)
    v, t, ring, _ = catalog_triangulation(x, y)
    S = C1Space(v, t, kind="argyris")

    kt, g1t, g2t = galsim_nfw_truth(v, halos=halos)
    rng = np.random.default_rng(seed + 1)
    g1 = g1t + rng.normal(0, noise_std, len(g1t))
    g2 = g2t + rng.normal(0, noise_std, len(g2t))

    t0 = time.perf_counter()
    rec = C1MAPReconstructor(S, lam=lam, wiener_length=wiener_length,
                             data_weight=(~ring).astype(float))
    k, _ = rec.reconstruct(g1, g2)
    dt = time.perf_counter() - t0

    n_used = int((~ring).sum())
    out = dict(method="Argyris (catalog)", n_eff=density_for_n_gal(n_used, radius),
               n_eff_nominal=float(n_eff), n_gal=n_used, radius_arcmin=float(radius),
               dofs=int(S.n_dofs), seconds=dt)
    out.update(_score(rec.kappa_at_vertices(k), kt, mask=~ring))
    out.update({f"mesh_{a}": b for a, b in mesh_quality(S).items()})
    return out


def p3_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0,
                   halos=((2.0e14, 4.0, (0.0, 0.0)),)):
    """Catalog-native P3 through the existing reconstruct_catalog path."""
    from .catalog import reconstruct_catalog
    from .truth import galsim_nfw_truth
    import time

    n_gal = n_gal_for_density(n_eff, radius)
    x, y = sample_catalog(n_gal, radius=radius, seed=seed)
    pts = np.stack([x, y], 1)
    kt, g1t, g2t = galsim_nfw_truth(pts, halos=halos)
    rng = np.random.default_rng(seed + 1)
    g1 = g1t + rng.normal(0, noise_std, n_gal)
    g2 = g2t + rng.normal(0, noise_std, n_gal)

    t0 = time.perf_counter()
    res = reconstruct_catalog(x, y, g1, g2, verbose=False, noise_std=noise_std)
    dt = time.perf_counter() - t0

    out = dict(method="P3 (catalog)", n_eff=density_for_n_gal(n_gal, radius),
               n_eff_nominal=float(n_eff), n_gal=int(n_gal),
               radius_arcmin=float(radius), dofs=int(res.ops.n_nodes), seconds=dt)
    ok = np.isfinite(res.kappa_gal)
    out.update(_score(res.kappa_gal[ok], kt[ok]))
    return out


def ks_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0, grid_size=32,
                   smoothing_px=1.0, halos=((2.0e14, 4.0, (0.0, 0.0)),)):
    """Kaiser-Squires on the same catalog, binned onto a grid."""
    from .catalog import kaiser_squires_binned
    from .truth import galsim_nfw_truth

    n_gal = n_gal_for_density(n_eff, radius)
    x, y = sample_catalog(n_gal, radius=radius, seed=seed)
    pts = np.stack([x, y], 1)
    kt, g1t, g2t = galsim_nfw_truth(pts, halos=halos)
    rng = np.random.default_rng(seed + 1)
    g1 = g1t + rng.normal(0, noise_std, n_gal)
    g2 = g2t + rng.normal(0, noise_std, n_gal)

    k = kaiser_squires_binned(x, y, g1, g2, grid_size=grid_size,
                              smoothing_px=smoothing_px, eval_pts=pts)
    out = dict(method="Kaiser-Squires", n_eff=density_for_n_gal(n_gal, radius),
               n_eff_nominal=float(n_eff), n_gal=int(n_gal),
               radius_arcmin=float(radius), dofs=grid_size**2, seconds=0.0)
    out.update(_score(k, kt))
    return out


def density_sweep(n_effs=(5.0, 10.0, 20.0, 30.0), noise_std=0.05, seeds=(0, 1, 2),
                  radius=3.0, methods=("argyris", "p3", "ks"), verbose=True):
    """Accuracy vs SOURCE DENSITY for each method, on the same catalogs.

    n_effs is in gal/arcmin^2. The defaults bracket the real surveys in
    SURVEY_NEFF, from DES Y3 (5.6) to Euclid (30). Field radius is in arcmin, so
    the galaxy count follows from the density and is not a free knob.

    SEEDS ARE NOT OPTIONAL, and that is a finding rather than a convenience. A
    single realisation of this experiment is not reproducible in the direction
    that matters: across seeds 0/1/2 the Argyris-vs-P3 equivalence factor swings
    from 0.58x to 2.9x, and P3 beats Argyris outright at n_eff = 20 in two runs
    out of three. Anything read off one seed is a draw from that spread, not a
    measurement. `seeds` defaults to three and `average_over_seeds` collapses
    them with a standard error, so the noise is visible instead of implied.

    Returns one result dict per (density, method, seed), each tagged with `seed`.
    """
    runners = dict(argyris=argyris_catalog_run, p3=p3_catalog_run, ks=ks_catalog_run)
    rows = []
    for seed in seeds:
        for n_eff in n_effs:
            n_gal = n_gal_for_density(n_eff, radius)
            for m in methods:
                if verbose:
                    print(f"  seed {seed}  n_eff={n_eff:5.1f} gal/arcmin^2 "
                          f"({n_gal:5d} gal)  {m} ...", flush=True)
                try:
                    r = runners[m](n_eff, noise_std=noise_std, seed=seed,
                                   radius=radius)
                except Exception as exc:
                    r = dict(method=m, n_eff=float(n_eff),
                             n_eff_nominal=float(n_eff), n_gal=n_gal,
                             radius_arcmin=float(radius),
                             error=f"{type(exc).__name__}: {exc}")
                r["seed"] = int(seed)
                rows.append(r)
    return rows


_AVG_KEYS = ("rel_l2", "shape_l2", "mean_err", "seconds", "n_eff", "n_gal", "dofs")


def average_over_seeds(rows):
    """Collapse a multi-seed sweep to one row per (method, nominal density).

    Adds `<key>_std` -- the STANDARD ERROR of the mean, not the sample spread,
    because what is being quoted is the mean curve. `n_seeds` records how many
    realisations survived; a cell where some seeds errored is averaged over the
    ones that ran and says so rather than silently changing meaning.
    """
    ok = [r for r in rows if "error" not in r]
    groups = {}
    for r in ok:
        groups.setdefault((r["method"], r["n_eff_nominal"]), []).append(r)

    out = []
    for (method, n_nom), grp in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = dict(method=method, n_eff_nominal=float(n_nom), n_seeds=len(grp),
                   seeds=sorted(int(g["seed"]) for g in grp),
                   radius_arcmin=grp[0]["radius_arcmin"])
        for k in _AVG_KEYS:
            v = np.array([g[k] for g in grp], float)
            row[k] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        for k in grp[0]:
            if k.startswith("mesh_"):
                row[k] = float(np.mean([g[k] for g in grp]))
        row["n_gal"] = int(round(row["n_gal"]))
        row["dofs"] = int(round(row["dofs"]))
        out.append(row)
    return out


def to_table(rows):
    """Density first, count second -- the count is a consequence of the density
    and the field area, and only the density is comparable to a survey.

    Accepts raw per-seed rows or the output of `average_over_seeds`; in the
    averaged case the standard error on the shape error is printed next to it,
    because that spread is the whole reason the claim is quoted as a range.
    """
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    avg = any("shape_l2_std" in r for r in ok)
    tail = f"{'+/-':>8}" if avg else f"{'seed':>6}"
    hdr = (f"{'method':<20}{'n_eff':>8}{'n_gal':>7}{'DOFs':>8}{'rel L2':>9}"
           f"{'shape L2':>10}{tail}{'mean err':>10}{'sec':>7}")
    lines = [hdr, f"{'':<20}{'/arcmin2':>8}", "-" * len(hdr)]
    for r in sorted(ok, key=lambda z: (z["n_eff_nominal"], z["method"],
                                       z.get("seed", 0))):
        t = (f"{r['shape_l2_std']:>8.4f}" if avg else f"{r.get('seed', 0):>6d}")
        lines.append(f"{r['method']:<20}{r['n_eff']:>8.2f}{r['n_gal']:>7}"
                     f"{r['dofs']:>8}{r['rel_l2']:>9.4f}{r['shape_l2']:>10.4f}"
                     f"{t}{r['mean_err']:>10.4f}{r['seconds']:>7.1f}")
    for r in bad:
        lines.append(f"{r['method']:<20}{r['n_eff_nominal']:>8.2f}"
                     f"{r['n_gal']:>7}  -- {r['error']}")
    return "\n".join(lines)
