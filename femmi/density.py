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
Argyris beats KAISER-SQUIRES at every swept density (5.5 to 21 sigma of catalog
scatter) and beats catalog-native P3 for n_eff >= 10 (3.8 to 9.9 sigma). At the
sparsest density the P3 gap is 1.4 sigma and is NOT claimed.

From n_eff = 10 upward its error is below anything either baseline reaches
anywhere in the swept range, so the equivalence factor is reported as a bound:
at CFHTLenS-like density (10 gal/arcmin^2) catalog-native Argyris already beats
what both baselines achieve at EUCLID density (30). That is a factor of more
than three. At the sparsest density, where a finite factor is measurable, it is
1.84x against P3 and 3.95x against KS.

Getting there required fixing a fairness bug rather than gathering more data:
reconstruct_catalog selects lambda per catalog, and the Argyris arm did not,
which cost about a fifth of its accuracy and made the P3 comparison look
merely suggestive. See c1_lambda and MATH.md 18.3j -- the investigation reached
a conclusion about lambda SELECTION that is a result in its own right.

The structured-mesh 8.5x (MATH.md 18.3h) still does not survive contact with
catalog geometry; that figure came from handing P3 an observation at every node.

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

3. ONE TRUTH FIELD. A single centred analytic NFW halo, which is the case a C^1
element should suit best. femmi.truth already provides lognormal and MassiveNuS
alternatives; until the sweep runs on those, the scope of the claim is a smooth
peaked field.

THE OPEN LEAD, NOW CLOSED
-------------------------
Argyris used to be flat between n_eff = 10 and 20 (0.5159 -> 0.5224) while both
baselines kept improving. That was the pinned lambda, and the fix moved the
whole curve: 0.6288, 0.4814, 0.4117, 0.3798, monotone again. What is left is the
sparsest density, where the per-catalog optimum genuinely varies and a single
constant cannot track it; cross-validation on held-out galaxies is the route
there, because it does not go through the flat residual curve that defeats
Morozov (MATH.md 18.3j).
"""

from __future__ import annotations
import numpy as np


def mesh_quality(space):
    """Diagnostics for a C^1 space on scattered data.

    Returns min/median triangle angle and the per-element Vandermonde condition
    number, which is what actually degrades on slivers.

    Two condition numbers, because they mean different things. `*_cond` is the
    RAW Vandermonde, i.e. how bad the element geometry is. `*_cond_eq` is after
    the two-sided equilibration that `elements.equilibrated_inverse` actually
    applies before inverting, i.e. how much of that badness reaches the
    arithmetic. The gap between them is what the scaling buys; the residual in
    `*_cond_eq` is genuine geometric degeneracy that no scaling can remove.
    """
    verts, tris = space.vertices, space.triangles
    angs, conds, conds_eq = [], [], []
    for i, tri in enumerate(tris):
        P = verts[tri]
        a = []
        for k in range(3):
            u1 = P[(k + 1) % 3] - P[k]; u2 = P[(k + 2) % 3] - P[k]
            a.append(np.degrees(np.arccos(np.clip(
                u1 @ u2 / (np.linalg.norm(u1) * np.linalg.norm(u2)), -1, 1))))
        angs.append(min(a))
        el = space.element(i)
        conds.append(float(getattr(el, "_cond_raw", np.nan)))
        conds_eq.append(float(getattr(el, "_cond_eq", np.nan)))
    angs = np.asarray(angs)
    conds = np.asarray(conds); conds_eq = np.asarray(conds_eq)
    # HCT solves a least-squares system rather than inverting a square
    # Vandermonde, so it reports no condition number at all. That is a real
    # difference between the elements, not missing data -- nanmedian of an
    # all-nan array warns and returns nan, so it is handled explicitly.
    agg = lambda v, fn: float(fn(v)) if np.isfinite(v).any() else float("nan")
    return dict(min_angle=float(angs.min()), median_angle=float(np.median(angs)),
                median_cond=agg(conds, np.nanmedian),
                max_cond=agg(conds, np.nanmax),
                median_cond_eq=agg(conds_eq, np.nanmedian),
                max_cond_eq=agg(conds_eq, np.nanmax),
                n_ill=int(np.nansum(conds > 1e10)),
                n_ill_eq=int(np.nansum(conds_eq > 1e10)),
                n_elements=len(tris))


# Effective source densities of real weak-lensing surveys, gal/arcmin^2, for
# putting any measured density on a scale a reader recognises.
SURVEY_NEFF = {"DES Y3": 5.6, "KiDS-1000": 6.2, "CFHTLenS": 11.0,
               "HSC Y3": 19.9, "LSST Y10": 27.0, "Euclid": 30.0}


# Kaiser-Squires grid resolution, CALIBRATED rather than assumed.
#
# This was the mirror of the lambda fairness bug, and it was worse. After
# MATH.md 18.3j the Argyris arm chose a calibrated lambda and the P3 arm ran its
# own per-catalog Morozov, while KS stayed pinned at grid_size=32,
# smoothing_px=1.0 -- numbers that were never measured. Grid resolution IS the
# KS regularisation knob (a coarser pixel averages more galaxies), so the
# comparison had become tuned-against-untuned in exactly the direction that
# flatters the claim.
#
# Calibrated on the same held-out catalogs as everything else (seeds 100-102,
# disjoint from every reported seed), sweeping grid_size in 6..48 and
# smoothing_px in 0..2, the optimum is INTERIOR at every density and the old
# defaults were badly wrong:
#
#   n_eff   best (grid, smooth)   err     err at (32, 1.0)   gain
#       5        (12, 1.0)      0.5943        0.8671        +31.5%
#      10        (12, 1.0)      0.4932        0.7768        +36.5%
#      20        (16, 1.0)      0.3617        0.5875        +38.4%
#      30        (24, 1.0)      0.4032        0.4730        +14.8%
#
# 15-38% of KS's error was the untuned grid. The first search bottomed out at
# its own lower edge and would have reported a boundary value as the optimum;
# extending it down to grid_size=6 moved the answer. Resolution rises with
# density, which is what it should do -- more galaxies support finer pixels
# before shot noise dominates -- and smoothing_px = 1.0 was right all along.
KS_CALIBRATION = {5.0: (12, 1.0), 10.0: (12, 1.0), 20.0: (16, 1.0),
                  30.0: (24, 1.0)}


def ks_params_for_density(n_eff):
    """(grid_size, smoothing_px) for KS at a given source density.

    Log-interpolated in n_eff between the calibrated anchors and clamped
    outside them, so an unswept density gets a sensible value rather than the
    old untuned default. Interpolating the LOG of grid_size keeps the pixel
    scale smooth in density, which is the quantity that actually matters.
    """
    ns = np.array(sorted(KS_CALIBRATION))
    gs = np.array([KS_CALIBRATION[n][0] for n in ns], float)
    sp = np.array([KS_CALIBRATION[n][1] for n in ns], float)
    ln = np.log(max(float(n_eff), 1e-6))
    grid = int(round(float(np.exp(np.interp(ln, np.log(ns), np.log(gs))))))
    smooth = float(np.interp(ln, np.log(ns), sp))
    return max(grid, 4), smooth


def n_gal_for_density(n_eff, radius_arcmin):
    """Galaxies in a circular field of the given radius at density n_eff."""
    return int(round(float(n_eff) * np.pi * float(radius_arcmin) ** 2))


def density_for_n_gal(n_gal, radius_arcmin):
    """Inverse of n_gal_for_density -- gal/arcmin^2."""
    return float(n_gal) / (np.pi * float(radius_arcmin) ** 2)


def _truth_at(pts, truth, halos, radius, seed, truth_kw=None):
    """(kappa, g1, g2) from femmi.truth at the given points.

    One place, so all three arms are guaranteed to score against the SAME field
    -- the comparison is meaningless otherwise, and three call sites is three
    chances for them to drift apart.
    """
    from .truth import independent_truth
    kw = dict(truth_kw or {})
    if truth == "nfw":
        kw.setdefault("halos", halos)
    else:
        # lognormal / massivenus are built on a square patch; it must cover the
        # disk or the corners of the field sample outside the generated map
        kw.setdefault("half_width", float(radius) * 1.05)
        kw.setdefault("seed", int(seed))
    return independent_truth(np.asarray(pts, float), source=truth, **kw)


def sample_catalog(n_gal, radius=3.0, seed=0, masks=(), clustering=0.0,
                   return_weights=False, weight_scatter=0.0):
    """Galaxy positions in a disk of the given radius (arcmin).

    Uniform and unmasked by default -- that is the configuration every reported
    result uses. The extras exist so the claim can be tested against the ways a
    real catalog differs from that, rather than assumed to survive them:

    masks : sequence of (x, y, r) circular holes -- bright stars, bad CCDs.
        These are the interesting perturbation, because a hole is an INTERIOR
        boundary and interior boundaries are where FEMMI's exact BEM far-field
        is supposed to beat KS's truncation. Measured, it ENLARGES the advantage
        (MATH.md 18.3ja).
    clustering : 0 gives Poisson positions. Above 0, a fraction of the galaxies
        are drawn in tight groups instead, which makes the sliver problem worse
        and is therefore the pessimistic case for the C^1 mesh.
    weight_scatter : lognormal scatter in per-galaxy weights (0 = equal weights).

    Returns (x, y) or (x, y, w) when return_weights.
    """
    rng = np.random.default_rng(seed)
    n_target = int(n_gal)

    if not masks:
        # EXACTLY the original two draws, in the original order. Rejection
        # sampling consumes the stream differently, so routing the unmasked case
        # through it would silently change every catalog for a given seed and
        # break reproducibility of every number already reported. That happened
        # once and was caught only because a spot-check disagreed with a
        # committed number; tests/test_density_extensions.py pins it.
        th = rng.uniform(0, 2 * np.pi, n_target)
        rr = radius * np.sqrt(rng.uniform(0, 1, n_target))
        x, y = rr * np.cos(th), rr * np.sin(th)
    else:
        # Rejection-sample so a masked catalog still reaches the requested
        # COUNT: a survey with a target n_eff and a masked field ends up denser
        # in the part that survives, which is what this reproduces.
        xs, ys = [], []
        have = 0
        while have < n_target:
            m = max(n_target * 2, 64)
            th = rng.uniform(0, 2 * np.pi, m)
            rr = radius * np.sqrt(rng.uniform(0, 1, m))
            px, py = rr * np.cos(th), rr * np.sin(th)
            keep = np.ones(m, bool)
            for cx, cy, cr in masks:
                keep &= np.hypot(px - cx, py - cy) > cr
            xs.append(px[keep]); ys.append(py[keep])
            have += int(keep.sum())
        x = np.concatenate(xs)[:n_target]
        y = np.concatenate(ys)[:n_target]

    if clustering > 0:
        n_c = int(round(clustering * n_target))
        if n_c > 1:
            n_seed = max(1, n_c // 8)
            pick = rng.choice(n_target, n_seed, replace=False)
            sigma = 0.02 * radius
            idx = rng.choice(pick, n_c)
            x[:n_c] = x[idx] + rng.normal(0, sigma, n_c)
            y[:n_c] = y[idx] + rng.normal(0, sigma, n_c)
            r = np.hypot(x, y)
            out = r > radius * 0.999          # keep the clustered ones inside
            x[out] *= radius * 0.999 / r[out]
            y[out] *= radius * 0.999 / r[out]

    if not return_weights:
        return x, y
    w = (np.ones(n_target) if weight_scatter <= 0
         else np.exp(rng.normal(0, weight_scatter, n_target)))
    return x, y, w / w.mean()


def _score(k_rec, k_true, mask=None):
    m = (k_true < 1.0) if mask is None else (mask & (k_true < 1.0))
    a, b = np.asarray(k_rec)[m], np.asarray(k_true)[m]
    dm = lambda z: z - np.nanmean(z)
    return dict(rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                shape_l2=float(np.linalg.norm(dm(a) - dm(b))
                               / np.linalg.norm(dm(b))),
                mean_err=float(abs(np.nanmean(a) - np.nanmean(b))))


def c1_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0, lam=None,
                   wiener_length=1.0, halos=((2.0e14, 4.0, (0.0, 0.0)),),
                   lam_grid=None, truth="nfw", truth_kw=None, catalog_kw=None,
                   kind="argyris"):
    """Catalog-native C^1: one vertex per galaxy, ring vertices carry no data.

    lam : None (default) uses femmi.c1_lambda.CALIBRATED_LAM = 1.2111, a single
        weight fitted on HELD-OUT catalogs (seeds 100-102) and then frozen. It
        replaced the old pinned 0.3, which was 11% worse overall and 21% worse
        at survey densities, and which produced the flat spot in MATH.md 18.3i.

        "auto" selects per catalog by Morozov/L-curve. That is available but NOT
        the default, because it measurably does not work here: blind on 24
        catalogs it beats a fixed lambda in only 10 and adds variance, since the
        residual is too flat in lambda to locate the optimum (MATH.md 18.3j).

        "cv" selects per catalog by 5-fold cross-validation on held-out
        galaxies. It WORKS -- 20 wins in 24 catalogs, and it recovers 77% of the
        remaining oracle gap at the sparsest density, which is exactly where the
        fixed constant is weak (MATH.md 18.3n). It is not the default only
        because it costs ~35 MAP solves per catalog against one, and buys 2.4%
        overall since the constant is already at the oracle for n_eff >= 20.
        Use it for sparse catalogs.

        A float pins it explicitly.

    The reported `seconds` covers lambda selection as well as the final solve
    when selection runs, since that is what the method costs when used honestly.
    `lam_used` and `lam_method` record what was chosen and by which rule.
    """
    from .elements import C1Space, catalog_triangulation
    from .c1_inverse import C1MAPReconstructor
    from .c1_lambda import select_c1_lambda, cv_lambda, CALIBRATED_LAM
    import time

    if lam is None:
        lam = CALIBRATED_LAM

    n_gal = n_gal_for_density(n_eff, radius)
    x, y = sample_catalog(n_gal, radius=radius, seed=seed,
                          **(catalog_kw or {}))
    v, t, ring, _ = catalog_triangulation(x, y)
    S = C1Space(v, t, kind=kind)

    kt, g1t, g2t = _truth_at(v, truth, halos, radius, seed, truth_kw)
    rng = np.random.default_rng(seed + 1)
    g1 = g1t + rng.normal(0, noise_std, len(g1t))
    g2 = g2t + rng.normal(0, noise_std, len(g2t))

    t0 = time.perf_counter()
    _selecting = lam in ("auto", "cv")
    rec = C1MAPReconstructor(S, lam=1.0 if _selecting else float(lam),
                             wiener_length=wiener_length,
                             data_weight=(~ring).astype(float))
    if lam == "auto":
        lam_used, info = select_c1_lambda(rec, g1, g2, noise_std=noise_std,
                                          lam_grid=lam_grid)
        rec.lam, lam_method = lam_used, info["method"]
    elif lam == "cv":
        lam_used, info = cv_lambda(rec, g1, g2, lam_grid=lam_grid, seed=seed)
        rec.lam, lam_method = lam_used, info["method"]
    else:
        lam_used, lam_method = float(lam), "fixed"
    k, _ = rec.reconstruct(g1, g2)
    dt = time.perf_counter() - t0

    n_used = int((~ring).sum())
    out = dict(method=f"{kind.capitalize()} (catalog)",
               n_eff=density_for_n_gal(n_used, radius),
               n_eff_nominal=float(n_eff), n_gal=n_used, radius_arcmin=float(radius),
               dofs=int(S.n_dofs), seconds=dt, lam_used=float(lam_used),
               lam_method=lam_method)
    out.update(_score(rec.kappa_at_vertices(k), kt, mask=~ring))
    out.update({f"mesh_{a}": b for a, b in mesh_quality(S).items()})
    return out


def argyris_catalog_run(n_eff, **kw):
    """Catalog-native Argyris (21 DOF/element, Hessian carried as vertex DOFs)."""
    return c1_catalog_run(n_eff, kind="argyris", **kw)


def hct_catalog_run(n_eff, **kw):
    """Catalog-native HCT (12 DOF/element, Hessian evaluated not carried).

    The cheap C^1 arm: a 12x12 element Vandermonde instead of 21x21, at ~36%
    fewer global DOFs on the same mesh. Measured, it does NOT pay -- Argyris
    beats it by 1.2/5.4/6.6 sigma with the gap growing, and wins per DOF too.
    But it runs at Argyris's calibrated lambda, so it is the untuned arm; see
    MATH.md 18.3jb.
    """
    return c1_catalog_run(n_eff, kind="hct", **kw)


def p3_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0,
                   halos=((2.0e14, 4.0, (0.0, 0.0)),), use_morozov=True,
                   truth="nfw", truth_kw=None, catalog_kw=None):
    """Catalog-native P3 through the existing reconstruct_catalog path.

    use_morozov is exposed only so the two arms can be pinned together. It has
    defaulted to True inside reconstruct_catalog since long before this module
    existed, which is precisely why a pinned Argyris lambda made the comparison
    unfair -- see argyris_catalog_run.
    """
    from .catalog import reconstruct_catalog
    import time

    n_gal = n_gal_for_density(n_eff, radius)
    x, y = sample_catalog(n_gal, radius=radius, seed=seed,
                          **(catalog_kw or {}))
    pts = np.stack([x, y], 1)
    kt, g1t, g2t = _truth_at(pts, truth, halos, radius, seed, truth_kw)
    rng = np.random.default_rng(seed + 1)
    g1 = g1t + rng.normal(0, noise_std, n_gal)
    g2 = g2t + rng.normal(0, noise_std, n_gal)

    t0 = time.perf_counter()
    res = reconstruct_catalog(x, y, g1, g2, verbose=False, noise_std=noise_std,
                              use_morozov=use_morozov)
    dt = time.perf_counter() - t0

    out = dict(method="P3 (catalog)", n_eff=density_for_n_gal(n_gal, radius),
               n_eff_nominal=float(n_eff), n_gal=int(n_gal),
               radius_arcmin=float(radius), dofs=int(res.ops.n_nodes), seconds=dt,
               lam_used=float(res.lam_reg),
               lam_method="morozov" if use_morozov else "fixed")
    ok = np.isfinite(res.kappa_gal)
    out.update(_score(res.kappa_gal[ok], kt[ok]))
    return out


def ks_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0, grid_size=None,
                   smoothing_px=None, halos=((2.0e14, 4.0, (0.0, 0.0)),),
                   truth="nfw", truth_kw=None, catalog_kw=None):
    """Kaiser-Squires on the same catalog, binned onto a grid.

    grid_size / smoothing_px default to the held-out calibration in
    KS_CALIBRATION rather than to the old pinned (32, 1.0), which cost KS
    15-38% of its accuracy and made every comparison against it unfair. Pass
    explicit values to pin them.
    """
    from .catalog import kaiser_squires_binned

    n_gal = n_gal_for_density(n_eff, radius)
    _gs, _sp = ks_params_for_density(n_eff)
    grid_size = _gs if grid_size is None else int(grid_size)
    smoothing_px = _sp if smoothing_px is None else float(smoothing_px)
    x, y = sample_catalog(n_gal, radius=radius, seed=seed,
                          **(catalog_kw or {}))
    pts = np.stack([x, y], 1)
    kt, g1t, g2t = _truth_at(pts, truth, halos, radius, seed, truth_kw)
    rng = np.random.default_rng(seed + 1)
    g1 = g1t + rng.normal(0, noise_std, n_gal)
    g2 = g2t + rng.normal(0, noise_std, n_gal)

    k = kaiser_squires_binned(x, y, g1, g2, grid_size=grid_size,
                              smoothing_px=smoothing_px, eval_pts=pts)
    out = dict(method="Kaiser-Squires", n_eff=density_for_n_gal(n_gal, radius),
               n_eff_nominal=float(n_eff), n_gal=int(n_gal),
               radius_arcmin=float(radius), dofs=grid_size**2, seconds=0.0,
               ks_grid_size=int(grid_size), ks_smoothing_px=float(smoothing_px))
    out.update(_score(k, kt))
    return out


def density_sweep(n_effs=(5.0, 10.0, 20.0, 30.0), noise_std=0.05, seeds=(0, 1, 2),
                  radius=3.0, methods=("argyris", "p3", "ks"), verbose=True,
                  truth="nfw", truth_kw=None, catalog_kw=None, **run_kw):
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
    runners = dict(argyris=argyris_catalog_run, hct=hct_catalog_run,
                   p3=p3_catalog_run, ks=ks_catalog_run)
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
                                   radius=radius, truth=truth,
                                   truth_kw=truth_kw, catalog_kw=catalog_kw,
                                   **run_kw)
                except Exception as exc:
                    r = dict(method=m, n_eff=float(n_eff),
                             n_eff_nominal=float(n_eff), n_gal=n_gal,
                             radius_arcmin=float(radius),
                             error=f"{type(exc).__name__}: {exc}")
                r["seed"] = int(seed)
                r["truth"] = str(truth)
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


def paired_comparison(rows, method_a, method_b, key="shape_l2"):
    """Per-density PAIRED comparison of two methods across catalog realisations.

    WHY PAIRED. Every arm sees the same catalogs, the same galaxy positions, the
    same noise realisation and the same truth. The catalog-to-catalog scatter is
    therefore SHARED, and it is large -- comparable to the differences being
    measured. Quoting

        diff / sqrt(se_A^2 + se_B^2)

    treats the two arms as independent samples and throws that cancellation
    away, which understates every significance in proportion to how much of the
    variance is common. The differences themselves are what should be averaged.

    This is not a refinement of taste: the win-count statistics already used for
    lambda selection (20/24 for CV against 10/24 for Morozov) are paired, and
    they were decisive precisely where the marginal error bars overlapped. A
    synthetic check makes the size of the error plain -- a true gap of 0.02 under
    shared scatter of 0.10 gives paired t of 9-11 and a marginal sigma of 0.3.

    Returns one dict per nominal density with the mean paired difference, its
    standard error, the paired t statistic, and the sign-test win count -- the
    last being distribution-free, which matters at n = 6.
    """
    def pick(m):
        out = {}
        for r in rows:
            # a row without a `seed` cannot be paired -- averaged rows are the
            # common case, and pairing them would silently compare one aggregate
            # against another and report a meaningless t
            if ("error" in r or "seed" not in r or key not in r
                    or not r.get("method", "").startswith(m)):
                continue
            out[(r["n_eff_nominal"], r["seed"])] = r[key]
        return out

    A, B = pick(method_a), pick(method_b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise ValueError(f"no catalogs shared by {method_a!r} and {method_b!r}; "
                         "paired statistics need per-seed rows, not averages")

    out = []
    for n_nom in sorted({k[0] for k in shared}):
        keys = [k for k in shared if k[0] == n_nom]
        d = np.array([B[k] - A[k] for k in keys])      # >0 means A is better
        n = len(d)
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out.append(dict(n_eff_nominal=float(n_nom), n_pairs=n,
                        mean_diff=float(d.mean()), se_diff=se,
                        t=float(d.mean() / se) if se and np.isfinite(se) else float("nan"),
                        wins=int((d > 0).sum()),
                        seeds=[k[1] for k in keys]))
    return out


def paired_table(rows, method_a, method_b, key="shape_l2"):
    """Rendered `paired_comparison`, for dropping straight into MATH.md."""
    stats = paired_comparison(rows, method_a, method_b, key=key)
    hdr = (f"{'n_eff':>7}{'pairs':>7}{'mean diff':>12}{'se':>10}"
           f"{'paired t':>10}{'wins':>8}")
    lines = [f"{method_a} vs {method_b}  ({key}, positive = {method_a} better)",
             hdr, "-" * len(hdr)]
    for st in stats:
        lines.append(f"{st['n_eff_nominal']:>7.1f}{st['n_pairs']:>7}"
                     f"{st['mean_diff']:>12.4f}{st['se_diff']:>10.4f}"
                     f"{st['t']:>10.2f}{st['wins']:>4}/{st['n_pairs']:<3}")
    return "\n".join(lines)


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
