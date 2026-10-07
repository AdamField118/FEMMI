"""Deterministic catalogues and independent truth dispatch; angles in arcmin."""
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


SURVEY_NEFF = {"DES Y3": 5.6, "KiDS-1000": 6.2, "CFHTLenS": 11.0,
               "HSC Y3": 19.9, "LSST Y10": 27.0, "Euclid": 30.0}


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
    from ..truth import independent_truth
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
    """Positions in a disk, with optional circular holes and clustering.

    Masks preserve the requested count, increasing density in the unmasked
    area. Weights are normalized to mean one. Effective density is
    (sum w)^2/sum(w^2)/gross area, which differs from count density.
    """
    if int(n_gal)<1 or not np.isfinite(radius) or radius<=0:
        raise ValueError('positive galaxy count and radius required')
    if not 0<=clustering<=1 or not np.isfinite(weight_scatter) or weight_scatter<0:
        raise ValueError('clustering must be in [0,1] and scatter nonnegative')
    for cx,cy,cr in masks:
        if not np.all(np.isfinite([cx,cy,cr])) or cr<0:
            raise ValueError('invalid circular mask')
        if cr>=radius+np.hypot(cx,cy):raise ValueError('mask covers the entire field')
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
        trials=0
        while have < n_target:
            trials+=1
            if trials>10000:raise ValueError("masked field has negligible accessible area")
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
            # Reject clustered perturbations outside the disk or inside a
            # hole. Clipping only to the outer circle reintroduced masked data.
            cx,cy=x[idx].copy(),y[idx].copy()
            todo=np.arange(n_c)
            for _ in range(10000):
                if not len(todo):break
                px=cx[todo]+rng.normal(0,sigma,len(todo))
                py=cy[todo]+rng.normal(0,sigma,len(todo))
                valid=np.hypot(px,py)<radius
                for mx,my,mr in masks:valid &= np.hypot(px-mx,py-my)>mr
                x[todo[valid]],y[todo[valid]]=px[valid],py[valid]
                todo=todo[~valid]
            if len(todo):raise ValueError('could not sample clustered unmasked positions')

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
