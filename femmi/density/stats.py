"""Per-catalogue reporting and paired statistics."""
import numpy as np
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
        groups.setdefault((r.get("scenario", "baseline"), r["method"], r["n_eff_nominal"]), []).append(r)

    out = []
    for (scenario, method, n_nom), grp in sorted(groups.items()):
        row = dict(scenario=scenario, method=method, n_eff_nominal=float(n_nom), n_seeds=len(grp),
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
    """Paired B-A differences, grouped by scenario and nominal density.

    Positive differences favour A. Prefixes are accepted only if unambiguous
    (legacy API). Duplicate keys and mismatched observation hashes are errors.
    Missing/failed pairs are counted, never presented as complete samples.
    Student-t intervals and exact two-sided sign p values are descriptive;
    six seeds do not justify universal superiority or a Gaussian sigma claim.
    """
    from scipy.stats import t as student_t, binomtest
    names={r.get('method','') for r in rows}
    def resolve(name):
        if name in names:return name
        hits=[n for n in names if n.startswith(name)]
        if len(hits)!=1:raise ValueError(f"ambiguous or absent method {name!r}")
        return hits[0]
    def pick(name):
        out={}
        for r in rows:
            if r.get('method')!=resolve(name) or 'seed' not in r:continue
            k=(r.get('scenario','baseline'),r['n_eff_nominal'],r['seed'])
            if k in out:raise ValueError(f"duplicate paired key {k}")
            out[k]=r
        return out
    A,B=pick(method_a),pick(method_b)
    allkeys=set(A)|set(B)
    if not allkeys:raise ValueError('paired statistics need per-seed rows, not averages')
    out=[]
    for scenario,n_nom in sorted({k[:2] for k in allkeys}):
        attempted=[k for k in allkeys if k[:2]==(scenario,n_nom)]
        keys=[]
        for k in sorted(set(A)&set(B)):
            if k[:2]!=(scenario,n_nom):continue
            a,b=A[k],B[k]
            if 'error' in a or 'error' in b or key not in a or key not in b:continue
            if a.get('catalogue_hash')!=b.get('catalogue_hash'):
                raise ValueError(f"different catalogues in paired key {k}")
            if not np.isfinite(a[key]) or not np.isfinite(b[key]):continue
            keys.append(k)
        if not keys:continue
        d=np.array([B[k][key]-A[k][key] for k in keys]);n=len(d)
        mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
        width=float(student_t.ppf(.975,n-1)*se) if n>1 else float('nan')
        wins=int((d>0).sum());losses=int((d<0).sum())
        out.append(dict(scenario=scenario,n_eff_nominal=float(n_nom),n_pairs=n,
            n_unpaired=len(attempted)-n,mean_diff=mean,se_diff=se,
            ci95=[mean-width,mean+width],t=mean/se if se>0 else float('nan'),
            wins=wins,ties=int((d==0).sum()),
            sign_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,
            seeds=[k[2] for k in keys],differences=d.tolist()))
    if not out:raise ValueError('no valid catalogs shared; paired statistics need per-seed rows')
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
                     f"{r.get('n_gal', 0):>7}  -- {r['error']}")
    return "\n".join(lines)
