"""Common spatial evaluation and accuracy/cost benchmark reporting."""
import numpy as np
from .density.sampling import _truth_at


def evaluation_field(config, seed, size=40):
    """Fixed pixel centres shared by all methods, including unobserved holes."""
    radius=config.get('radius',3.)
    axis=(np.arange(size)+.5)*2*radius/size-radius
    x,y=np.meshgrid(axis,axis); points=np.column_stack([x.ravel(),y.ravel()])
    k,a,b=_truth_at(points,config.get('truth','nfw'),config.get('halos',((2e14,4.,(0.,0.)),)),
                    radius,seed,config.get('truth_kw'))
    valid=(np.hypot(*points.T)<radius)&np.isfinite(k)&(k<1.)
    hole=np.zeros(len(points),bool)
    for cx,cy,r in config.get('catalog_kw',{}).get('masks',[]):
        hole |= np.hypot(points[:,0]-cx,points[:,1]-cy)<r
    return points,k,a,b,dict(field=valid,interior=valid&(np.hypot(*points.T)<.8*radius)&~hole,
        boundary=valid&(np.hypot(*points.T)>=.8*radius),mask=valid&hole)


def spatial_metrics(values,truth,points,regions,radius):
    """DC-removed errors plus a compensated central aperture contrast.

    One field-wide offset is removed for regional errors; regions are never
    independently recentered. The mass contrast is invariant under a sheet.
    """
    values,truth=np.asarray(values),np.asarray(truth)
    valid=regions['field']
    if not np.all(np.isfinite(values[valid])):
        raise ValueError('nonfinite reconstruction on common evaluation support')
    delta=values-truth; offset=float(delta[valid].mean()); centered=delta-offset
    out={'field_offset':offset}
    for name,mask in regions.items():
        out[name+'_pixels']=int(mask.sum())
        out[name+'_rmse']=float(np.sqrt(np.mean(centered[mask]**2))) if mask.any() else None
    denom=np.linalg.norm(truth[valid]-truth[valid].mean())
    out['field_shape_l2']=float(np.linalg.norm(centered[valid])/denom) if denom else None
    r=np.hypot(*points.T)
    core=valid&(r<radius/4); ann=valid&(r>=radius/2)&(r<3*radius/4)
    out['aperture_contrast_error']=float(delta[core].mean()-delta[ann].mean()) if core.any() and ann.any() else None
    return out


def summarize(directory):
    """Write paired summaries and explicit failure counts from saved evaluations."""
    from pathlib import Path
    import json
    from .calibration import write_json
    from .density.stats import paired_comparison
    root=Path(directory); rows=[]
    for path in sorted(root.glob('*/evaluation.json')):
        rows.extend(json.loads(path.read_text()))
    names=sorted({r['method'] for r in rows})
    pairs=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for metric in ('shape_l2','field_shape_l2','seconds'):
                try:
                    stats=paired_comparison(rows,a,b,key=metric)
                except ValueError as exc:
                    if 'no valid catalogs shared' not in str(exc):raise
                    stats=[]
                pairs.append(dict(a=a,b=b,metric=metric,statistics=stats))
    report=dict(attempted=len(rows),failures=sum('error' in r for r in rows),paired=pairs)
    write_json(root/'summary.json',report)
    lines=['# SMPy comparison results','',
        'Timings are per-run wall times, including FEM setup. These are local measurements, not target-hardware speed claims.',
        '', '| Scenario | Method | Fits | Failures | Mean field shape L2 | Mean total seconds |',
        '|---|---|---:|---:|---:|---:|']
    for scenario in sorted({r['scenario'] for r in rows}):
        for name in names:
            group=[r for r in rows if r['scenario']==scenario and r['method']==name]
            ok=[r for r in group if 'error' not in r]
            score=np.mean([r['field_shape_l2'] for r in ok]) if ok else float('nan')
            sec=np.mean([r['seconds'] for r in ok]) if ok else float('nan')
            lines.append(f'| {scenario} | {name} | {len(ok)} | {len(group)-len(ok)} | {score:.5g} | {sec:.5g} |')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report
