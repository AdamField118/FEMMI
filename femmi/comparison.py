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
    out['aperture_contrast_abs_error']=abs(out['aperture_contrast_error']) if out['aperture_contrast_error'] is not None else None
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
            for metric in ('shape_l2','field_shape_l2','interior_rmse','boundary_rmse','mask_rmse','aperture_contrast_abs_error','seconds'):
                try:
                    stats=paired_comparison(rows,a,b,key=metric)
                except ValueError as exc:
                    if 'no valid catalogs shared' not in str(exc):raise
                    stats=[]
                pairs.append(dict(a=a,b=b,metric=metric,statistics=stats))
    aperture_rows=[]
    for path in sorted(root.glob('*/aperture.json')):aperture_rows.extend(json.loads(path.read_text()))
    aperture_pairs=[];anames=sorted({r['method'] for r in aperture_rows})
    for i,a in enumerate(anames):
        for b in anames[i+1:]:
            try:st=paired_comparison(aperture_rows,a,b,key='aperture_rmse')
            except ValueError as exc:
                if 'no valid catalogs shared' not in str(exc):raise
                st=[]
            aperture_pairs.append(dict(a=a,b=b,statistics=st))
    ks_policies=[]
    for path in sorted(root.glob('*/calibration.json')):
        cal=json.loads(path.read_text());cfg=cal.get('config',{})
        if 'smpy_ks_plus' in cal.get('calibrations',{}):
            chosen=cal['calibrations']['smpy_ks_plus']
            ks_policies.append(dict(scenario=cfg.get('name',path.parent.name),
                forward_transform=cfg.get('ks_plus_forward','upstream'),
                iteration_policy=cfg.get('ks_plus_iteration_policy','historical'),
                selected_iterations=chosen.get('selected_iterations',cfg.get('ks_plus_iterations',100)),
                threshold_tau=cfg.get('ks_plus_threshold_tau'),
                budget_limited=chosen.get('iteration_budget_limited',False),
                plateau_accepted=cal.get('ks_plus_iteration_stability',{}).get('accepted')))
    report=dict(ks_plus_policies=ks_policies,attempted=len(rows),failures=sum('error' in r for r in rows),paired=pairs,
        aperture=dict(attempted=len(aperture_rows),failures=sum('error' in r for r in aperture_rows),paired=aperture_pairs))
    write_json(root/'summary.json',report)
    lines=['# SMPy comparison results','',
        'Timings are per-run wall times, including FEM setup. These are local measurements, not target-hardware speed claims.',
        '', '| Scenario | Method | Fits | Failures | Mean field shape L2 | Mean total seconds |',
        '|---|---|---:|---:|---:|---:|']
    for scenario in sorted({r['scenario'] for r in rows}):
        for name in names:
            group=[r for r in rows if r['scenario']==scenario and r['method']==name]
            ok=[r for r in group if 'error' not in r]
            scores=[r['field_shape_l2'] for r in ok if r.get('field_shape_l2') is not None]
            score=np.mean(scores) if scores else float('nan')
            sec=np.mean([r['seconds'] for r in ok]) if ok else float('nan')
            lines.append(f'| {scenario} | {name} | {len(ok)} | {len(group)-len(ok)} | {score:.5g} | {sec:.5g} |')
    if ks_policies:
        lines += ['', '## KS+ method identity and stopping policy', '']
        for policy in ks_policies:
            lines.append(f"- {policy['scenario']}: forward={policy['forward_transform']}; "
                         f"policy={policy['iteration_policy']}; iterations={policy['selected_iterations']}; "
                         f"tau={policy['threshold_tau']}; plateau passed={policy['plateau_accepted']}; "
                         f"at largest candidate budget={policy['budget_limited']}.")
        lines += ['', 'Corrected KS+ includes a FEMMI adapter fix to the pinned SMPy forward transform. '
                  'A calibrated-budget result is a finite-budget estimator comparison, not a convergence claim.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report
