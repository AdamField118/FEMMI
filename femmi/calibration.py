"""Held-out, paired calibration of catalog-native FEM and binned KS.

Catalogues are generated ONCE before constructing any mesh. Every arm observes
the same positions, weights and both noise components. Truth is only used for
synthetic calibration/evaluation, never by the reconstructor.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import platform
import importlib.metadata
import os
import numpy as np
from .density.sampling import sample_catalog, n_gal_for_density, _truth_at, _score


@dataclass
class Catalogue:
    x: np.ndarray
    y: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    weight: np.ndarray
    truth: np.ndarray
    radius: float
    seed: int
    nominal_density: float

    @property
    def fingerprint(self):
        h=hashlib.sha256()
        for a in (self.x,self.y,self.g1,self.g2,self.weight,self.truth):
            h.update(np.asarray(a,dtype='<f8').tobytes())
        return h.hexdigest()

    def save(self,path):
        np.savez_compressed(path,**vars(self))


def make_catalogue(n_eff,seed,radius=3.,noise_std=.05,truth='nfw',
                   halos=((2e14,4.,(0.,0.)),),truth_kw=None,catalog_kw=None):
    if not np.isfinite(noise_std) or noise_std<0:
        raise ValueError('noise_std must be finite and nonnegative')
    kw=dict(catalog_kw or {});kw.pop('return_weights',None)
    x,y,w=sample_catalog(n_gal_for_density(n_eff,radius),radius,seed,
                         return_weights=True,**kw)
    # A shared, explicitly recorded deduplication prevents method-specific
    # source selection and protects random catalogues from nearly singular cells.
    from .mesh import _dedup_indices
    keep=_dedup_indices(np.column_stack([x,y]),.02*radius/np.sqrt(len(x)))
    x,y,w=x[keep],y[keep],w[keep];w=w/w.mean()
    kt,a,b=_truth_at(np.column_stack([x,y]),truth,halos,radius,seed,truth_kw)
    rng=np.random.default_rng(seed+1)
    a=a+rng.normal(size=len(x))*noise_std/np.sqrt(w)
    b=b+rng.normal(size=len(x))*noise_std/np.sqrt(w)
    return Catalogue(x,y,a,b,w,kt,float(radius),int(seed),float(n_eff))


def catalogue_mapper(catalogue, kind):
    from .mapping import FEMMapper, MapperConfig
    return FEMMapper(catalogue, MapperConfig(kind, 1., 1., catalogue.radius))


def fem_fit(model, lam, length):
    result = model.reconstruct(lam=lam, length=length)
    model.last_result = result
    return result.kappa, result.diagnostics, result.diagnostics['solve_seconds']


def result_row(c,method,values,info,seconds,dofs,setup_seconds=0.):
    info={k:v for k,v in info.items() if k not in ("setup_seconds","solve_seconds")}
    row=dict(method=method,seed=c.seed,n_eff_nominal=c.nominal_density,
             n_eff=float(c.weight.sum()**2/np.dot(c.weight,c.weight)/(np.pi*c.radius**2)),
             n_gal=len(c.x),radius_arcmin=c.radius,catalogue_hash=c.fingerprint,
             seconds=float(seconds+setup_seconds),setup_seconds=float(setup_seconds),
             solve_seconds=float(seconds),dofs=int(dofs),**info)
    row.update(_score(values,c.truth))
    if not all(np.isfinite(row[k]) for k in ('rel_l2','shape_l2','mean_err')):
        raise ValueError('nonfinite comparison metric')
    return row


def ks_fit(c,grid,smooth):
    from .catalog import kaiser_squires_binned
    start=time.perf_counter()
    k=kaiser_squires_binned(c.x,c.y,c.g1,c.g2,weight=c.weight,grid_size=int(grid),
           smoothing_px=float(smooth),extent=(-c.radius,c.radius,-c.radius,c.radius),
           eval_pts=np.column_stack([c.x,c.y]))
    return k,dict(converged=True,ks_grid_size=int(grid),ks_smoothing_px=float(smooth)),time.perf_counter()-start


class CalibrationFailure(RuntimeError):
    """A failed search retains every evaluated candidate for the audit trail."""
    def __init__(self,message,candidates):
        super().__init__(message)
        self.candidates=candidates


def adaptive_grid(evaluate,axes,max_expansions=6,integer_first=False,refine=True):
    """Cartesian search, expanding every winning edge before accepting it.

    Zero smoothing/length is a physical endpoint; other unresolved edges are
    reported explicitly. All candidate scores and failures are retained.
    """
    axes=[sorted(set(float(x) for x in a)) for a in axes]
    if len(axes)!=2 or any(len(a)<2 or min(a)<0 for a in axes):
        raise ValueError("two nonnegative axes with at least two entries required")
    cache={};history=[];refined=False
    for step in range(max_expansions+3):
        for b in axes[1]:
            for a in axes[0]:
                if (a,b) not in cache:
                    cache[a,b]=evaluate(a,b)
        valid={k:v for k,v in cache.items() if np.isfinite(v['score'])}
        if not valid:
            reasons=[v.get('error','nonfinite score') for v in cache.values()]
            raise CalibrationFailure('every calibration candidate failed: '+repr(reasons[:3]),
                [dict(parameters=list(k),**v) for k,v in sorted(cache.items())])
        best=min(valid,key=lambda k:(valid[k]['score'],k))
        edges=[]
        for j in range(2):
            if best[j]==axes[j][0] and best[j]>0 and not(integer_first and j==0 and best[j]<=4):
                edges.append((j,'low'))
            if best[j]==axes[j][-1]:edges.append((j,'high'))
        history.append(dict(step=step,axes=[a.copy() for a in axes],best=list(best),edges=edges))
        if not edges:
            if refine and not refined:
                for j in range(2):
                    i=axes[j].index(best[j]);extra=[]
                    for neighbour in axes[j][max(0,i-1):i]+axes[j][i+1:i+2]:
                        value=np.sqrt(best[j]*neighbour) if best[j]*neighbour>0 else (best[j]+neighbour)/2
                        extra.append(float(round(value)) if integer_first and j==0 else float(value))
                    axes[j]=sorted(set(axes[j]+extra))
                refined=True
                continue
            break
        if step>=max_expansions:break
        for j,side in edges:
            if integer_first and j==0:
                value=max(4,round(axes[j][0]/1.5)) if side=='low' else round(axes[j][-1]*1.5)
            elif axes[j][0]==0 and len(axes[j])>1:
                value=axes[j][-1]*2 if side=='high' else 0.
            else:
                value=axes[j][0]/3 if side=='low' else axes[j][-1]*3
            axes[j]=sorted(set(axes[j]+[float(value)]))
    candidates=[dict(parameters=list(k),**v) for k,v in sorted(cache.items())]
    return dict(parameters=list(best),score=valid[best]['score'],
                boundary_unresolved=bool(edges),history=history,candidates=candidates)


def calibrate_and_evaluate(config,output):
    """Run one scenario/density; checkpoint calibration before blind evaluation."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cal=list(config.get('calibration_seeds',[100,101,102]))
    ev=list(config.get('evaluation_seeds',range(6)))
    if not cal or not ev or set(cal)&set(ev) or len(set(cal))!=len(cal) or len(set(ev))!=len(ev):
        raise ValueError('calibration/evaluation seeds must be unique and disjoint')
    methods=config.get('methods',['p3','argyris','hct','ks'])
    allowed={'p3','argyris','hct','ks','smpy_ks','smpy_ks_plus'}
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=allowed:
        raise ValueError('supply unique supported convergence methods')
    if any(m.startswith('smpy_') for m in methods):
        from .smpy import verify_installation
        verify_installation()
    kwargs={k:v for k,v in config.items() if k in ('n_eff','radius','noise_std','truth','truth_kw','catalog_kw','halos')}
    cal_kwargs=dict(kwargs);eval_kwargs=dict(kwargs)
    for stage,target in (('calibration',cal_kwargs),('evaluation',eval_kwargs)):
        if stage+'_truth_kw' in config:target['truth_kw']=config[stage+'_truth_kw']
    manifest={}
    if kwargs.get('truth')=='massivenus':
        from .neural_prior.massivenus import find_maps
        for stage,target in (('calibration',cal_kwargs),('evaluation',eval_kwargs)):
            tk=target.get('truth_kw',{})
            if not tk.get('data_dir'):raise ValueError('MassiveNuS requires separate calibration/evaluation map directories')
            manifest[stage]=[dict(path=str(Path(f).resolve()),sha256=hashlib.sha256(Path(f).read_bytes()).hexdigest())
                             for f in find_maps(tk['data_dir'],tk.get('map_glob'))]
        if {r['sha256'] for r in manifest['calibration']}&{r['sha256'] for r in manifest['evaluation']}:
            raise ValueError('MassiveNuS calibration/evaluation map contents overlap')
    cats=[make_catalogue(seed=s,**cal_kwargs) for s in cal]
    for c in cats:c.save(output/f'catalogue-cal-{c.seed}.npz')
    report=dict(config=config,calibrations={},data_manifest=manifest,provenance=dict(
        c1_coupled_solve="diagonally equilibrated SuperLU; matched transpose scaling",
        solver_acceptance="fresh normal-equation relative residual <=1e-6 (internal CG target 1e-8)",
        base_commit="9f0394aec74e51ee3792fc07eb462dbfffcf919a",
        python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','galsim')},
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','FEMMI_BEM_BACKEND')},
        metric="unweighted galaxy-position DC-removed relative L2; truth kappa<1",
        regularizer="lambda * kappa.T @ (M + length**2 K) @ kappa",
        noise="independent component sigma/sqrt(normalized weight)",
        boundary="ring radius 1.12R; common source list; no measured guard nodes"))
    def grid_fit(c,method,a,b):
        if method=='ks': return ks_fit(c,a,b)
        from .smpy import reconstruct
        return reconstruct(c,method,a,b,iterations=config.get('ks_plus_iterations',100))[:3]
    for method in methods:
        gridded=method in ('ks','smpy_ks','smpy_ks_plus')
        models=[] if gridded else [catalogue_mapper(c,method) for c in cats]
        def evaluate(a,b):
            rows=[]
            for j,c in enumerate(cats):
                try:
                    k,info,sec=grid_fit(c,method,a,b) if gridded else fem_fit(models[j],a,b)
                    rows.append(result_row(c,method,k,info,sec,int(a)**2 if gridded else models[j].dofs))
                except (RuntimeError,ValueError) as exc:
                    return dict(score=float('inf'),rows=rows,error=str(exc))
            return dict(score=float(np.mean([r['shape_l2'] for r in rows])),rows=rows)
        axes=(config.get('ks_axes',[[6,12,24,48],[0,.5,1.,2.]]) if gridded
              else config.get('fem_axes',[[.03,.3,3.,30.],[.2,.6,1.8,5.4]]))
        try:
            report['calibrations'][method]=adaptive_grid(evaluate,axes,config.get('max_expansions',6),gridded,config.get('refine',True))
        except CalibrationFailure as exc:
            report['calibrations'][method]=dict(error=str(exc),candidates=exc.candidates,
                boundary_unresolved=True)
            write_json(output/'calibration.json',report)
            raise
        write_json(output/'calibration.json',report)
        print(method,report['calibrations'][method]['parameters'],
              'edge',report['calibrations'][method]['boundary_unresolved'],flush=True)
        models.clear()
    if any(r['boundary_unresolved'] for r in report['calibrations'].values()) and not config.get('allow_unresolved',False):
        raise CalibrationFailure('expand unresolved calibration grids before evaluation', report['calibrations'])
    rows=[]
    for seed in ev:
        c=make_catalogue(seed=seed,**eval_kwargs);c.save(output/f'catalogue-eval-{seed}.npz')
        for method in methods:
            a,b=report['calibrations'][method]['parameters']
            try:
                if method in ('ks','smpy_ks','smpy_ks_plus'):
                    k,info,sec=grid_fit(c,method,a,b);dofs=int(a)**2;setup=0.
                else:
                    model=catalogue_mapper(c,method)
                    k,info,sec=fem_fit(model,a,b);dofs=model.dofs;setup=model.setup_seconds
                row=result_row(c,method,k,info,sec,dofs,setup)
                arrays=dict(kappa=k)
                if config.get('evaluation_grid'):
                    from .comparison import evaluation_field,spatial_metrics
                    points,kt,_,_,regions=evaluation_field(eval_kwargs,seed,config['evaluation_grid'])
                    if method in ('smpy_ks','smpy_ks_plus'):
                        from .smpy import reconstruct,sample_grid
                        _,_,_,grid,bmode=reconstruct(c,method,a,b,iterations=config.get('ks_plus_iterations',100))
                        values=sample_grid(grid,points,c.radius)
                        arrays.update(grid=grid,bmode=bmode)
                    elif method=='ks':
                        from .catalog import kaiser_squires_binned
                        values=kaiser_squires_binned(c.x,c.y,c.g1,c.g2,weight=c.weight,
                            grid_size=int(a),smoothing_px=b,extent=(-c.radius,c.radius,-c.radius,c.radius),eval_pts=points)
                    else:
                        # Reuse the accepted solve, never refit for spatial scoring.
                        values=model.evaluate(model.last_result.coefficients,points)
                    row.update(spatial_metrics(values,kt,points,regions,c.radius))
                    arrays.update(evaluation_points=points,field_kappa=values,field_truth=kt,
                                  field_valid=regions['field'],mask_region=regions['mask'])
                np.savez_compressed(output/f'map-{method}-{seed}.npz',**arrays)
            except (RuntimeError,ValueError) as exc:
                row=dict(method=method,seed=seed,n_eff_nominal=c.nominal_density,
                         catalogue_hash=c.fingerprint,error=str(exc))
            row['scenario']=config.get('name','baseline')
            row['calibration_boundary_unresolved']=report['calibrations'][method]['boundary_unresolved']
            rows.append(row);write_json(output/'evaluation.json',rows)
    return report,rows


def write_json(path,value):
    # JSON null is the explicit representation of failed/nonfinite statistics.
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
        return x
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2)+'\n');tmp.replace(path)
