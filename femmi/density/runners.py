"""Legacy public runners; explicit calibrated benchmarks use calibration.py."""
import numpy as np
from .sampling import *
from .sampling import _truth_at, _score
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


def _catalogue(n_eff,seed,radius,noise_std,truth,halos,truth_kw,catalog_kw):
    from ..calibration import make_catalogue
    return make_catalogue(n_eff,seed,radius,noise_std,truth,halos,truth_kw,catalog_kw)


def _fixed(c,kind,lam,length):
    from ..calibration import FEMCatalogueModel,result_row
    model=FEMCatalogueModel(c,kind)
    k,info,sec=model.fit(lam,length)
    info['lam_method']='fixed'
    return result_row(c,f"{kind.capitalize()} (catalog)" if kind!='p3' else 'P3 (catalog)',
                      k,info,sec,model.dofs,model.setup_seconds)


def c1_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0, lam=None,
                   wiener_length=1.0, halos=((2.0e14, 4.0, (0.0, 0.0)),),
                   lam_grid=None, truth="nfw", truth_kw=None, catalog_kw=None,
                   kind="argyris"):
    """Compatibility runner on the shared weighted catalogue.

    Fixed Gaussian MAP uses a residual-checked quadratic solve. The historical
    default lambda is retained for reproducibility, not advertised as optimal.
    Use calibration artifacts for fair comparisons. 'auto' and 'cv' remain
    available via the older selectors and explicitly check final convergence.
    """
    from ..c1_lambda import CALIBRATED_LAM
    c=_catalogue(n_eff,seed,radius,noise_std,truth,halos,truth_kw,catalog_kw)
    if lam is None:lam=CALIBRATED_LAM
    if lam not in ('auto','cv'):return _fixed(c,kind,float(lam),wiener_length)
    import time
    from ..elements import C1Space,catalog_triangulation
    from ..c1_inverse import C1MAPReconstructor
    from ..c1_lambda import select_c1_lambda,cv_lambda
    from ..calibration import result_row
    start=time.perf_counter()
    nb=max(18,3*int(np.ceil(2*np.sqrt(len(c.x))/3)))
    v,t,ring,idx=catalog_triangulation(c.x,c.y,radius=1.12*radius,
        center=(0.,0.),n_boundary=nb,dedup=0.)
    space=C1Space(v,t,kind)
    a=np.zeros(len(v));b=a.copy();w=a.copy()
    a[idx]=c.g1;b[idx]=c.g2;w[idx]=c.weight
    rec=C1MAPReconstructor(space,lam=1.,wiener_length=wiener_length,
        data_weight=w,maxiter=10000,degree=5 if kind=='argyris' else 3)
    if lam=='auto':chosen,info=select_c1_lambda(rec,a,b,noise_std=noise_std,lam_grid=lam_grid)
    else:chosen,info=cv_lambda(rec,a,b,lam_grid=lam_grid,seed=seed)
    rec.lam=chosen
    k,res=rec.reconstruct(a,b)
    if not res.success:raise RuntimeError(f"C1 legacy selection final solve failed: {res.message}")
    return result_row(c,f"{kind.capitalize()} (catalog)",rec.kappa_at_vertices(k)[idx],
        dict(lam_used=float(chosen),lam_method=lam,converged=True),
        time.perf_counter()-start,space.n_dofs)


def argyris_catalog_run(n_eff,**kw):
    """Argyris compatibility runner; publication uses held-out joint tuning."""
    return c1_catalog_run(n_eff,kind='argyris',**kw)


def hct_catalog_run(n_eff,**kw):
    """HCT compatibility runner; publication calibrates HCT independently."""
    return c1_catalog_run(n_eff,kind='hct',**kw)


def p3_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0,
                   halos=((2.0e14, 4.0, (0.0, 0.0)),), use_morozov=True,
                   truth="nfw", truth_kw=None, catalog_kw=None,
                   lam=1e-2,wiener_length=None):
    """Shared-catalogue P3; explicit fixed lambda/length supports matched tuning.

    Morozov is retained as a separate selection method for legacy callers.
    """
    from ..catalog import reconstruct_catalog
    from ..calibration import result_row
    import time
    c=_catalogue(n_eff,seed,radius,noise_std,truth,halos,truth_kw,catalog_kw)
    length=.2*1.12*radius if wiener_length is None else wiener_length
    if not use_morozov:return _fixed(c,'p3',lam,length)
    start=time.perf_counter()
    nb=max(18,3*int(np.ceil(2*np.sqrt(len(c.x))/3)))
    res=reconstruct_catalog(c.x,c.y,c.g1,c.g2,weight=c.weight,use_weights=True,
        radius=1.12*radius,n_boundary=nb,dedup_radius=0.,guard_ring=False,
        lam_reg=lam,wiener_length=length,noise_std=noise_std,use_morozov=True,
        maxiter=10000,verbose=False)
    if not np.all(np.isfinite(res.kappa_gal)):raise RuntimeError('P3 dropped shared sources')
    return result_row(c,'P3 (catalog)',res.kappa_gal,
        dict(lam_used=float(res.lam_reg),lam_method='morozov'),
        time.perf_counter()-start,res.ops.n_nodes)


def ks_catalog_run(n_eff, noise_std=0.05, seed=0, radius=3.0, grid_size=None,
                   smoothing_px=None, halos=((2.0e14, 4.0, (0.0, 0.0)),),
                   truth="nfw", truth_kw=None, catalog_kw=None):
    """Weighted KS on the same selected catalogue, with actual elapsed time.

    Historical anchors apply only to the original NFW/noise/field setting;
    new calibrated comparisons use the saved scenario-specific parameters.
    """
    from ..calibration import ks_fit,result_row
    c=_catalogue(n_eff,seed,radius,noise_std,truth,halos,truth_kw,catalog_kw)
    gs,sp=ks_params_for_density(n_eff)
    gs=gs if grid_size is None else int(grid_size)
    sp=sp if smoothing_px is None else float(smoothing_px)
    k,info,sec=ks_fit(c,gs,sp)
    return result_row(c,'Kaiser-Squires',k,info,sec,gs**2)
