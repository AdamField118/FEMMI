"""Pinned SMPy adapters with an explicitly labeled KS+ forward correction.

The catalogue is binned by weighted mean once per requested grid. KS+ receives
weight sums as availability and performs one outer pass for shear inputs.
"""
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
from .catalog import bin_shear_to_grid

SMPY_COMMIT = '26d231f5b4b22b41e76cb3bd3143d799c2e7ebfe'
METHODS = {'smpy_ks':'kaiser_squires', 'smpy_ks_plus':'ks_plus',
           'smpy_aperture':'aperture_mass'}


def verify_installation():
    """Require the immutable Git install used for scientific comparisons."""
    try:
        dist = importlib.metadata.distribution('SMPy')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('install requirements-benchmark.txt to run SMPy comparisons') from exc
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    revision = direct.get('vcs_info',{}).get('commit_id')
    if revision != SMPY_COMMIT:
        raise RuntimeError(f'SMPy must be installed from pinned Git commit {SMPY_COMMIT}; got {revision}')
    return dict(version=dist.version,commit=revision)


def create_maps(g1, g2, weight, method, smooth=0., *, iterations=100,
                aperture_scale=3., threshold_tau=None, ks_plus_forward="corrected"):
    """Use upstream mappers on [y,x] grids; E/B remain separate outputs."""
    from smpy.config import Config
    from smpy.mapping_methods.kaiser_squires.kaiser_squires import KaiserSquiresMapper
    from smpy.mapping_methods.ks_plus.ks_plus import KSPlusMapper
    from smpy.mapping_methods.aperture_mass.aperture_mass import ApertureMassMapper
    if method not in METHODS:
        raise ValueError(f'unknown SMPy method {method}')
    if not np.isfinite(smooth) or smooth<0:
        raise ValueError('smoothing must be finite and nonnegative')
    a,b,w = (np.asarray(v,dtype=float) for v in (g1,g2,weight))
    if a.ndim!=2 or a.shape!=b.shape or a.shape!=w.shape or not all(np.all(np.isfinite(v)) for v in (a,b,w)) or np.any(w<0):
        raise ValueError('finite matching grids and nonnegative weights required')
    name=METHODS[method]; cfg=Config.from_defaults(name).to_dict()
    cfg['general'].update(save_plots=False,save_fits=False)
    options=cfg['methods'][name]
    if method=='smpy_aperture':
        if not np.isfinite(aperture_scale) or aperture_scale<=0:
            raise ValueError('aperture_scale must be positive')
        options['filter']=dict(type='schneider',scale=aperture_scale,l=3,truncation=1.)
    else:
        options['smoothing']=dict(type='gaussian' if smooth else None,sigma=float(smooth))
    if method=='smpy_ks_plus':
        if not isinstance(iterations,int) or iterations<1:
            raise ValueError('iterations must be a positive integer')
        if threshold_tau is not None and (not np.isfinite(threshold_tau) or threshold_tau <= 0):
            raise ValueError('threshold_tau must be finite and positive')
        if ks_plus_forward not in ('corrected', 'upstream'):
            raise ValueError('ks_plus_forward must be corrected or upstream')
        options.update(reduced_shear_iterations=1,inpainting_iterations=iterations,
                       threshold_tau=threshold_tau)
    cls={'smpy_ks':KaiserSquiresMapper,'smpy_ks_plus':KSPlusMapper,
         'smpy_aperture':ApertureMassMapper}[method]
    if method == 'smpy_ks_plus' and ks_plus_forward == 'corrected':
        # Pinned upstream drops the B contribution in the forward transform.
        # Keep all scheduling, constraints, masks and inverse transforms upstream.
        class CorrectedKSPlusMapper(KSPlusMapper):
            _kappa_to_gamma = staticmethod(ks_plus_forward_shear)
        cls = CorrectedKSPlusMapper
    mapper=cls(cfg)
    mapper._weight_grid=w
    e,b=mapper.create_maps(a,b)
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(b)):
        raise RuntimeError('SMPy returned nonfinite maps')
    if method == 'smpy_ks_plus':
        cfg['femmi_adapter'] = dict(forward_transform=ks_plus_forward,
            correction='include B in gamma1=D1 E-D2 B, gamma2=D2 E+D1 B'
                       if ks_plus_forward == 'corrected' else None)
    return e,b,cfg


def sample_grid(grid, points, radius):
    from scipy.ndimage import map_coordinates
    n=grid.shape[0]; xy=np.asarray(points)
    index=(xy+radius)/(2*radius)*n-.5
    # Pixel centres: nearest extension within the boundary half-pixels.
    out=map_coordinates(grid,[index[:,1],index[:,0]],order=1,mode='nearest')
    out[np.any(np.abs(xy)>radius,axis=1)]=np.nan
    return out


def reconstruct(c, method, grid, smooth, *, iterations=100, threshold_tau=None, ks_plus_forward="corrected"):
    if int(grid)!=grid or grid<4:
        raise ValueError('grid must be an integer >=4')
    start=time.perf_counter()
    a,b,w,_=bin_shear_to_grid(c.x,c.y,c.g1,c.g2,weight=c.weight,
        grid_size=int(grid),extent=(-c.radius,c.radius,-c.radius,c.radius))
    e,b,cfg=create_maps(a,b,w,method,smooth,iterations=iterations,
        threshold_tau=threshold_tau,ks_plus_forward=ks_plus_forward)
    values=sample_grid(e,np.column_stack([c.x,c.y]),c.radius)
    return values,dict(completed=True,ks_grid_size=int(grid),ks_smoothing_px=float(smooth),
        smpy_method=METHODS[method],smpy_commit=SMPY_COMMIT,smpy_config=cfg,
        smpy_variant=('KS+ with corrected E/B forward' if ks_plus_forward=='corrected' else 'unmodified upstream KS+') if method=='smpy_ks_plus' else 'unmodified upstream',
        inpainting_iterations=iterations if method=='smpy_ks_plus' else None,
        reduced_shear_iterations=1 if method=='smpy_ks_plus' else None),time.perf_counter()-start,e,b


def ks_plus_forward_shear(kappa_e, kappa_b):
    """Correct the pinned upstream KS+ E/B forward transform.

    The inverse and its real-valued Nyquist convention remain upstream. Away
    from DC/Nyquist, forward followed by inverse recovers both components.
    This function is a documented adapter correction, not unmodified SMPy.
    """
    e, b = np.asarray(kappa_e), np.asarray(kappa_b)
    k1, k2 = np.meshgrid(np.fft.fftfreq(e.shape[1]), np.fft.fftfreq(e.shape[0]))
    q = k1*k1+k2*k2
    d1 = np.divide(k1*k1-k2*k2, q, out=np.zeros_like(q), where=q>0)
    d2 = np.divide(2*k1*k2, q, out=np.zeros_like(q), where=q>0)
    eh, bh = np.fft.fft2(e), np.fft.fft2(b)
    return np.fft.ifft2(d1*eh-d2*bh).real, np.fft.ifft2(d2*eh+d1*bh).real
