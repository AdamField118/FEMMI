"""Pinned SMPy adapters. No fallback to a FEMMI implementation.

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
                aperture_scale=3.):
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
        options.update(reduced_shear_iterations=1,inpainting_iterations=iterations)
    cls={'smpy_ks':KaiserSquiresMapper,'smpy_ks_plus':KSPlusMapper,
         'smpy_aperture':ApertureMassMapper}[method]
    mapper=cls(cfg)
    mapper._weight_grid=w
    e,b=mapper.create_maps(a,b)
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(b)):
        raise RuntimeError('SMPy returned nonfinite maps')
    return e,b,cfg


def sample_grid(grid, points, radius):
    from scipy.ndimage import map_coordinates
    n=grid.shape[0]; xy=np.asarray(points)
    index=(xy+radius)/(2*radius)*n-.5
    # Pixel centres: nearest extension within the boundary half-pixels.
    out=map_coordinates(grid,[index[:,1],index[:,0]],order=1,mode='nearest')
    out[np.any(np.abs(xy)>radius,axis=1)]=np.nan
    return out


def reconstruct(c, method, grid, smooth, *, iterations=100):
    if int(grid)!=grid or grid<4:
        raise ValueError('grid must be an integer >=4')
    start=time.perf_counter()
    a,b,w,_=bin_shear_to_grid(c.x,c.y,c.g1,c.g2,weight=c.weight,
        grid_size=int(grid),extent=(-c.radius,c.radius,-c.radius,c.radius))
    e,b,cfg=create_maps(a,b,w,method,smooth,iterations=iterations)
    values=sample_grid(e,np.column_stack([c.x,c.y]),c.radius)
    return values,dict(completed=True,ks_grid_size=int(grid),ks_smoothing_px=float(smooth),
        smpy_method=METHODS[method],smpy_commit=SMPY_COMMIT,smpy_config=cfg,
        inpainting_iterations=iterations if method=='smpy_ks_plus' else None,
        reduced_shear_iterations=1 if method=='smpy_ks_plus' else None),time.perf_counter()-start,e,b
