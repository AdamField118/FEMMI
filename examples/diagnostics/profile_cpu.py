"""Profile the production mapper; cold/warm runs must be separate processes.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/diagnostics/profile_cpu.py \
  --kind hct --sources 200 --repeats 3 --warm-numba --output profile.json
Use a new NUMBA_CACHE_DIR for compilation and reuse it to time disk-cache load.
"""
import argparse
import cProfile
from contextlib import contextmanager,ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


@contextmanager
def phase_timer(module,name,phases,label):
    original=getattr(module,name)
    def timed(*args,**kwargs):
        start=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:phases[label]=phases.get(label,0.)+time.perf_counter()-start
    setattr(module,name,timed)
    try:yield
    finally:setattr(module,name,original)


def run(args):
    import numpy as np
    import scipy.sparse.linalg as spla
    from femmi import FlatCatalog,MapperConfig,FEMMapper,bem,bem_hp,operators,c1_coupling
    from femmi.catalog import analytic_gaussian_catalog
    phases={}
    if args.warm_numba:
        from femmi._bem_assembly import _numba_kernel,_reference
        ts=time.perf_counter();kernel=_numba_kernel()
        if kernel is None:raise RuntimeError('--warm-numba requires the speed extra')
        from numba import typeof
        phases['numba_import_s']=time.perf_counter()-ts
        _,w,phi,_,_=_reference(3,2,True,True)
        inputs=(np.array([[0,1,2,3]],dtype=np.int64),np.zeros((1,2,2)),
            np.ones(1),np.zeros((1,2)),w,phi,np.zeros((1,4,4)),4,True)
        ts=time.perf_counter();kernel.compile(tuple(typeof(x) for x in inputs))
        phases['jit_compile_or_cache_load_s']=time.perf_counter()-ts
        ts=time.perf_counter();kernel(*inputs);phases['jit_first_execution_s']=time.perf_counter()-ts
    ts=time.perf_counter()
    cat=analytic_gaussian_catalog(n_gal=args.sources,field_radius=args.radius,
        sigma=.7,amp=.2,shape_noise=.01,seed=2718)
    c=FlatCatalog(cat['x'],cat['y'],cat['g1'],cat['g2'],np.ones(len(cat['x'])))
    phases['catalogue_s']=time.perf_counter()-ts
    config=MapperConfig(args.kind,args.lam,args.length,args.radius)
    with ExitStack() as stack:
        for module,name,label in [(bem,'assemble_single_layer','single_layer_s'),
            (bem,'assemble_double_layer','double_layer_s'),
            (bem_hp,'assemble_single_layer_hp','single_layer_s'),
            (bem_hp,'assemble_double_layer_hp','double_layer_s'),
            (c1_coupling,'assemble_c1','volume_assembly_s'),
            (c1_coupling,'trace_operator','trace_s'),
            (spla,'splu','coupled_factorization_s')]:
            stack.enter_context(phase_timer(module,name,phases,label))
        mapper=FEMMapper(c,config)
    phases['setup_s']=mapper.setup_seconds
    arrays={key:getattr(c,key) for key in ("x","y","g1","g2")};fits=[]
    with phase_timer(spla,'splu',phases,'prior_factorization_s'):
        for i in range(args.repeats):
            # Fixed geometry, new noise each time; no warm-start advantage.
            rng=np.random.default_rng(800+i)
            a=c.g1+rng.normal(0,.001,c.n);b=c.g2+rng.normal(0,.001,c.n)
            ts=time.perf_counter();result=mapper.reconstruct(a,b)
            seconds=time.perf_counter()-ts
            fits.append(dict(result.diagnostics,seconds=seconds,
                value_rmse=float(np.sqrt(np.mean((result.kappa-cat['kappa_true'])**2)))))
            arrays[f'kappa_{i}']=result.coefficients
            arrays[f'prediction_{i}']=np.r_[result.predicted_g1,result.predicted_g2]
    phases['reconstruction_s']=sum(r['seconds'] for r in fits)
    phases['total_s']=phases['catalogue_s']+phases['setup_s']+phases['reconstruction_s']
    phases['total_including_jit_s']=phases['total_s']+sum(phases.get(k,0.) for k in
        ('numba_import_s','jit_compile_or_cache_load_s','jit_first_execution_s'))
    phases['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    from dataclasses import asdict
    out=dict(config=asdict(config),kind=args.kind,sources=c.n,dofs=mapper.dofs,phases=phases,fits=fits,
        backend=os.environ.get('FEMMI_BEM_BACKEND','auto'),profiled=args.profile,
        python=sys.version,platform=platform.platform(),cpu=platform.processor(),
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS')},
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','jax','femmi')},
        note='Phase subtimers are nested in setup/reconstruction; do not sum them twice.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['p3','argyris','hct'],required=True)
    p.add_argument('--sources',type=int,default=200)
    p.add_argument('--radius',type=float,default=3.)
    p.add_argument('--lam',type=float,default=.3)
    p.add_argument('--length',type=float,default=.6)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warm-numba',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.sources<3 or args.repeats<1:p.error('sources>=3 and repeats>=1 required')
    if args.profile:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        cProfile.runctx('run(args)',globals(),locals(),str(args.output.with_suffix('.prof')))
    else:run(args)
