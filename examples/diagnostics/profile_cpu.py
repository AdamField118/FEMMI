"""Reproducible CPU setup/reconstruction benchmark (not a science comparison).

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Use --source-root to
benchmark an untouched checkout with this same driver. Timed runs are separate
from --profile runs; cProfile overhead must not enter speedup estimates.
"""
from __future__ import annotations
import argparse
import cProfile
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


@contextmanager
def phase_timer(module, name, phases, label):
    original = getattr(module, name)
    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            phases[label] = phases.get(label, 0.) + time.perf_counter() - start
    setattr(module, name, timed)
    try:
        yield
    finally:
        setattr(module, name, original)


def run(args):
    import numpy as np
    import scipy.sparse.linalg as spla
    import scipy.optimize as sopt
    from contextlib import ExitStack
    from femmi import bem, bem_hp, operators, c1_coupling
    from femmi.elements import C1Space, circular_triangulation
    from femmi.c1_inverse import C1MAPReconstructor
    from femmi.inverse import MAPReconstructor
    from femmi.forward import DifferentiableForward

    # Apply identical, explicit stopping criteria to BOTH old/new estimators
    # without changing their production defaults or requiring new baseline APIs.
    if args.gtol is not None:
        original_minimize = sopt.minimize
        def minimize(*a, **kw):
            kw['options'] = dict(kw.get('options',{}),gtol=args.gtol,
                                 ftol=1e-14,maxcor=20)
            return original_minimize(*a,**kw)
        sopt.minimize = minimize

    phases = {}
    # Keep compiler/cache loading separate from assembly. For a genuine cold
    # compilation use a NEW NUMBA_CACHE_DIR and --warm-numba; repeat with the
    # same directory to measure disk-cache loading. It is included in cold total.
    if args.warm_numba:
        from femmi._bem_assembly import _numba_kernel, _reference
        ts = time.perf_counter()
        kernel = _numba_kernel()
        if kernel is None:
            raise RuntimeError('--warm-numba requires the speed extra')
        from numba import typeof
        phases['numba_import_s'] = time.perf_counter()-ts
        xi,w,phi,_,_ = _reference(3,2,True,True)
        inputs=(np.array([[0,1,2,3]],dtype=np.int64),np.zeros((1,2,2)),
                np.ones(1),np.zeros((1,2)),w,phi,np.zeros((1,4,4)),4,True)
        ts = time.perf_counter()
        kernel.compile(tuple(typeof(x) for x in inputs))
        phases['jit_compile_or_cache_load_s'] = time.perf_counter()-ts
        ts = time.perf_counter()
        kernel(*inputs)
        phases['jit_first_execution_s'] = time.perf_counter()-ts
    start = time.perf_counter()
    with ExitStack() as stack:
        for module, name, label in [
            (bem, 'assemble_single_layer', 'single_layer_s'),
            (bem, 'assemble_double_layer', 'double_layer_s'),
            (bem_hp, 'assemble_single_layer_hp', 'single_layer_s'),
            (bem_hp, 'assemble_double_layer_hp', 'double_layer_s'),
            (c1_coupling, 'assemble_c1', 'volume_assembly_s'),
            (c1_coupling, 'trace_operator', 'trace_s'),
            (spla, 'splu', 'sparse_factorization_s'),
            (operators, '_reference_data', 'p3_reference_s'),
        ]:
            stack.enter_context(phase_timer(module, name, phases, label))
        if args.kind == 'p3':
            ops = operators.build_operators(args.size, args.size, verbose=False)
            xy = np.asarray(ops.mesh.nodes)
            weight = np.ones(ops.n_nodes)
            weight[np.asarray(ops.mesh.boundary)] = 0.
            rec = MAPReconstructor(DifferentiableForward(ops, lam_reg=args.lam),
                                   data_weight=weight, wiener_length=args.length,
                                   maxiter=args.maxiter, callback_every=0)
            forward = ops.forward
            n = ops.n_nodes
            nb = len(ops.mesh.boundary)
        else:
            v, t = circular_triangulation(args.size)
            space = C1Space(v, t, kind=args.kind)
            rec = C1MAPReconstructor(space, lam=args.lam, wiener_length=args.length,
                                     maxiter=args.maxiter,
                                     degree=5 if args.kind == 'argyris' else 3)
            xy = v
            forward = rec.shear_of
            n = space.n_dofs
            nb = rec.ops.bnd.n_boundary_dofs
    phases['setup_s'] = time.perf_counter() - start
    data_start = time.perf_counter()
    # Analytic Gaussian kappa/shear, independent of the FEM forward. Stable
    # origin limit; this only fixes a deterministic workload, not calibration.
    r2 = np.sum(xy**2, axis=1)
    scale = .7
    truth = .2 * np.exp(-r2/(2*scale**2))
    mean = np.divide(.4*scale**2*(-np.expm1(-r2/(2*scale**2))), r2,
                     out=np.full_like(r2, .2), where=r2>0)
    tangential = mean-truth
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    rng = np.random.default_rng(2718)
    g1 = -tangential*np.cos(2*ang) + rng.normal(0, .01, len(xy))
    g2 = -tangential*np.sin(2*ang) + rng.normal(0, .01, len(xy))
    phases['catalogue_s'] = time.perf_counter()-data_start
    arrays = {'g1':g1, 'g2':g2}
    fits = []
    for j in range(args.repeats):
        ts = time.perf_counter()
        k, result = rec.reconstruct(g1, g2, verbose=False)
        elapsed = time.perf_counter()-ts
        if args.kind == 'p3':
            objective, gradient = rec._make_obj_and_grad(g1,g2)[0](k)
            iters, success = result.n_iter, result.converged
            kv = k
        else:
            objective, gradient = rec._obj_grad(k,g1,g2)
            iters, success = result.nit, bool(result.success)
            kv = rec.kappa_at_vertices(k)
        pred = np.concatenate(forward(k))
        arrays[f'kappa_{j}'] = k
        arrays[f'prediction_{j}'] = pred
        fits.append(dict(seconds=elapsed, objective=objective, iterations=iters,
                         gradient_inf=float(np.max(np.abs(gradient))),
                         converged=success, value_rmse=float(np.sqrt(np.mean((kv-truth)**2)))))
    phases['reconstruction_s'] = sum(f['seconds'] for f in fits)
    phases['total_s'] = phases['setup_s']+phases['catalogue_s']+phases['reconstruction_s']
    phases['total_including_jit_s'] = phases['total_s']+sum(phases.get(k,0.) for k in
        ['numba_import_s','jit_compile_or_cache_load_s','jit_first_execution_s'])
    # Parity probes outside the reported total.
    probe = rng.normal(size=n)
    arrays['forward_probe'] = np.concatenate(forward(probe))
    if args.kind == 'p3':
        arrays['gradient_probe'] = rec._make_obj_and_grad(g1,g2)[0](probe)[1]
    else:
        arrays['gradient_probe'] = rec._obj_grad(probe,g1,g2)[1]
    phases['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    versions = {p:importlib.metadata.version(p) for p in ['numpy','scipy','jax']}
    try:
        versions['numba'] = importlib.metadata.version('numba')
    except importlib.metadata.PackageNotFoundError:
        versions['numba'] = None
    output = dict(kind=args.kind,size=args.size,dofs=n,boundary_dofs=nb,
                  maxiter=args.maxiter,repeats=args.repeats, phases=phases,fits=fits,
                  gtol=args.gtol,
                  lam=args.lam,wiener_length=args.length,
                  source_root=str(args.source_root),python=sys.version,versions=versions,
                  platform=platform.platform(),processor=platform.processor(),
                  threads={v:os.environ.get(v) for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']},
                  backend=os.environ.get('FEMMI_BEM_BACKEND','auto'),profiled=bool(args.profile))
    output['numba_warmed_before_setup'] = args.warm_numba
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['p3','argyris','hct'],required=True)
    p.add_argument('--size',type=int,required=True,help='P3 square nx; C1 circle boundary vertices')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--maxiter',type=int,default=300)
    p.add_argument('--gtol',type=float,help='common L-BFGS gtol; ftol=1e-14, maxcor=20')
    p.add_argument('--lam',type=float,default=.03)
    p.add_argument('--length',type=float,default=.5)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path)
    p.add_argument('--warm-numba',action='store_true',help='separately time JIT initialization before setup')
    args=p.parse_args()
    sys.path.insert(0,str(args.source_root.resolve()))
    if args.profile:
        profiler=cProfile.Profile()
        profiler.runcall(run,args)
        profiler.dump_stats(str(args.profile))
    else:
        run(args)
