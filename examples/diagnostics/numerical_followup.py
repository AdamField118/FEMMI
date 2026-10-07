"""Measured BC spectra, matched-residual direct/GMRES and warm dense/ACA timings."""
import os,sys,time,json,argparse,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.sparse.linalg import splu
from femmi.calibration import write_json

def timed(fn):
    t=time.perf_counter();v=fn();return v,time.perf_counter()-t

def run(output):
    from femmi.calderon import dual_conditioning
    from femmi.operators import build_operators
    from femmi.iterative import CoupledOperator,fem_block_preconditioner,solve_coupled
    from femmi.bem_hp import build_circular_boundary_mesh,assemble_single_layer_hp,assemble_single_layer_auto
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(platform=platform.platform(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','FEMMI_BEM_BACKEND')},bc=[],solvers=[],aca=[])
    for irregular in (False,True):
        for n in (16,32,64,128):
            t=np.arange(n)*2*np.pi/n
            if irregular:t+=.25*np.sin(t)
            xy=np.column_stack([np.cos(t),np.sin(t)])
            report['bc'].append(dict(irregular=irregular,**dual_conditioning(xy)))
    write_json(output/'numerics.json',report)
    for nx in (4,8,16,24,32):
        ops,assembly=timed(lambda:build_operators(nx=nx,ny=nx,verbose=False))
        lu,factor=timed(lambda:splu(ops.A_coupled.tocsc()))
        row=dict(nx=nx,n_dofs=ops.n_nodes,assembly_including_initial_lu_seconds=assembly,
                 direct_factor_seconds=factor,solves=[])
        try:
            pre,pretime=timed(lambda:fem_block_preconditioner(ops))
            row['ilu_seconds']=pretime;op=CoupledOperator(ops)
            for seed in range(5):
                rhs=-2*(ops.M@np.random.default_rng(seed).normal(size=ops.n_nodes))
                rhs[ops._rhs_zero_nodes()]=0.
                xd,td=timed(lambda:lu.solve(rhs))
                (xi,info),ti=timed(lambda:solve_coupled(ops,rhs,tol=1e-10,maxiter=10,
                        operator=op,precond=pre,return_info=True))
                dr=float(np.linalg.norm(ops.A_coupled@xd-rhs)/np.linalg.norm(rhs))
                row['solves'].append(dict(seed=seed,direct_seconds=td,gmres_seconds=ti,
                    direct_residual=dr,relative_solution_error=float(np.linalg.norm(xi-xd)/np.linalg.norm(xd)),**info))
        except (RuntimeError,ValueError) as exc:row['error']=str(exc)
        report['solvers'].append(row);write_json(output/'numerics.json',report)
        print('solver',nx,row['n_dofs'],flush=True)
    # Warm both assembly routes before collecting independent repeated timings.
    for degree in (3,5):
        warm=build_circular_boundary_mesh(6,degree=degree)
        assemble_single_layer_hp(warm,degree)
        assemble_single_layer_auto(warm,degree,use_aca=True,tol=1e-9)
        for nb in (96,192,384):
            mesh=build_circular_boundary_mesh(max(4,nb//degree),degree=degree)
            trials=[]
            for _ in range(3):
                vd,td=timed(lambda:assemble_single_layer_hp(mesh,degree))
                va,ta=timed(lambda:assemble_single_layer_auto(mesh,degree,use_aca=True,tol=1e-9))
                trials.append(dict(dense_seconds=td,aca_seconds=ta,relative_error=float(np.linalg.norm(va-vd)/np.linalg.norm(vd))))
            report['aca'].append(dict(degree=degree,n_b=mesh.n_boundary_dofs,trials=trials))
            write_json(output/'numerics.json',report);print('ACA',degree,nb,flush=True)
    append_catalog_aca(report,output/'numerics.json')
    return report


def append_catalog_aca(report,path):
    from femmi.elements import C1Space,catalog_triangulation
    from femmi.c1_coupling import boundary_loop
    from femmi.density import sample_catalog,n_gal_for_density
    from femmi.bem_hp import build_boundary_mesh,assemble_single_layer_hp,assemble_single_layer_auto
    for density in (5,20):
        x,y=sample_catalog(n_gal_for_density(density,3),seed=0)
        v,t,_,_=catalog_triangulation(x,y)
        space=C1Space(v,t,'argyris');mesh=build_boundary_mesh(v[boundary_loop(space)],5)
        assemble_single_layer_hp(mesh,5);assemble_single_layer_auto(mesh,5,use_aca=True,tol=1e-9)
        trials=[]
        for _ in range(3):
            vd,td=timed(lambda:assemble_single_layer_hp(mesh,5))
            va,ta=timed(lambda:assemble_single_layer_auto(mesh,5,use_aca=True,tol=1e-9))
            trials.append(dict(dense_seconds=td,aca_seconds=ta,relative_error=float(np.linalg.norm(va-vd)/np.linalg.norm(vd))))
        report['aca'].append(dict(geometry='catalogue ring',density=density,degree=5,n_b=mesh.n_boundary_dofs,trials=trials))
        write_json(path,report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='benchmarks/calibration/numerics')
    run(p.parse_args().output)
