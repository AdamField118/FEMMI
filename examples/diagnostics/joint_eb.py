#!/usr/bin/env python3
"""Reproduce E/B injection and prior-sensitivity controls; write untracked JSON."""
import argparse
import json
from pathlib import Path
import numpy as np
from femmi import FEMMapper, MapperConfig, FlatCatalog
from femmi.catalog import analytic_gaussian_shear
from femmi.diagnostics import mass_norm
from femmi.eb import sampled_eb_overlap


def run(seeds, sources, output):
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        xy=rng.uniform(-1,1,(sources,2));xy=xy[np.linalg.norm(xy,axis=1)<1]
        for masked in (False,True):
            pos=xy[np.linalg.norm(xy-[.15,.2],axis=1)>.25] if masked else xy
            ke,a,b=analytic_gaussian_shear(pos,sigma=.3,amp=.1,center=(.2,-.1))
            kb,u,v=analytic_gaussian_shear(pos,sigma=.25,amp=.03,center=(-.3,.25))
            for kind in ('p3','argyris','hct'):
                c=FlatCatalog(*pos.T,a,b,np.ones(len(pos)))
                mapper=FEMMapper(c,MapperConfig(kind,.03,.3,1.,maxiter=4000))
                overlap=sampled_eb_overlap(mapper)
                # Vary E/B ratio on pure E; a mixed injection checks real B response.
                for injection in ('E','E+B'):
                    d1,d2=(a,b) if injection=='E' else (a-v,b+u)
                    raw=mapper.reconstruct(d2,-d1)
                    for ratio in (.1,1.,10.):
                        fit=mapper.reconstruct_eb(d1,d2,lam_b=.03*ratio)
                        def dc_error(pred,truth):
                            delta=pred-truth;delta-=delta.mean()
                            return float(np.sqrt(np.mean(delta**2)))
                        rows.append(dict(seed=seed,masked=masked,method=kind,
                            injection=injection,b_to_e_prior_ratio=ratio,
                            active_sources=len(pos),e_rank=overlap['e_rank'],
                            observation_dimension=overlap['observation_dimension'],
                            smallest_relative_singular_value=overlap['singular_values'][-1]/overlap['singular_values'][0],
                            raw_b_l2=mass_norm(mapper,raw.coefficients),
                            joint_b_l2=mass_norm(mapper,fit.b_coefficients),
                            e_source_rmse=dc_error(fit.kappa_e,ke),
                            b_source_rmse=dc_error(fit.kappa_b,0*kb if injection=='E' else kb),
                            diagnostics=fit.diagnostics))
                print(seed,masked,kind,flush=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(dict(seeds=seeds,requested_sources=sources,
        description='Noiseless analytic Gaussian shear; descriptive controls, not calibrated performance',
        rows=rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[728,729])
    p.add_argument('--sources',type=int,default=60)
    p.add_argument('--output',default='results/joint-eb.json')
    args=p.parse_args();run(args.seeds,args.sources,args.output)
