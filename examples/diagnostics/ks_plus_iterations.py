#!/usr/bin/env python3
"""Measure native/corrected KS+ schedule sensitivity on fixed catalogues."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time
import numpy as np
from femmi.calibration import make_catalogue
from femmi.smpy import reconstruct, verify_installation


def run(args):
    upstream=verify_installation();rows=[]
    for seed in args.seeds:
        for masked in (False,True):
            c=make_catalogue(3,seed,radius=3,truth='nfw',
                catalog_kw={'masks':[(0,0,.6)]} if masked else {})
            for forward in args.forward:
                for tau in (None,args.tau):
                    previous=None
                    for n in args.iterations:
                        start=time.perf_counter()
                        with contextlib.redirect_stdout(io.StringIO()):
                            *_,e,b=reconstruct(c,'smpy_ks_plus',args.grid,0.,
                                iterations=n,threshold_tau=tau,ks_plus_forward=forward)
                        axis=(np.arange(args.grid)+.5)*2/args.grid-1
                        x,y=np.meshgrid(axis,axis);field=x*x+y*y<1
                        z=np.r_[e[field]-e[field].mean(),b[field]-b[field].mean()]
                        row=dict(seed=seed,masked=masked,forward=forward,
                            tau=tau,iterations=n,seconds=time.perf_counter()-start,
                            change_from_previous=None if previous is None else
                            float(np.linalg.norm(z-previous)/max(np.linalg.norm(z),1e-300)))
                        rows.append(row);print(json.dumps(row),flush=True);previous=z
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(smpy=upstream,grid=args.grid,rows=rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[21001])
    p.add_argument('--iterations',type=int,nargs='+',default=[100,400,1600])
    p.add_argument('--grid',type=int,default=12)
    p.add_argument('--tau',type=float,default=25.)
    p.add_argument('--forward',nargs='+',choices=['upstream','corrected'],default=['upstream','corrected'])
    p.add_argument('--output',default='results/ks-plus-iterations.json')
    args=p.parse_args()
    if args.iterations!=sorted(set(args.iterations)) or min(args.iterations)<1:
        p.error('iterations must be increasing distinct positive counts')
    run(args)
