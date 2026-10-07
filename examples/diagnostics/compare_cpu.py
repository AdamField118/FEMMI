"""Require matched settings, converged solves and numerical parity before timing ratios."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(before,after):
    b=json.loads(before.read_text());a=json.loads(after.read_text())
    for key in ('kind','sources','dofs','config'):
        if a.get(key)!=b.get(key):raise ValueError(f'mismatched benchmark setting: {key}')
    if a['profiled'] or b['profiled']:raise ValueError('profiled runs cannot establish speedups')
    if len(a['fits'])!=len(b['fits']):raise ValueError('mismatched repeat count')
    parity={}
    with np.load(before.with_suffix('.npz')) as B,np.load(after.with_suffix('.npz')) as A:
        for key in ('x','y','g1','g2'):np.testing.assert_array_equal(A[key],B[key])
        for i,(bf,af) in enumerate(zip(b['fits'],a['fits'])):
            if not bf['converged'] or not af['converged']:raise AssertionError('unconverged fit')
            for r in (bf,af):
                if r['relative_residual']>r['residual_tolerance']:raise AssertionError('residual exceeds tolerance')
            np.testing.assert_allclose(af['objective'],bf['objective'],rtol=1e-7,atol=1e-12)
            for field in ('prediction','kappa'):
                key=f'{field}_{i}'
                error=float(np.linalg.norm(A[key]-B[key])/max(np.linalg.norm(B[key]),1e-300))
                parity[key]=error
                if error>1e-5:raise AssertionError(f'{key} relative error {error}')
    timings={key:dict(before=b['phases'][key],after=a['phases'][key],speedup=b['phases'][key]/a['phases'][key])
        for key in ('setup_s','single_layer_s','coupled_factorization_s','reconstruction_s','total_s')}
    return dict(before=str(before),after=str(after),parity=parity,timings=timings)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=json.dumps(compare(args.before,args.after),indent=2)+'\n'
    if args.output:args.output.write_text(result)
    print(result)
