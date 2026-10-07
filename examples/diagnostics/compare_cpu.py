"""Compare paired profile_cpu runs; fail on material operator/quality changes.

This checks measured numerical output, never enforces hardware timing ratios.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(before, after):
    b=json.loads(before.read_text()); a=json.loads(after.read_text())
    for key in ['kind','size','maxiter','repeats','lam','wiener_length','gtol']:
        if a.get(key)!=b.get(key):
            raise ValueError(f'mismatched benchmark setting: {key}')
    parity={}
    with np.load(before.with_suffix('.npz')) as B, np.load(after.with_suffix('.npz')) as A:
        for key in ['g1','g2']:
            np.testing.assert_array_equal(A[key],B[key])
        for key in ['forward_probe','gradient_probe']:
            error=float(np.linalg.norm(A[key]-B[key])/max(np.linalg.norm(B[key]),1e-300))
            parity[key+'_relative']=error
            if error>1e-9:
                raise AssertionError(f'{key} relative error {error}')
        for j,(bf,af) in enumerate(zip(b['fits'],a['fits'])):
            if not bf['converged'] or not af['converged']:
                raise AssertionError('matched-quality comparison requires both fits to converge')
            # Stopping criteria allow small variations in weak directions.
            # Inspect objective and predicted data, not just optimizer status.
            np.testing.assert_allclose(af['objective'],bf['objective'],rtol=1e-5,atol=1e-10)
            key=f'prediction_{j}'
            error=float(np.linalg.norm(A[key]-B[key])/max(np.linalg.norm(B[key]),1e-300))
            parity[key+'_relative']=error
            if error>1e-3:
                raise AssertionError(f'prediction relative error {error}')
            parity[f'kappa_{j}_relative']=float(np.linalg.norm(A[f'kappa_{j}']-B[f'kappa_{j}'])/
                                                max(np.linalg.norm(B[f'kappa_{j}']),1e-300))
            stride={'p3':1,'argyris':6,'hct':3}[b['kind']]
            av=A[f'kappa_{j}'][:len(A['g1'])*stride:stride]
            bv=B[f'kappa_{j}'][:len(B['g1'])*stride:stride]
            value_error=float(np.linalg.norm(av-bv)/max(np.linalg.norm(bv),1e-300))
            parity[f'kappa_values_{j}_relative']=value_error
            if value_error>1e-3:
                raise AssertionError(f'vertex kappa relative error {value_error}')
    timings={key:dict(before=b['phases'][key],after=a['phases'][key],
                      speedup=b['phases'][key]/a['phases'][key])
             for key in ['setup_s','single_layer_s','sparse_factorization_s','reconstruction_s','total_s']}
    return dict(before=str(before),after=str(after),parity=parity,timings=timings)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path)
    p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=json.dumps(compare(args.before,args.after),indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    print(result)
