"""Serial fresh-process NumPy/Numba comparisons at matched reconstruction quality."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from compare_cpu import compare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--sources',type=int,default=200)
    p.add_argument('--processes',type=int,default=3)
    args=p.parse_args()
    if args.processes<1:p.error('processes must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    driver=Path(__file__).with_name('profile_cpu.py')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    comparisons=[]
    with tempfile.TemporaryDirectory(prefix='femmi-numba-') as cache:
        env['NUMBA_CACHE_DIR']=cache
        for kind in ('p3','argyris','hct'):
            for backend in ('numba','numpy'):
                for i in range(args.processes):
                    out=args.output/f'{kind}-{backend}-{i}.json'
                    cmd=[sys.executable,str(driver),'--kind',kind,'--sources',str(args.sources),
                         '--repeats','3','--output',str(out)]
                    if backend=='numba':cmd.append('--warm-numba')
                    print(out,flush=True)
                    with out.with_suffix('.log').open('w') as log:
                        subprocess.run(cmd,env=dict(env,FEMMI_BEM_BACKEND=backend),stdout=log,stderr=subprocess.STDOUT,check=True)
            for i in range(args.processes):
                comparisons.append(compare(args.output/f'{kind}-numpy-{i}.json',args.output/f'{kind}-numba-{i}.json'))
    (args.output/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
