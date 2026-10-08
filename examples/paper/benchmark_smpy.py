"""Held-out comparisons against immutable SMPy; no substitute implementations.

python examples/paper/benchmark_smpy.py --config configs/benchmarks/publication.json --output results/smpy
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from femmi.protocol import run_suite

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--names',nargs='+')
    args=p.parse_args()
    configs=json.loads(args.config.read_text())
    if isinstance(configs,dict):configs=[configs]
    selected=[c for c in configs if not args.names or c['name'] in args.names]
    if not selected:p.error('no selected scenarios')
    run_suite(selected,args.output)
