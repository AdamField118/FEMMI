"""Held-out comparisons against immutable SMPy; no substitute implementations.

python examples/paper/benchmark_smpy.py --config benchmarks/smpy/configs.json --output results/smpy
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from femmi.smpy import verify_installation
from femmi.calibration import calibrate_and_evaluate,write_json
from femmi.comparison import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--names',nargs='+')
    args=p.parse_args()
    provenance=verify_installation()
    configs=json.loads(args.config.read_text())
    if isinstance(configs,dict):configs=[configs]
    selected=[c for c in configs if not args.names or c['name'] in args.names]
    if not selected:p.error('no selected scenarios')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'smpy-version.json',provenance)
    for config in selected:
        calibrate_and_evaluate(config,args.output/config['name'])
    summarize(args.output)
