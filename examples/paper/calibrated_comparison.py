"""Reproducible held-out calibration/evaluation; every arm uses identical data.

Run: python examples/paper/calibrated_comparison.py --config CONFIG --output DIR
JSON configs are supplied in benchmarks/calibration/configs.json.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from femmi.calibration import calibrate_and_evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--names',nargs='+',help='scenario names from the config list')
    args=p.parse_args()
    configs=json.loads(Path(args.config).read_text())
    if isinstance(configs,dict):configs=[configs]
    for config in configs:
        if args.names and config['name'] not in args.names:continue
        print(config['name'],flush=True)
        calibrate_and_evaluate(config,Path(args.output)/config['name'])
