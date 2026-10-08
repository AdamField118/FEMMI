"""Regenerate paired summaries from a completed protocol run without refitting."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from femmi.comparison import summarize

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Directory containing scenario subdirectories')
    args=p.parse_args()
    summarize(args.root)
