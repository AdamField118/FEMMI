"""
examples/benchmark_grid.py
Run the {element} x {prior} x {method} grid and print the ranked table.

Everything is scored on the SAME independent truth (femmi.truth), the same noise
realisation and the same mesh, with Kaiser-Squires included automatically as the
baseline. Cost is reported as global DOFs and wall-clock beside accuracy, since
accuracy-per-DOF is the only fair axis for comparing elements.

    python examples/benchmark_grid.py
    python examples/benchmark_grid.py --nx 18 --priors wiener tv --noise 0.03
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from femmi.benchmark import sweep, to_table


def main():
    ap = argparse.ArgumentParser(description="FEMMI configuration benchmark")
    ap.add_argument("--nx", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="nfw",
                    choices=["nfw", "lognormal", "massivenus"],
                    help="independent truth field. 'nfw' is smooth (favours "
                         "Wiener); 'lognormal' is peaked and non-Gaussian, which "
                         "is where TV/sparsity are supposed to win. Run both.")
    ap.add_argument("--elements", nargs="+", default=["p3"])
    ap.add_argument("--priors", nargs="+", default=["wiener", "tv", "sparse", "maxent"])
    ap.add_argument("--methods", nargs="+", default=["map"])
    ap.add_argument("--sort", default="rel_l2_dc_removed",
                    choices=["rel_l2", "rel_l2_dc_removed", "mean_err", "seconds"])
    ap.add_argument("--no-ks", action="store_true", help="omit the KS baseline")
    args = ap.parse_args()

    grid = dict(element=args.elements, prior=args.priors, method=args.methods)
    rows = sweep(grid, nx=args.nx, noise_std=args.noise, seed=args.seed,
                 source=args.source, include_ks=not args.no_ks)
    print()
    print(f"truth: {args.source}   nx={args.nx}   noise={args.noise}")
    print(to_table(rows, sort_by=args.sort))
    print("\n'shape L2' is the DC-removed error -- the part of the comparison that")
    print("survives the mass-sheet limitation documented in MATH.md 6.3a.")


if __name__ == "__main__":
    main()
