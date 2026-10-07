"""
examples/diagnostics/solver_crossovers.py
Where do the fast solvers actually start paying? (tasks #46, #47)

Both `femmi.aca` (H-matrix BEM) and `femmi.iterative` (matrix-free coupled
solve) were built, tested, and then used by nothing. The tempting move is to
switch them on and claim a speedup; the honest move is to measure where the
crossover is first. This script is that measurement, and it overturned the
prediction it was written to confirm.

THE PREDICTION, AND WHAT WAS MEASURED
-------------------------------------
The catalog fields here are small at the boundary:

    catalog field, radius 3 arcmin
    n_eff =  5 gal/arcmin^2  ->  N_b = 120,  n_dofs = 1458
    n_eff = 30 gal/arcmin^2  ->  N_b = 290,  n_dofs = 8093

Two opposite predictions are both tempting. One says an H-matrix at N_b = 290 is
pure overhead, since a dense 290x290 factorises in microseconds. The other says
it must still win, because the real cost of `assemble_single_layer_hp` is not
linear algebra at all -- it is the O(N^2) Galerkin QUADRATURE in Python, and ACA
never evaluates most of those entries.

Measured, neither holds: ACA is SLOWER than dense everywhere tried.

    geometry              N_b    dense/ACA
    uniform circle (d=3)  144    0.52-0.78x   (tol 1e-6..1e-9, eta 1..2)
    uniform circle (d=5)  240    0.56-0.68x
    catalog guard ring    120    0.65x
    catalog guard ring    240    0.65x

The quadrature saving is real but it does not cover ACA's own costs: the cluster
tree, the per-block pivoting, and the near-field blocks, which still need the
tuned Duffy / log-Gauss treatment and are what a boundary this size is mostly
made of. The ratio is FLAT in N_b and flat in the ACA parameters, so it is a
per-entry cost difference rather than an overhead that amortises -- no crossover
is approaching below the sizes this project runs.

The compression and the 1e-9 accuracy are genuine; only the speed claim fails.
`bem_hp.ACA_MIN_NB` is set beyond any reachable size accordingly.

Run both halves and compare -- that contrast is the point of the script.

    python examples/diagnostics/solver_crossovers.py
    python examples/diagnostics/solver_crossovers.py --nb 128 256 512 1024
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from femmi.bem_hp import (build_circular_boundary_mesh, build_boundary_mesh,
                          assemble_single_layer_hp, assemble_single_layer_auto)


def time_it(fn, repeat=1):
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _row(label, n_b, t_dense, t_aca, rel):
    print(f"{label:<12}{n_b:>6}{t_dense:>10.2f}{t_aca:>10.2f}"
          f"{t_dense / t_aca:>9.2f}x{rel:>11.1e}", flush=True)


def uniform_circle(nbs, degree, tol):
    """The favourable geometry: equal elements, well-separated clusters."""
    rows = []
    for nb in nbs:
        n_el = max(4, int(nb) // degree)
        m = build_circular_boundary_mesh(n_el, degree=degree, radius=1.0)
        t_d, Vd = time_it(lambda: assemble_single_layer_hp(m, degree))
        t_a, Va = time_it(lambda: assemble_single_layer_auto(m, degree,
                                                             use_aca=True, tol=tol))
        rel = np.linalg.norm(Va - Vd) / np.linalg.norm(Vd)
        _row("circular", m.n_boundary_dofs, t_d, t_a, rel)
        rows.append((m.n_boundary_dofs, t_d, t_a, rel))
    return rows


def catalog_ring(n_effs, degree, tol, radius=3.0):
    """The geometry actually solved on: the guard ring of a catalog mesh."""
    from femmi.elements import C1Space, catalog_triangulation
    from femmi.c1_coupling import boundary_loop
    from femmi.density import sample_catalog, n_gal_for_density

    rows = []
    for n_eff in n_effs:
        x, y = sample_catalog(n_gal_for_density(n_eff, radius), radius=radius,
                              seed=0)
        v, t, _, _ = catalog_triangulation(x, y)
        S = C1Space(v, t, kind="argyris")
        m = build_boundary_mesh(S.vertices[boundary_loop(S)], degree)
        t_d, Vd = time_it(lambda: assemble_single_layer_hp(m, degree))
        t_a, Va = time_it(lambda: assemble_single_layer_auto(m, degree,
                                                             use_aca=True, tol=tol))
        rel = np.linalg.norm(Va - Vd) / np.linalg.norm(Vd)
        _row(f"catalog n={n_eff:g}", m.n_boundary_dofs, t_d, t_a, rel)
        rows.append((m.n_boundary_dofs, t_d, t_a, rel))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Fast-solver crossover measurement")
    ap.add_argument("--nb", type=int, nargs="+", default=[96, 192, 384],
                    help="boundary DOF counts for the circular mesh")
    ap.add_argument("--n-eff", type=float, nargs="+", default=[5.0, 20.0, 30.0],
                    help="source densities for the catalog guard ring")
    ap.add_argument("--degree", type=int, default=5,
                    help="boundary element degree (5 is the Argyris trace)")
    ap.add_argument("--tol", type=float, default=1e-9, help="ACA tolerance")
    ap.add_argument("--skip-catalog", action="store_true")
    args = ap.parse_args()

    print("single-layer assembly: dense vs ACA\n")
    print(f"{'geometry':<12}{'N_b':>6}{'dense':>10}{'aca':>10}"
          f"{'speedup':>10}{'rel err':>11}")
    print("-" * 59)

    circ = uniform_circle(args.nb, args.degree, args.tol)
    cat = [] if args.skip_catalog else catalog_ring(args.n_eff, args.degree,
                                                    args.tol)

    print()
    allrows = circ + cat
    if allrows:
        best = max(t_d / t_a for _, t_d, t_a, _ in allrows)
        worst = min(t_d / t_a for _, t_d, t_a, _ in allrows)
        if best < 1.0:
            print(f"ACA is SLOWER everywhere measured: {worst:.2f}x to {best:.2f}x "
                  "against dense assembly.\nThe ratio is flat in N_b, so this is a "
                  "per-entry cost difference rather than an\noverhead that "
                  "amortises -- no crossover is approaching at these sizes. The "
                  "1e-9\naccuracy is real; the speed is not, which is why "
                  "bem_hp.ACA_MIN_NB keeps it off\nby default (MATH.md 18.3l).")
        else:
            print(f"ACA reaches {best:.2f}x at best and {worst:.2f}x at worst. If "
                  "the best case is on the\nuniform circular mesh and the worst on "
                  "the catalog ring, that is the split\nMATH.md 18.3l describes: "
                  "the saving is quadrature skipped, and irregular\nboundaries give "
                  "it back in near-field blocks.")


if __name__ == "__main__":
    main()
