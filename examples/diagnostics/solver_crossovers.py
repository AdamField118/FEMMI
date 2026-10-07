"""Dense/ACA single-layer assembly timing on circles and catalogue rings.

Historical timings predate the CPU Numba backend and are not current speed
claims. Run numerical_followup.py for warmed repeated measurements, saved raw
results, actual direct/GMRES costs and dual-grid spectra. Neither a flat ratio
nor a limited mesh range determines an unmeasured asymptotic crossover.
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
        print(f"Dense/ACA assembly ratio: {worst:.3g}x to {best:.3g}x in the tested cases. "
              "Repeat with warm compilation and controlled load before a speed claim. "
              "No crossover is inferred outside the measured range.")



if __name__ == "__main__":
    main()
