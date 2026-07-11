"""
examples/bem_dtn_diagnostic.py
Decides whether the scalar length-factor coupling fix (couple_scale) is
SUFFICIENT, or whether the symmetric Steklov-Poincare operator must be derived.

This is a boundary-only test (no volume mesh, no reconstruction), so it is cheap
even at high boundary resolution -- push n_boundary and the mode count on an HPC
node to get a clean verdict.

Method
------
The exterior Steklov-Poincare (Dirichlet-to-Neumann) map S sends a boundary
trace u to the outward normal derivative of its decaying harmonic extension. The
2D decaying harmonics are known exactly: u_n = cos(n*theta)/r^n, with
    du_n/dn = -n cos(n*theta) / r^(n+1)   (outward normal = +r on a disk).
The FEM-BEM coupling contributes, in Galerkin form, C u ~ M_b (du/dn). We fit the
best scalar factor s (which absorbs the coupling's wrong 1/L scaling and any
overall constant) and report the RESIDUAL that remains after that best scalar.

Verdict
-------
  * If the residual -> 0 as n_boundary grows (for every mode), a single scalar
    length factor reproduces the true DtN and couple_scale is provably
    sufficient. (The best-fit s itself is not the constant to copy -- it absorbs
    an M_b normalisation that scales with resolution -- so calibrate the constant
    separately, e.g. by matching a reference solution, not from s directly.)
  * If the residual is resolution-INDEPENDENT and mode-dependent (it plateaus),
    the non-symmetric coupling is missing operator structure and NO scalar can
    fix it -> derive and implement the symmetric Steklov-Poincare operator
    S = D + (1/2 M + K')^T V^{-1} (1/2 M + K')  (D = hypersingular operator).

On the shipped coupling this residual plateaus at ~0.1-0.2, i.e. the scalar fix
is a good stopgap (right scaling, ~15% magnitude error) but the symmetric
operator is needed for the accuracy the method targets. A sign variant
(0.5 M - K) is included because it markedly lowers the n=1 residual here, hinting
at a double-layer sign convention worth checking during the derivation.

Run:
    python examples/bem_dtn_diagnostic.py
    python examples/bem_dtn_diagnostic.py --nb 160 400 800 --modes 1 2 3 4
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.operators import build_operators_circular
from femmi.bem       import assemble_bem_matrices


def dtn_residuals(radius, n_boundary, modes, rhs_sign=+1):
    ops = build_operators_circular(radius=radius, n_boundary=n_boundary, verbose=False)
    V, K, M = assemble_bem_matrices(ops.bnd_mesh, n_quad_sl=25, n_quad_dl=8)
    C = np.linalg.solve(V, 0.5 * M + rhs_sign * K)
    bn = ops.bnd_mesh.nodes
    th = np.arctan2(bn[:, 1], bn[:, 0]); r = np.hypot(bn[:, 0], bn[:, 1])
    out = {}
    for n in modes:
        u    = np.cos(n * th) / r**n
        dudn = -n * np.cos(n * th) / r**(n + 1)
        Cu   = C @ u
        T    = M @ dudn
        s    = float(np.dot(Cu, T) / np.dot(Cu, Cu))         # best scalar factor
        out[n] = dict(s_over_R=s / radius,
                      resid=float(np.linalg.norm(s * Cu - T) / np.linalg.norm(T)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--nb", type=int, nargs="+", default=[120, 240, 480])
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    print("Exterior DtN Galerkin residual  (does a scalar factor suffice?)")
    print(f"  radius={args.radius}  modes={args.modes}\n")
    for sign, tag in [(+1, "0.5M+K (shipped)"), (-1, "0.5M-K (sign variant)")]:
        print(f"coupling rhs = {tag}")
        table = {nb: dtn_residuals(args.radius, nb, args.modes, sign) for nb in args.nb}
        header = "  n_boundary | " + " | ".join(f"n={n} resid  s/R" for n in args.modes)
        print(header)
        for nb in args.nb:
            cells = " | ".join(f"{table[nb][n]['resid']:.3f}  {table[nb][n]['s_over_R']:.4f}"
                               for n in args.modes)
            print(f"  {nb:9d}  | {cells}")
        # verdict on the shipped coupling
        if sign == +1:
            first, last = args.nb[0], args.nb[-1]
            worsens = all(table[last][n]['resid'] >= 0.8 * table[first][n]['resid']
                          for n in args.modes)
            print("  -> residual does NOT vanish with resolution "
                  "(structural): a scalar factor is insufficient; derive the "
                  "symmetric Steklov-Poincare operator."
                  if worsens else
                  "  -> residual shrinks with resolution: a scalar length factor "
                  "may suffice (read the constant off s/R).")
        print()


if __name__ == "__main__":
    main()
