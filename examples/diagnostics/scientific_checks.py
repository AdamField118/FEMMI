"""Diagnose rotated-shear leakage and halo superposition at fixed geometry.

All outputs go to the requested directory. Vary sources, padding, and prior
independently; these controls are numerical diagnostics, not hyperparameter tuning.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from femmi import FlatCatalog, MapperConfig, FEMMapper
from femmi.catalog import analytic_gaussian_shear
from femmi.diagnostics import operator_checks, mass_norm
from femmi.calibration import write_json


def run(args):
    rows = []
    for n in args.sources:
        xy = np.random.default_rng(args.seed).uniform(-1.4, 1.4, (n, 2))
        radius = float(np.linalg.norm(xy, axis=1).max()) * (1 + 1e-12)
        if args.truth == "gaussian":
            components = [
                analytic_gaussian_shear(xy, sigma=0.4, amp=0.05, center=center)
                for center in ((-0.45, 0.25), (0.55, -0.25))
            ]
        else:
            from femmi.truth import galsim_nfw_truth

            components = [
                galsim_nfw_truth(xy, halos=[(1e14, 4.0, center)])
                for center in ((-0.45, 0.25), (0.55, -0.25))
            ]
        c = FlatCatalog(*xy.T, components[0][1], components[0][2], np.ones(n))
        for kind in args.methods:
            for padding in args.padding:
                for length in args.lengths:
                    m = FEMMapper(
                        c,
                        MapperConfig(
                            kind, args.lam, length, radius, boundary_padding=padding
                        ),
                    )
                    checks = operator_checks(m, args.seed)
                    e1 = m.reconstruct(components[0][1], components[0][2])
                    e2 = m.reconstruct(components[1][1], components[1][2])
                    g1 = components[0][1] + components[1][1]
                    g2 = components[0][2] + components[1][2]
                    total = m.reconstruct(g1, g2)
                    superposition = mass_norm(
                        m, total.coefficients - e1.coefficients - e2.coefficients
                    ) / max(mass_norm(m, total.coefficients), 1e-300)
                    for name, fit, truth in [
                        ("single", e1, components[0][0]),
                        ("two", total, components[0][0] + components[1][0]),
                    ]:
                        response = m.diagnose_b(
                            fit.observed_g1, fit.observed_g2, e_fit=fit
                        )
                        # Generate a discrete E input using the *same* operator.
                        manufactured = m.diagnose_b(fit.predicted_g1, fit.predicted_g2)
                        inside = np.linalg.norm(xy, axis=1) < 1.0
                        error = fit.kappa - truth
                        strict = (
                            FEMMapper(
                                c,
                                replace(m.config, rtol=1e-10, residual_tolerance=1e-7),
                            )
                            if args.strict
                            else None
                        )
                        strict_change = None
                        if strict:
                            refined = strict.reconstruct(
                                fit.observed_g1, fit.observed_g2
                            )
                            strict_change = mass_norm(
                                m, refined.coefficients - fit.coefficients
                            ) / max(mass_norm(m, fit.coefficients), 1e-300)
                        row = dict(
                            sources=n,
                            method=kind,
                            truth=args.truth,
                            scenario=name,
                            padding=padding,
                            length=length,
                            lam=args.lam,
                            superposition_relative=superposition,
                            strict_relative_change=strict_change,
                            checks=checks,
                            b_response=response["diagnostics"],
                            manufactured_b=manufactured["diagnostics"],
                            inner_shape_rmse=float(np.std(error[inside])),
                            boundary_shape_rmse=float(np.std(error[~inside])),
                            solver=fit.diagnostics,
                        )
                        if (
                            checks["adjoint_relative"] > 1e-8
                            or checks["cross_skew_relative"] > 1e-8
                            or superposition > 2e-5
                            or response["diagnostics"]["closure_relative"] > 2e-5
                        ):
                            raise RuntimeError(f"algebraic diagnostic failed: {row}")
                        rows.append(row)
                        write_json(
                            args.output / "diagnostics.json",
                            dict(
                                arguments=vars(args) | {"output": str(args.output)},
                                rows=rows,
                            ),
                        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", type=int, default=[64, 180])
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["p3", "argyris", "hct"],
        default=["p3", "argyris", "hct"],
    )
    p.add_argument("--padding", nargs="+", type=float, default=[1.12, 1.5])
    p.add_argument("--lengths", nargs="+", type=float, default=[0.3])
    p.add_argument("--lam", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--truth", choices=["gaussian", "nfw"], default="gaussian")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
