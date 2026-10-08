"""Algebraic checks and conditional E-to-B response for a fixed mapper."""

import numpy as np


def mass_norm(mapper, coefficients):
    """Finite-element L2 norm, including derivative DOFs in C1 spaces."""
    k = np.asarray(coefficients)
    return float(np.sqrt(max(float(k @ (mapper.solver.M @ k)), 0.0)))


def b_response(mapper, g1=None, g2=None, *, e_fit=None, b_fit=None):
    if (g1 is None) != (g2 is None):
        raise ValueError("supply both shear components")
    c = mapper.catalogue
    a, b = (c.g1, c.g2) if g1 is None else (np.asarray(g1), np.asarray(g2))
    for fit, observed in ((e_fit, (a, b)), (b_fit, (b, -a))):
        if fit is not None and (
            fit.mapper is not mapper
            or not np.array_equal(fit.observed_g1, observed[0])
            or not np.array_equal(fit.observed_g2, observed[1])
        ):
            raise ValueError("supplied fit does not match mapper and observations")
    e = e_fit or mapper.reconstruct(a, b)
    lam, length = e.config.lam, e.config.length
    if b_fit is not None and (b_fit.config.lam, b_fit.config.length) != (lam, length):
        raise ValueError("E and B fits must use the same prior")
    raw = b_fit or mapper.reconstruct(b, -a, lam=lam, length=length)
    leakage = mapper.reconstruct(
        e.predicted_g2, -e.predicted_g1, lam=lam, length=length
    )
    residual = mapper.reconstruct(
        b - e.predicted_g2, -a + e.predicted_g1, lam=lam, length=length
    )
    scale = max(
        mass_norm(mapper, raw.coefficients),
        mass_norm(mapper, leakage.coefficients),
        mass_norm(mapper, residual.coefficients),
        1e-300,
    )
    closure = (
        mass_norm(
            mapper, raw.coefficients - leakage.coefficients - residual.coefficients
        )
        / scale
    )
    return dict(
        e=e,
        raw=raw,
        leakage=leakage,
        residual=residual,
        diagnostics=dict(
            interpretation="rotated-shear response conditional on the fitted E model; not pure B",
            e_l2=mass_norm(mapper, e.coefficients),
            raw_b_l2=mass_norm(mapper, raw.coefficients),
            predicted_leakage_l2=mass_norm(mapper, leakage.coefficients),
            residual_b_l2=mass_norm(mapper, residual.coefficients),
            closure_relative=float(closure),
            lam=lam,
            length=length,
        ),
    )


def operator_checks(mapper, seed=0):
    """Adjoint and E/B cross-response identity without forming dense F."""
    rng = np.random.default_rng(seed)
    s = mapper.solver
    k = rng.normal(size=mapper.dofs)
    a = rng.normal(size=len(s.weight))
    b = rng.normal(size=len(a))
    u, v = s.forward(k)
    left = float(a @ u + b @ v)
    right = float(k @ s.transpose(a, b))
    adjoint = abs(left - right) / max(abs(left), abs(right), 1.0)
    cross = s.transpose(s.weight * v, -s.weight * u)
    # k.T (F1.T W F2 - F2.T W F1) k is zero by skew symmetry.
    skew = abs(float(k @ cross)) / max(np.linalg.norm(k) * np.linalg.norm(cross), 1.0)
    return dict(
        adjoint_relative=float(adjoint),
        cross_skew_relative=float(skew),
        cross_response_norm=float(np.linalg.norm(cross)),
    )
