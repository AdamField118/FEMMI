"""Matched Schneider aperture statistics for convergence and shear maps."""

import numpy as np
from scipy.ndimage import convolve, binary_erosion


def convergence_aperture(kappa, radius_pixels, order=3):
    """Apply the compensated U paired with SMPy's polynomial Q filter.

    U(x)=(l+2)/(pi R^2) (1-x^2)^l [1-(l+2)x^2], x<=1.
    Subtract the sampled kernel mean within its support for exact discrete
    compensation. This quadrature correction vanishes under grid refinement.
    """
    if not np.isfinite(radius_pixels) or radius_pixels < 2:
        raise ValueError("aperture radius must span at least two pixels")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    size = int(np.ceil(2 * radius_pixels))
    size += size % 2 == 0
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    t = (x * x + y * y) / radius_pixels**2
    support = t <= 1
    u = np.zeros_like(t)
    u[support] = (
        (order + 2)
        / (np.pi * radius_pixels**2)
        * (1 - t[support]) ** order
        * (1 - (order + 2) * t[support])
    )
    u[support] -= u.sum() / support.sum()
    return convolve(
        np.nan_to_num(kappa, nan=0.0), u, mode="constant", cval=0.0
    ), support


def _target(truth, regions, config, radius):
    size = config["evaluation_grid"]
    aperture = float(config.get("aperture_radius_arcmin", radius / 4))
    rp = aperture / (2 * radius / size)
    target, footprint = convergence_aperture(np.asarray(truth).reshape(size, size), rp)
    good = binary_erosion(
        regions["field"].reshape(size, size), structure=footprint, border_value=0
    )
    if not good.any():
        raise ValueError("no complete apertures on common scoring support")
    return target, good, aperture, rp


def quadrature_control(truth, g1, g2, regions, config, radius):
    """Dense noiseless Q-versus-U discrepancy, separate from catalogue errors."""
    from .smpy import create_maps

    size = config["evaluation_grid"]
    target, good, _, rp = _target(truth, regions, config, radius)
    shear_target, _, _ = create_maps(
        np.asarray(g1).reshape(size, size),
        np.asarray(g2).reshape(size, size),
        np.ones((size, size)),
        "smpy_aperture",
        aperture_scale=rp,
    )
    error = shear_target[good] - target[good]
    return dict(
        rmse=float(np.sqrt(np.mean(error**2))),
        relative_l2=float(
            np.linalg.norm(error) / max(np.linalg.norm(target[good]), 1e-300)
        ),
        pixels=int(good.sum()),
        radius_pixels=rp,
        interpretation="dense noiseless Q-versus-U discretization control; not a fitted method",
    )


def matched_aperture(c, values, truth, regions, config, method):
    """Score a kappa map or upstream shear aperture on identical full support."""
    from .smpy import create_maps
    from .catalog import bin_shear_to_grid

    size = config["evaluation_grid"]
    radius = c.radius
    target, good, aperture, rp = _target(truth, regions, config, radius)
    if method == "smpy_aperture":
        a, b, w, _ = bin_shear_to_grid(
            c.x,
            c.y,
            c.g1,
            c.g2,
            weight=c.weight,
            grid_size=size,
            extent=(-radius, radius, -radius, radius),
        )
        estimate, _, _ = create_maps(a, b, w, method, aperture_scale=rp)
    else:
        estimate, _ = convergence_aperture(np.asarray(values).reshape(size, size), rp)
    error = estimate[good] - target[good]
    return dict(
        method=method,
        seed=c.seed,
        scenario=config.get("name", "baseline"),
        catalogue_hash=c.fingerprint,
        n_eff_nominal=c.nominal_density,
        aperture_radius_arcmin=aperture,
        aperture_radius_pixels=rp,
        order=3,
        aperture_rmse=float(np.sqrt(np.mean(error**2))),
        aperture_bias=float(np.mean(error)),
        aperture_pixels=int(good.sum()),
        target="discretely compensated Schneider U applied to common truth grid",
    )
