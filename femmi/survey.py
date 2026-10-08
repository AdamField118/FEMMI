"""Survey FITS-to-FITS API, following SMPy 26d231f's public workflow.

Numerical reconstruction remains catalogue-native. Output images are samples of
that FE solution, not a second gridded inverse problem. See docs/survey-io.md for
verified upstream defects and the explicit geometry/calibration differences.
"""

from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import time
import random
import numpy as np
from .io import (
    FlatCatalog,
    read_fits_catalog,
    gnomonic_project,
    gnomonic_deproject,
    ARCMIN_PER_RAD,
)
from .mapping import FEMMapper, MapperConfig


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _software_versions():
    from importlib.metadata import version, PackageNotFoundError

    versions = {}
    for name in ("femmi", "numpy", "scipy", "astropy", "jax", "numba"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _positive(value, name):
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _bounds(a, b, names, units):
    return dict(
        coord1_min=float(np.min(a)),
        coord1_max=float(np.max(a)),
        coord2_min=float(np.min(b)),
        coord2_max=float(np.max(b)),
        coord1_name=names[0],
        coord2_name=names[1],
        units=units,
    )


def _sky_to_xy(ra, dec, centre, projection):
    dra = (np.asarray(ra) - centre[0] + 180) % 360 - 180
    if projection == "smpy":
        return dra * np.cos(np.deg2rad(dec)) * 60, (np.asarray(dec) - centre[1]) * 60
    x, y = gnomonic_project(np.deg2rad(ra), np.deg2rad(dec), *np.deg2rad(centre))
    return x * ARCMIN_PER_RAD, y * ARCMIN_PER_RAD


def _prepare(
    data,
    coord_system,
    coord1,
    coord2,
    g1_col,
    g2_col,
    weight_col,
    input_hdu,
    convention,
    projection,
    centre,
    pixel_scale_arcmin,
    pixel_origin,
    read_options,
):
    columns = dict(ra=coord1, dec=coord2, g1=g1_col, g2=g2_col, weight=weight_col)
    cat = read_fits_catalog(
        data,
        column_map=columns,
        hdu=input_hdu,
        coord_system=coord_system,
        **read_options,
    )
    if cat.n < 3:
        raise ValueError("fewer than three valid selected sources")
    sign = -1 if convention == "smpy" and coord_system == "radec" else 1
    if coord_system == "radec":
        # SMPy uses midrange centering; unwrap first to fix fields crossing RA=0.
        anchor = cat.center()[0]
        ra = anchor + (cat.ra - anchor + 180) % 360 - 180
        centre = (
            tuple(centre)
            if centre is not None
            else ((ra.min() + ra.max()) / 2, (cat.dec.min() + cat.dec.max()) / 2)
        )
        if projection == "tan":
            flat = cat.to_tangent_plane(center=centre, flip_g2=sign < 0)
        else:
            x, y = _sky_to_xy(cat.ra, cat.dec, centre, projection)
            flat = FlatCatalog(
                x,
                y,
                cat.g1,
                sign * cat.g2,
                cat.weight,
                z=cat.z,
                center=centre,
                name=cat.name,
                meta=dict(cat.meta),
                row_index=cat.row_index,
                object_id=cat.object_id,
            )
        true = _bounds(ra, cat.dec, ("RA", "Dec"), "deg")
        scaled = _bounds(flat.x / 60, flat.y / 60, ("Scaled RA", "Scaled Dec"), "deg")
        pixel_centre = None
    else:
        scale = _positive(
            pixel_scale_arcmin, "pixel_scale_arcmin for pixel coordinates"
        )
        # Physical mesh units are arcmin; pixel catalogue convention matches SMPy.
        pixel_centre = (
            (np.floor(cat.x.min()) + np.ceil(cat.x.max())) / 2,
            (np.floor(cat.y.min()) + np.ceil(cat.y.max())) / 2,
        )
        flat = replace(
            cat,
            x=(cat.x - pixel_centre[0]) * scale,
            y=(cat.y - pixel_centre[1]) * scale,
            units="arcmin",
        )
        true = _bounds(
            np.array([np.floor(cat.x.min()), np.ceil(cat.x.max())]),
            np.array([np.floor(cat.y.min()), np.ceil(cat.y.max())]),
            ("X", "Y"),
            "pixels",
        )
        scaled = dict(true)
        centre = (0.0, 0.0)
    flat.meta.update(
        coord_system=coord_system,
        shear_convention=convention,
        projection=projection,
        source_plane="effective",
        redshifts_used_in_operator=False,
        pixel_origin=pixel_origin,
    )
    return cat, flat, centre, pixel_centre, scaled, true


def _grid(
    flat,
    centre,
    pixel_centre,
    scaled,
    true,
    coord_system,
    projection,
    pixel_scale,
    downsample_factor,
    pixel_scale_arcmin,
    pixel_origin,
    reference_image,
    reference_hdu,
    max_pixels,
    footprint,
):
    from astropy.wcs import WCS
    from astropy.io import fits

    if reference_image:
        with fits.open(reference_image) as hdul:
            hdu = hdul[reference_hdu]
            if hdu.data is None or hdu.data.ndim != 2:
                raise ValueError("reference image must be two dimensional")
            shape = hdu.data.shape
            wcs = WCS(hdu.header, hdul)
            if any(
                getattr(wcs, k) is not None
                for k in ("cpdis1", "cpdis2", "det2im1", "det2im2")
            ):
                raise ValueError(
                    "reference WCS with lookup-table distortions is unsupported; reproject it to TAN/SIP first"
                )
        if not wcs.has_celestial or coord_system != "radec":
            raise ValueError("reference image requires celestial WCS and RA/Dec input")
    else:
        width = true["coord1_max"] - true["coord1_min"]
        height = true["coord2_max"] - true["coord2_min"]
        wcs = WCS(naxis=2)
        if coord_system == "radec":
            step = _positive(pixel_scale, "pixel_scale in arcmin")
            xmin, xmax = scaled["coord1_min"] * 60, scaled["coord1_max"] * 60
            ymin, ymax = scaled["coord2_min"] * 60, scaled["coord2_max"] * 60
            nx = max(1, int(np.ceil((xmax - xmin) / step)))
            ny = max(1, int(np.ceil((ymax - ymin) / step)))
            if projection == "smpy":
                source_dec = footprint.y / 60 + centre[1]
                source_ra = centre[0] + footprint.x / (
                    60 * np.cos(np.deg2rad(source_dec))
                )
                tx, ty = _sky_to_xy(source_ra, source_dec, centre, "tan")
                xmin, xmax = float(tx.min()), float(tx.max())
                ymin, ymax = float(ty.min()), float(ty.max())
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            if dx <= 0 or dy <= 0:
                raise ValueError("catalogue must span both coordinate axes")
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.wcs.cunit = ["deg", "deg"]
            wcs.wcs.crval = centre
            wcs.wcs.cdelt = [dx / 60, dy / 60]
            # First pixel centre is xmin+dx/2; FITS pixels start at one.
            wcs.wcs.crpix = [0.5 - xmin / dx, 0.5 - ymin / dy]
            wcs.wcs.radesys = "ICRS"
        else:
            down = _positive(downsample_factor, "downsample_factor")
            nx = max(1, int(np.ceil(width / down)))
            ny = max(1, int(np.ceil(height / down)))
            dx = width / nx
            dy = height / ny
            if dx <= 0 or dy <= 0:
                raise ValueError("catalogue must span both coordinate axes")
            wcs.wcs.ctype = ["LINEAR", "LINEAR"]
            wcs.wcs.cunit = ["pix", "pix"]
            wcs.wcs.crpix = [1, 1]
            wcs.wcs.crval = [true["coord1_min"] + dx / 2, true["coord2_min"] + dy / 2]
            wcs.wcs.cdelt = [dx, dy]
        shape = (ny, nx)
    if np.prod(shape) > max_pixels:
        raise ValueError(
            "output grid exceeds max_pixels; increase pixel scale or explicitly raise the limit"
        )
    yy, xx = np.indices(shape)
    a, b = wcs.all_pix2world(xx, yy, 0)
    if coord_system == "radec":
        from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
        from astropy.coordinates import SkyCoord

        sky = pixel_to_skycoord(xx, yy, wcs, origin=0).icrs
        x, y = _sky_to_xy(sky.ra.deg, sky.dec.deg, centre, projection)
        # Source bins use this very same WCS, including reference-image rotations.
        ra, dec = (
            gnomonic_deproject(
                flat.x / ARCMIN_PER_RAD, flat.y / ARCMIN_PER_RAD, *np.deg2rad(centre)
            )
            if projection == "tan"
            else (None, None)
        )
        if projection == "tan":
            ra, dec = np.rad2deg(ra), np.rad2deg(dec)
        else:
            dec = flat.y / 60 + centre[1]
            ra = centre[0] + flat.x / (60 * np.cos(np.deg2rad(dec)))
        sx, sy = skycoord_to_pixel(
            SkyCoord(ra, dec, unit="deg", frame="icrs"), wcs, origin=0
        )
    else:
        x = (a - pixel_centre[0]) * pixel_scale_arcmin
        y = (b - pixel_centre[1]) * pixel_scale_arcmin
        sx, sy = wcs.all_world2pix(
            flat.x / pixel_scale_arcmin + pixel_centre[0],
            flat.y / pixel_scale_arcmin + pixel_centre[1],
            0,
        )
    return (
        wcs,
        shape,
        np.column_stack([x.ravel(), y.ravel()]),
        np.column_stack([sx, sy]),
    )


def _coverage(pixels, weight, shape):
    # Include a source on the outer half-pixel boundary; SMPy's digitize path drops maxima.
    ix = np.floor(pixels[:, 0] + 0.5).astype(int)
    iy = np.floor(pixels[:, 1] + 0.5).astype(int)
    ix[np.isclose(pixels[:, 0], -0.5, rtol=0, atol=1e-8)] = 0
    iy[np.isclose(pixels[:, 1], -0.5, rtol=0, atol=1e-8)] = 0
    ix[np.isclose(pixels[:, 0], shape[1] - 0.5, rtol=0, atol=1e-8)] = shape[1] - 1
    iy[np.isclose(pixels[:, 1], shape[0] - 0.5, rtol=0, atol=1e-8)] = shape[0] - 1
    ok = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
    index = iy[ok] * shape[1] + ix[ok]
    n = int(np.prod(shape))
    count = np.bincount(index, minlength=n).reshape(shape)
    w = np.bincount(index, weights=weight[ok], minlength=n).reshape(shape)
    w2 = np.bincount(index, weights=weight[ok] ** 2, minlength=n).reshape(shape)
    eff = np.divide(w * w, w2, out=np.zeros_like(w), where=w2 > 0)
    return count, w, eff, ok


def _smooth(image, sigma):
    from scipy.ndimage import gaussian_filter

    if sigma is None or sigma == 0:
        return image.copy()
    valid = np.isfinite(image)
    out = gaussian_filter(np.where(valid, image, 0.0), sigma=sigma)
    norm = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.divide(out, norm, out=np.full_like(out, np.nan), where=norm > 0)
    out[~valid] = np.nan
    return out


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def map_mass(
    data,
    method="p3",
    coord_system="radec",
    pixel_scale=None,
    downsample_factor=None,
    output_dir=".",
    output_base_name="femmi_output",
    g1_col="g1",
    g2_col="g2",
    weight_col=None,
    mode="E",
    create_snr=False,
    create_counts_map=False,
    overlay_counts_map=False,
    save_fits=False,
    save_plots=True,
    print_timing=False,
    smoothing=None,
    *,
    lam=None,
    length=None,
    input_hdu=1,
    coord1=None,
    coord2=None,
    pixel_scale_arcmin=None,
    pixel_origin=1,
    centre=None,
    radius=None,
    shear_convention="smpy",
    projection="smpy",
    response=None,
    input_kind="shear",
    ra_unit=None,
    dec_unit=None,
    id_col=None,
    selection=None,
    flag_col=None,
    reject_bits=0,
    z_col=None,
    z_range=None,
    max_shear=None,
    source_plane="effective",
    reference_image=None,
    reference_hdu=0,
    max_pixels=4_000_000,
    masks=(),
    num_shuffles=100,
    shuffle_type="spatial",
    seed=0,
    snr_smoothing=2.0,
    rtol=1e-8,
    residual_tolerance=1e-6,
    maxiter=2000,
    xray_image=None,
    xray_levels=None,
    xray_contours=None,
    plotting=None,
    snr_plot_title="Signal-to-Noise Map",
    overwrite=False,
    boundary_padding=1.12,
    boundary_nodes=None,
    b_diagnostics=False,
):
    """SMPy-style survey workflow returning maps, boundaries, WCS and diagnostics.

    ``pixel_scale`` is output arcmin/pixel. Pixel catalogues additionally require
    ``pixel_scale_arcmin`` (arcmin per input pixel) and ``downsample_factor``.
    Prior lambda/length are explicit; there is no scientifically universal default.
    Mask circles are (x,y,r) in the computational arcmin plane. Redshift columns
    are retained for selection/provenance, never used as per-source efficiencies.
    """
    start = time.perf_counter()
    timings = {}
    from .survey_output import validate_plot_options

    validate_plot_options(plotting, xray_contours)
    if centre is not None and (
        np.shape(centre) != (2,)
        or not np.all(np.isfinite(centre))
        or abs(centre[1]) > 90
    ):
        raise ValueError("centre must contain finite RA/Dec degrees")
    if not isinstance(max_pixels, int) or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if method not in ("p3", "argyris", "hct"):
        raise ValueError("method must be p3, argyris or hct")
    if coord_system not in ("radec", "pixel"):
        raise ValueError("coord_system must be radec or pixel")
    if shear_convention not in ("smpy", "east_north"):
        raise ValueError("unknown shear_convention")
    if projection not in ("smpy", "tan"):
        raise ValueError("projection must be smpy or tan")
    if source_plane != "effective":
        raise ValueError("only an effective source plane is modeled")
    if pixel_origin not in (0, 1):
        raise ValueError("pixel_origin must be 0 or 1")
    modes = [mode] if isinstance(mode, str) else list(mode)
    if b_diagnostics and "B" not in modes:
        raise ValueError("b_diagnostics requires B in mode")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"E", "B"}:
        raise ValueError("mode must contain E and/or B once")
    lam = _positive(lam, "lam")
    length = float(length) if length is not None else -1
    if not np.isfinite(length) or length < 0:
        raise ValueError("length must be finite and nonnegative")
    for sigma in (smoothing, snr_smoothing):
        if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
            raise ValueError("smoothing must be nonnegative")
    if create_snr and (not isinstance(num_shuffles, int) or num_shuffles < 2):
        raise ValueError("num_shuffles must be >=2")
    if shuffle_type not in ("spatial", "orientation"):
        raise ValueError("unknown shuffle_type")
    if seed == "random":
        import secrets

        seed = secrets.randbits(32)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer or 'random'")
    if Path(output_base_name).name != output_base_name:
        raise ValueError("output_base_name must be a filename stem")
    coord1 = coord1 or ("ra" if coord_system == "radec" else "X_IMAGE")
    coord2 = coord2 or ("dec" if coord_system == "radec" else "Y_IMAGE")
    read_options = dict(
        response=response,
        input_kind=input_kind,
        ra_unit=ra_unit,
        dec_unit=dec_unit,
        id_col=id_col,
        selection=selection,
        flag_col=flag_col,
        reject_bits=reject_bits,
        z_range=z_range,
        max_shear=max_shear,
    )
    # Optional redshift mapping is applied by the reader through a separate option.
    read_options["z_col"] = z_col
    cat, flat, centre, pixel_centre, scaled, true = _prepare(
        data,
        coord_system,
        coord1,
        coord2,
        g1_col,
        g2_col,
        weight_col,
        input_hdu,
        shear_convention,
        projection,
        centre,
        pixel_scale_arcmin,
        pixel_origin,
        read_options,
    )
    footprint = flat  # Masking must not move the output grid or exterior boundary.
    holes = np.asarray(masks, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(holes)) or np.any(holes[:, 2] <= 0):
        raise ValueError("mask circles require finite coordinates and positive radii")
    keep = np.ones(flat.n, bool)
    for cx, cy, r in holes:
        keep &= np.hypot(flat.x - cx, flat.y - cy) >= r
    flat.meta.setdefault("rejected_rows", {})["spatial_mask"] = flat.row_index[
        ~keep
    ].tolist()
    flat = flat.select(keep)
    flat.meta["n_dropped"] = flat.meta["n_input"] - flat.n
    if flat.n < 3:
        raise ValueError("fewer than three sources outside masks")
    timings["ingestion_s"] = time.perf_counter() - start
    t = time.perf_counter()
    wcs, shape, points, pixels = _grid(
        flat,
        centre,
        pixel_centre,
        scaled,
        true,
        coord_system,
        projection,
        pixel_scale,
        downsample_factor,
        pixel_scale_arcmin,
        pixel_origin,
        reference_image,
        reference_hdu,
        max_pixels,
        footprint,
    )
    count, weight_sum, neff, in_image = _coverage(pixels, flat.weight, shape)
    timings["grid_s"] = time.perf_counter() - t
    # The computational domain must contain sources; the output can extend beyond it.
    field_radius = (
        float(radius)
        if radius is not None
        else float(np.max(np.hypot(footprint.x, footprint.y))) * (1 + 1e-10)
    )
    cfg = MapperConfig(
        method,
        lam,
        length,
        field_radius,
        rtol=rtol,
        residual_tolerance=residual_tolerance,
        maxiter=maxiter,
        boundary_padding=boundary_padding,
        boundary_nodes=boundary_nodes,
    )
    t = time.perf_counter()
    mapper = FEMMapper(flat, cfg)
    timings["assembly_factorization_s"] = time.perf_counter() - t
    t = time.perf_counter()
    fits = {}
    maps = {}
    for m in modes:
        a, b = (flat.g1, flat.g2) if m == "E" else (flat.g2, -flat.g1)
        fits[m] = mapper.reconstruct(a, b)
    timings["reconstruction_s"] = time.perf_counter() - t
    b_response = None
    if b_diagnostics:
        t = time.perf_counter()
        b_response = mapper.diagnose_b(e_fit=fits.get("E"), b_fit=fits["B"])
        timings["b_diagnostics_s"] = time.perf_counter() - t
    t = time.perf_counter()
    for m, fit in fits.items():
        maps[m] = _smooth(fit.evaluate(points).reshape(shape), smoothing)
    timings["evaluation_s"] = time.perf_counter() - t
    valid = np.isfinite(next(iter(maps.values())))
    outside = np.hypot(*points.T).reshape(shape) > field_radius
    valid &= ~outside
    for image in maps.values():
        image[~valid] = np.nan
    mask_region = np.zeros(shape, bool)
    for cx, cy, r in holes:
        mask_region |= (np.hypot(points[:, 0] - cx, points[:, 1] - cy) < r).reshape(
            shape
        )
    coverage = np.zeros(shape, np.uint8)
    coverage[valid & (count == 0)] = 1  # unobserved/interpolated
    coverage[valid & (count > 0)] = 2  # data present
    coverage[valid & mask_region] = 3  # explicitly masked, prior-dependent
    snr_maps = {}
    variance_maps = {}
    null_means = {}
    t = time.perf_counter()
    if create_snr:
        rng = np.random.default_rng(seed)
        mean = {m: np.zeros(shape) for m in modes}
        m2 = {m: np.zeros(shape) for m in modes}
        for i in range(num_shuffles):
            if shuffle_type == "orientation":
                phase = np.arctan2(flat.g2, flat.g1) + rng.uniform(0, 2 * np.pi, flat.n)
                amp = np.hypot(flat.g1, flat.g2)
                a, b = amp * np.cos(phase), amp * np.sin(phase)
                worker = mapper
            else:
                # Match SMPy's Python-random coordinate permutation. A new weight
                # layout is needed when heterogeneous source weights move.
                order = list(range(flat.n))
                random.Random(seed + i).shuffle(order)
                shuffled = replace(flat, x=flat.x[order], y=flat.y[order])
                worker = FEMMapper(shuffled, cfg)
                a, b = flat.g1, flat.g2
            for m in modes:
                fit = (
                    worker.reconstruct(a, b) if m == "E" else worker.reconstruct(b, -a)
                )
                image = _smooth(fit.evaluate(points).reshape(shape), smoothing)
                image[~valid] = np.nan
                image = _smooth(image, snr_smoothing)
                if not np.all(np.isfinite(image[valid])):
                    raise RuntimeError("null map has invalid pixels on science support")
                value = np.where(valid, image, 0.0)
                delta = value - mean[m]
                mean[m] += delta / (i + 1)
                m2[m] += delta * (value - mean[m])
        for m in modes:
            var = m2[m] / num_shuffles  # SMPy uses population variance (ddof=0).
            var[~valid] = np.nan
            variance_maps[m] = var
            snr_maps[m] = np.divide(
                _smooth(maps[m], snr_smoothing),
                np.sqrt(var),
                out=np.full(shape, np.nan),
                where=valid & (var > 0),
            )
            mean[m][~valid] = np.nan
            null_means[m] = mean[m]
    timings["null_reconstructions_s"] = time.perf_counter() - t
    # Pixel solid angle includes projection distortion and reference-image rotation.
    if coord_system == "radec":
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        from astropy.wcs.utils import pixel_to_skycoord

        yy, xx = np.indices(shape)
        c = pixel_to_skycoord(xx, yy, wcs)
        cx = pixel_to_skycoord(xx + 0.5, yy, wcs)
        cy = pixel_to_skycoord(xx, yy + 0.5, wcs)
        ax, ay = c.spherical_offsets_to(cx)
        bx, by = c.spherical_offsets_to(cy)
        area = 4 * np.abs(
            ax.to_value(u.arcmin) * by.to_value(u.arcmin)
            - ay.to_value(u.arcmin) * bx.to_value(u.arcmin)
        )
    else:
        area = np.full(
            shape, abs(np.linalg.det(wcs.pixel_scale_matrix)) * pixel_scale_arcmin**2
        )
    eff_density = np.divide(neff, area, out=np.zeros(shape), where=area > 0)
    metadata = dict(
        schema_version=1,
        mapper=asdict(cfg),
        tangent_point_degrees=centre,
        pixel_centre=pixel_centre,
        software=_software_versions(),
        timing_scope="output_s and total_s exclude final timing-manifest serialization",
        catalogue=flat.meta,
        n_selected=flat.n,
        n_sources_in_output=int(in_image.sum()),
        smpy_reference="26d231f5b4b22b41e76cb3bd3143d799c2e7ebfe",
        input_sha256=_file_hash(data),
        output=dict(
            shape=shape,
            smoothing=smoothing,
            reference_image=str(reference_image) if reference_image else None,
            pixel_scale=pixel_scale,
            pixel_scale_arcmin=pixel_scale_arcmin,
            downsample_factor=downsample_factor,
            masks=holes.tolist(),
        ),
        null=dict(
            enabled=create_snr,
            num_shuffles=num_shuffles,
            shuffle_type=shuffle_type,
            seed=seed,
            smoothing=snr_smoothing,
            variance_ddof=0,
            interpretation="randomized-catalogue noise, not posterior uncertainty",
        ),
        diagnostics={m: fit.diagnostics for m, fit in fits.items()},
    )
    if b_response is not None:
        metadata["b_response"] = b_response["diagnostics"]
    b_maps = {}
    if b_response is not None:
        for key in ("leakage", "residual"):
            b_maps[key] = _smooth(
                b_response[key].evaluate(points).reshape(shape), smoothing
            )
            b_maps[key][~valid] = np.nan
    metadata["output"].update(
        output_base_name=output_base_name,
        save_fits=save_fits,
        save_plots=save_plots,
        create_counts_map=create_counts_map,
        overlay_counts_map=overlay_counts_map,
        plotting=plotting,
        snr_plot_title=snr_plot_title,
        xray_image=str(xray_image) if xray_image else None,
        xray_contours=xray_contours,
    )
    result = dict(
        maps=maps,
        b_diagnostic_maps=b_maps,
        scaled_boundaries=scaled,
        true_boundaries=true,
        wcs=wcs,
        counts_map=count,
        weight_map=weight_sum,
        effective_density_map=eff_density,
        coverage=coverage,
        snr_maps=snr_maps,
        variance_maps=variance_maps,
        null_means=null_means,
        catalogue=flat,
        input_catalogue=cat,
        reconstructions=fits,
        metadata=metadata,
        timings=timings,
    )
    t = time.perf_counter()
    if save_fits or save_plots:
        from .survey_output import write_products

        write_products(
            result,
            output_dir,
            output_base_name,
            method,
            save_fits,
            save_plots,
            create_counts_map,
            overlay_counts_map,
            xray_image,
            xray_levels,
            xray_contours,
            plotting,
            overwrite,
        )
    timings["output_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - start
    if save_fits or save_plots:
        manifest = Path(output_dir) / method / f"{output_base_name}_{method}_run.json"
        manifest.write_text(
            json.dumps(dict(metadata, timings=timings), default=_json_default, indent=2)
            + "\n"
        )
    if print_timing:
        print(json.dumps(timings, indent=2))
    return result
