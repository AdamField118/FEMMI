"""WCS-registered FITS and plotting products for the survey API.

Names and defaults follow SMPy; all images share the reconstruction WCS.
"""

from pathlib import Path
import numpy as np


def validate_plot_options(plotting, contours):
    """Reject unsupported nested options before any expensive solve or write."""
    p = plotting or {}
    allowed = {
        "figsize",
        "fontsize",
        "cmap",
        "xlabel",
        "ylabel",
        "plot_title",
        "gridlines",
        "vmin",
        "vmax",
        "scaling",
    }
    if set(p) - allowed:
        raise ValueError(f"unsupported plotting settings: {sorted(set(p) - allowed)}")
    scaling = p.get("scaling") or {}
    if set(scaling) - {"type", "gamma", "percentile", "convergence", "snr"}:
        raise ValueError("unsupported plot scaling option")
    if scaling.get("type", "linear") not in ("linear", "power", "symlog"):
        raise ValueError("plot scaling must be linear, power or symlog")
    for mode in ("convergence", "snr"):
        if set(scaling.get(mode) or {}) - {"linthresh", "linscale"}:
            raise ValueError("unsupported symlog scaling option")
    ctr = contours or {}
    if set(ctr) - {
        "ctr_file",
        "show_on_convergence",
        "show_on_snr",
        "color",
        "linewidth",
        "alpha",
    }:
        raise ValueError("unsupported xray_contours option")


def _header(result, unit, product):
    header = result["wcs"].to_header(relax=True)
    meta = result["metadata"]
    cfg = meta["mapper"]
    header["BUNIT"] = unit
    header["PRODUCT"] = product
    header["METHOD"] = cfg["method"]
    header["REG_LAM"] = cfg["lam"]
    header["REG_LEN"] = (cfg["length"], "Prior correlation length in arcmin")
    header["SHEARCON"] = meta["catalogue"]["shear_convention"]
    header["PROJMOD"] = meta["catalogue"]["projection"]
    header["SRCPLANE"] = "effective"
    header["REDSHIFT"] = (False, "Per-source redshifts used in forward operator")
    header["NSOURCE"] = meta["n_selected"]
    if product.startswith("JOINT_"):
        joint = meta["joint_eb"]
        header["EBMETHOD"] = "joint MAP"
        header["REG_LAM"] = joint["lam_e"] if product == "JOINT_E" else joint["lam_b"]
        header["REG_LEN"] = joint["length_e"] if product == "JOINT_E" else joint["length_b"]
        header["E_LAMBDA"] = joint["lam_e"]
        header["B_LAMBDA"] = joint["lam_b"]
        header["E_LENGTH"] = joint["length_e"]
        header["B_LENGTH"] = joint["length_b"]
        header["HISTORY"] = "Joint posterior mode, conditional on both priors; not pure E/B."
        header["HISTORY"] = "No uncertainty for joint products; rotated-fit SNR is a different estimator."
    elif product.endswith("_B") or product.startswith("B_"):
        header["BINTERP"] = "rotated shear"
        header["HISTORY"] = (
            "B products are finite-field responses, not an orthogonal E/B decomposition."
        )
    header["INPSHA"] = meta["input_sha256"]
    header["SMPYREF"] = meta["smpy_reference"]
    header["HISTORY"] = (
        "Linear shear reconstruction; response applied only when requested."
    )
    header["HISTORY"] = (
        "See companion run JSON for selection, calibration, solver and timing metadata."
    )
    header["HISTORY"] = (
        "COVERAGE: 0 outside support; 1 unobserved; 2 observed; 3 explicitly masked."
    )
    if product.startswith(("SNR", "NULL")):
        header["NSHUFFLE"] = meta["null"]["num_shuffles"]
        header["RNGSEED"] = meta["null"]["seed"]
        header["SHUFFLE"] = meta["null"]["shuffle_type"]
        header["VARDDOF"] = 0
        header["HISTORY"] = (
            "Randomized-catalogue noise at fixed prior, not posterior uncertainty."
        )
    return header


def _source_table(result):
    from astropy.table import Table

    c = result["catalogue"]
    original = result["input_catalogue"]
    lookup = {int(row): i for i, row in enumerate(original.row_index)}
    index = np.array([lookup[int(row)] for row in c.row_index])
    tab = Table(
        dict(ROW_INDEX=c.row_index, X=c.x, Y=c.y, G1=c.g1, G2=c.g2, WEIGHT=c.weight)
    )
    tab["X"].unit = tab["Y"].unit = "arcmin"
    if c.meta["coord_system"] == "radec":
        tab["RA"], tab["DEC"] = original.ra[index], original.dec[index]
        tab["RA"].unit = tab["DEC"].unit = "deg"
    else:
        tab["X_IMAGE"], tab["Y_IMAGE"] = original.x[index], original.y[index]
        tab["X_IMAGE"].unit = tab["Y_IMAGE"].unit = "pix"
    joint = result.get("joint_reconstruction")
    if joint is not None:
        tab["JOINT_E"] = joint.kappa_e
        tab["JOINT_B"] = joint.kappa_b
        tab["JOINT_PRED_G1"] = joint.predicted_g1
        tab["JOINT_PRED_G2"] = joint.predicted_g2
        tab["JOINT_RES_G1"] = c.g1-joint.predicted_g1
        tab["JOINT_RES_G2"] = c.g2-joint.predicted_g2
    if c.object_id is not None:
        tab["OBJECT_ID"] = c.object_id
    if c.z is not None:
        tab["REDSHIFT"] = c.z
    for mode, fit in result["reconstructions"].items():
        tab[f"{mode}_PRED1"], tab[f"{mode}_PRED2"] = fit.predicted_g1, fit.predicted_g2
        tab[f"{mode}_RES1"] = fit.observed_g1 - fit.predicted_g1
        tab[f"{mode}_RES2"] = fit.observed_g2 - fit.predicted_g2
        tab[f"{mode}_KAPPA"] = fit.kappa
    tab.meta["COMMENT"] = (
        "Zero-based input row IDs. G1/G2 in computational frame; B residuals use (G2,-G1)."
    )
    return tab


def read_ds9_contours(path):
    """Read DS9 exported .ctr polylines (SMPy's supported contour-file format).

    World contours are transformed using WCS, not bounding-box interpolation.
    Image/physical contour coordinates are retained in their native frame.
    """
    lines, points, frame = [], [], "fk5"
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() in ("fk5", "wcs", "image", "physical"):
            frame = line.lower()
        elif line.lower() == "line":
            if points:
                lines.append(np.asarray(points))
                points = []
        else:
            try:
                pair = [float(v) for v in line.replace(",", " ").split()[:2]]
            except ValueError:
                continue
            if len(pair) == 2:
                points.append(pair)
    if points:
        lines.append(np.asarray(points))
    return lines, frame


def _plot(
    result,
    image,
    path,
    label,
    overlay_counts,
    xray_image,
    xray_levels,
    xray_contours,
    plotting,
    is_snr=False,
):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import PowerNorm, SymLogNorm
    from astropy.io import fits
    from astropy.wcs import WCS

    p = dict(plotting or {})
    if is_snr:
        p["plot_title"] = result["metadata"]["output"]["snr_plot_title"]
    if label == "COUNTS":
        p["plot_title"] = "Counts Map"
        p["scaling"] = {"type": "linear"}
        overlay_counts = True
        xray_image = None
        xray_contours = None
    fig = Figure(figsize=p.get("figsize", (12, 8)))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection=result["wcs"])
    limits = {key: p[key] for key in ("vmin", "vmax") if p.get(key) is not None}
    scaling = p.get("scaling") or {}
    if scaling.get("percentile") is not None:
        vals = image[np.isfinite(image)]
        if vals.size:
            lo, hi = np.percentile(vals, scaling["percentile"])
            limits.setdefault("vmin", lo)
            limits.setdefault("vmax", hi)
    kind = scaling.get("type", "linear")
    if kind == "power":
        limits = dict(norm=PowerNorm(scaling.get("gamma", 2), **limits))
    elif kind == "symlog":
        opts = scaling.get("snr" if is_snr else "convergence") or {}
        limits = dict(
            norm=SymLogNorm(
                opts.get("linthresh", 5 if is_snr else 0.1),
                linscale=opts.get("linscale", 0.5 if is_snr else 1),
                **limits,
            )
        )
    elif kind != "linear":
        raise ValueError("plot scaling must be linear, power or symlog")
    artist = ax.imshow(image, origin="lower", cmap=p.get("cmap", "viridis"), **limits)
    fig.colorbar(artist, ax=ax, label=label)
    ax.set_title(p.get("plot_title", label), fontsize=p.get("fontsize", 15))
    for axis, default in (
        (
            "x",
            result["wcs"].wcs.lngtyp
            if result["wcs"].has_celestial
            else "X (catalogue pixels)",
        ),
        (
            "y",
            result["wcs"].wcs.lattyp
            if result["wcs"].has_celestial
            else "Y (catalogue pixels)",
        ),
    ):
        text = p.get(axis + "label", "auto")
        getattr(ax, "set_" + axis + "label")(default if text == "auto" else text or "")
    ax.grid(bool(p.get("gridlines", True)), alpha=0.25)
    if overlay_counts:
        for (y, x), count in np.ndenumerate(result["counts_map"]):
            if count:
                ax.text(x, y, str(count), ha="center", va="center", fontsize=6)
    if xray_image:
        with fits.open(xray_image) as hdul:
            hdu = hdul[0]
            wcs = WCS(hdu.header)
            if not wcs.has_celestial or not result["wcs"].has_celestial:
                raise ValueError(
                    "X-ray FITS overlay requires celestial WCS on both images"
                )
            ax.contour(
                hdu.data,
                levels=xray_levels,
                transform=ax.get_transform(wcs),
                colors="cyan",
                linewidths=0.8,
            )
    ctr = xray_contours or {}
    if ctr.get("ctr_file") and ctr.get(
        "show_on_snr" if is_snr else "show_on_convergence", False
    ):
        segments, frame = read_ds9_contours(ctr["ctr_file"])
        if frame in ("fk5", "wcs"):
            if not result["wcs"].has_celestial:
                raise ValueError("sky contours require a celestial output WCS")
            transform = ax.get_transform("fk5" if frame == "fk5" else "world")
            offset = 0
        else:
            if result["wcs"].has_celestial:
                raise ValueError(
                    "image/physical contours need a pixel catalogue or a celestial contour file"
                )
            transform = ax.get_transform("world")
            offset = 0
        for line in segments:
            ax.plot(
                line[:, 0] - offset,
                line[:, 1] - offset,
                transform=transform,
                color=ctr.get("color", "cyan"),
                linewidth=ctr.get("linewidth", 0.8),
                alpha=ctr.get("alpha", 0.7),
            )
    # SMPy's astronomical display convention: RA increases to the left.
    if (
        result["wcs"].has_celestial
        and result["wcs"].wcs.lngtyp == "RA"
        and result["wcs"].pixel_scale_matrix[0, 0] > 0
    ):
        ax.invert_xaxis()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clear()


def write_products(
    result,
    output_dir,
    base,
    method,
    save_fits,
    save_plots,
    create_counts,
    overlay_counts,
    xray_image,
    xray_levels,
    xray_contours,
    plotting,
    overwrite,
):
    from astropy.io import fits

    directory = Path(output_dir) / method
    prefix = f"{base}_{method}"
    products = {}
    for m, image in result["maps"].items():
        products[f"{m.lower()}_mode"] = (image, "1", f"KAPPA_{m}")
    for m, image in result.get("joint_maps", {}).items():
        products[f"joint_{m.lower()}_mode"] = (image, "1", f"JOINT_{m}")
    for key, image in result.get("b_diagnostic_maps", {}).items():
        products[f"b_{key}"] = (image, "1", f"B_{key.upper()}")
    for m, image in result["snr_maps"].items():
        products[f"snr_{m.lower()}_mode"] = (image, "1", f"SNR_{m}")
        products[f"null_variance_{m.lower()}_mode"] = (
            result["variance_maps"][m],
            "1",
            f"NULLVAR_{m}",
        )
        products[f"null_mean_{m.lower()}_mode"] = (
            result["null_means"][m],
            "1",
            f"NULLMEAN_{m}",
        )
    products.update(
        weight_sum=(result["weight_map"], "1", "WEIGHT_SUM"),
        effective_density=(
            result["effective_density_map"],
            "arcmin-2",
            "EFFECTIVE_DENSITY",
        ),
        coverage=(result["coverage"], "1", "COVERAGE"),
    )
    if create_counts:
        products["counts"] = (result["counts_map"], "count", "COUNTS")
    plot_names = [
        k for k in products if k.endswith("_mode") and not k.startswith("null_")
    ]
    if create_counts:
        plot_names.append("counts")
    paths = [directory / f"{prefix}_run.json"]
    if save_fits:
        paths += [directory / f"{prefix}_{k}.fits" for k in products]
        paths.append(directory / f"{prefix}_sources.fits")
    if save_plots:
        paths += [directory / f"{prefix}_{k}.png" for k in plot_names]
    if not overwrite:
        for path in paths:
            if path.exists():
                raise FileExistsError(
                    f"output exists: {path}; choose another name or set overwrite=True"
                )
    directory.mkdir(parents=True, exist_ok=True)
    if save_fits:
        for name, (array, unit, product) in products.items():
            header = _header(result, unit, product)
            extensions = [fits.PrimaryHDU(array, header)]
            if name != "coverage":
                extensions.append(
                    fits.ImageHDU(
                        result["coverage"],
                        result["wcs"].to_header(relax=True),
                        name="COVERAGE",
                    )
                )
            fits.HDUList(extensions).writeto(
                directory / f"{prefix}_{name}.fits", overwrite=overwrite, checksum=True
            )
        tab = fits.BinTableHDU(_source_table(result), name="SOURCES")
        tab.header["INPSHA"] = result["metadata"]["input_sha256"]
        tab.header["INPHDU"] = str(result["metadata"]["catalogue"]["hdu"])
        tab.header["GFRAME"] = "computational"
        tab.header["PIXORIG"] = result["metadata"]["catalogue"]["pixel_origin"]
        fits.HDUList([fits.PrimaryHDU(), tab]).writeto(
            directory / f"{prefix}_sources.fits", overwrite=overwrite, checksum=True
        )
    if save_plots:
        for name in plot_names:
            _plot(
                result,
                products[name][0],
                directory / f"{prefix}_{name}.png",
                products[name][2],
                overlay_counts and name != "counts",
                xray_image,
                xray_levels,
                xray_contours,
                plotting,
                is_snr=name.startswith("snr_"),
            )
    result["output_files"] = [str(path) for path in paths]
