"""Translate SMPy's nested YAML vocabulary into the survey API."""

from copy import deepcopy
from pathlib import Path
import inspect
import yaml


def _smoothing(value):
    if not isinstance(value, dict):
        return value
    if set(value) - {"type", "sigma"}:
        raise ValueError("smoothing accepts only type and sigma")
    if value.get("type") is None:
        return None
    if value["type"] != "gaussian":
        raise ValueError("only Gaussian output smoothing is supported")
    return value["sigma"]


def load_survey_config(path):
    """Read a strict SMPy-shaped config; unknown settings never disappear silently.

    FEM-specific settings live under methods.p3/argyris/hct and catalogue
    calibration/selection settings under catalogue. Paths follow SMPy's behavior:
    relative to the current working directory.
    """
    from .survey import map_mass

    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping")
    cfg = deepcopy(cfg)
    if set(cfg) - {"general", "methods", "snr", "plotting", "catalogue"}:
        raise ValueError(
            f"unknown configuration sections: {sorted(set(cfg) - {'general', 'methods', 'snr', 'plotting', 'catalogue'})}"
        )
    general = cfg.get("general", {})
    coord = general.pop("coordinate_system", "radec")
    if coord not in ("radec", "pixel"):
        raise ValueError("coordinate_system must be radec or pixel")
    geometry = general.pop(coord, {})
    general.pop("pixel" if coord == "radec" else "radec", None)
    if coord == "radec":
        geometry["pixel_scale"] = geometry.pop("resolution", 0.4)
    else:
        geometry.setdefault("downsample_factor", 1)
        # Catalogue-coordinate axes remain truthful after downsampling.
        if geometry.pop("pixel_axis_reference", "catalog") != "catalog":
            raise ValueError(
                "pixel_axis_reference supports catalog only; output WCS retains catalogue coordinates"
            )
    options = dict(coord_system=coord, **geometry)
    rename = {"input_path": "data", "output_directory": "output_dir"}
    options.update({rename.get(k, k): v for k, v in general.items()})
    method = options.setdefault("method", "p3")
    methods = cfg.get("methods", {})
    if method not in methods:
        raise ValueError(f"methods.{method} must specify lam and length")
    options.update(methods[method])
    options.update(cfg.get("catalogue", {}))
    snr = cfg.get("snr", {})
    if "plot_title" in snr:
        snr["snr_plot_title"] = snr.pop("plot_title")
    if "smoothing" in snr:
        snr["snr_smoothing"] = _smoothing(snr.pop("smoothing"))
    options.update(snr)
    if "smoothing" in options:
        options["smoothing"] = _smoothing(options["smoothing"])
    plotting = cfg.get("plotting", {})
    if "xray_contours" in plotting:
        options["xray_contours"] = plotting.pop("xray_contours")
    if plotting:
        options["plotting"] = plotting
    allowed = set(inspect.signature(map_mass).parameters)
    if set(options) - allowed:
        raise ValueError(f"unknown survey options: {sorted(set(options) - allowed)}")
    return options
