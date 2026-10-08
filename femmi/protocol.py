"""Frozen benchmark inputs and guards for held-out publication comparisons."""

import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def validate_config(config):
    """Validate design choices before generating catalogues or fitting anything."""
    from .smpy import METHODS

    allowed_keys = {
        "name",
        "methods",
        "n_eff",
        "radius",
        "noise_std",
        "truth",
        "truth_kw",
        "catalog_kw",
        "halos",
        "calibration_seeds",
        "evaluation_seeds",
        "evaluation_grid",
        "max_expansions",
        "ks_axes",
        "fem_axes",
        "refine",
        "ks_plus_iterations",
        "ks_plus_threshold_tau",
        "ks_plus_forward",
        "ks_plus_iteration_policy",
        "ks_plus_iteration_candidates",
        "ks_plus_iteration_check",
        "ks_plus_stability_tolerance",
        "selection_metric",
        "aperture_comparison",
        "aperture_radius_arcmin",
        "aperture_quadrature_tolerance",
        "mapper_options",
        "exploratory",
        "allow_unresolved",
        "allow_unstable_iterations",
        "calibration_truth_kw",
        "evaluation_truth_kw",
    }
    unknown = set(config) - allowed_keys
    if unknown:
        raise ValueError(f"unknown protocol options: {sorted(unknown)}")
    for key in ("calibration_seeds", "evaluation_seeds"):
        if not isinstance(config.get(key), list) or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in config[key]
        ):
            raise ValueError(f"{key} must be a list of nonnegative integers")
    for key in ("n_eff", "radius"):
        value = config.get(key, 3.0 if key == "radius" else None)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    allowed = {"p3", "argyris", "hct", "ks", *METHODS} - {"smpy_aperture"}
    methods = config.get("methods", [])
    if not methods or not set(methods) <= allowed or len(set(methods)) != len(methods):
        raise ValueError("choose distinct supported convergence methods")
    cal = config.get("calibration_seeds", [])
    ev = config.get("evaluation_seeds", [])
    if (
        not cal
        or not ev
        or set(cal) & set(ev)
        or len(set(cal)) != len(cal)
        or len(set(ev)) != len(ev)
    ):
        raise ValueError(
            "calibration and evaluation seeds must be nonempty, unique and disjoint"
        )
    if config.get("selection_metric", "source_shape_l2") != "source_shape_l2":
        raise ValueError("selection_metric must be source_shape_l2")
    grid = config.get("evaluation_grid", 0)
    if not isinstance(grid, int) or grid < 8:
        raise ValueError("evaluation_grid must be an integer >=8")
    if config.get("allow_unresolved") and not config.get("exploratory", False):
        raise ValueError("unresolved search edges require exploratory=true")
    if config.get("allow_unstable_iterations") and not config.get("exploratory", False):
        raise ValueError("unstable iterations require exploratory=true")
    if not config.get("exploratory", False) and (len(cal) < 5 or len(ev) < 20):
        raise ValueError(
            "publication protocol requires >=5 calibration and >=20 evaluation seeds"
        )
    if "smpy_ks_plus" in methods:
        policy = config.get("ks_plus_iteration_policy", "stable")
        if policy not in ("stable", "calibrated_budget"):
            raise ValueError("KS+ iteration policy must be stable or calibrated_budget")
        if config.get("ks_plus_forward", "corrected") not in ("corrected", "upstream"):
            raise ValueError("KS+ forward must be corrected or upstream")
        tau = config.get("ks_plus_threshold_tau")
        if tau is None or not np.isfinite(tau) or tau <= 0:
            raise ValueError("set a positive ks_plus_threshold_tau independent of iteration budget")
        check = config.get("ks_plus_iteration_check", [])
        def valid_counts(values):
            return (isinstance(values, list) and len(values) == len(set(values))
                    and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in values))
        if not valid_counts(check):
            raise ValueError("KS+ checks require distinct positive integer counts")
        candidates = (config.get("ks_plus_iteration_candidates", []) if policy == "calibrated_budget"
                      else [config.get("ks_plus_iterations", 100)])
        if (not valid_counts(candidates) or not candidates
                or (policy == "calibrated_budget" and len(candidates) < 2)
                or not set(candidates) <= set(check)
                or len([n for n in check if n > max(candidates)]) < 2):
            raise ValueError("KS+ checks must include every candidate and two larger reference budgets")
    if (
        config.get("aperture_radius_arcmin", config.get("radius", 3.0) / 4)
        / (2 * config.get("radius", 3.0) / grid)
        < 2
    ):
        raise ValueError("aperture radius must span at least two evaluation pixels")
    tol = config.get("aperture_quadrature_tolerance", 0.05)
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("aperture_quadrature_tolerance must be in (0,1)")
    return config


def fits_roundtrip(c, path):
    """Persist one sky catalogue and ingest it through the production survey path.

    Truth is a separate column used only for scoring. Both estimator families
    subsequently receive the same returned observations and weights.
    """
    from astropy.table import Table
    from .survey import _prepare
    from .calibration import Catalogue

    dec = -30 + c.y / 60
    ra = 35 + c.x / (60 * np.cos(np.deg2rad(dec)))
    tab = Table(
        dict(
            ra=ra,
            dec=dec,
            g1=c.g1,
            g2=-c.g2,
            weight=c.weight,
            SOURCE_ID=np.arange(len(c.x)),
            KAPPA_TRUE=c.truth,
        )
    )
    tab["ra"].unit = tab["dec"].unit = "deg"
    tab.write(path, overwrite=False)
    _, flat, _, _, _, _ = _prepare(
        path,
        "radec",
        "ra",
        "dec",
        "g1",
        "g2",
        "weight",
        1,
        "smpy",
        "smpy",
        (35.0, -30.0),
        None,
        1,
        dict(id_col="SOURCE_ID"),
    )
    if flat.n != len(c.x) or not np.array_equal(flat.row_index, np.arange(flat.n)):
        raise ValueError("survey ingestion changed benchmark source selection")
    return Catalogue(
        flat.x,
        flat.y,
        flat.g1,
        flat.g2,
        flat.weight,
        c.truth,
        c.radius,
        c.seed,
        c.nominal_density,
    )


def iteration_stability(catalogues, parameters, counts, selected, tolerance=0.05,
                        *, threshold_tau, ks_plus_forward="corrected"):
    """Fixed-schedule, unsmoothed E AND B stability on calibration catalogues.

    Require the selected map and every longer checkpoint to agree with the
    longest run. Two longer runs avoid declaring a single accidental crossing
    a plateau. This is an empirical finite-budget check, not a convergence proof.
    """
    from .smpy import reconstruct
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("iteration tolerance must be in (0,1)")
    if not np.isfinite(threshold_tau) or threshold_tau <= 0:
        raise ValueError("threshold_tau must be positive")
    if selected not in counts or len(set(n for n in counts if n > selected)) < 2:
        raise ValueError("include the selected count and two larger references")
    rows = []
    for c in catalogues:
        grids = {}
        for count in sorted(set(counts)):
            *_, e, b = reconstruct(c, "smpy_ks_plus", parameters[0], 0.,
                iterations=count, threshold_tau=threshold_tau, ks_plus_forward=ks_plus_forward)
            grids[count] = (e, b)
        reference = grids[max(counts)]
        axis = (np.arange(reference[0].shape[0]) + 0.5) * 2 / reference[0].shape[0] - 1
        x, y = np.meshgrid(axis, axis)
        field = x*x+y*y < 1
        ref = [v[field]-v[field].mean() for v in reference]
        # Common E+B scale remains meaningful for a near-zero B component.
        den = max(np.linalg.norm(np.concatenate(ref)), np.finfo(float).tiny)
        for count, pair in grids.items():
            change = [float(np.linalg.norm(v[field]-v[field].mean()-r)/den)
                      for v, r in zip(pair, ref)]
            rows.append(dict(seed=c.seed, iterations=count,
                relative_change=float(np.hypot(*change)),
                e_change_over_eb_scale=change[0], b_change_over_eb_scale=change[1],
                field_mean_change=[float(v[field].mean()-r[field].mean())
                                   for v,r in zip(pair,reference)],
                full_grid_relative_change=float(np.linalg.norm(np.stack(pair)-np.stack(reference)) /
                    max(np.linalg.norm(np.stack(reference)), np.finfo(float).tiny))))
    checked = [r["relative_change"] for r in rows if selected <= r["iterations"] < max(counts)]
    return dict(reference_iterations=max(counts), selected_iterations=selected,
        threshold_tau=threshold_tau, forward_transform=ks_plus_forward,
        smoothing_for_check=0., tolerance=tolerance,
        accepted=bool(checked and max(checked) <= tolerance), rows=rows,
        interpretation="fixed-threshold-schedule unsmoothed E+B field plateau; not a solver residual or accuracy guarantee")


def run_suite(configs, output):
    """Write a frozen protocol, then execute fresh scenario directories."""
    from .calibration import write_json, calibrate_and_evaluate
    from .comparison import summarize
    from .smpy import verify_installation

    configs = [validate_config(dict(c)) for c in configs]
    names = [c["name"] for c in configs]
    if len(set(names)) != len(names) or any(
        Path(n).name != n or n in ("", ".", "..") for n in names
    ):
        raise ValueError("scenario names must be unique filename components")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "benchmark output must be empty; choose a fresh directory"
        )
    upstream = (
        verify_installation()
        if any(
            c.get("aperture_comparison")
            or any(m.startswith("smpy_") for m in c["methods"])
            for c in configs
        )
        else None
    )
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(configs, sort_keys=True).encode()
    source = source_provenance()
    write_json(
        root / "protocol.json",
        dict(
            schema_version=1,
            configs=configs,
            sha256=hashlib.sha256(payload).hexdigest(),
            source=source,
            smpy=upstream,
            selection_metric="source-position DC-removed relative L2 on calibration catalogues",
            evaluation="common field grid; one global offset; paired differences and failures",
            catalogue_transport="one FITS catalogue per realization through production ingestion",
        ),
    )
    for config in configs:
        config = dict(config, catalogue_transport="fits")
        calibrate_and_evaluate(config, root / config["name"])
    return summarize(root)


def source_provenance():
    """Fingerprint current source inputs, including uncommitted new modules."""
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout

    try:
        revision = git("rev-parse", "HEAD").strip()
        paths = set(
            git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        )
        files = {}
        for name in sorted(paths):
            path = root / name
            if path.is_file() and path.suffix in (
                ".py",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".txt",
            ):
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(
            revision=revision,
            dirty=bool(git("status", "--porcelain").strip()),
            source_sha256=hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()
            ).hexdigest(),
            files=files,
        )
    except (OSError, subprocess.CalledProcessError):
        return dict(
            revision=None,
            dirty=None,
            source_sha256=None,
            note="Git checkout unavailable; archive the installed package and environment separately",
        )
