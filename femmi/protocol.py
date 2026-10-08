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
        check = config.get("ks_plus_iteration_check")
        chosen = config.get("ks_plus_iterations", 100)
        if (
            not check
            or any(
                not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in check
            )
            or chosen not in check
            or max(check) <= chosen
        ):
            raise ValueError(
                "KS+ iteration check must include the selected count and a larger reference"
            )
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


def iteration_stability(catalogues, parameters, counts, selected, tolerance=0.05):
    """Compare KS+ iteration counts on calibration data at fixed chosen grid/prior."""
    from .smpy import reconstruct

    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("iteration tolerance must be in (0,1)")
    rows = []
    for c in catalogues:
        grids = {}
        for count in sorted(set(counts)):
            *_, grid, _ = reconstruct(c, "smpy_ks_plus", *parameters, iterations=count)
            grids[count] = grid
        reference = grids[max(counts)]
        axis = (np.arange(reference.shape[0]) + 0.5) * 2 / reference.shape[0] - 1
        x, y = np.meshgrid(axis, axis)
        field = x * x + y * y < 1
        ref = reference[field] - reference[field].mean()
        den = max(np.linalg.norm(ref), 1e-300)
        for count, grid in grids.items():
            current = grid[field] - grid[field].mean()
            rows.append(
                dict(
                    seed=c.seed,
                    iterations=count,
                    relative_change=float(np.linalg.norm(current - ref) / den),
                    field_mean_change=float(
                        grid[field].mean() - reference[field].mean()
                    ),
                    full_grid_relative_change=float(
                        np.linalg.norm(grid - reference)
                        / max(np.linalg.norm(reference), 1e-300)
                    ),
                )
            )
    chosen = [r["relative_change"] for r in rows if r["iterations"] == selected]
    return dict(
        reference_iterations=max(counts),
        selected_iterations=selected,
        tolerance=tolerance,
        accepted=bool(max(chosen) <= tolerance),
        rows=rows,
        interpretation="DC-removed physical-field map stability under iteration budget changes; upstream threshold schedule also changes; not a solver residual",
    )


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
