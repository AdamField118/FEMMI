"""
femmi/config.py
One layered YAML config that specifies the WHOLE pipeline -- the forward
operator, the data, the inverse (MAP or posterior sampling), the prior, and the
outputs -- so a run is `femmi run --config my_run.yaml` instead of a wall of
command-line flags.

    from femmi.config import load_config
    cfg = load_config("configs/default.yaml")   # user file merged over defaults
    cfg.get("inverse.method")                    # dot-notation access

A user YAML is deep-merged on top of DEFAULTS, so a config only lists what it
changes. `load_config(None)` returns the defaults. See configs/default.yaml for
the annotated, copyable version of this schema.
"""

from __future__ import annotations
import copy
from typing import Any, Optional

try:
    import yaml
except ImportError:      # pragma: no cover
    yaml = None


# Built-in defaults. Every knob the pipeline reads lives here, grouped by stage.
DEFAULTS: dict = {

    # --- forward operator: the FEM-BEM mesh and coupling ---------------------
    "forward": {
        "geometry": "square",     # square | circular | catalog
        "nx": 20,                 # square: cells per side
        "half_width": 2.5,        # square: domain is [-half_width, half_width]^2
        "radius": 2.5,            # circular/catalog: boundary radius (catalog: auto if null)
        "n_boundary": 96,         # boundary elements
        "coupling": "steinbach",  # BEM coupling (steinbach is the correct, default one)
        "sigma_scale": 1.0,       # steinbach length-scale multiplier
    },

    # --- data: where the shear catalog comes from ---------------------------
    "data": {
        "source": "synthetic",    # synthetic | catalog_fits | frontier
        # synthetic (square geometry):
        "kappa_field": "gaussian", # gaussian (analytic smooth blob) | lognormal
        #   (non-Gaussian field matching the neural prior -- the fair neural test)
        "lognormal": {"kappa_std": 0.35, "slope": 2.5, "sigma_g": 0.9, "n_pix": 128},
        "n_gal": 1500,
        "kappa_sigma": 0.5,
        "shape_noise": 0.06,
        "seed": 0,
        # catalog_fits:
        "fits": None,             # path to a shear-catalog FITS
        "hdu": 1,
        "flip_g2": False,
        # frontier (CATS lens-model maps):
        "frontier_dir": None,
        "frontier_source": "deflect",   # psi | deflect | kappa
        "kappa_max": 0.8,
        "rmax": None,
        "reduced_shear": False,
        # optional circular missing-data mask (any source), null = no mask:
        "mask_radius": None,
        "mask_center": [0.0, 0.0],
    },

    # --- inverse: MAP point estimate or full posterior sampling -------------
    "inverse": {
        "method": "map",          # map | sample
        "lam": None,              # regularisation; null -> select via Morozov (MAP only)
        "wiener_length": 0.5,     # Matern-1/2 prior length (Wiener prior)
        "morozov": True,          # auto-select lambda (MAP + Wiener prior)
        "noise_source": "bmode",  # mad | bmode  (Morozov noise estimate)
        "maxiter": 400,
    },

    # --- prior: the regulariser ---------------------------------------------
    "prior": {
        "kind": "wiener",         # wiener | tv | sparse | maxent | neural
        "tv":     {"eps": 1.0e-3},
        "sparse": {"transform": "field", "eps": 1.0e-3},
        "maxent": {"model": 1.0e-2},
        "neural": {"n_pix": 32, "base": 16, "ckpt": None,    # ckpt null -> cached model
                   "hybrid": False,                          # hybrid: learn only the
        #          non-Gaussian residual on an analytic Gaussian prior (Remy 2020 eq. 6)
                   "boundary_taper": 0.08,                   # taper the score to 0 in a
        #          boundary band (fraction of domain) to kill mesh<->grid edge artifacts
                   "train_data": "synthetic",                # synthetic | massivenus
                   "data_dir": None,                         # MassiveNuS map directory
        #          (train_data=massivenus -> the exact simulation suite from the paper)
                   "map_glob": None,                         # filename filter e.g. '*z1.00*'
                   "pool_maps": 512},                        # #maps held in RAM (bounded)
    },

    # --- sampler: only used when inverse.method == sample -------------------
    "sampler": {
        "method": "auto",         # auto | rto | annealed_hmc | langevin
        "n_samples": 300,         # rto: number of exact posterior draws
        "n_levels": 10, "steps_per_level": 12, "n_chains": 40,
        "n_leapfrog": 5, "keep_final": 4,
        "sigma_max": 1.0, "sigma_min": 0.02, "n_delta_logp": 4,
        "seed": 1,
    },

    # --- output --------------------------------------------------------------
    "output": {
        "dir": "runs",
        "name": "run",            # basename for saved arrays / figures
        "save_kappa": True,       # write kappa (and std, if sampling) as .npz
        "save_samples": True,     # also store individual posterior draws in the .npz
        "save_figure": True,      # write a summary figure
    },
}


class Config:
    """A merged config with dot-notation access (`cfg.get('inverse.method')`)."""

    def __init__(self, data: dict):
        self._d = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node = self._d
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self._d
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    def section(self, name: str) -> dict:
        return dict(self._d.get(name, {}))

    def as_dict(self) -> dict:
        return copy.deepcopy(self._d)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> Config:
    """Load a config: the built-in DEFAULTS with the YAML at `path` merged on top."""
    data = copy.deepcopy(DEFAULTS)
    if path:
        if yaml is None:
            raise ImportError("pyyaml is required to read config files (pip install pyyaml)")
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        data = _deep_merge(data, user)
    return Config(data)
