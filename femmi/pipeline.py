"""
femmi/pipeline.py
The general run pipeline: given one config (femmi.config), build the forward
operator, get the data, build the prior, run the inverse (MAP or posterior
sampling), and save the outputs. This is what `femmi run --config ...` calls, and
what specific scripts (e.g. the paper-artifact recreation) reuse so they share
exactly one description of a run.

Everything is driven by the config sections: forward, data, inverse, prior,
sampler, output (see configs/default.yaml).
"""

from __future__ import annotations
import os
import tempfile
import numpy as np

from .operators import build_operators, build_operators_catalog, dirichlet_from_operators
from .forward   import DifferentiableForward
from .inverse   import MAPReconstructor
from .catalog   import (reconstruct_catalog, analytic_gaussian_catalog,
                        analytic_gaussian_shear, load_frontier_model, field_to_catalog)


# --------------------------------------------------------------------------- #
# data + operator assembly (config -> ops and node-placed shear)
# --------------------------------------------------------------------------- #
def build_forward_and_data(cfg):
    """From the `forward` and `data` config, return everything the inverse needs:

    dict(ops, g1n, g2n, weight, truth_nodes, catalog_mesh, nodes, noise_std,
         to_galaxies) where g1n/g2n/weight are placed on the mesh nodes, and
    to_galaxies(kappa_nodes) maps a node field back to source order (identity for
    the structured-grid geometry).
    """
    geom = cfg.get("forward.geometry")
    src  = cfg.get("data.source")
    sn   = float(cfg.get("data.shape_noise"))

    if geom == "square":
        # structured grid; synthetic Gaussian truth placed at the nodes.
        if src != "synthetic":
            raise ValueError("forward.geometry='square' only supports data.source='synthetic'; "
                             "use geometry='catalog' for FITS/frontier data.")
        hw = cfg.get("forward.half_width"); nx = cfg.get("forward.nx")
        ops = build_operators(nx, nx, -hw, hw, -hw, hw, verbose=False,
                              coupling=cfg.get("forward.coupling"),
                              sigma_scale=cfg.get("forward.sigma_scale"))
        nodes = np.array(ops.mesh.nodes)
        kt, g1t, g2t = analytic_gaussian_shear(nodes, sigma=cfg.get("data.kappa_sigma"))
        kt = np.asarray(kt); g1t = np.asarray(g1t); g2t = np.asarray(g2t)
        rng = np.random.default_rng(cfg.get("data.seed"))
        g1n = g1t + rng.normal(0, sn, len(g1t)); g2n = g2t + rng.normal(0, sn, len(g2t))
        weight = np.ones(len(nodes))
        _apply_mask(cfg, nodes, g1n, g2n, weight)
        return dict(ops=ops, g1n=g1n, g2n=g2n, weight=weight, truth_nodes=kt,
                    catalog_mesh=None, nodes=nodes, noise_std=sn,
                    to_galaxies=lambda kn: kn)

    # --- catalog-native: get (x, y, g1, g2) then build the catalog mesh -----
    x, y, g1, g2, truth_gal, center = _get_catalog(cfg)
    ops, cm = build_operators_catalog(
        x, y, center=center, radius=cfg.get("forward.radius"),
        n_boundary=cfg.get("forward.n_boundary"), verbose=False,
        coupling=cfg.get("forward.coupling"), sigma_scale=cfg.get("forward.sigma_scale"))
    n = ops.n_nodes; gn, si = cm.galaxy_nodes, cm.source_index
    g1n = np.zeros(n); g1n[gn] = np.asarray(g1)[si]
    g2n = np.zeros(n); g2n[gn] = np.asarray(g2)[si]
    weight = np.zeros(n); weight[gn] = 1.0
    truth_nodes = np.full(n, np.nan)
    if truth_gal is not None:
        truth_nodes[gn] = np.asarray(truth_gal)[si]

    def to_galaxies(kappa_nodes):
        out = np.full(len(x), np.nan); out[si] = np.asarray(kappa_nodes)[gn]; return out

    return dict(ops=ops, g1n=g1n, g2n=g2n, weight=weight, truth_nodes=truth_nodes,
                catalog_mesh=cm, nodes=np.array(ops.mesh.nodes), noise_std=sn,
                to_galaxies=to_galaxies)


def _get_catalog(cfg):
    """Return (x, y, g1, g2, kappa_true_or_None, center) for the catalog path."""
    src = cfg.get("data.source")
    if src == "synthetic":
        cat = analytic_gaussian_catalog(n_gal=cfg.get("data.n_gal"),
                                        sigma=cfg.get("data.kappa_sigma"),
                                        shape_noise=cfg.get("data.shape_noise"),
                                        seed=cfg.get("data.seed"))
        return (cat["x"], cat["y"], cat["g1"], cat["g2"], cat["kappa_true"],
                cat.get("center", (0.0, 0.0)))
    if src == "catalog_fits":
        from .io import read_fits_catalog
        path = cfg.get("data.fits")
        if not path:
            raise ValueError("data.source='catalog_fits' requires data.fits (a FITS path)")
        flat = read_fits_catalog(path, hdu=cfg.get("data.hdu")).to_tangent_plane(
            units="arcmin", flip_g2=cfg.get("data.flip_g2"))
        return (flat.x, flat.y, flat.g1, flat.g2, None, (0.0, 0.0))
    if src == "frontier":
        d = cfg.get("data.frontier_dir")
        if not d:
            raise ValueError("data.source='frontier' requires data.frontier_dir")
        field = load_frontier_model(d, source=cfg.get("data.frontier_source"))
        cat = field_to_catalog(field, n_gal=cfg.get("data.n_gal"),
                               shape_noise=cfg.get("data.shape_noise"),
                               kappa_max=cfg.get("data.kappa_max"),
                               reduced_shear=cfg.get("data.reduced_shear"),
                               rmax_arcmin=cfg.get("data.rmax"), seed=1)
        return (cat["x"], cat["y"], cat["g1"], cat["g2"], cat["kappa_true"],
                cat.get("center", (0.0, 0.0)))
    raise ValueError(f"unknown data.source={src!r}")


def _apply_mask(cfg, nodes, g1n, g2n, weight):
    r = cfg.get("data.mask_radius")
    if not r:
        return
    cx, cy = cfg.get("data.mask_center")
    hole = np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy) < r
    g1n[hole] = 0.0; g2n[hole] = 0.0; weight[hole] = 0.0


# --------------------------------------------------------------------------- #
# prior
# --------------------------------------------------------------------------- #
def build_prior(cfg, ops):
    """Return a Prior object, or None for the default Wiener prior (which the
    reconstructor/sampler parameterise directly via wiener_length)."""
    kind = cfg.get("prior.kind")
    if kind in (None, "wiener"):
        return None
    from .priors import make_prior
    kw = cfg.section("prior").get(kind, {})
    return make_prior(kind, ops, **kw)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(cfg, verbose=True):
    """Execute the whole pipeline described by `cfg`. Returns a result dict."""
    d = build_forward_and_data(cfg)
    ops, g1n, g2n, weight = d["ops"], d["g1n"], d["g2n"], d["weight"]
    prior = build_prior(cfg, ops)
    lam = cfg.get("inverse.lam")
    fwd = DifferentiableForward(ops, lam_reg=(1e-2 if lam is None else lam))

    method = cfg.get("inverse.method")
    if method == "map":
        from .regularization import estimate_noise_level
        noise_std = None
        if cfg.get("inverse.morozov") and prior is None and lam is None:
            # Morozov applies to the quadratic (Wiener) prior only.
            if cfg.get("inverse.noise_source") == "bmode":
                rec0 = MAPReconstructor(fwd, wiener_length=cfg.get("inverse.wiener_length"),
                                        data_weight=weight, callback_every=0)
                noise_std = rec0.estimate_noise_bmode(g1n, g2n,
                                                      maxiter=min(200, cfg.get("inverse.maxiter")))
            else:
                gsel = weight > 0
                noise_std = estimate_noise_level(np.concatenate([g1n[gsel], g2n[gsel]]), method="mad")
        rec = MAPReconstructor(fwd, maxiter=cfg.get("inverse.maxiter"), callback_every=0,
                               wiener_length=(cfg.get("inverse.wiener_length") if prior is None else 0.0),
                               noise_std=noise_std, data_weight=weight, prior=prior)
        kappa_nodes, res = rec.reconstruct(g1n, g2n, verbose=verbose)
        result = dict(kappa=d["to_galaxies"](kappa_nodes), kappa_nodes=kappa_nodes,
                      std=None, converged=res.converged)
    elif method == "sample":
        from .sampling import sample_posterior
        s = cfg.section("sampler")
        ps = sample_posterior(
            fwd, g1n, g2n, noise_std=d["noise_std"], prior=prior,
            wiener_length=cfg.get("inverse.wiener_length"), data_weight=weight,
            lam=(None if lam is None else lam), method=s.get("method", "auto"),
            n_samples=s.get("n_samples"), n_levels=s.get("n_levels"),
            steps_per_level=s.get("steps_per_level"), n_chains=s.get("n_chains"),
            n_leapfrog=s.get("n_leapfrog"), keep_final=s.get("keep_final"),
            sigma_max=s.get("sigma_max"), sigma_min=s.get("sigma_min"),
            n_delta_logp=s.get("n_delta_logp"), seed=s.get("seed"), verbose=verbose)
        # Point estimate: for exact RTO the posterior mean IS the (smooth) MAP, so
        # report that instead of the DC-noisy empirical mean; for annealed HMC use
        # the sample mean. The std map is always from the samples.
        point = ps.map_kappa if ps.method == "rto" else ps.mean
        result = dict(kappa=d["to_galaxies"](point), kappa_nodes=point,
                      std=d["to_galaxies"](ps.std), std_nodes=ps.std,
                      samples=ps.samples, sampler_method=ps.method, info=ps.info)
    else:
        raise ValueError(f"inverse.method must be 'map' or 'sample', got {method!r}")

    result.update(truth=d["to_galaxies"](d["truth_nodes"]), nodes=d["nodes"],
                  truth_nodes=d["truth_nodes"])
    _report(cfg, result, verbose)
    _save(cfg, result)
    return result


def _report(cfg, result, verbose):
    if not verbose:
        return
    t = result.get("truth_nodes")
    k = result["kappa_nodes"]
    if t is not None and np.any(np.isfinite(t)):
        m = np.isfinite(t) & np.isfinite(k)
        if m.sum():
            l2 = np.linalg.norm(k[m] - t[m]) / (np.linalg.norm(t[m]) + 1e-30)
            print(f"  relative L2 vs truth = {l2:.3f}")
    if result.get("std_nodes") is not None:
        print(f"  mean posterior std = {np.nanmean(result['std_nodes']):.3e}")


def _resolve_out_dir(cfg):
    """Return a writable output directory (absolute path).

    output.dir is resolved against the current working directory. If it can't be
    created (e.g. a SLURM job whose cwd landed in an unwritable spool dir), fall
    back to ./femmi_outputs under the cwd, then to the system temp dir, so a long
    compute run never dies at the final write.
    """
    want = os.path.abspath(cfg.get("output.dir"))
    for cand in (want, os.path.abspath("femmi_outputs"),
                 os.path.join(tempfile.gettempdir(), "femmi_outputs")):
        try:
            os.makedirs(cand, exist_ok=True)
            if os.access(cand, os.W_OK):
                if cand != want:
                    print(f"  [warn] cannot write to {want!r}; using {cand!r} instead")
                return cand
        except OSError:
            continue
    raise RuntimeError(f"no writable output directory (tried {want!r} and fallbacks)")


def _save(cfg, result):
    if not cfg.get("output.save_kappa"):
        return
    out_dir = _resolve_out_dir(cfg)
    path = os.path.join(out_dir, cfg.get("output.name") + ".npz")
    payload = dict(kappa=result["kappa"], nodes=result["nodes"])
    if result.get("std") is not None:
        payload["std"] = result["std"]
    if result.get("truth") is not None:
        payload["truth"] = result["truth"]
    np.savez(path, **payload)
    print(f"  saved {os.path.normpath(path)}")
