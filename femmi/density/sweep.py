"""Multi-seed runner dispatch."""
from .sampling import n_gal_for_density
from .runners import *
def density_sweep(n_effs=(5.0, 10.0, 20.0, 30.0), noise_std=0.05, seeds=(0, 1, 2),
                  radius=3.0, methods=("argyris", "p3", "ks"), verbose=True,
                  truth="nfw", truth_kw=None, catalog_kw=None, **run_kw):
    """Legacy multi-seed dispatch; use calibrated_comparison.py for publication.

    Nominal density determines source count per gross area. Every arm receives
    the same selected weighted catalogue. Selection rules remain explicit.
    """
    runners = dict(argyris=argyris_catalog_run, hct=hct_catalog_run,
                   p3=p3_catalog_run, ks=ks_catalog_run)
    rows = []
    for seed in seeds:
        for n_eff in n_effs:
            n_gal = n_gal_for_density(n_eff, radius)
            for m in methods:
                if verbose:
                    print(f"  seed {seed}  n_eff={n_eff:5.1f} gal/arcmin^2 "
                          f"({n_gal:5d} gal)  {m} ...", flush=True)
                try:
                    r = runners[m](n_eff, noise_std=noise_std, seed=seed,
                                   radius=radius, truth=truth,
                                   truth_kw=truth_kw, catalog_kw=catalog_kw,
                                   **run_kw)
                except Exception as exc:
                    r = dict(method=m, n_eff=float(n_eff),
                             n_eff_nominal=float(n_eff), n_gal=n_gal,
                             radius_arcmin=float(radius),
                             error=f"{type(exc).__name__}: {exc}")
                r["seed"] = int(seed)
                r["truth"] = str(truth)
                rows.append(r)
    return rows
